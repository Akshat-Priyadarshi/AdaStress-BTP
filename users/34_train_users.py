"""
34_train_users.py

WHAT THIS IS
    Stage 2 of AdaStress, run on the five users who have survey answers.

    The encoder was trained once on WESAD with no labels and then frozen.
    This trains only what sits on top of it: the context gate and the
    classifier. That is the repository's own design, and the paper says the
    same thing (Section IV-B): "only the gating and classification parameters
    are updated, while the pretrained encoder remains fixed".

WHOSE CODE IS WHOSE
    From the repository, imported and unchanged:
        adastress.models.load_encoder        the frozen encoder
        adastress.models.ContextGateFusion   the gate
        adastress.models.MLPClassifier       the classifier
        adastress.models.StressModel         the three joined together
        adastress.config.load_config
        adastress.utils.set_seed

    Ours:
        reading the labelled windows and matching them to the 1 Hz signal
        leave-one-user-out, because the repository splits at random with no
        test set at all

THE TWO PIECES EACH WINDOW NEEDS
    signal   60 seconds of the six channels EmbracePlus and WESAD both
             record, from data/processed/user_6ch/<user>.csv. This is what
             the frozen encoder reads.

    context  one row of the twelve derived columns, from the labelled
             windows file. This is what the gate reads.

    They are matched on time. A labelled window with no signal underneath it
    is dropped.

LEAVE ONE USER OUT
    Train on four people, test on the fifth, rotate so each is tested once.

    Testing on somebody you trained on measures nothing: of course the model
    fits them, you showed it their answers. With five users this is the only
    way to know whether a number is real.

    The repository does not do this. Its train_classifier.py uses
    random_split with no held-out test set, so it reports the validation set
    it selected on.

TWO FOLDS THAT WILL LOOK STRANGE, AND ARE NOT BUGS
    user26 answered Low on all 18 occasions. Holding user26 out gives a fold
    where always answering Low scores 100 percent, and where balanced
    accuracy is the only honest number.

    user18 never answered Low at all. So a model trained on the others, which
    do contain Low, will predict Low for user18 and be wrong. This is the
    real difficulty in the data: five people used the scale differently.

    Both are reported rather than averaged away.

RUNS WORTH COMPARING
    --context off       body signals only. The honest floor.
    --context on        body plus the twelve derived columns, gate on.

    The leak check found no context column beating the guessing baseline, so
    these two may well come out close. That would itself be the finding, and
    it is better to have predicted it than to explain it afterwards.

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/34_train_users.py --config configs/w6.yaml \\
            --run-id U01_nocontext --context off

        python3 btp/34_train_users.py --config configs/w6.yaml \\
            --run-id U02_context --context on
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.config import load_config                            # noqa: E402
from adastress.models import (ContextGateFusion, MLPClassifier,     # noqa: E402
                              StressModel, load_encoder)
from adastress.utils import set_seed                                # noqa: E402

LABELLED = "data/processed/labels/labelled_windows.csv"
SIGNAL_DIR = "data/processed/user_6ch"
LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

NOT_CONTEXT = {"Subject_ID", "datetime_utc", "datetime_ist", "date",
               "response_time", "STAI6", "STAI6_RAW", "LABEL", "LABEL_NAME"}

TIME_COL = "datetime_ist"


def load_windows(labelled_path, signal_dir, signal_cols, window, context_cols,
                 suffix=""):
    """Pair each labelled window with the 60 seconds of signal beneath it.

    The labelled file says when a window is and what the person reported.
    The signal file holds the readings. They are matched on time, and a
    labelled window with no signal under it is dropped.
    """
    lab = pd.read_csv(labelled_path)
    lab[TIME_COL] = pd.to_datetime(lab[TIME_COL], errors="coerce")
    lab = lab.dropna(subset=[TIME_COL])

    Xs, Cs, ys, subs, times, resp = [], [], [], [], [], []

    for user, g in lab.groupby("Subject_ID", sort=True):
        spath = os.path.join(signal_dir, "%s%s.csv" % (user, suffix))
        if not os.path.exists(spath):
            print("  %-8s no signal file, skipped" % user)
            continue

        sig = pd.read_csv(spath)
        sig[TIME_COL] = pd.to_datetime(sig[TIME_COL], errors="coerce")
        sig = sig.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

        arr = sig[signal_cols].to_numpy(dtype=float)
        arr = pd.DataFrame(arr).interpolate(limit_direction="both") \
                .fillna(0.0).to_numpy()
        tidx = sig[TIME_COL].to_numpy()

        # where in the signal does each labelled window start
        starts = np.searchsorted(tidx, g[TIME_COL].to_numpy(), side="left")

        kept = 0
        for pos, (_i, row) in zip(starts, g.iterrows()):
            if pos + window > len(arr):
                continue
            # the window must actually be 60 contiguous seconds
            span = (tidx[pos + window - 1] - tidx[pos]) / np.timedelta64(1, "s")
            if not (window - 5 <= span <= window + 5):
                continue
            Xs.append(arr[pos:pos + window].T)
            ys.append(int(row["LABEL"]))
            subs.append(user)
            times.append(row[TIME_COL])
            # which survey answer this window belongs to. Needed so a whole
            # answer can be kept on one side of a split.
            resp.append("%s|%s" % (user, row.get("response_time", "")))
            if context_cols:
                Cs.append(row[context_cols].to_dict())
            kept += 1

        print("  %-8s %5d of %5d labelled windows have signal underneath"
              % (user, kept, len(g)))

    if not Xs:
        raise SystemExit("No labelled window could be paired with signal.")

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys)
    subs = np.asarray(subs)
    C = pd.DataFrame(Cs) if context_cols else None
    return X, y, C, subs, np.asarray(times), np.asarray(resp)


def build_context(C_train, C_all, context_cols):
    """Turn the context frame into numbers, fitted on the training users only.

    Fitting on training users only matters: fitting on everything would let
    the held-out person's own statistics influence the scaling, which is a
    quiet way of leaking.

    Returns the fitted transformers alongside the numbers. They have to be
    kept, because a saved model cannot be used on new data without them:
    there would be no way to turn "sitting" into the same column the gate
    was trained to read.
    """
    num = [c for c in context_cols
           if pd.api.types.is_numeric_dtype(C_all[c])]
    cat = [c for c in context_cols if c not in num]

    sc = oh = None
    pieces_all = []
    if num:
        sc = StandardScaler().fit(C_train[num].fillna(0.0))
        pieces_all.append(sc.transform(C_all[num].fillna(0.0)))
    if cat:
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh.fit(C_train[cat].astype(str))
        pieces_all.append(oh.transform(C_all[cat].astype(str)))

    out = np.concatenate(pieces_all, axis=1).astype(np.float32)
    return out, {"numeric_cols": num, "categorical_cols": cat,
                 "scaler": sc, "onehot": oh}


def make_folds(args, subs, resp, n, users):
    """Decide what trains and what is tested on.

    Three ways, and they answer different questions.

    user
        Hold one person out completely, rotate. Asks: does this work on
        somebody the model has never seen? That is what a deployed system
        has to do.

    random
        Pool every window from everybody and split at random, 70/10/20. This
        is what was done for WESAD, so it makes the two comparable.

        Be careful reading it. Windows overlap by 30 seconds, and each survey
        answer labels about 60 windows in a row. Splitting at random scatters
        those near-copies across both sides, so a window in the test set may
        share half its samples with one the model trained on. On WESAD this
        was worth 0.975 against 0.833 once the overlap was handled, and here
        the effect will be stronger.

    response
        Pool everybody, but keep every window belonging to one survey answer
        on the same side. The same person can still appear in train and test,
        which is what makes it comparable to random, but the same moment
        cannot. This is the honest version of a pooled split.
    """
    if args.split == "user":
        return [(u, ~(subs == u), subs == u) for u in users]

    tr_pct, va_pct, te_pct = [float(x) for x in args.ratios.split("/")]
    total = tr_pct + va_pct + te_pct
    tr_pct, va_pct = tr_pct / total, va_pct / total

    folds = []
    for si in range(args.seeds):
        rng = np.random.default_rng(args.seed + si)

        if args.split == "random":
            order = rng.permutation(n)
        else:
            # shuffle whole survey answers, then flatten
            groups = [np.where(resp == r)[0] for r in np.unique(resp)]
            rng.shuffle(groups)
            order = np.concatenate(groups)

        n_tr = int(round(tr_pct * n))
        n_va = int(round(va_pct * n))
        tr = np.zeros(n, dtype=bool)
        te = np.zeros(n, dtype=bool)
        tr[order[:n_tr]] = True
        te[order[n_tr + n_va:]] = True
        folds.append(("seed%d" % (args.seed + si), tr, te))
    return folds


def main():
    ap = argparse.ArgumentParser(
        description="AdaStress stage 2 on the user data: train the gate and "
                    "classifier, encoder stays frozen.")
    ap.add_argument("--config", required=True,
                    help="the 6-channel config, e.g. configs/w6.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--signal-suffix", default="",
                    help="added to the user name when looking for the signal "
                         "file. Default looks for <user>.csv; pass _dequant "
                         "for the Chronos-normalised files written by "
                         "btp/35_tokenise_1hz.py.")
    ap.add_argument("--no-scaler", action="store_true",
                    help="skip the WESAD scaler. Use this with the Chronos "
                         "files, which are already normalised per window: "
                         "scaling them again would undo the thing being "
                         "tested.")
    ap.add_argument("--context", choices=["on", "off"], default="on")
    ap.add_argument("--split", choices=["user", "random", "response"],
                    default="user",
                    help=("user: hold one person out at a time. "
                          "random: pool everyone and split 70/10/20 at "
                          "random, as was done for WESAD. "
                          "response: pool everyone but keep each survey "
                          "answer whole."))
    ap.add_argument("--ratios", default="70/10/20",
                    help="train/val/test, for the pooled splits")
    ap.add_argument("--seeds", type=int, default=1,
                    help="how many random draws to average, for the pooled "
                         "splits. A single draw tells you very little.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="default comes from the config")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-model", action="store_true",
                    help="write model.pth for the first fold, so it can be "
                         "used for inference or for measuring what each "
                         "context column contributes. Only the first fold "
                         "is kept: the others differ only in which rows they "
                         "trained on, and a single file is what a deployed "
                         "system would carry.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    window = cfg.window.window_sec * cfg.window.fs
    epochs = args.epochs or cfg.classifier.epochs
    signal_cols = list(cfg.data.signal_columns)

    run_dir = os.path.join("runs", args.run_id)
    if os.path.exists(run_dir):
        raise SystemExit("Run folder exists: %s. Use a new --run-id." % run_dir)
    os.makedirs(run_dir)

    # which context columns are available
    head = pd.read_csv(args.labelled, nrows=1)
    context_cols = ([c for c in head.columns if c not in NOT_CONTEXT]
                    if args.context == "on" else [])

    print("=" * 72)
    print("RUN       : %s" % args.run_id)
    print("config    : %s" % args.config)
    print("encoder   : %s  (frozen, trained on WESAD without labels)"
          % cfg.paths.encoder_ckpt)
    print("channels  : %d  %s" % (len(signal_cols), signal_cols))
    print("window    : %d rows at %d Hz" % (window, cfg.window.fs))
    print("context   : %s" % (", ".join(context_cols) if context_cols
                              else "none"))
    print("split     : %s%s" % (args.split,
                                "  %s, %d seed(s)" % (args.ratios, args.seeds)
                                if args.split != "user" else
                                "  one person held out at a time"))
    if args.split == "random":
        print("            careful: windows overlap by 30 s and each survey")
        print("            answer labels about 60 of them, so near-copies")
        print("            land on both sides. Compare against --split")
        print("            response, which keeps each answer whole.")
    print("=" * 72)
    print()

    print("pairing labelled windows with signal:")
    X, y, C, subs, times, resp = load_windows(args.labelled, args.signal_dir,
                                              signal_cols, window,
                                              context_cols,
                                              args.signal_suffix)
    users = sorted(set(subs))
    print()
    print("windows: %s   users: %s" % (X.shape, ", ".join(users)))
    print("labels :", {LABEL_NAMES[c]: int((y == c).sum())
                       for c in sorted(set(y))})
    print()

    # The repo's own scaler, fitted on WESAD, so the encoder sees the same
    # kind of numbers it was trained on.
    #
    # Skipped for the Chronos files, which are already centred and scaled
    # within each window. Scaling those again would undo the very thing the
    # experiment is testing.
    if args.no_scaler:
        print("WESAD scaler skipped: the input is already normalised per")
        print("window, which is what this run is testing.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0,
                          neginf=0.0).astype(np.float32)
    else:
        scaler = joblib.load(cfg.paths.scaler_ckpt)
        flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
        X = scaler.transform(flat).reshape(X.shape[0], X.shape[2],
                                           X.shape[1]).transpose(0, 2, 1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0,
                          neginf=0.0).astype(np.float32)
    print("input after scaling: mean %.3f, sd %.3f, range %.1f to %.1f"
          % (X.mean(), X.std(), X.min(), X.max()))
    print()

    classes = sorted(set(y))
    results = []
    saved = False

    for held, tr, te in make_folds(args, subs, resp, len(y), users):
        if te.sum() < 10 or tr.sum() < 50:
            print("fold %s skipped, too few windows" % held)
            continue

        set_seed(args.seed)

        # context fitted on the training users only
        if context_cols:
            Call, ctx_fitted = build_context(C[tr], C, context_cols)
            ctx_dim = Call.shape[1]
        else:
            Call = np.zeros((len(y), 1), dtype=np.float32)
            ctx_dim = 0
            ctx_fitted = None

        # the number this fold has to beat
        lab, cnt = np.unique(y[tr], return_counts=True)
        baseline = float((y[te] == lab[cnt.argmax()]).mean())

        tr_ds = TensorDataset(torch.tensor(X[tr]), torch.tensor(Call[tr]),
                              torch.tensor(y[tr]))
        te_ds = TensorDataset(torch.tensor(X[te]), torch.tensor(Call[te]),
                              torch.tensor(y[te]))
        tr_dl = DataLoader(tr_ds, batch_size=cfg.classifier.batch_size,
                           shuffle=True)
        te_dl = DataLoader(te_ds, batch_size=cfg.classifier.batch_size)

        encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1],
                               cfg.ssl.lstm_hidden, device)
        gate = (ContextGateFusion(encoder.output_dim, ctx_dim).to(device)
                if ctx_dim > 0 else None)
        clf = MLPClassifier(encoder.output_dim, len(classes),
                            dropout=cfg.classifier.dropout).to(device)
        model = StressModel(encoder, clf, gate).to(device)

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.adaptable_parameters(), lr=cfg.classifier.lr)

        for _ep in range(epochs):
            model.classifier.train()
            if model.gate is not None:
                model.gate.train()
            for xb, cb, yb in tr_dl:
                loss = criterion(model(xb, cb if ctx_dim > 0 else None), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        preds, trues, gates = [], [], []
        with torch.no_grad():
            for xb, cb, yb in te_dl:
                # The repository's StressModel.forward takes only (x, context)
                # and returns the logits. It exposes the gate separately
                # through gate.gate_values(h, c), which is what we use here
                # rather than changing their model.
                out = model(xb, cb if ctx_dim > 0 else None)
                if gate is not None:
                    h = model.embed(xb)
                    gates.append(gate.gate_values(h, cb).mean(dim=1).numpy())
                preds.append(out.argmax(1).numpy())
                trues.append(yb.numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)

        acc = float(accuracy_score(true, pred))
        bal = float(balanced_accuracy_score(true, pred))
        mf1 = float(f1_score(true, pred, average="macro", zero_division=0))
        gate_mean = (float(np.concatenate(gates).mean()) if gates else None)

        n_cls_test = len(set(true))
        note = ""
        if n_cls_test == 1:
            note = ("  <- only one class in this fold, read balanced accuracy")

        print("%-9s test %4d  acc %.3f  balanced %.3f  macroF1 %.3f  "
              "(guessing %.3f)%s"
              % (held, te.sum(), acc, bal, mf1, baseline, note))
        if gate_mean is not None:
            print("            gate %.3f  (near 1 means it leaned on the body,"
                  " near 0 on context)" % gate_mean)

        # Save the first fold, if asked. The bundle carries everything
        # needed to rebuild the model elsewhere: the architecture's shape,
        # the paths to the frozen encoder and its scaler, and the fitted
        # context transformers, without which the gate cannot be fed.
        #
        # Keys match what btp/21_train_subjectwise.py writes, so
        # btp/50_inference.py reads either without changes.
        if args.save_model and not saved:
            bundle = {
                "classifier": clf.state_dict(),
                "gate": gate.state_dict() if gate is not None else None,
                "classes": classes,
                "context_dim": int(ctx_dim),
                "context_columns": context_cols,
                "context_fitted": ctx_fitted,
                "input_channels": int(X.shape[1]),
                "signal_columns": signal_cols,
                "encoder_ckpt": cfg.paths.encoder_ckpt,
                "scaler_ckpt": cfg.paths.scaler_ckpt,
                "wesad_scaler_applied": not args.no_scaler,
                "lstm_hidden": cfg.ssl.lstm_hidden,
                "window": int(window),
                "stride": int(cfg.window.get("stride_sec", window // 2)
                              * cfg.window.fs),
                "seed": args.seed,
                "trained_on": "user data, %s split, fold %s"
                              % (args.split, held),
                "labelled_file": args.labelled,
                "signal_dir": args.signal_dir,
            }
            mpath = os.path.join(run_dir, "model.pth")
            torch.save(bundle, mpath)
            saved = True
            print("            saved %s  (fold %s, %d context values)"
                  % (mpath, held, ctx_dim))

        results.append({
            "fold": held, "n_test": int(te.sum()), "n_train": int(tr.sum()),
            "accuracy": acc, "balanced_accuracy": bal, "macro_f1": mf1,
            "majority_baseline": baseline, "gate_mean": gate_mean,
            "classes_in_test": n_cls_test,
            "confusion_matrix": confusion_matrix(
                true, pred, labels=classes).tolist(),
            "report": classification_report(
                true, pred, labels=classes,
                target_names=[LABEL_NAMES[c] for c in classes],
                zero_division=0, output_dict=True),
        })

    if not results:
        raise SystemExit("No fold could be run.")

    print()
    print("=" * 72)
    print("ACROSS THE FIVE FOLDS")
    print("=" * 72)
    accs = [r["accuracy"] for r in results]
    bals = [r["balanced_accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    base = [r["majority_baseline"] for r in results]

    print("  %-24s %9s %10s %9s %10s"
          % ("", "accuracy", "balanced", "macroF1", "guessing"))
    print("  %-24s %9.3f %10.3f %9.3f %10.3f"
          % ("MEAN", np.mean(accs), np.mean(bals), np.mean(f1s), np.mean(base)))
    print("  %-24s %9.3f %10.3f %9.3f"
          % ("SPREAD (best - worst)", max(accs) - min(accs),
             max(bals) - min(bals), max(f1s) - min(f1s)))
    print()

    single = [r["fold"] for r in results if r["classes_in_test"] == 1]
    if single:
        print("  %s had only one class in the test fold. Accuracy there is"
              % ", ".join(single))
        print("  whatever the majority is, so balanced accuracy is the number")
        print("  to read, and the fold should be flagged in any report.")
        print()

    print("  Balanced accuracy is the one to compare: it averages the recall")
    print("  of each class, so a model that always answers the commonest one")
    print("  scores 0.333 rather than looking respectable.")
    print()

    json.dump({"run_id": args.run_id, "config": args.config,
               "context": args.context, "context_columns": context_cols,
               "split": args.split, "ratios": args.ratios,
               "seeds": args.seeds,
               "signal_dir": args.signal_dir,
               "signal_suffix": args.signal_suffix,
               "wesad_scaler_applied": not args.no_scaler,
               "channels": signal_cols, "window": window, "epochs": epochs,
               "seed": args.seed,
               "mean_accuracy": float(np.mean(accs)),
               "mean_balanced_accuracy": float(np.mean(bals)),
               "mean_macro_f1": float(np.mean(f1s)),
               "mean_baseline": float(np.mean(base)),
               "accuracy_spread": float(max(accs) - min(accs)),
               "model_saved": bool(args.save_model and saved),
               "per_fold": results},
              open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print("saved %s/metrics.json" % run_dir)


if __name__ == "__main__":
    main()
