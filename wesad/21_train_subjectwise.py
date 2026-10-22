"""
21_train_subjectwise.py

WHY THIS FILE EXISTS
    The repository's own scripts/train_classifier.py gave 94.4% on real WESAD,
    above the 88.4% the paper reports. That number cannot be taken at face
    value, for three separate reasons. This script measures how much each one
    is worth.

    It does NOT reimplement their pipeline. It imports their config loader,
    their dataset builder, their model classes and their evaluate function.
    The only things done differently are listed below, and each one is marked.

THE THREE PROBLEMS WITH THE 94.4%

    1. The split is random, not by person.
       train_classifier.py uses random_split, so the same subject appears in
       both training and validation. The model can recognise the person
       instead of the stress.

    2. Windows overlap, and are then split randomly.
       config.yaml uses a 60 second window with a 30 second stride, so
       neighbouring windows share half their data. Splitting those randomly
       puts almost-identical windows on both sides. This is a stronger leak
       than it sounds: it is close to putting the same window in train and
       test.

    3. There is no test set at all.
       The best epoch is chosen on the validation set, and then the same
       validation set is reported as the result.

    A fourth, separate issue, measured earlier: Task predicts the stress label
    with 100% accuracy on held-out subjects, because the label comes from one
    questionnaire per session and Task IS the session. Task is one of the
    three context columns this config feeds to the gate. Use --context off to
    measure without it.

A BUG IN THEIR CODE, WORTH REPORTING

    train_classifier.py keeps the best epoch like this:

        best_state = {"classifier": model.classifier.state_dict(), ...}

    state_dict() hands back the live tensors, not copies. So as training
    continues, best_state changes with the model, and by the end it holds the
    final weights. Restoring it does nothing.

    You can see it in their own output: the best macro F1 was 0.9626 at epoch
    19, but the reported number was 0.9418, which is exactly epoch 20. This
    script clones the tensors, so the best epoch is genuinely kept.

WHAT THIS SCRIPT CHANGES, AND NOTHING ELSE
    C1  split by subject instead of at random
    C2  a real held-out test set: choose the epoch on validation, report on test
    C3  clone the weights when keeping the best epoch, so it actually works
    C4  report the majority-class baseline, balanced accuracy and a confusion
        matrix, none of which their script prints

    Same encoder, same frozen weights, same hyperparameters, same data.

HOW TO RUN
    From the AdaStress_BTP folder, after pretrain_encoder.py has run:

        python3 btp/21_train_subjectwise.py --config configs/btp_13ch.yaml \
            --run-id L02_subjectwise_context

        python3 btp/21_train_subjectwise.py --config configs/btp_13ch.yaml \
            --run-id L03_subjectwise_nocontext --context off

        python3 btp/21_train_subjectwise.py --config configs/btp_13ch.yaml \
            --run-id L04_subjectwise_nooverlap --stride 60

    L04 uses a 60 second stride so windows do not overlap at all. Comparing
    L02 with L04 shows how much of the score came from overlapping windows.
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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix,
                             classification_report)
import joblib

# Import the repository's own code. Nothing here is reimplemented.
VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.config import load_config                    # noqa: E402
from adastress.datasets import (LabelledWindowDataset,       # noqa: E402
                                prepare_labelled_dataset)
from adastress.models import (ContextGateFusion, MLPClassifier,   # noqa: E402
                              StressModel, load_encoder)
from adastress.train_classifier import evaluate             # noqa: E402
from adastress.utils import set_seed                        # noqa: E402

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

# The same split used for every earlier experiment, so results stay
# comparable. Nine subjects to train on, three to choose the epoch, three
# held back and never looked at until the end.
#
# S9 is in the training group on purpose: it has no MediumStress rows at all,
# so putting it in the test group would make the test set unscoreable.
# The 15 subjects in five groups of three.
GROUPS = {
    "G1": ["S2", "S3", "S4"],
    "G2": ["S5", "S6", "S7"],
    "G3": ["S8", "S9", "S10"],
    "G4": ["S11", "S13", "S14"],
    "G5": ["S15", "S16", "S17"],
}

# Five rotations, so every subject is tested exactly once.
#
# Why this matters. In rotation A the test people were S15, S16 and S17, and
# accuracy came out 0.867 while validation stayed between 0.24 and 0.46 for
# all 20 epochs. Validation worse than test should not happen.
#
# The reason is how the label is built: each subject's five questionnaire
# scores are cut into three groups separately for that person. So S11's calm
# baseline session is labelled Medium and S14's is High, while the training
# people's baseline is Low. S15, S16 and S17 happen to share the training
# people's mapping, so the model transfers to them and fails on S11 and S14.
#
# If 0.867 is a real result it survives rotation. If it only happens when the
# held-out people's mapping matches the training group, it will fall sharply
# as soon as S11 or S14 is in the test set.
ROTATIONS = {
    "A": {"test": "G5", "val": "G4"},   # the split used so far
    "B": {"test": "G4", "val": "G5"},   # S11, S13, S14 tested
    "C": {"test": "G3", "val": "G4"},   # S9 is here, it has no Medium rows
    "D": {"test": "G2", "val": "G3"},
    "E": {"test": "G1", "val": "G2"},
}


def make_split(rotation):
    """Build train / val / test subject lists for one rotation."""
    te = ROTATIONS[rotation]["test"]
    va = ROTATIONS[rotation]["val"]
    tr = [g for g in GROUPS if g not in (te, va)]
    return {
        "train": [s for g in tr for s in GROUPS[g]],
        "val": list(GROUPS[va]),
        "test": list(GROUPS[te]),
    }


def run_one(args, rotation, split, quiet=False):
    """Train and score for one rotation. Returns a dict of results."""
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cpu")

    run_id = args.run_id if rotation == args.rotation else \
        "%s_rot%s" % (args.run_id, rotation)
    run_dir = os.path.join("runs", run_id)
    if os.path.exists(run_dir):
        raise SystemExit("Run folder already exists: %s. Use a new --run-id."
                         % run_dir)
    os.makedirs(run_dir)

    window = cfg.window.window_sec * cfg.window.fs
    stride = args.stride if args.stride else cfg.window.stride_sec * cfg.window.fs
    use_context = (args.context == "on") and cfg.classifier.use_context_gate

    # Which context columns actually get fed in.
    #
    # The config lists Task, SessionPhase and Activity. Task on its own
    # predicts the stress label with 100% accuracy on held-out people,
    # because the label comes from one questionnaire per session and Task IS
    # the session. So "context on" versus "context off" does not separate the
    # two questions we care about:
    #
    #   how much does the paper's context help?
    #   how much of that is only Task?
    #
    # --context-cols answers the second by dropping Task and keeping the rest.
    if use_context:
        if args.context_cols:
            wanted = [c.strip() for c in args.context_cols.split(",")
                      if c.strip()]
            unknown = [c for c in wanted if c not in cfg.data.context_columns]
            if unknown:
                raise SystemExit(
                    "Not in the config's context_columns: %s\nAvailable: %s"
                    % (unknown, cfg.data.context_columns))
            context_cols = wanted
        else:
            context_cols = list(cfg.data.context_columns)
    else:
        context_cols = None

    print("=" * 64)
    print("RUN       : %s" % run_id)
    print("rotation  : %s   test=%s  val=%s"
          % (rotation, split["test"], split["val"]))
    print("window    : %d rows, stride %d%s"
          % (window, stride, "  (no overlap)" if stride >= window else
             "  (overlap %d rows)" % (window - stride)))
    print("context   : %s" % (", ".join(context_cols) if context_cols
                              else "none"))
    if context_cols and "Task" not in context_cols:
        print("            Task deliberately left out. On its own it")
        print("            predicts the label perfectly.")
    print("=" * 64)

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=context_cols,
        scaler=scaler,
        window_size=window,
        stride=stride,
        subject_column=cfg.data.subject_column,
    )

    has_context = C is not None
    context_dim = C.shape[1] if has_context else 0

    subjects = np.array(subjects)
    m_tr = np.isin(subjects, split["train"])
    m_va = np.isin(subjects, split["val"])
    m_te = np.isin(subjects, split["test"])

    print("  train %d / val %d / test %d windows"
          % (m_tr.sum(), m_va.sum(), m_te.sum()))

    vals, cnts = np.unique(y[m_tr], return_counts=True)
    majority = int(vals[cnts.argmax()])
    baseline = float((y[m_te] == majority).mean())

    dataset = LabelledWindowDataset(X, y, C)
    tr_loader = DataLoader(Subset(dataset, np.where(m_tr)[0]),
                           batch_size=cfg.classifier.batch_size, shuffle=True)
    va_loader = DataLoader(Subset(dataset, np.where(m_va)[0]),
                           batch_size=cfg.classifier.batch_size)
    te_loader = DataLoader(Subset(dataset, np.where(m_te)[0]),
                           batch_size=cfg.classifier.batch_size)

    encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1],
                           cfg.ssl.lstm_hidden, device)
    gate = (ContextGateFusion(encoder.output_dim, context_dim).to(device)
            if (use_context and context_dim > 0) else None)
    classifier = MLPClassifier(encoder.output_dim, len(classes),
                               dropout=cfg.classifier.dropout).to(device)
    model = StressModel(encoder, classifier, gate).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.adaptable_parameters(), lr=cfg.classifier.lr)

    best_f1, best_epoch, best_state = -1.0, 0, None
    for epoch in range(cfg.classifier.epochs):
        model.classifier.train()
        if model.gate is not None:
            model.gate.train()
        for batch in tr_loader:
            if has_context:
                xb, cb, yb = batch
                cb = cb.to(device)
            else:
                xb, yb = batch
                cb = None
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb, cb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        va_acc, va_f1 = evaluate(model, va_loader, device, has_context)
        if not quiet:
            print("  epoch %2d: val_acc=%.4f val_macro_f1=%.4f"
                  % (epoch + 1, va_acc, va_f1))
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch + 1
            best_state = {
                "classifier": {k: v.clone() for k, v in
                               model.classifier.state_dict().items()},
                "gate": ({k: v.clone() for k, v in
                          model.gate.state_dict().items()}
                         if model.gate is not None else None),
            }

    model.classifier.load_state_dict(best_state["classifier"])
    if model.gate is not None and best_state["gate"] is not None:
        model.gate.load_state_dict(best_state["gate"])

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in te_loader:
            if has_context:
                xb, cb, yb = batch
                cb = cb.to(device)
            else:
                xb, yb = batch
                cb = None
            preds.append(model(xb.to(device), cb).argmax(1).cpu().numpy())
            trues.append(yb.numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(trues)

    acc = float(accuracy_score(true, pred))
    bal = float(balanced_accuracy_score(true, pred))
    mf1 = float(f1_score(true, pred, average="macro", zero_division=0))
    cm = confusion_matrix(true, pred, labels=[0, 1, 2]).tolist()

    print("  RESULT test=%s  acc=%.4f  balanced=%.4f  macroF1=%.4f  "
          "(guessing %.4f)  best val f1=%.4f at epoch %d"
          % (split["test"], acc, bal, mf1, baseline, best_f1, best_epoch))

    te_subj = subjects[m_te]
    per_subject = {}
    for sid in sorted(set(te_subj)):
        sel = te_subj == sid
        per_subject[sid] = float((pred[sel] == true[sel]).mean())
    print("  per person: %s"
          % {k: round(v, 3) for k, v in per_subject.items()})

    if not quiet:
        print()
        print("  confusion matrix, rows are the truth:")
        print("    %-18s %9s %9s %9s"
              % ("", "pred Low", "pred Med", "pred High"))
        for i, name in enumerate(LABEL_NAMES):
            print("    %-18s %9d %9d %9d"
                  % ("true " + name, cm[i][0], cm[i][1], cm[i][2]))
        print()
        print(classification_report(true, pred, labels=[0, 1, 2],
                                    target_names=LABEL_NAMES,
                                    zero_division=0))

    torch.save({"classifier": model.classifier.state_dict(),
                "gate": model.gate.state_dict() if model.gate else None,
                "classes": classes, "context_dim": context_dim,
                "input_channels": X.shape[1]},
               os.path.join(run_dir, "classifier.pth"))

    result = {
        "run_id": run_id, "rotation": rotation, "config": args.config,
        "split": "subject_held_out", "context": args.context,
        "context_columns": context_cols,
        "window": window, "stride": stride,
        "windows_overlap": bool(stride < window),
        "subjects_train": split["train"], "subjects_val": split["val"],
        "subjects_test": split["test"],
        "n_train": int(m_tr.sum()), "n_val": int(m_va.sum()),
        "n_test": int(m_te.sum()),
        "majority_baseline_accuracy": baseline,
        "accuracy": acc, "balanced_accuracy": bal, "macro_f1": mf1,
        "best_epoch_on_val": best_epoch, "best_val_macro_f1": float(best_f1),
        "confusion_matrix": cm, "per_subject_test_accuracy": per_subject,
        "seed": args.seed,
    }
    json.dump(result, open(os.path.join(run_dir, "metrics.json"), "w"),
              indent=2)
    print("  saved to %s/" % run_dir)
    print()
    return result


def run_all(args):
    """Run all five rotations, so every subject is tested exactly once."""
    results = []
    for rot in ["A", "B", "C", "D", "E"]:
        results.append(run_one(args, rot, make_split(rot), quiet=True))

    print("=" * 72)
    label = args.context_cols if args.context_cols else args.context
    print("ALL FIVE ROTATIONS   context=%s" % label)
    print("=" * 72)
    print("%-4s %-22s %9s %10s %9s %10s"
          % ("rot", "tested on", "accuracy", "balanced", "macroF1", "guessing"))
    for r in results:
        print("%-4s %-22s %9.4f %10.4f %9.4f %10.4f"
              % (r["rotation"], ", ".join(r["subjects_test"]),
                 r["accuracy"], r["balanced_accuracy"], r["macro_f1"],
                 r["majority_baseline_accuracy"]))

    accs = [r["accuracy"] for r in results]
    bals = [r["balanced_accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    bases = [r["majority_baseline_accuracy"] for r in results]
    print("-" * 72)
    print("%-27s %9.4f %10.4f %9.4f %10.4f"
          % ("MEAN", np.mean(accs), np.mean(bals), np.mean(f1s),
             np.mean(bases)))
    print("%-27s %9.4f %10.4f %9.4f"
          % ("SPREAD (best - worst)", max(accs) - min(accs),
             max(bals) - min(bals), max(f1s) - min(f1s)))
    print()
    print("accuracy for every subject, each tested exactly once:")
    every = {}
    for r in results:
        every.update(r["per_subject_test_accuracy"])
    for sid in sorted(every, key=lambda x: int(x[1:])):
        print("  %-5s %.4f" % (sid, every[sid]))
    print()
    print("How to read the spread. A single rotation tells you very little if")
    print("the spread is large: it would mean the score depends on which")
    print("people happened to be held out, not on the model. The label is cut")
    print("into three groups separately for each person, so a held-out person")
    print("whose mapping matches the training group scores well and one whose")
    print("mapping differs scores badly.")
    print()

    out = os.path.join("runs", "%s_rotation_summary.json" % args.run_id)
    json.dump({"context": args.context,
               "context_columns": args.context_cols,
               "mean_accuracy": float(np.mean(accs)),
               "mean_balanced_accuracy": float(np.mean(bals)),
               "mean_macro_f1": float(np.mean(f1s)),
               "mean_baseline": float(np.mean(bases)),
               "accuracy_spread": float(max(accs) - min(accs)),
               "per_rotation": results,
               "per_subject": every},
              open(out, "w"), indent=2)
    print("summary written to %s" % out)


def make_pooled_masks(subjects, n, args, seed):
    """Split every window into train / validation / test in one pile.

    This is what the released train_classifier.py does: it ignores who the
    window came from. The same person can land in train and in test. We were
    asked to use this, on the grounds that the same person at a different time
    is a different reading.

    Two ways of doing it:

      random   every window thrown in at random. Exactly the instruction.

      blocked  the same ratios, but neighbouring windows move together. With a
               60 second window and a 30 second stride, window i and window
               i+1 share 30 seconds of the SAME samples. A plain random split
               can therefore put two windows that are half-identical on
               opposite sides, which is not "a different time", it is the same
               data twice. Blocked keeps runs of neighbouring windows on one
               side so that only the block edges touch.
    """
    tr_pct, va_pct, te_pct = [float(x) for x in args.ratios.split("/")]
    total = tr_pct + va_pct + te_pct
    tr_pct, va_pct = tr_pct / total, va_pct / total

    rng = np.random.default_rng(seed)

    if args.pool_mode == "random":
        order = rng.permutation(n)
    else:
        # build contiguous blocks inside each person, then shuffle the blocks
        blocks = []
        start = 0
        for subj in pd.unique(pd.Series(subjects)):
            idx = np.where(subjects == subj)[0]
            for i in range(0, len(idx), args.block_size):
                blocks.append(idx[i:i + args.block_size])
        rng.shuffle(blocks)
        order = np.concatenate(blocks)

    n_tr = int(round(tr_pct * n))
    n_va = int(round(va_pct * n))

    m_tr = np.zeros(n, dtype=bool)
    m_va = np.zeros(n, dtype=bool)
    m_te = np.zeros(n, dtype=bool)
    m_tr[order[:n_tr]] = True
    m_va[order[n_tr:n_tr + n_va]] = True
    m_te[order[n_tr + n_va:]] = True
    return m_tr, m_va, m_te


def run_pooled(args):
    """Train with everyone pooled and split at random, 70/10/20 by default."""
    cfg = load_config(args.config)
    device = torch.device("cpu")

    window = cfg.window.window_sec * cfg.window.fs
    stride = args.stride if args.stride else cfg.window.stride_sec * cfg.window.fs
    use_context = (args.context == "on") and cfg.classifier.use_context_gate

    if use_context:
        if args.context_cols:
            context_cols = [c.strip() for c in args.context_cols.split(",")
                            if c.strip()]
        else:
            context_cols = list(cfg.data.context_columns)
    else:
        context_cols = None

    print("=" * 68)
    print("RUN        : %s" % args.run_id)
    print("split      : pooled, %s, %s" % (args.ratios, args.pool_mode))
    print("             everyone in one pile; the same person can appear in")
    print("             train and test, as the released code does")
    if args.pool_mode == "blocked":
        print("             neighbouring windows kept together, %d at a time"
              % args.block_size)
    print("window     : %d rows, stride %d%s"
          % (window, stride,
             "  (overlap %d rows)" % (window - stride) if stride < window
             else "  (no overlap)"))
    print("context    : %s" % (", ".join(context_cols) if context_cols
                               else "none"))
    print("channels   : %d" % len(cfg.data.signal_columns))
    print("seeds      : %d" % args.seeds)
    print("=" * 68)
    print()

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=context_cols,
        scaler=scaler,
        window_size=window,
        stride=stride,
        subject_column=cfg.data.subject_column,
    )
    has_context = C is not None
    context_dim = C.shape[1] if has_context else 0
    subjects = np.array(subjects)
    n = len(y)

    print("windows: %s, classes: %s, context width: %d" % (X.shape, classes,
                                                           context_dim))
    print()

    dataset = LabelledWindowDataset(X, y, C)
    results = []

    for si in range(args.seeds):
        seed = args.seed + si
        set_seed(seed)

        m_tr, m_va, m_te = make_pooled_masks(subjects, n, args, seed)

        vals, cnts = np.unique(y[m_tr], return_counts=True)
        baseline = float((y[m_te] == int(vals[cnts.argmax()])).mean())

        tr_loader = DataLoader(Subset(dataset, np.where(m_tr)[0]),
                               batch_size=cfg.classifier.batch_size,
                               shuffle=True)
        va_loader = DataLoader(Subset(dataset, np.where(m_va)[0]),
                               batch_size=cfg.classifier.batch_size)
        te_loader = DataLoader(Subset(dataset, np.where(m_te)[0]),
                               batch_size=cfg.classifier.batch_size)

        encoder = load_encoder(cfg.paths.encoder_ckpt, X.shape[1],
                               cfg.ssl.lstm_hidden, device)
        gate = (ContextGateFusion(encoder.output_dim, context_dim).to(device)
                if (use_context and context_dim > 0) else None)
        classifier = MLPClassifier(encoder.output_dim, len(classes),
                                   dropout=cfg.classifier.dropout).to(device)
        model = StressModel(encoder, classifier, gate).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.adaptable_parameters(),
                               lr=cfg.classifier.lr)

        best_f1, best_epoch, best_state = -1.0, 0, None
        for epoch in range(cfg.classifier.epochs):
            model.classifier.train()
            if model.gate is not None:
                model.gate.train()
            for batch in tr_loader:
                if has_context:
                    xb, cb, yb = batch
                    cb = cb.to(device)
                else:
                    xb, yb = batch
                    cb = None
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb, cb), yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            va_acc, va_f1 = evaluate(model, va_loader, device, has_context)

            # clone, so the best epoch is genuinely kept. The released code
            # stores state_dict() directly, which hands back the live tensors,
            # so its saved copy changes as training continues.
            if va_f1 > best_f1:
                best_f1, best_epoch = va_f1, epoch + 1
                best_state = {
                    "classifier": {k: v.clone() for k, v in
                                   model.classifier.state_dict().items()},
                    "gate": ({k: v.clone() for k, v in
                              model.gate.state_dict().items()}
                             if model.gate is not None else None),
                }

        model.classifier.load_state_dict(best_state["classifier"])
        if model.gate is not None and best_state["gate"] is not None:
            model.gate.load_state_dict(best_state["gate"])

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in te_loader:
                if has_context:
                    xb, cb, yb = batch
                    cb = cb.to(device)
                else:
                    xb, yb = batch
                    cb = None
                preds.append(model(xb.to(device), cb).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)

        acc = float(accuracy_score(true, pred))
        bal = float(balanced_accuracy_score(true, pred))
        mf1 = float(f1_score(true, pred, average="macro", zero_division=0))

        # Keep everything needed to use this model on new data later. The
        # released code saves only the classifier, which is not enough: the
        # gate, the class order, the channel count and the context width are
        # all needed too, or the weights will load into the wrong shape.
        last_model = {
            "classifier": {k: v.clone() for k, v in
                           model.classifier.state_dict().items()},
            "gate": ({k: v.clone() for k, v in model.gate.state_dict().items()}
                     if model.gate is not None else None),
            "classes": classes,
            "context_dim": context_dim,
            "context_columns": context_cols,
            "input_channels": int(X.shape[1]),
            "signal_columns": list(cfg.data.signal_columns),
            "encoder_ckpt": cfg.paths.encoder_ckpt,
            "scaler_ckpt": cfg.paths.scaler_ckpt,
            "lstm_hidden": cfg.ssl.lstm_hidden,
            "window": window,
            "stride": stride,
            "seed": seed,
        }

        print("seed %d: train %d / val %d / test %d   "
              "acc=%.4f  balanced=%.4f  macroF1=%.4f  (guessing %.4f)  "
              "best val f1=%.4f at epoch %d"
              % (seed, m_tr.sum(), m_va.sum(), m_te.sum(), acc, bal, mf1,
                 baseline, best_f1, best_epoch))

        results.append({
            "seed": seed, "accuracy": acc, "balanced_accuracy": bal,
            "macro_f1": mf1, "majority_baseline_accuracy": baseline,
            "best_epoch_on_val": best_epoch,
            "best_val_macro_f1": float(best_f1),
            "n_train": int(m_tr.sum()), "n_val": int(m_va.sum()),
            "n_test": int(m_te.sum()),
            "confusion_matrix": confusion_matrix(
                true, pred, labels=list(range(len(classes)))).tolist(),
        })

    accs = [r["accuracy"] for r in results]
    bals = [r["balanced_accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    bases = [r["majority_baseline_accuracy"] for r in results]

    print()
    print("=" * 68)
    print("%-26s %9s %10s %9s %10s"
          % ("", "accuracy", "balanced", "macroF1", "guessing"))
    print("%-26s %9.4f %10.4f %9.4f %10.4f"
          % ("MEAN over %d seed(s)" % args.seeds, np.mean(accs), np.mean(bals),
             np.mean(f1s), np.mean(bases)))
    if args.seeds > 1:
        print("%-26s %9.4f %10.4f %9.4f"
              % ("SPREAD (best - worst)", max(accs) - min(accs),
                 max(bals) - min(bals), max(f1s) - min(f1s)))
    print("=" * 68)
    print()

    run_dir = os.path.join("runs", args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # The model from the last seed, so this run can be applied to the user
    # data without retraining. Everything the loader needs is in one file.
    torch.save(last_model, os.path.join(run_dir, "model.pth"))
    print("saved %s/model.pth  (last seed, %d channels, context width %d)"
          % (run_dir, last_model["input_channels"], last_model["context_dim"]))

    json.dump({
        "run_id": args.run_id, "config": args.config,
        "split": "pooled", "ratios": args.ratios,
        "pool_mode": args.pool_mode, "block_size": args.block_size,
        "context": args.context, "context_columns": context_cols,
        "window": window, "stride": stride,
        "n_channels": len(cfg.data.signal_columns),
        "seeds": args.seeds,
        "mean_accuracy": float(np.mean(accs)),
        "mean_balanced_accuracy": float(np.mean(bals)),
        "mean_macro_f1": float(np.mean(f1s)),
        "mean_baseline": float(np.mean(bases)),
        "accuracy_spread": float(max(accs) - min(accs)),
        "per_seed": results,
    }, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print("saved %s/metrics.json" % run_dir)


def main():
    ap = argparse.ArgumentParser(
        description="Train the repo's classifier with a split by person "
                    "instead of at random.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--context", choices=["on", "off"], default="on",
                    help="on uses whatever context_columns the config lists. "
                         "off turns the gate off completely.")
    ap.add_argument("--context-cols", default=None,
                    help="comma separated list of context columns to use, "
                         "instead of all the ones in the config. This is how "
                         "you drop Task on its own while keeping the rest, "
                         "e.g. --context-cols SessionPhase,Activity")
    ap.add_argument("--stride", type=int, default=None,
                    help="override the stride. 60 means windows do not "
                         "overlap at all")
    # ---- how the data is split ----
    #
    # subject : hold three people out completely. Nobody appears in two sets.
    #
    # pooled  : put every window from every person in one pile and split that
    #           at random, which is what the repository's own
    #           train_classifier.py does (random_split, 80/20, subject ignored)
    #           and what we were asked to use. The same person can appear in
    #           train and test. The reasoning is that this is time-series data,
    #           so the same person at a different time is a different reading.
    #
    #           Note the paper text says the opposite (Section VII-A-1: "tested
    #           on held-out subjects"), so pooled follows the released code
    #           rather than the paper.
    ap.add_argument("--split", choices=["subject", "pooled"], default="subject",
                    help="subject: hold people out. pooled: one pile, split "
                         "at random, as the released code does.")
    ap.add_argument("--ratios", default="70/10/20",
                    help="train/val/test percentages for --split pooled.")
    ap.add_argument("--pool-mode", choices=["random", "blocked"],
                    default="random",
                    help="random: plain random, as asked. blocked: same "
                         "ratios, but windows that share samples stay on the "
                         "same side. With a 60s window and 30s stride "
                         "neighbouring windows share 30 seconds of identical "
                         "samples, so a plain random split can put nearly the "
                         "same window in train and test.")
    ap.add_argument("--block-size", type=int, default=10,
                    help="how many neighbouring windows move together when "
                         "--pool-mode blocked.")
    ap.add_argument("--seeds", type=int, default=1,
                    help="how many seeds to run and average, as the paper "
                         "does with five.")
    ap.add_argument("--rotation", default="A",
                    choices=["A", "B", "C", "D", "E", "all"],
                    help="which group of people is held out for testing. "
                         "'all' runs all five, so every subject is tested "
                         "once, and prints a summary at the end.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.split == "pooled":
        run_pooled(args)
        return

    if args.rotation == "all":
        run_all(args)
        return

    split = make_split(args.rotation)
    run_one(args, args.rotation, split)


if __name__ == "__main__":
    main()
