"""
08_infer_user1.py

WHY THIS FILE EXISTS
    user1 has no stress label in any of its files (decision D2). Two
    recording sessions would give about two questionnaire answers even if they
    were found on paper. So nothing here can be trained, and no accuracy can
    be measured.

    What can be done is apply the WESAD-trained models to user1 and compare
    how they behave. That is what this script does.

WHAT IT COMPARES
    Experiment A, runs/W_zero_seed42
        Context is the WESAD schema plus the four columns user1 has and WESAD
        does not. During WESAD training those four were always exactly zero.
        On user1 they hold real numbers for the first time. This is the direct
        test of whether zero-padding survives contact with real data.

    Experiment B, runs/W_derived_seed42
        Context is movement worked out from the accelerometer, by the same
        rule on both datasets. Nothing is padded.

WHAT IT REPORTS
    For each model: the spread of predicted classes, the stress score, the
    gate value, and how each of those differs from the same model's behaviour
    on the WESAD test subjects. A model whose gate collapses or whose scores
    pile up at one end has broken, even though there is no accuracy to prove
    it with.

TWO CHOICES THIS SCRIPT DOES NOT MAKE FOR YOU
    1. --context-scale
       On WESAD, steps, MET, ACT_COUNT and ACT_INT were always 0, next to
       one-hot columns that are 0 or 1. On user1, ACT_COUNT reaches 227.
       Feeding 227 into a slot the model only ever saw as 0 is exactly the
       derailment we are testing for.

         as_is  use the real numbers, unchanged. The naive version, and the
                one that shows the problem if there is one.
         unit   squash each of the four onto roughly 0 to 1 first, so they sit
                in the same range as the one-hot columns.

       Run both. Reporting only one would be picking an answer.

    2. --derived-scale
       Experiment B's two movement numbers were z-scored during WESAD
       training. Whose mean and standard deviation should be used on user1?

         wesad  reuse WESAD's, so the transformation is literally identical
         user1  work out user1's own, so the numbers are comparable in spread

       Both are defensible and they answer different questions. Run both.

OUTPUT
    runs/<out-id>/
        predictions.csv     one row per window: probabilities, score, gate
        metrics_user1.json  the summary numbers
    docs/31_user1_inference.md   the comparison table

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/08_infer_user1.py \
            --run-dirs runs/W_zero_seed42 runs/W_derived_seed42 \
            --context-scale as_is

        python3 btp/08_infer_user1.py \
            --run-dirs runs/W_zero_seed42 runs/W_derived_seed42 \
            --context-scale unit --out-suffix unit
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

USER1_CSV = "data/processed/user1/user1_all.csv"
REPORT = "docs/31_user1_inference.md"

WINDOW_ROWS = 15
STRIDE_ROWS = 15
LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

# The four columns WESAD does not have. Order must match training.
USER1_ONLY_CONTEXT = ["steps", "MET", "ACT_COUNT", "ACT_INT"]

# ACT_INT holds words. They have a natural order, so we number them that way
# and write it down. Anything unrecognised becomes 0, the same as absent.
ACT_INT_ORDER = {"sedentary": 0.0, "light": 1.0, "moderate": 2.0,
                 "vigorous": 3.0}


# =============================================================================
# The two model pieces, copied from btp/05_train.py so this script can run on
# its own. Identical definitions, or the saved weights would not load.
# =============================================================================
class EncoderCNNLSTM(nn.Module):
    def __init__(self, in_channels, proj_dim=128, lstm_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True,
                            bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128), nn.ReLU(),
            nn.Linear(128, proj_dim))

    def forward(self, x):
        h = self.net(x).permute(0, 2, 1)
        h, _ = self.lstm(h)
        return F.normalize(self.proj(h.mean(1)), dim=1)


class GatedFusionClassifier(nn.Module):
    def __init__(self, emb_dim, ctx_dim, num_classes):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, emb_dim)
        self.gate = nn.Linear(emb_dim * 2, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes))

    def forward(self, h, c, return_gate=False):
        cproj = self.ctx_proj(c)
        g = torch.sigmoid(self.gate(torch.cat([h, cproj], dim=1)))
        out = self.classifier(g * h + (1 - g) * cproj)
        return (out, g) if return_gate else out


def windows_per_session(df, feature_matrix):
    """Cut 15 row windows inside each session, never across two."""
    Xw, rows = [], []
    pos = 0
    for session, group in df.groupby("subject_id", sort=True):
        n = len(group)
        block = feature_matrix[pos:pos + n]
        for s in np.arange(0, n - WINDOW_ROWS + 1, STRIDE_ROWS):
            Xw.append(block[s:s + WINDOW_ROWS].T)
            rows.append({"session": session,
                         "window_start_row": int(s),
                         "global_row": int(pos + s)})
        pos += n
    if not Xw:
        return np.zeros((0, feature_matrix.shape[1], WINDOW_ROWS)), pd.DataFrame()
    return np.stack(Xw).astype(np.float32), pd.DataFrame(rows)


def user1_extra_columns(df, win_info, scale_mode):
    """Build the four columns WESAD did not have, one value per window.

    On WESAD these were all exactly zero during training. Here they are real.
    """
    out = np.zeros((len(win_info), len(USER1_ONLY_CONTEXT)), dtype=np.float32)
    notes = {}

    for j, name in enumerate(USER1_ONLY_CONTEXT):
        if name not in df.columns:
            notes[name] = "column absent, left at zero"
            continue

        if name == "ACT_INT":
            series = df[name].map(ACT_INT_ORDER)
        else:
            series = pd.to_numeric(df[name], errors="coerce")

        n_missing = int(series.isna().sum())
        series = series.fillna(0.0)     # absent, same as WESAD's zeros

        values = series.to_numpy(dtype=float)
        # the context is read from the first row of the window, as the
        # original code does
        picked = values[win_info["global_row"].to_numpy()]

        if scale_mode == "unit":
            # squash onto roughly 0 to 1 using the 95th percentile, so a
            # single spike does not flatten everything else
            top = np.percentile(values[values > 0], 95) if (values > 0).any() else 1.0
            top = top if top > 0 else 1.0
            picked = np.clip(picked / top, 0.0, 1.5)
            notes[name] = ("divided by the 95th percentile, %.3f; missing %d"
                           % (top, n_missing))
        else:
            notes[name] = ("used as it is, range %.2f to %.2f; missing %d"
                           % (float(picked.min()), float(picked.max()),
                              n_missing))

        out[:, j] = picked

    return out, notes


def build_context_for_user1(df, win_info, run_dir, ctx_info, ctx_fit,
                            scale_mode, derived_scale):
    """Build user1's context with the SAME encoder the model was trained with.

    Using a fresh encoder here would give a different column order. The shapes
    would still match, the code would run, and every number would be wrong
    while looking completely normal.
    """
    mode = ctx_info["mode"]

    if mode == "none":
        return np.zeros((len(win_info), 1), dtype=np.float32), {}

    if mode in ("original", "zero"):
        ohe = ctx_fit["ohe"]

        # WESAD's Task and SessionPhase do not exist for user1 and have no
        # equivalent. We pass a value the encoder has never seen. Because it
        # was fitted with handle_unknown="ignore", those two blocks come out
        # as all zeros, which is exactly the "absent" representation.
        frame = pd.DataFrame({
            "Task": ["__absent__"] * len(df),
            "SessionPhase": ["__absent__"] * len(df),
            "Activity": df["Activity"].astype(str).to_numpy(),
        })

        C_all = ohe.transform(frame.to_numpy()).astype(np.float32)
        C = C_all[win_info["global_row"].to_numpy()]

        notes = {}
        block = ohe.get_feature_names_out(["Task", "SessionPhase", "Activity"])
        n_task = sum(1 for b in block if b.startswith("Task_"))
        n_phase = sum(1 for b in block if b.startswith("SessionPhase_"))
        notes["Task block"] = "%d columns, all zero (no equivalent in user1)" % n_task
        notes["SessionPhase block"] = "%d columns, all zero (D11)" % n_phase
        notes["Activity block"] = "real values"

        if mode == "zero":
            extra, extra_notes = user1_extra_columns(df, win_info, scale_mode)
            C = np.concatenate([C, extra], axis=1)
            notes.update(extra_notes)

        return C, notes

    if mode == "derived":
        src = "ACC_Mag_Wrist_mean"
        values = df[src].to_numpy(dtype=float)
        feats = []
        for g in win_info["global_row"].to_numpy():
            block = values[g:g + WINDOW_ROWS]
            feats.append([np.nanmean(block), np.nanstd(block)])
        C = np.nan_to_num(np.array(feats, dtype=np.float32))

        if derived_scale == "wesad":
            sc = ctx_fit["ctx_scaler"]
            note = ("z-scored with WESAD's own mean and standard deviation, "
                    "so the transformation is identical")
        else:
            sc = StandardScaler().fit(C)
            note = ("z-scored with user1's own mean and standard deviation, "
                    "so the spread is comparable")

        C = sc.transform(C).astype(np.float32)
        return C, {"scaling": note}

    raise SystemExit("Unknown context mode in %s: %s" % (run_dir, mode))


def run_one(run_dir, df, scale_mode, derived_scale, out_suffix):
    """Apply one trained model to user1."""
    print("=" * 64)
    print("MODEL: %s" % run_dir)
    print("=" * 64)

    need = ["sig_cols.json", "scaler.pkl", "encoder.pth", "classifier.pth",
            "context_fit.pkl", "context_names.json", "metrics.json"]
    for f in need:
        if not os.path.exists(os.path.join(run_dir, f)):
            raise SystemExit(
                "%s has no %s.\nRe-run that training with the current "
                "btp/05_train.py, which saves the context encoder (R20)."
                % (run_dir, f))

    sig_cols = json.load(open(os.path.join(run_dir, "sig_cols.json")))
    scaler = joblib.load(os.path.join(run_dir, "scaler.pkl"))
    ctx_fit = joblib.load(os.path.join(run_dir, "context_fit.pkl"))
    ctx_info = json.load(open(os.path.join(run_dir, "context_names.json")))
    wesad = json.load(open(os.path.join(run_dir, "metrics.json")))

    print("context mode   : %s" % ctx_info["mode"])
    print("context width  : %d (must match at the end)" % ctx_info["width"])
    print("signal channels: %d" % len(sig_cols))

    missing = [c for c in sig_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            "user1 is missing %d channels this model needs:\n  %s\n"
            "Was the model trained with --channels wrist_shared?"
            % (len(missing), missing))
    print("all %d channels present in user1" % len(sig_cols))
    print()

    X = scaler.transform(df[sig_cols].astype(float))
    X = pd.DataFrame(X).interpolate(limit_direction="both").fillna(0).values

    Xw, win_info = windows_per_session(df, X)
    print("windows: %s  (cut inside each session)" % (Xw.shape,))
    print(win_info["session"].value_counts().to_string())
    print()

    C, notes = build_context_for_user1(df, win_info, run_dir, ctx_info,
                                       ctx_fit, scale_mode, derived_scale)
    print("context built: width %d" % C.shape[1])
    for k, v in notes.items():
        print("  %-22s %s" % (k, v))

    if C.shape[1] != ctx_info["width"]:
        raise SystemExit(
            "Context width is %d but the model was trained with %d. "
            "Stopping: the numbers would look fine and be wrong."
            % (C.shape[1], ctx_info["width"]))
    print("  width matches training")
    print()

    # How far outside the training range does user1's context land?
    #
    # This is the number that explains a degenerate result. During WESAD
    # training every context column was either a one-hot 0/1 or a z-scored
    # value, so almost everything sat between -3 and 3. If user1's values land
    # far outside that, the gate and classifier are being fed inputs they
    # never saw, and the softmax saturates. Entropy of exactly 0.000 is the
    # giveaway.
    col_names = ctx_info.get("names") or []
    extreme = []
    print("context values reaching the model:")
    print("  %-26s %10s %10s %10s" % ("column", "min", "max", "mean"))
    for j in range(C.shape[1]):
        name = col_names[j] if j < len(col_names) else "col_%d" % j
        lo, hi, mu = float(C[:, j].min()), float(C[:, j].max()), float(C[:, j].mean())
        flag = ""
        if max(abs(lo), abs(hi)) > 5:
            flag = "  <-- far outside the training range"
            extreme.append(name)
        print("  %-26s %10.3f %10.3f %10.3f%s" % (name, lo, hi, mu, flag))
    print()
    if extreme:
        print("  WARNING: %d column(s) land far outside anything seen in"
              % len(extreme))
        print("  training: %s" % ", ".join(extreme))
        print("  The model is being fed inputs it never saw. Expect a")
        print("  saturated, degenerate prediction.")
        print()

    device = torch.device("cpu")
    enc = EncoderCNNLSTM(in_channels=Xw.shape[1]).to(device)
    enc.load_state_dict(torch.load(os.path.join(run_dir, "encoder.pth"),
                                   map_location=device))
    enc.eval()
    with torch.no_grad():
        H = torch.cat([enc(torch.tensor(Xw[i:i + 64])) for i in
                       range(0, len(Xw), 64)]).cpu().numpy()

    clf = GatedFusionClassifier(H.shape[1], C.shape[1], 3).to(device)
    clf.load_state_dict(torch.load(os.path.join(run_dir, "classifier.pth"),
                                   map_location=device))
    clf.eval()
    with torch.no_grad():
        logits, gates = clf(torch.tensor(H), torch.tensor(C), return_gate=True)
        probs = F.softmax(logits, dim=1).numpy()
        gate_per_window = gates.mean(dim=1).numpy()

    pred = probs.argmax(1)
    score = probs[:, 1] + 2 * probs[:, 2]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)

    counts = {LABEL_NAMES[c]: int((pred == c).sum()) for c in [0, 1, 2]}
    print("predicted classes: %s" % counts)
    print("  most common class is %.1f%% of windows"
          % (100.0 * max(counts.values()) / len(pred)))
    print()
    print("stress score  mean %.3f, median %.3f, range %.3f to %.3f"
          % (score.mean(), np.median(score), score.min(), score.max()))
    print("gate value    mean %.3f  (WESAD test set: %.3f)"
          % (gate_per_window.mean(), wesad.get("mean_gate", float("nan"))))
    print("uncertainty   mean %.3f  (WESAD test set: %.3f)"
          % (entropy.mean(), wesad.get("mean_entropy", float("nan"))))
    print()

    print("per session:")
    out = win_info.copy()
    out["P_Low"], out["P_Medium"], out["P_High"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["predicted_class"] = pred
    out["predicted_label"] = [LABEL_NAMES[c] for c in pred]
    out["stress_score"] = score
    out["uncertainty"] = entropy
    out["gate"] = gate_per_window
    for session, g in out.groupby("session"):
        print("  %-22s windows %3d  score %.3f  gate %.3f  classes %s"
              % (session, len(g), g["stress_score"].mean(), g["gate"].mean(),
                 dict(g["predicted_label"].value_counts())))
    print()

    out_id = os.path.basename(run_dir.rstrip("/")) + "_on_user1"
    if out_suffix:
        out_id += "_" + out_suffix
    out_dir = os.path.join("runs", out_id)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    summary = {
        "out_id": out_id,
        "source_run": run_dir,
        "context_mode": ctx_info["mode"],
        "context_width": int(C.shape[1]),
        "context_scale": scale_mode,
        "derived_scale": derived_scale,
        "n_windows": int(len(pred)),
        "predicted_counts": counts,
        "largest_class_share": float(max(counts.values()) / len(pred)),
        "stress_score_mean": float(score.mean()),
        "stress_score_median": float(np.median(score)),
        "stress_score_min": float(score.min()),
        "stress_score_max": float(score.max()),
        "gate_mean": float(gate_per_window.mean()),
        "entropy_mean": float(entropy.mean()),
        "wesad_gate_mean": wesad.get("mean_gate"),
        "wesad_entropy_mean": wesad.get("mean_entropy"),
        "wesad_stress_score_mean": wesad.get("mean_stress_score"),
        "wesad_accuracy": wesad.get("accuracy"),
        "context_notes": {k: str(v) for k, v in notes.items()},
        "context_min": float(C.min()),
        "context_max": float(C.max()),
        "context_columns_out_of_range": extreme,
    }
    json.dump(summary, open(os.path.join(out_dir, "metrics_user1.json"), "w"),
              indent=2)
    print("saved %s/" % out_dir)
    print()
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Apply WESAD-trained models to user1. No labels exist, "
                    "so no accuracy is reported.")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--user1-csv", default=USER1_CSV)
    ap.add_argument("--context-scale", choices=["as_is", "unit"],
                    default="as_is")
    ap.add_argument("--derived-scale", choices=["wesad", "user1"],
                    default="wesad")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    if not os.path.exists(args.user1_csv):
        raise SystemExit("Not found: %s\nRun btp/07_prep_user1.py first."
                         % args.user1_csv)

    df = pd.read_csv(args.user1_csv)
    print("user1 rows: %d, columns: %d" % (len(df), len(df.columns)))
    print("sessions: %s" % sorted(df["subject_id"].unique()))
    print("context scale for the four extra columns: %s" % args.context_scale)
    print("scaling for the derived movement context : %s" % args.derived_scale)
    print()
    print("NOTE: user1 has no stress label (D2). Nothing below is an accuracy.")
    print()

    results = [run_one(rd, df, args.context_scale, args.derived_scale,
                       args.out_suffix) for rd in args.run_dirs]

    # ---------------- comparison ----------------
    print("=" * 64)
    print("COMPARISON")
    print("=" * 64)
    print("%-26s %8s %8s %8s %8s %10s"
          % ("model", "score", "gate", "uncert", "biggest", "WESAD acc"))
    for r in results:
        print("%-26s %8.3f %8.3f %8.3f %7.0f%% %10.3f"
              % (r["context_mode"] + " (" + r["context_scale"] + ")",
                 r["stress_score_mean"], r["gate_mean"], r["entropy_mean"],
                 100 * r["largest_class_share"], r["wesad_accuracy"] or 0))
    print()
    print("How to read this. There is no accuracy, so we look for signs of")
    print("breakage instead:")
    print("  - one class taking nearly every window means the model has")
    print("    stopped discriminating")
    print("  - a gate far from its WESAD value means the model is weighting")
    print("    context very differently from how it was trained")
    print("  - uncertainty much higher than on WESAD means it is guessing")
    print()

    L = []
    L.append("# user1 inference")
    L.append("")
    L.append("Written by `btp/08_infer_user1.py`.")
    L.append("")
    L.append("**user1 has no stress label** (decision D2). Two recording")
    L.append("sessions would give about two questionnaire answers even if they")
    L.append("were found on paper. So nothing here is an accuracy. These are")
    L.append("WESAD-trained models applied to user1, compared by how they")
    L.append("behave.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| user1 rows | %d |" % len(df))
    L.append("| sessions | %s |" % ", ".join(sorted(df["subject_id"].unique())))
    L.append("| scale for the four extra columns | `%s` |" % args.context_scale)
    L.append("| scale for derived movement | `%s` |" % args.derived_scale)
    L.append("")
    L.append("## Comparison")
    L.append("")
    L.append("| model | context | windows | mean stress score | gate | gate on WESAD | uncertainty | biggest class | context out of range |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        oor = r.get("context_columns_out_of_range") or []
        L.append("| `%s` | %s | %d | %.3f | %.3f | %.3f | %.3f | %.0f%% | %s |"
                 % (os.path.basename(r["source_run"]), r["context_mode"],
                    r["n_windows"], r["stress_score_mean"], r["gate_mean"],
                    r["wesad_gate_mean"] or 0, r["entropy_mean"],
                    100 * r["largest_class_share"],
                    ", ".join(oor) if oor else "none"))
    L.append("")
    L.append("There is no accuracy to report, so the signs of breakage are:")
    L.append("")
    L.append("- **one class taking nearly every window**: the model has")
    L.append("  stopped telling windows apart")
    L.append("- **a gate far from its WESAD value**: the model is weighting")
    L.append("  context very differently from how it was trained")
    L.append("- **uncertainty much higher than on WESAD**: it is guessing")
    L.append("")
    for r in results:
        L.append("## %s" % os.path.basename(r["source_run"]))
        L.append("")
        L.append("| | |")
        L.append("|---|---|")
        L.append("| context mode | `%s` |" % r["context_mode"])
        L.append("| context width | %d |" % r["context_width"])
        L.append("| predicted classes | %s |" % r["predicted_counts"])
        L.append("| stress score | mean %.3f, median %.3f, %.3f to %.3f |"
                 % (r["stress_score_mean"], r["stress_score_median"],
                    r["stress_score_min"], r["stress_score_max"]))
        L.append("| gate on user1 | %.3f |" % r["gate_mean"])
        L.append("| gate on WESAD test set | %.3f |" % (r["wesad_gate_mean"] or 0))
        L.append("| this model's WESAD accuracy | %.3f |" % (r["wesad_accuracy"] or 0))
        L.append("")
        L.append("How each context block was built:")
        L.append("")
        for k, v in r["context_notes"].items():
            L.append("- **%s**: %s" % (k, v))
        L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("Report written to %s" % REPORT)


if __name__ == "__main__":
    main()
