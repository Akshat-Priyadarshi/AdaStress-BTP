"""
22_floor_check.py

WHY THIS FILE EXISTS
    Before spending more effort on the neural network, we need to know whether
    the problem is the model or the data.

    This fits a plain random forest on the same windows, with the same split,
    using nothing but the average and the spread of each channel. No
    self-supervised pretraining, no context gate, no encoder.

      - If the random forest scores about the same as the neural network, the
        model is not the bottleneck. The data as represented cannot support a
        better answer, and the next move is better features, not a better
        network.

      - If the random forest scores much higher, the neural pipeline is
        underperforming and the problem is on the model side.

    This is a floor, not a result. It is never reported as an AdaStress
    number.

WHY IT CAN BE TRUSTED AS A COMPARISON
    It calls the repository's own prepare_labelled_dataset, so the windows are
    byte for byte the ones the neural network saw: same window length, same
    stride, same scaler, same subject grouping. The only thing that changes is
    what is fitted on top.

    It also uses the same five rotations as btp/21_train_subjectwise.py, so
    every subject is tested exactly once and the numbers line up row for row.

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/22_floor_check.py --config configs/btp_13ch_cond.yaml
        python3 btp/22_floor_check.py --config configs/btp_13ch.yaml

    Seconds per rotation. No GPU, no training loop.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.config import load_config                  # noqa: E402
from adastress.datasets import prepare_labelled_dataset   # noqa: E402

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

# The same five groups and rotations as btp/21_train_subjectwise.py, so the
# numbers can be compared row for row.
GROUPS = {
    "G1": ["S2", "S3", "S4"],
    "G2": ["S5", "S6", "S7"],
    "G3": ["S8", "S9", "S10"],
    "G4": ["S11", "S13", "S14"],
    "G5": ["S15", "S16", "S17"],
}
ROTATIONS = {
    "A": {"test": "G5", "val": "G4"},
    "B": {"test": "G4", "val": "G5"},
    "C": {"test": "G3", "val": "G4"},
    "D": {"test": "G2", "val": "G3"},
    "E": {"test": "G1", "val": "G2"},
}


def make_split(rotation):
    te, va = ROTATIONS[rotation]["test"], ROTATIONS[rotation]["val"]
    tr = [g for g in GROUPS if g not in (te, va)]
    return {"train": [s for g in tr for s in GROUPS[g]],
            "val": list(GROUPS[va]), "test": list(GROUPS[te])}


def window_features(X):
    """Turn [N, channels, time] into [N, 2 * channels].

    Just the average and the spread of each channel over the window. This is
    deliberately the simplest thing that could work: if it matches the neural
    network, the network is not adding anything.
    """
    return np.concatenate([X.mean(axis=2), X.std(axis=2)], axis=1)


def main():
    ap = argparse.ArgumentParser(
        description="Plain random forest on the same windows, as a floor.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", choices=["on", "off"], default="off",
                    help="off by default. The point is to test the physiology "
                         "on its own.")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    window = cfg.window.window_sec * cfg.window.fs
    stride = cfg.window.stride_sec * cfg.window.fs

    print("=" * 68)
    print("FLOOR CHECK, not an AdaStress result")
    print("config : %s" % args.config)
    print("data   : %s" % cfg.data.labelled_csv)
    print("window : %d rows, stride %d   (same as the neural runs)"
          % (window, stride))
    print("context: %s" % args.context)
    print("=" * 68)
    print()

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=(cfg.data.context_columns
                         if args.context == "on" else None),
        scaler=scaler,
        window_size=window,
        stride=stride,
        subject_column=cfg.data.subject_column,
    )

    feats = window_features(X)
    if C is not None:
        feats = np.concatenate([feats, C], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    subjects = np.array(subjects)
    print("windows: %s -> %d features per window" % (X.shape, feats.shape[1]))
    print("classes: %s" % classes)
    print()

    results = []
    for rot in ["A", "B", "C", "D", "E"]:
        split = make_split(rot)
        m_tr = np.isin(subjects, split["train"] + split["val"])
        m_te = np.isin(subjects, split["test"])

        # The random forest needs no validation set to pick an epoch, so the
        # validation people are folded into training. It sees exactly the same
        # people the neural network saw, and is tested on exactly the same
        # held-out people.
        vals, cnts = np.unique(y[m_tr], return_counts=True)
        baseline = float((y[m_te] == int(vals[cnts.argmax()])).mean())

        clf = RandomForestClassifier(n_estimators=args.trees,
                                     random_state=args.seed, n_jobs=-1,
                                     class_weight="balanced")
        clf.fit(feats[m_tr], y[m_tr])
        pred = clf.predict(feats[m_te])
        true = y[m_te]

        acc = float(accuracy_score(true, pred))
        bal = float(balanced_accuracy_score(true, pred))
        mf1 = float(f1_score(true, pred, average="macro", zero_division=0))

        te_subj = subjects[m_te]
        per_subject = {s: float((pred[te_subj == s] == true[te_subj == s]).mean())
                       for s in sorted(set(te_subj))}

        print("rot %s  test=%-22s acc=%.4f  balanced=%.4f  macroF1=%.4f  "
              "(guessing %.4f)" % (rot, ", ".join(split["test"]), acc, bal,
                                   mf1, baseline))
        print("        per person: %s"
              % {k: round(v, 3) for k, v in per_subject.items()})

        results.append({"rotation": rot, "subjects_test": split["test"],
                        "accuracy": acc, "balanced_accuracy": bal,
                        "macro_f1": mf1,
                        "majority_baseline_accuracy": baseline,
                        "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                        "confusion_matrix": confusion_matrix(
                            true, pred, labels=[0, 1, 2]).tolist(),
                        "per_subject_test_accuracy": per_subject})

    accs = [r["accuracy"] for r in results]
    bals = [r["balanced_accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    bases = [r["majority_baseline_accuracy"] for r in results]

    print()
    print("=" * 68)
    print("%-24s %9s %10s %9s %10s"
          % ("", "accuracy", "balanced", "macroF1", "guessing"))
    print("%-24s %9.4f %10.4f %9.4f %10.4f"
          % ("MEAN over 5 rotations", np.mean(accs), np.mean(bals),
             np.mean(f1s), np.mean(bases)))
    print("%-24s %9.4f %10.4f %9.4f"
          % ("SPREAD (best - worst)", max(accs) - min(accs),
             max(bals) - min(bals), max(f1s) - min(f1s)))
    print("=" * 68)
    print()
    print("How to read this:")
    print("  Compare the mean above against the neural network's mean on the")
    print("  same data and the same rotations.")
    print()
    print("  About the same  -> the model is not the bottleneck. The data as")
    print("                     represented cannot support a better answer.")
    print("                     Next step is better features, not a better")
    print("                     network.")
    print("  Much higher     -> the neural pipeline is underperforming.")
    print()

    tag = os.path.splitext(os.path.basename(args.config))[0]
    out = os.path.join("runs", "floorcheck_%s_context_%s.json"
                       % (tag, args.context))
    os.makedirs("runs", exist_ok=True)
    json.dump({"config": args.config, "context": args.context,
               "labelled_csv": cfg.data.labelled_csv,
               "window": window, "stride": stride,
               "mean_accuracy": float(np.mean(accs)),
               "mean_balanced_accuracy": float(np.mean(bals)),
               "mean_macro_f1": float(np.mean(f1s)),
               "mean_baseline": float(np.mean(bases)),
               "accuracy_spread": float(max(accs) - min(accs)),
               "per_rotation": results}, open(out, "w"), indent=2)
    print("saved %s" % out)


if __name__ == "__main__":
    main()
