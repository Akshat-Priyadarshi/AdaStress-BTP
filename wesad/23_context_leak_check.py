"""
23_context_leak_check.py

WHY THIS FILE EXISTS
    On the PANAS datasets we measured that Task alone predicts the stress
    label with 100% accuracy on held-out people. The reason was structural:
    the label came from one questionnaire per session, and Task IS the
    session, so they were two names for one thing.

    WESAD's own condition labels come from the same five sessions. So the same
    problem is likely there too, and probably worse, because the condition
    label is defined BY the protocol block rather than derived from a
    questionnaire answered during it.

    We have to measure that before running any context experiment on this
    dataset. If Task gives the answer away again, then any context-aware
    accuracy is a leakage measurement, and the only honest number is the
    no-context one.

WHAT IT DOES
    For every context column, and for subject identity, it asks: how well does
    this one column predict the label, on its own, with no model at all?

    Learn the most common label for each value of the column using the
    training people, then apply that lookup to the held-out people. That is
    the score any model gets for free just by reading that column.

    Uses the repository's own prepare_labelled_dataset, so the windows and the
    context values are exactly the ones the neural network saw, and the same
    five rotations as the training scripts.

HOW TO READ IT
    lookup accuracy near 1.0
        The column is the label under another name. Any accuracy from a model
        that can see it is meaningless.

    lookup accuracy near the guessing baseline
        The column carries nothing. Adding it can only add noise.

    in between
        Genuine partial information. Report the no-context model beside every
        context-aware one so the reader can see where the accuracy came from.

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/23_context_leak_check.py --config configs/btp_13ch_cond.yaml
        python3 btp/23_context_leak_check.py --config configs/btp_13ch.yaml

    A few seconds. It trains nothing.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2_contingency

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.config import load_config                  # noqa: E402
from adastress.datasets import prepare_labelled_dataset   # noqa: E402

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
    return {"train": [s for g in tr for s in GROUPS[g]] + list(GROUPS[va]),
            "test": list(GROUPS[te])}


def lookup_accuracy(values_tr, y_tr, values_te, y_te):
    """Score from reading one column, with no model.

    Learn the most common label for each value from the training people, then
    apply it to the held-out people. Values never seen fall back to the most
    common label overall.
    """
    table = {}
    for v in np.unique(values_tr):
        sel = values_tr == v
        lab, cnt = np.unique(y_tr[sel], return_counts=True)
        table[v] = lab[cnt.argmax()]

    lab, cnt = np.unique(y_tr, return_counts=True)
    fallback = lab[cnt.argmax()]

    pred = np.array([table.get(v, fallback) for v in values_te])
    return float((pred == y_te).mean())


def cramers_v(a, b):
    """How strongly two label-like columns move together. 0 to 1."""
    table = pd.crosstab(pd.Series(a), pd.Series(b))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    smaller = min(table.shape) - 1
    if n == 0 or smaller == 0:
        return float("nan")
    return float(np.sqrt((chi2 / n) / smaller))


def main():
    ap = argparse.ArgumentParser(
        description="Measure how much each context column gives the label "
                    "away, before any context experiment is run.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    window = cfg.window.window_sec * cfg.window.fs
    stride = cfg.window.stride_sec * cfg.window.fs
    ctx_cols = list(cfg.data.context_columns)

    print("=" * 70)
    print("CONTEXT LEAK CHECK   (no model is trained)")
    print("config  : %s" % args.config)
    print("data    : %s" % cfg.data.labelled_csv)
    print("window  : %d rows, stride %d" % (window, stride))
    print("columns : %s" % ", ".join(ctx_cols))
    print("=" * 70)
    print()

    scaler = joblib.load(cfg.paths.scaler_ckpt)
    X, y, C, subjects, classes = prepare_labelled_dataset(
        csv_path=cfg.data.labelled_csv,
        signal_columns=cfg.data.signal_columns,
        label_column=cfg.data.label_column,
        context_columns=ctx_cols,
        scaler=scaler,
        window_size=window,
        stride=stride,
        subject_column=cfg.data.subject_column,
    )
    subjects = np.array(subjects)
    print("windows: %d, classes: %s, context width: %d"
          % (len(y), classes, C.shape[1]))
    print()

    # subject identity is not context, but it is the other way a model can
    # cheat, so it is measured alongside
    columns = {ctx_cols[i]: C[:, i] for i in range(C.shape[1])}
    columns["subject_id"] = subjects

    per_col = {name: [] for name in columns}
    baselines = []

    print("%-14s %8s %8s %8s %8s %8s %9s"
          % ("column", "rot A", "rot B", "rot C", "rot D", "rot E", "MEAN"))
    print("-" * 70)

    for rot in ["A", "B", "C", "D", "E"]:
        split = make_split(rot)
        m_tr = np.isin(subjects, split["train"])
        m_te = np.isin(subjects, split["test"])

        lab, cnt = np.unique(y[m_tr], return_counts=True)
        baselines.append(float((y[m_te] == lab[cnt.argmax()]).mean()))

        for name, vals in columns.items():
            per_col[name].append(
                lookup_accuracy(vals[m_tr], y[m_tr], vals[m_te], y[m_te]))

    for name in columns:
        vals = per_col[name]
        print("%-14s %8.3f %8.3f %8.3f %8.3f %8.3f %9.3f"
              % (name, vals[0], vals[1], vals[2], vals[3], vals[4],
                 float(np.mean(vals))))

    print("%-14s %8.3f %8.3f %8.3f %8.3f %8.3f %9.3f"
          % ("guessing", baselines[0], baselines[1], baselines[2],
             baselines[3], baselines[4], float(np.mean(baselines))))
    print()

    print("How strongly each column moves with the label (0 to 1):")
    vs = {}
    for name, vals in columns.items():
        vs[name] = cramers_v(vals, y)
        print("  %-14s %.3f" % (name, vs[name]))
    print()

    base_mean = float(np.mean(baselines))
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    verdicts = {}
    for name in columns:
        m = float(np.mean(per_col[name]))
        if m >= 0.95:
            v = "GIVES THE ANSWER AWAY. Any model that sees it is meaningless."
        elif m > base_mean + 0.10:
            v = ("real head start over guessing (+%.3f). Report the "
                 "no-context model beside it." % (m - base_mean))
        elif m > base_mean + 0.02:
            v = "small real information (+%.3f)." % (m - base_mean)
        else:
            v = "carries nothing. At or below guessing."
        verdicts[name] = v
        print("  %-14s %.3f  %s" % (name, m, v))
    print()

    tag = os.path.splitext(os.path.basename(args.config))[0]
    out = os.path.join("runs", "contextleak_%s.json" % tag)
    os.makedirs("runs", exist_ok=True)
    json.dump({"config": args.config,
               "labelled_csv": cfg.data.labelled_csv,
               "n_windows": int(len(y)),
               "per_column_per_rotation": {k: v for k, v in per_col.items()},
               "per_column_mean": {k: float(np.mean(v))
                                   for k, v in per_col.items()},
               "cramers_v": vs,
               "baseline_per_rotation": baselines,
               "baseline_mean": base_mean,
               "verdict": verdicts}, open(out, "w"), indent=2)
    print("saved %s" % out)


if __name__ == "__main__":
    main()
