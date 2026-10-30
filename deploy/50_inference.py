"""
50_inference.py

WHAT THIS IS
    Runs a trained model over a signal file one window at a time, the way a
    deployed monitor would: read sixty seconds, predict, wait, repeat.

    Adapted from the reference inference.py. Two differences.

    That one reads embeddings that were computed in advance, and its
    classifier is a fixed 128-in, three-out MLP. Ours starts from the raw
    signal and has to rebuild the encoder, gate and classifier, so the shape
    is read out of the checkpoint rather than written here. Point it at a
    six-channel model or a seven-channel one and it adapts.

    The notification and dashboard parts are left out, as asked. What is
    kept is the shape that matters for a demonstration: one window at a
    time, the latency of each, and a line only when the answer changes.

WHAT THE MODEL IS
    Everything comes out of the checkpoint:

        signal_columns   which channels, and in which order
        encoder_ckpt     the frozen encoder, trained on WESAD without labels
        scaler_ckpt      the scaling the encoder was trained with
        gate             present only if the model was trained with context
        classifier       the head
        window, stride   sixty seconds, moving thirty

    The encoder, gate and classifier classes are the repository's own,
    imported unchanged from vendor/adastress.

THE SCORE
    The model gives three probabilities. They are turned into one number:

        stress_score = 1 x P(Medium) + 2 x P(High)        0 to 2

    That formula is ours, not the paper's. AdaStress outputs the three
    probabilities and stops there.

    A score of 1.0 means one of two different things: the model is confident
    the answer is Medium, or it is split evenly across all three. The
    uncertainty printed beside it separates those, so read them together.

WHAT TO KNOW BEFORE SHOWING THIS TO ANYONE
    The checkpoint in runs/P06b_w6_pooled_noctx was trained on WESAD. On
    held-out WESAD subjects it reaches 0.935 balanced accuracy. On our own
    users, measured against what they actually reported, leave-one-user-out
    gives 0.310 against a chance level of 0.333.

    So these scores are the model's opinion, not a measurement of anybody's
    stress.

    One pattern worth mentioning before someone spots it: the score tends to
    peak between four and eight in the morning. At night the body is still
    and skin temperature shifts, and in WESAD those same patterns meant the
    stress task, because that is when people sat doing arithmetic under
    pressure. Overnight scores should be treated with suspicion.

HOW TO RUN
    From the AdaStress_BTP folder.

    Over a whole file, as fast as it will go:

        python3 btp/50_inference.py \\
            --model runs/P06b_w6_pooled_noctx/model.pth \\
            --signals data/processed/user_6ch/user27.csv \\
            --sleep 0

    As a live demonstration, one window every two seconds:

        python3 btp/50_inference.py \\
            --model runs/P06b_w6_pooled_noctx/model.pth \\
            --signals data/processed/user_6ch/user27.csv \\
            --sleep 2 --limit 30
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import joblib

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.models import (ContextGateFusion, MLPClassifier,     # noqa: E402
                              StressModel, load_encoder)

LABEL_NAMES = ["Low", "Medium", "High"]
TIME_COL = "datetime_ist"
DEVICE = torch.device("cpu")


def load_model(path):
    """Rebuild the model from a checkpoint saved by our training scripts.

    The bundle carries its own shape, so nothing about the architecture is
    assumed here.
    """
    b = torch.load(path, map_location="cpu", weights_only=False)

    needed = ["classifier", "classes", "input_channels", "signal_columns",
              "encoder_ckpt", "scaler_ckpt", "lstm_hidden", "window"]
    missing = [k for k in needed if k not in b]
    if missing:
        raise SystemExit("this checkpoint is missing %s.\nIt may not have "
                         "been written by btp/21_train_subjectwise.py."
                         % missing)

    for key in ("encoder_ckpt", "scaler_ckpt"):
        if not os.path.exists(b[key]):
            raise SystemExit("%s points at %s, which is not there. Run from "
                             "the project root, where runs/ lives."
                             % (key, b[key]))

    encoder = load_encoder(b["encoder_ckpt"], b["input_channels"],
                           b["lstm_hidden"], DEVICE)

    ctx_dim = int(b.get("context_dim", 0))
    gate = None
    if ctx_dim > 0 and b.get("gate") is not None:
        gate = ContextGateFusion(encoder.output_dim, ctx_dim).to(DEVICE)
        gate.load_state_dict(b["gate"])

    clf = MLPClassifier(encoder.output_dim, len(b["classes"])).to(DEVICE)
    clf.load_state_dict(b["classifier"])

    model = StressModel(encoder, clf, gate).to(DEVICE)
    model.eval()

    scaler = joblib.load(b["scaler_ckpt"])
    return model, scaler, b


def load_signals(path, columns, window, stride):
    """Cut the file into windows, inside each day so none spans a gap."""
    df = pd.read_csv(path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit("the signal file lacks %s.\nThis model expects: %s"
                         % (missing, columns))

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    df = df.reset_index(drop=True)

    arr = df[columns].to_numpy(dtype=float)
    arr = pd.DataFrame(arr).interpolate(limit_direction="both") \
            .fillna(0.0).to_numpy()

    groups = (df.groupby("date").groups.items() if "date" in df.columns
              else [(None, df.index)])

    windows, stamps = [], []
    for _d, idx in groups:
        lo, hi = int(min(idx)), int(max(idx)) + 1
        for s in range(lo, hi - window + 1, stride):
            windows.append(arr[s:s + window].T)
            stamps.append(df[TIME_COL].iloc[s] if TIME_COL in df.columns
                          else s)

    if not windows:
        raise SystemExit("no complete %d-second window in %s"
                         % (window, path))
    return np.stack(windows).astype(np.float32), stamps


def main():
    ap = argparse.ArgumentParser(
        description="Run a trained model over a signal file, one window at "
                    "a time.")
    ap.add_argument("--model", required=True,
                    help="a checkpoint from btp/21_train_subjectwise.py")
    ap.add_argument("--signals", required=True,
                    help="a 1 Hz CSV, e.g. data/processed/user_6ch/user27.csv")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait between windows. 0 runs straight "
                         "through; 2 makes a readable demonstration.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many windows")
    ap.add_argument("--all", action="store_true",
                    help="print every window, not only the changes")
    ap.add_argument("--out", default=None,
                    help="write every prediction to this CSV")
    args = ap.parse_args()

    print("=" * 70)
    print("Stress inference")
    print("=" * 70)

    model, scaler, meta = load_model(args.model)
    cols = list(meta["signal_columns"])
    window = int(meta["window"])
    stride = int(meta.get("stride", window // 2))
    ctx_dim = int(meta.get("context_dim", 0))

    print("model    : %s" % args.model)
    print("channels : %d  %s" % (len(cols), cols))
    print("window   : %d seconds, stride %d" % (window, stride))
    print("encoder  : %s  (frozen)" % meta["encoder_ckpt"])
    if ctx_dim > 0:
        print("context  : %d values, the gate is active" % ctx_dim)
    else:
        print("context  : none. This model scores from the body signals")
        print("           alone, so the part of AdaStress that weighs")
        print("           context against physiology is not in use.")
    print()

    X, stamps = load_signals(args.signals, cols, window, stride)
    print("signals  : %s" % args.signals)
    print("windows  : %d  (%.1f hours)" % (len(X), len(X) * stride / 3600.0))

    flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    X = scaler.transform(flat).reshape(X.shape[0], X.shape[2],
                                       X.shape[1]).transpose(0, 2, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0,
                      neginf=0.0).astype(np.float32)
    print("after scaling: mean %.2f, sd %.2f, range %.1f to %.1f"
          % (X.mean(), X.std(), X.min(), X.max()))
    if abs(X.mean()) > 2 or X.std() > 5:
        print("  That is a long way from what the model was trained on, so")
        print("  treat what follows with suspicion.")
    print()
    print("-" * 70)

    n = len(X) if args.limit is None else min(args.limit, len(X))
    last, rows, lat = None, [], []

    for i in range(n):
        x = torch.tensor(X[i:i + 1])
        c = torch.zeros((1, max(ctx_dim, 1))) if ctx_dim > 0 else None

        t0 = time.time()
        with torch.no_grad():
            p = torch.softmax(model(x, c), dim=1).numpy()[0]
        ms = (time.time() - t0) * 1000.0
        lat.append(ms)

        pred = int(p.argmax())
        score = float(p[1] * 1.0 + p[2] * 2.0)
        # how far from certain: 0 means one class took everything
        uncert = float(1.0 - p.max())

        when = (stamps[i].strftime("%Y-%m-%d %H:%M")
                if hasattr(stamps[i], "strftime") else str(stamps[i]))

        rows.append({"window": i, "time": when,
                     "P_Low": round(float(p[0]), 4),
                     "P_Medium": round(float(p[1]), 4),
                     "P_High": round(float(p[2]), 4),
                     "predicted": LABEL_NAMES[pred],
                     "stress_score": round(score, 3),
                     "uncertainty": round(uncert, 3),
                     "latency_ms": round(ms, 2)})

        if args.all or pred != last:
            mark = "" if pred == last else "  <- changed"
            print("[%5d] %s  %-6s  score %.2f  uncertain %.2f  %5.1f ms%s"
                  % (i, when, LABEL_NAMES[pred], score, uncert, ms, mark))
            last = pred

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("-" * 70)
    print()

    r = pd.DataFrame(rows)
    print("windows scored : %d" % len(r))
    print("latency        : median %.1f ms, worst %.1f ms"
          % (np.median(lat), np.max(lat)))
    print()
    print("what it said:")
    for name, cnt in r["predicted"].value_counts().items():
        print("  %-7s %6d  (%.1f%%)" % (name, cnt, 100 * cnt / len(r)))
    print()
    print("stress score   : mean %.2f, median %.2f"
          % (r["stress_score"].mean(), r["stress_score"].median()))
    print("uncertainty    : mean %.2f" % r["uncertainty"].mean())

    biggest = r["predicted"].value_counts().iloc[0] / len(r)
    if biggest > 0.9:
        print()
        print("  %.0f%% of windows got the same answer. Either the person's"
              % (100 * biggest))
        print("  state really did not change, or an input landed outside what")
        print("  the model was trained on. Check the scaling line above.")

    if args.out:
        r.to_csv(args.out, index=False)
        print()
        print("saved %s" % args.out)


if __name__ == "__main__":
    main()
