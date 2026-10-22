"""
26_apply_to_users.py

WHY THIS FILE EXISTS
    Before spending time on fine-tuning, we look at what the WESAD model does
    on the user data as it is. Nothing is trained here. No labels are needed.

    The question is simple: does the model behave sensibly on real wrist data,
    or does it fall over? If it falls over, fine-tuning will not rescue it and
    we want to know now rather than after three days of work.

    We have seen a model fall over before. On an earlier transfer, one run
    produced the identical answer for all 120 windows with an uncertainty of
    exactly 0.000, because an input landed seventy standard deviations outside
    anything it had seen in training. This script checks for that first, and
    says so plainly if it happens.

WHAT IT LOOKS AT, SINCE THERE IS NO ACCURACY

    1. Are the inputs in range?
       Each channel is scaled with WESAD's own scaler. If a channel comes out
       far outside what the model saw in training, everything after it is
       meaningless. This is reported BEFORE the predictions, because it is the
       cause rather than a symptom.

    2. Does it spread its answers?
       One class taking almost every window means the model has stopped
       telling windows apart.

    3. Is it as confident as it was on WESAD?
       Much more confident is as bad a sign as much less. Confidently wrong is
       worse than unsure.

    4. Does the gate sit where it did in training?
       The gate is the model's own weighting between the body signal and the
       context. A big move means it is working differently from how it was
       trained.

    5. Do the scores follow the clock?
       With no labels, the one sanity check available is whether the stress
       score moves with time of day in a way a person would recognise. Higher
       during work hours than at 3am is believable. Flat is not informative.
       Reversed is worth investigating.

OUTPUT
    runs/<run-id>_on_users/
        predictions.csv     one row per window
        summary.json
    docs/51_apply_to_users.md

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/26_apply_to_users.py \\
            --model runs/P06_w6_pooled_noctx/model.pth \\
            --users data/processed/user_6ch/users_all.csv

    A minute or two for 49,000 windows on CPU.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.models import (ContextGateFusion, MLPClassifier,   # noqa: E402
                              StressModel, load_encoder)

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]
TIME_COL = "datetime_ist"


def make_windows(df, signal_cols, window, stride):
    """Cut windows inside each user-day, so none spans a gap in recording."""
    Xw, rows = [], []
    for (user, date), g in df.groupby(["Subject_ID", "date"], sort=True):
        g = g.reset_index(drop=True)
        arr = g[signal_cols].to_numpy(dtype=float)
        n = len(g)
        for s in range(0, n - window + 1, stride):
            Xw.append(arr[s:s + window].T)
            mid = g[TIME_COL].iloc[s + window // 2]
            rows.append({"Subject_ID": user, "date": date,
                         "window_start": int(s), "mid_time": mid})
    if not Xw:
        return np.zeros((0, len(signal_cols), window)), pd.DataFrame()
    return np.stack(Xw).astype(np.float32), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Apply a WESAD-trained model to the user data. Nothing "
                    "is trained; the users have no labels yet.")
    ap.add_argument("--model", required=True,
                    help="runs/<id>/model.pth from btp/21_train_subjectwise.py")
    ap.add_argument("--users", default="data/processed/user_6ch/users_all.csv")
    ap.add_argument("--scale", choices=["wesad", "user"], default="wesad",
                    help=(
                        "wesad: WESAD's own scaler, so the transformation is "
                        "identical to training. "
                        "user: work the mean and spread out from the user "
                        "data itself. Use this when a channel is recorded in "
                        "different units on the two devices, which squashes "
                        "it to nothing under WESAD's scaler."))
    ap.add_argument("--out-id", default=None)
    args = ap.parse_args()

    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    signal_cols = bundle["signal_columns"]
    window, stride = bundle["window"], bundle["stride"]
    classes = bundle["classes"]
    context_dim = bundle["context_dim"]

    # The scale choice goes in the folder name, otherwise running both
    # overwrites the first with the second and we lose the comparison.
    out_id = args.out_id or (os.path.basename(os.path.dirname(args.model))
                             + "_on_users_" + args.scale)
    out_dir = os.path.join("runs", out_id)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("APPLYING A WESAD MODEL TO THE USER DATA")
    print("model    : %s" % args.model)
    print("channels : %d  %s" % (len(signal_cols), signal_cols))
    print("window   : %d rows, stride %d" % (window, stride))
    print("context  : %s" % (bundle.get("context_columns") or "none"))
    print("=" * 70)
    print()
    print("The users have no stress labels yet, so nothing below is an")
    print("accuracy. These are signs of whether the model is working.")
    print()

    df = pd.read_csv(args.users)
    missing = [c for c in signal_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            "The user file is missing %d channels this model needs:\n  %s\n"
            "Was the model trained with the 6-channel config?" % (len(missing),
                                                                  missing))

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(["Subject_ID", TIME_COL])
    print("user rows: %d, users: %s"
          % (len(df), sorted(df["Subject_ID"].unique())))

    # ---- scale with WESAD's own scaler, then check the result is in range --
    # Two ways to put the channels on a scale the model recognises, and they
    # answer different questions.
    #
    #   wesad  WESAD's own mean and spread, identical to training. Right when
    #          both devices record a channel the same way.
    #
    #   user   The user data's own mean and spread. Right when they do not.
    #          WESAD's wrist accelerometer is stored in raw device counts and
    #          EmbracePlus converts to g, a difference of roughly 64 times, so
    #          under WESAD's scaler the user's accelerometer comes out almost
    #          flat and the model cannot see movement at all.
    vals = df[signal_cols].astype(float)
    vals = vals.interpolate(limit_direction="both").fillna(0.0).to_numpy()

    if args.scale == "wesad":
        # .to_numpy() above also stops sklearn warning about feature names:
        # the scaler was fitted on a plain array during WESAD training
        scaler = joblib.load(bundle["scaler_ckpt"])
        scaled = scaler.transform(vals)
        print()
        print("scaling: WESAD's own, identical to training")
    else:
        mu, sd0 = np.nanmean(vals, axis=0), np.nanstd(vals, axis=0)
        sd0[sd0 == 0] = 1.0
        scaled = (vals - mu) / sd0
        print()
        print("scaling: the user data's own mean and spread")

    print()
    print("Are the inputs in range? Each channel after WESAD's own scaling.")
    print("In training almost everything sat between about -3 and 3.")
    print()
    print("  A channel whose sd is near zero is dead: averaging has flattened")
    print("  it, and the model can read nothing from it.")
    print()
    print("  %-14s %9s %9s %9s %9s %s"
          % ("channel", "min", "max", "mean", "sd", "status"))
    out_of_range = []
    dead = []
    for i, c in enumerate(signal_cols):
        col = scaled[:, i]
        lo, hi = float(col.min()), float(col.max())
        sd = float(col.std())
        flag = ""
        if sd < 0.10:
            flag = "<-- DEAD, barely varies"
            dead.append(c)
        elif max(abs(lo), abs(hi)) > 10:
            flag = "<-- far outside training"
            out_of_range.append(c)
        print("  %-14s %9.2f %9.2f %9.2f %9.2f %s"
              % (c, lo, hi, float(col.mean()), sd, flag))
    print()
    if dead:
        print("  WARNING: %d channel(s) barely vary at all: %s"
              % (len(dead), ", ".join(dead)))
        print("  Averaging 64 readings a second down to one has cancelled")
        print("  them out. BVP swings above and below zero with every")
        print("  heartbeat, so a one-second average is close to nothing left.")
        print("  The model is effectively running on the channels that remain.")
        print()
    if out_of_range:
        print("  WARNING: %d channel(s) land far outside anything the model"
              % len(out_of_range))
        print("  saw in training: %s" % ", ".join(out_of_range))
        print("  Expect a stuck, one-class answer. This is the cause, not a")
        print("  symptom.")
    else:
        print("  All channels are in a range the model has seen. Good.")
    print()

    df_scaled = df.copy()
    df_scaled[signal_cols] = scaled

    Xw, info = make_windows(df_scaled, signal_cols, window, stride)
    print("windows: %s  (cut inside each user-day)" % (Xw.shape,))
    print(info["Subject_ID"].value_counts().to_string())
    print()

    # ---- rebuild the model exactly as it was trained ----
    device = torch.device("cpu")
    encoder = load_encoder(bundle["encoder_ckpt"], bundle["input_channels"],
                           bundle["lstm_hidden"], device)
    gate = (ContextGateFusion(encoder.output_dim, context_dim).to(device)
            if context_dim > 0 else None)
    classifier = MLPClassifier(encoder.output_dim, len(classes)).to(device)
    model = StressModel(encoder, classifier, gate).to(device)
    model.classifier.load_state_dict(bundle["classifier"])
    if gate is not None and bundle["gate"] is not None:
        model.gate.load_state_dict(bundle["gate"])
    model.eval()

    probs_all, gates_all = [], []
    with torch.no_grad():
        for i in range(0, len(Xw), 256):
            xb = torch.tensor(Xw[i:i + 256])
            cb = (torch.zeros(len(xb), context_dim) if context_dim > 0
                  else None)
            if gate is not None:
                logits, g = model(xb, cb, return_gate=True)
                gates_all.append(g.mean(dim=1).numpy())
            else:
                logits = model(xb, cb)
            probs_all.append(F.softmax(logits, dim=1).numpy())

    probs = np.concatenate(probs_all)
    pred = probs.argmax(1)
    score = probs[:, 1] + 2 * probs[:, 2]
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    gates = (np.concatenate(gates_all) if gates_all
             else np.full(len(pred), np.nan))

    counts = {LABEL_NAMES[c]: int((pred == c).sum()) for c in range(len(classes))}
    biggest = max(counts.values()) / len(pred)

    print("=" * 70)
    print("WHAT THE MODEL SAID")
    print("=" * 70)
    print("  predicted classes : %s" % counts)
    print("  biggest class     : %.1f%% of all windows" % (100 * biggest))
    print("  stress score      : mean %.3f, median %.3f, range %.3f to %.3f"
          % (score.mean(), np.median(score), score.min(), score.max()))
    print("  uncertainty       : mean %.3f" % entropy.mean())
    if not np.isnan(gates).all():
        print("  gate              : mean %.3f" % np.nanmean(gates))
    print()

    if biggest > 0.95:
        print("  VERDICT: stuck. One class takes almost every window, so the")
        print("  model has stopped telling them apart.")
    elif entropy.mean() < 0.01:
        print("  VERDICT: saturated. The uncertainty is essentially zero, so")
        print("  every window gets the same confident answer.")
    else:
        print("  VERDICT: the model is producing a spread of answers. Worth")
        print("  continuing to fine-tuning.")
    print()

    info["P_Low"], info["P_Medium"], info["P_High"] = (probs[:, 0], probs[:, 1],
                                                       probs[:, 2])
    info["predicted"] = [LABEL_NAMES[c] for c in pred]
    info["stress_score"] = score
    info["uncertainty"] = entropy
    info["gate"] = gates
    info["hour"] = pd.to_datetime(info["mid_time"]).dt.hour

    print("Per user:")
    print("  %-9s %9s %8s %8s   %s" % ("user", "windows", "score", "uncert",
                                       "classes"))
    per_user = {}
    for u, g in info.groupby("Subject_ID"):
        per_user[u] = {"windows": int(len(g)),
                       "stress_score_mean": float(g["stress_score"].mean()),
                       "entropy_mean": float(g["uncertainty"].mean()),
                       # int() because value_counts returns numpy int64, which
                       # the json module refuses to write
                       "counts": {str(k): int(v) for k, v in
                                  g["predicted"].value_counts().items()}}
        print("  %-9s %9d %8.3f %8.3f   %s"
              % (u, len(g), g["stress_score"].mean(), g["uncertainty"].mean(),
                 dict(g["predicted"].value_counts())))
    print()

    # ---- the one sanity check available without labels ----
    print("Does the stress score follow the clock?")
    print("With no labels this is the only check we have. Higher during work")
    print("hours than at 3am would be believable. Flat tells us nothing.")
    print()
    by_hour = info.groupby("hour")["stress_score"].agg(["mean", "count"])
    print("  %-6s %9s %9s" % ("hour", "score", "windows"))
    for h, r in by_hour.iterrows():
        bar = "#" * int(r["mean"] * 20)
        print("  %-6d %9.3f %9d  %s" % (h, r["mean"], int(r["count"]), bar))
    swing = float(by_hour["mean"].max() - by_hour["mean"].min())
    print()
    print("  highest hour minus lowest hour: %.3f" % swing)
    if swing < 0.05:
        print("  Flat across the day. The score is not tracking anything")
        print("  about daily life.")
    else:
        print("  The score does move across the day. Whether it moves the")
        print("  right way needs the survey answers to say.")
    print()

    info.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    summary = {
        "model": args.model, "users_csv": args.users,
        "n_windows": int(len(pred)), "channels": signal_cols,
        "window": window, "stride": stride,
        "context_dim": int(context_dim),
        "scale": args.scale,
        "channels_out_of_range": out_of_range,
        "channels_dead": dead,
        "predicted_counts": counts,
        "largest_class_share": float(biggest),
        "stress_score_mean": float(score.mean()),
        "entropy_mean": float(entropy.mean()),
        "gate_mean": (None if np.isnan(gates).all()
                      else float(np.nanmean(gates))),
        "hour_swing": swing,
        "per_user": per_user,
        "by_hour": {int(h): float(r["mean"]) for h, r in by_hour.iterrows()},
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"),
              indent=2, default=str)
    print("saved %s/" % out_dir)


if __name__ == "__main__":
    main()
