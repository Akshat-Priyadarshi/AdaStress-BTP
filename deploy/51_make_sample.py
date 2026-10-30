"""
51_make_sample.py

WHAT THIS DOES
    Packs one person's day into a single folder that can be copied to
    another machine and run there with nothing else from this project.

        sample_data/<user>_<date>/
            signals.csv      the seven channels, one row per second
            context.csv      the eight context columns, every 30 seconds
            truth.csv        what the person actually reported, for checking
            model.pth        the trained model, copied in
            encoder.pth      the frozen encoder it needs
            scaler.pkl       the scaling it was trained with
            README.md        what each file is and how to run it

    The model checkpoint stores the encoder and scaler as relative paths, so
    they are copied in beside it and the paths rewritten. The folder works
    wherever it is put.

WHY A WHOLE DAY AND NOT A FEW MINUTES
    A demonstration where the answer never changes shows nothing. This picks
    a day where the person reported more than one stress level, so there is
    something for the model to get right or wrong.

    The day is chosen by hand with --user and --date, because which one
    makes a good demonstration is a judgement, not something to leave to a
    script.

WHAT truth.csv IS FOR
    The survey answers for that day, as STAI-6 scores and classes. The model
    never saw them, so comparing against them is a real test rather than a
    replay.

    Be careful how it is used in front of people. One day of one person is
    not evidence about the model; it is an illustration of the pipeline. The
    numbers that carry weight are the leave-one-user-out ones, and they are
    in the README.

HOW TO RUN
    python3 btp/51_make_sample.py --user user25 --date 2026-09-16 \\
        --model runs/DEMO_ctx8_random/model.pth
"""

import os
import json
import shutil
import argparse

import numpy as np
import pandas as pd
import torch

SIGNAL_DIR = "data/processed/user_7ch"
CONTEXT_FILE = "data/processed/labels/labelled_windows_ctx8.csv"
OUT_ROOT = "sample_data"
TIME_COL = "datetime_ist"


def main():
    ap = argparse.ArgumentParser(
        description="Pack one person's day into a folder that runs anywhere.")
    ap.add_argument("--user", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--model", required=True)
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--context", default=CONTEXT_FILE)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    out = os.path.join(args.out_root, "%s_%s" % (args.user, args.date))
    os.makedirs(out, exist_ok=True)

    print("=" * 70)
    print("Packing %s, %s" % (args.user, args.date))
    print("=" * 70)

    # ---------- the model, and what it needs ----------
    b = torch.load(args.model, map_location="cpu", weights_only=False)
    for key, name in (("encoder_ckpt", "encoder.pth"),
                      ("scaler_ckpt", "scaler.pkl")):
        src = b[key]
        if not os.path.exists(src):
            raise SystemExit("%s points at %s, which is not there" % (key, src))
        shutil.copy(src, os.path.join(out, name))
        b[key] = name              # rewrite so the folder is self-contained
    torch.save(b, os.path.join(out, "model.pth"))

    cols = list(b["signal_columns"])
    ctx_cols = list(b.get("context_columns", []))
    print("model     : %s" % args.model)
    print("channels  : %d  %s" % (len(cols), cols))
    print("context   : %d columns, %d values after encoding"
          % (len(ctx_cols), b.get("context_dim", 0)))
    print()

    # ---------- the signals ----------
    spath = os.path.join(args.signal_dir, "%s.csv" % args.user)
    if not os.path.exists(spath):
        raise SystemExit("no signal file at %s" % spath)

    sig = pd.read_csv(spath)
    sig[TIME_COL] = pd.to_datetime(sig[TIME_COL], errors="coerce")
    sig = sig.dropna(subset=[TIME_COL])
    day = sig[sig[TIME_COL].dt.date.astype(str) == args.date]

    missing = [c for c in cols if c not in day.columns]
    if missing:
        raise SystemExit("the signal file lacks %s" % missing)
    if day.empty:
        raise SystemExit("no signal on %s for %s" % (args.date, args.user))

    keep = [TIME_COL] + cols
    day = day[keep].sort_values(TIME_COL).reset_index(drop=True)
    day.to_csv(os.path.join(out, "signals.csv"), index=False)

    span = (day[TIME_COL].max() - day[TIME_COL].min()).total_seconds() / 3600.0
    print("signals.csv  %6d rows, %.1f hours, %s to %s"
          % (len(day), span, day[TIME_COL].min().strftime("%H:%M"),
             day[TIME_COL].max().strftime("%H:%M")))

    # ---------- the context and the truth ----------
    ctx = pd.read_csv(args.context)
    ctx[TIME_COL] = pd.to_datetime(ctx[TIME_COL], errors="coerce")
    cday = ctx[(ctx["Subject_ID"] == args.user) &
               (ctx[TIME_COL].dt.date.astype(str) == args.date)].copy()

    if cday.empty:
        print("context.csv  nothing for this day, so the gate cannot be used")
        n_resp = 0
    else:
        cday = cday.sort_values(TIME_COL)
        have = [c for c in ctx_cols if c in cday.columns]
        cday[[TIME_COL] + have].to_csv(os.path.join(out, "context.csv"),
                                       index=False)
        print("context.csv  %6d rows, every 30 s, %d columns"
              % (len(cday), len(have)))

        truth = cday[[TIME_COL, "response_time", "STAI6", "LABEL",
                      "LABEL_NAME"]].copy()
        truth.to_csv(os.path.join(out, "truth.csv"), index=False)
        n_resp = truth["response_time"].nunique()
        print("truth.csv    %6d labelled windows from %d survey answers"
              % (len(truth), n_resp))
        print()
        print("what the person reported that day:")
        for name, cnt in cday["LABEL_NAME"].value_counts().items():
            print("   %-14s %4d windows" % (name, cnt))
        print()
        print("context values present:")
        for c in have:
            u = cday[c].nunique()
            # SLEEP holds numbers and the word "Unknown" together, so ask
            # whether it is numeric rather than trusting the dtype
            vals = pd.to_numeric(cday[c], errors="coerce")
            if vals.notna().all():
                print("   %-16s %d distinct, %.1f to %.1f"
                      % (c, u, vals.min(), vals.max()))
            else:
                print("   %-16s %d distinct: %s"
                      % (c, u, list(cday[c].astype(str)
                                    .value_counts().head(4).index)))
    print()

    # ---------- a readme that travels with it ----------
    L = []
    L.append("# Sample: %s, %s" % (args.user, args.date))
    L.append("")
    L.append("One person's day, packed so it runs on another machine with")
    L.append("nothing else from the project.")
    L.append("")
    L.append("| file | what it is |")
    L.append("|---|---|")
    L.append("| `signals.csv` | %d channels, one row per second |" % len(cols))
    L.append("| `context.csv` | %d context columns, every 30 seconds |"
             % len(ctx_cols))
    L.append("| `truth.csv` | what the person reported, for checking |")
    L.append("| `model.pth` | the trained model |")
    L.append("| `encoder.pth` | the frozen encoder, trained on WESAD |")
    L.append("| `scaler.pkl` | the scaling the encoder was trained with |")
    L.append("")
    L.append("## Running it")
    L.append("")
    L.append("```")
    L.append("python3 52_edge_demo.py --sample %s" % out)
    L.append("```")
    L.append("")
    L.append("Needs `numpy`, `pandas`, `torch`, `scikit-learn`, `joblib`, and")
    L.append("the `vendor/adastress` package for the model classes.")
    L.append("")
    L.append("**numpy must stay below 2.** torch and pandas here were built")
    L.append("against 1.x and all of them break together if it is upgraded.")
    L.append("")
    L.append("## What the numbers mean")
    L.append("")
    L.append("Three probabilities per window, turned into one score:")
    L.append("")
    L.append("```")
    L.append("stress_score = 1 x P(Medium) + 2 x P(High)     0 to 2")
    L.append("```")
    L.append("")
    L.append("That formula is ours. AdaStress outputs the three")
    L.append("probabilities and nothing more.")
    L.append("")
    L.append("## How much to trust this")
    L.append("")
    L.append("This is one day of one person. It shows the pipeline working,")
    L.append("not how well the model works.")
    L.append("")
    L.append("The numbers that carry weight, measured across five people")
    L.append("against their own reported stress:")
    L.append("")
    L.append("| test | balanced accuracy | chance |")
    L.append("|---|---:|---:|")
    L.append("| held-out WESAD subjects | 0.935 | 0.531 |")
    L.append("| pooled random split | 0.890 | 0.333 |")
    L.append("| survey answers kept whole | 0.380 | 0.333 |")
    L.append("| a person never seen before | 0.286 | 0.333 |")
    L.append("")
    L.append("The random split is high because windows overlap by 30 seconds")
    L.append("and each survey answer labels about 60 of them, so near")
    L.append("identical windows land on both sides. All 100 survey answers")
    L.append("had windows in both training and test. The model recognises")
    L.append("moments it has already seen.")
    L.append("")
    L.append("**Transfer to a new person does not work on this data.** Five")
    L.append("people used a self-report scale in five different parts of its")
    L.append("range. One reported low stress on all 18 occasions; another")
    L.append("never reported low at all.")
    L.append("")
    L.append("One more thing worth saying before anyone notices it: the score")
    L.append("tends to peak between four and eight in the morning. At night")
    L.append("the body is still and skin temperature shifts, and in WESAD")
    L.append("those same patterns meant the stress task.")
    L.append("")

    open(os.path.join(out, "README.md"), "w").write("\n".join(L) + "\n")

    size = sum(os.path.getsize(os.path.join(out, f))
               for f in os.listdir(out)) / 1e6
    print("=" * 70)
    print("packed into %s  (%.1f MB)" % (out, size))
    for f in sorted(os.listdir(out)):
        print("   %-14s %7.2f MB"
              % (f, os.path.getsize(os.path.join(out, f)) / 1e6))
    print()
    if n_resp:
        print("Good for a demonstration: %d survey answers that day, so the"
              % n_resp)
        print("score has something to be right or wrong about.")
    print()


if __name__ == "__main__":
    main()
