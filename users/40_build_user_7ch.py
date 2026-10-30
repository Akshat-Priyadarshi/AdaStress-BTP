"""
40_build_user_7ch.py

WHAT THIS DOES
    Adds a breathing-rate channel to the six already in
    data/processed/user_6ch/, and writes the result to
    data/processed/user_7ch/.

        ACC_X_wrist, ACC_Y_wrist, ACC_Z_wrist, BVP, EDA_wrist, Temp_wrist
        + RESP

    so that the users match the seven channels the WESAD encoder is trained
    on.

WHERE RESP COMES FROM, AND THE CATCH
    The raw EmbracePlus files were deleted after the context was derived, so
    the breathing rate cannot be recomputed at one value per second right
    now. What exists is the RESP column already in
    data/processed/context/<user>_context.csv, at one value every 30
    seconds.

    This script carries that forward onto the 1 Hz grid, so each value is
    repeated 30 times.

    Be honest about what that means. Within one 60 second window the channel
    holds at most two distinct values with a step between them, where a true
    1 Hz breathing rate would move continuously. The encoder reads a coarse
    staircase rather than a waveform.

    It is enough to answer the question that matters first: does a breathing
    channel change the result at all? If it does, the raw files are worth
    downloading again to compute it properly. If it does not, 44 GB has been
    saved.

    Use --mode raw once the raw files are back, and this script will compute
    it per second instead.

AND THE OTHER CATCH, WHICH IS NOT ABOUT RESOLUTION
    On WESAD, RESP is computed from the chest respiration belt, which
    measures the chest actually expanding. Here it comes from the pulse,
    because a wristband cannot measure breathing directly.

    Measured on WESAD, where both are available, those two agree at a
    correlation of 0.185. So the channel means something slightly different
    at each end, and the encoder's filters for it are learned from the
    cleaner version.

    That is a real limitation and belongs in any report using these files,
    alongside the 0.185.

OUTPUT
    data/processed/user_7ch/<user>.csv
    data/processed/user_7ch/users_all.csv
    docs/62_user_7ch.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/40_build_user_7ch.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

SIGNAL_DIR = "data/processed/user_6ch"
CONTEXT_DIR = "data/processed/context"
OUT_DIR = "data/processed/user_7ch"
REPORT = "docs/62_user_7ch.md"

TIME_COL = "datetime_ist"
USERS = ["user17", "user18", "user25", "user26", "user27"]

SIX = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
       "BVP", "EDA_wrist", "Temp_wrist"]
SEVEN = SIX + ["RESP"]

# A breathing rate outside this is not a person breathing.
MIN_BR, MAX_BR = 5.0, 45.0


def build_user(user, signal_dir, context_dir):
    """Join the 1 Hz signals with the 30-second breathing rate."""
    spath = os.path.join(signal_dir, "%s.csv" % user)
    cpath = os.path.join(context_dir, "%s_context.csv" % user)

    if not os.path.exists(spath):
        print("    no 1 Hz signal file")
        return None, None
    if not os.path.exists(cpath):
        print("    no context file, so no breathing rate")
        return None, None

    sig = pd.read_csv(spath)
    missing = [c for c in SIX if c not in sig.columns]
    if missing:
        print("    missing channels %s" % missing)
        return None, None

    sig[TIME_COL] = pd.to_datetime(sig[TIME_COL], errors="coerce")
    sig = sig.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    sig = sig.reset_index(drop=True)

    ctx = pd.read_csv(cpath, usecols=lambda c: c in (TIME_COL, "RESP"))
    if "RESP" not in ctx.columns:
        print("    the context file has no RESP column")
        return None, None

    ctx[TIME_COL] = pd.to_datetime(ctx[TIME_COL], errors="coerce")
    ctx = ctx.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

    # drop anything that is not a plausible breathing rate before it is
    # carried across 30 seconds of signal
    bad = ((ctx["RESP"] < MIN_BR) | (ctx["RESP"] > MAX_BR)).sum()
    ctx.loc[(ctx["RESP"] < MIN_BR) | (ctx["RESP"] > MAX_BR), "RESP"] = np.nan

    # each 30-second value carried forward over the seconds it covers, and
    # no further: a gap in the context is a gap in the signal too
    merged = pd.merge_asof(sig, ctx[[TIME_COL, "RESP"]], on=TIME_COL,
                           direction="backward",
                           tolerance=pd.Timedelta(seconds=30))

    stats = {
        "user": user,
        "rows": int(len(merged)),
        "resp_present": int(merged["RESP"].notna().sum()),
        "resp_coverage": round(float(merged["RESP"].notna().mean()), 3),
        "resp_dropped_implausible": int(bad),
        "resp_median": (round(float(merged["RESP"].median()), 1)
                        if merged["RESP"].notna().any() else float("nan")),
        "resp_distinct": int(merged["RESP"].nunique()),
    }
    return merged, stats


def main():
    ap = argparse.ArgumentParser(
        description="Add a breathing-rate channel to the six already built, "
                    "so the users match the seven the encoder expects.")
    ap.add_argument("--signal-dir", default=SIGNAL_DIR)
    ap.add_argument("--context-dir", default=CONTEXT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--users", default=",".join(USERS))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    print("=" * 72)
    print("Adding a breathing channel to the user signals")
    print("=" * 72)
    print("RESP comes from the 30-second context files, carried forward onto")
    print("the 1 Hz grid, because the raw files were deleted after the")
    print("context was derived. Each value therefore repeats 30 times.")
    print()
    print("Enough to answer whether a breathing channel changes anything at")
    print("all. If it does, the raw files are worth downloading again to")
    print("compute it properly.")
    print("=" * 72)
    print()

    frames, summary = [], []
    for u in users:
        print("%s" % u)
        df, st = build_user(u, args.signal_dir, args.context_dir)
        if df is None:
            print()
            continue

        out = df.copy()
        path = os.path.join(args.out_dir, "%s.csv" % u)
        out.to_csv(path, index=False)
        frames.append(out)
        summary.append(st)

        print("    %d rows, RESP on %.0f%% of them, median %.1f breaths/min"
              % (st["rows"], 100 * st["resp_coverage"], st["resp_median"]))
        if st["resp_dropped_implausible"]:
            print("    %d context values dropped as outside %.0f to %.0f"
                  % (st["resp_dropped_implausible"], MIN_BR, MAX_BR))
        print("    saved %s" % path)
        print()

    if not frames:
        raise SystemExit("Nothing produced.")

    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(os.path.join(args.out_dir, "users_all.csv"), index=False)

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  %-9s %10s %10s %10s %10s"
          % ("user", "rows", "RESP %", "median", "distinct"))
    for s in summary:
        print("  %-9s %10d %9.0f%% %10.1f %10d"
              % (s["user"], s["rows"], 100 * s["resp_coverage"],
                 s["resp_median"], s["resp_distinct"]))
    print("  " + "-" * 52)
    print("  %-9s %10d" % ("TOTAL", len(allf)))
    print()

    have = [c for c in SEVEN if c in allf.columns]
    print("channels written: %d" % len(have))
    print("  %s" % ", ".join(have))
    if len(have) != 7:
        print("  WARNING: expected 7, to match configs/w7.yaml")
    print()
    print("value ranges:")
    print(allf[have].agg(["min", "max", "mean"]).round(3).to_string())
    print()
    print("  RESP should sit roughly 10 to 25 breaths a minute. A resting")
    print("  adult is 12 to 20.")
    print()

    n_steps = int(allf["RESP"].diff().ne(0).sum())
    print("RESP changes value %d times in %d rows, about once every %.0f"
          % (n_steps, len(allf), len(allf) / max(n_steps, 1)))
    print("seconds. A true 1 Hz breathing rate would change far more often.")
    print("This is the staircase the 30-second source produces.")
    print()

    L = []
    L.append("# User signals with a breathing channel")
    L.append("")
    L.append("Written by `btp/40_build_user_7ch.py`, so the users match the")
    L.append("seven channels `configs/w7.yaml` expects.")
    L.append("")
    L.append("## Two limitations, both real")
    L.append("")
    L.append("**Resolution.** The raw files were deleted after the context was")
    L.append("derived, so the breathing rate could not be recomputed per")
    L.append("second. It comes from the 30-second context files, carried")
    L.append("forward, so each value repeats 30 times. Within a 60 second")
    L.append("window the channel holds at most two distinct values.")
    L.append("")
    L.append("**Source.** On WESAD the breathing rate is computed from the")
    L.append("chest respiration belt, which measures the chest expanding.")
    L.append("Here it is derived from the pulse, because a wristband cannot")
    L.append("measure breathing directly. Measured on WESAD, where both are")
    L.append("available, those two agree at a correlation of **0.185**.")
    L.append("")
    L.append("So the channel means something slightly different at each end,")
    L.append("and the encoder's filters for it are learned from the cleaner")
    L.append("version. Both points belong in any report using these files.")
    L.append("")
    L.append("## Per user")
    L.append("")
    L.append("| user | rows | RESP coverage | median breaths/min | distinct values |")
    L.append("|---|---:|---:|---:|---:|")
    for s in summary:
        L.append("| %s | %d | %.0f%% | %.1f | %d |"
                 % (s["user"], s["rows"], 100 * s["resp_coverage"],
                    s["resp_median"], s["resp_distinct"]))
    L.append("")
    L.append("## What this is for")
    L.append("")
    L.append("To answer whether a breathing channel changes the result at")
    L.append("all. If it does, the raw files are worth downloading again to")
    L.append("compute it at one value per second. If it does not, 44 GB of")
    L.append("downloading has been saved.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"per_user": summary, "channels": have,
               "resp_source": "30s context, carried forward"},
              open(os.path.join(args.out_dir, "build_stats.json"), "w"),
              indent=2)
    print("saved %s/users_all.csv" % args.out_dir)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
