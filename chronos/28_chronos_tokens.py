"""
28_chronos_tokens.py

WHY THIS FILE EXISTS
    We were asked to start generating time-series tokens with Chronos, from
    the raw data rather than our 1 Hz averaged version, using PR, BVP, skin
    temperature and the other channels in the raw files, with the stress score
    added.

    Nothing in the AdaStress repository does this. Chronos appears nowhere in
    its code, its requirements or its paper. This is a new direction, and it
    is recorded as one.

WHAT A TOKEN IS, IN PLAIN WORDS
    A language model reads text as a list of numbers from a fixed vocabulary.
    Chronos does the same to a signal:

      1. take one chunk of one channel
      2. divide it by its own size, so a loud and a quiet version of the same
         shape become identical
      3. round each value into one of 4096 bins

    So a chunk of BVP becomes a list of integers between 2 and 4095.

WHY THIS HELPS, BEYOND FOLLOWING THE INSTRUCTION
    Two things we measured get fixed.

    The units. WESAD stores accelerometer in device counts, EmbracePlus
    converts to g, about 64 times apart. With WESAD's scaler the accelerometer
    went dead (sd 0.01); with the user's own scaler BVP exploded to plus or
    minus 138. Step 2 is per chunk, so the units cancel: the same signal 64
    times apart produces identical tokens.

    The averaging. Our 1 Hz version averages BVP over a whole second, and a
    pulse swings above and below zero every beat, so the mean cancels it:
    about 8% of the variation survived. Reading the raw 64 Hz signal keeps the
    pulse. This is exactly what the instruction to use raw data fixes.

THE TWO TOKENISERS, AND WHY BOTH
    SensorLLM ships two, and they disagree:

      MeanScaleUniformBins       scaled = x / mean(|x|)      upstream Chronos
      StanNormalizeUniformBins   scaled = (x - mean) / std   SensorLLM's own

    Which to use is a real choice, so both are produced and both reported.

CHUNKING
    Chronos reads at most 512 values at once. A 60 second window of 64 Hz BVP
    is 3840 values, seven times too long. So each window is cut into chunks of
    480, which divides 3840 evenly and stays under the limit. SensorLLM splits
    into segments the same way.

    One 60 second window produces:

      BVP               64 Hz   3840 values   8 chunks
      accelerometer x   64 Hz   3840 values   8 chunks
      accelerometer y   64 Hz   3840 values   8 chunks
      accelerometer z   64 Hz   3840 values   8 chunks
      EDA                4 Hz    240 values   1 chunk
      PR (derived)       4 Hz    240 values   1 chunk
      temperature        1 Hz     60 values   1 chunk

THE STRESS SCORE
    Attached per window from the model's own predictions. There is no survey
    ground truth yet, so this is what the model said, not what the person
    reported. The column is called stress_score_predicted throughout so it
    cannot be mistaken for a measurement later.

THE NUMBER THIS SCRIPT EXISTS TO PRODUCE
    How many DISTINCT tokens each channel uses.

    A channel using 5 of a possible 4096 is flat: there is nothing in it for a
    model to read. One using hundreds is alive. This says, in one table,
    whether raw BVP comes alive where 1 Hz BVP was dead.

OUTPUT
    data/processed/chronos_tokens/<user>_<method>.npz
    data/processed/chronos_tokens/<user>_windows.csv
    docs/53_chronos_tokens.md

HOW TO RUN
    Start with the smallest user to check the design:

        python3 btp/28_chronos_tokens.py --users user20

    Then all of them:

        python3 btp/28_chronos_tokens.py
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/embraceplus"
OUT_DIR = "data/processed/chronos_tokens"
PRED_CSV = "runs/P06b_w6_pooled_noctx_on_users_user/predictions.csv"
REPORT = "docs/53_chronos_tokens.md"

# Chronos vocabulary, exactly as the released config has it.
N_TOKENS = 4096
N_SPECIAL = 2          # 0 is PAD, 1 is EOS, so real values start at 2
LOW, HIGH = -15.0, 15.0

CHUNK = 480            # under Chronos's 512 limit, divides 3840 evenly
WINDOW_SEC = 60
STRIDE_SEC = 30

# channel name -> (raw file suffix, column inside it, rate in Hz)
CHANNELS = {
    "BVP":      ("bvp",           "bvp",             64),
    "ACC_X":    ("accelerometer", "accelerometer_x", 64),
    "ACC_Y":    ("accelerometer", "accelerometer_y", 64),
    "ACC_Z":    ("accelerometer", "accelerometer_z", 64),
    "EDA":      ("eda",           "eda",              4),
    "PR":       ("PR",            "PR",               4),
    "SkinTemp": ("temperature",   "temperature",      1),
}

TIME_COL = "datetime_ist"


def make_bin_edges():
    """Chronos's bins: 4093 centres evenly spread over -15 to 15."""
    centers = np.linspace(LOW, HIGH, N_TOKENS - N_SPECIAL - 1)
    return np.concatenate([[-1e20], (centers[1:] + centers[:-1]) / 2, [1e20]])


BIN_EDGES = make_bin_edges()


def tokenise(values, method):
    """Turn one chunk of one channel into token IDs.

    Both methods divide the chunk by something worked out from the chunk
    itself. That is what makes the result independent of the device's units.
    """
    x = np.asarray(values, dtype=float)
    if np.isnan(x).all():
        return np.zeros(len(x), dtype=np.int16)
    x = np.nan_to_num(x, nan=0.0)

    if method == "MeanScaleUniformBins":
        scale = float(np.mean(np.abs(x)))
        scaled = x / (scale if scale > 0 else 1.0)
    else:                                   # StanNormalizeUniformBins
        mu, sd = float(x.mean()), float(x.std())
        scaled = (x - mu) / (sd if sd > 0 else 1.0)

    ids = np.searchsorted(BIN_EDGES, scaled, side="left") + N_SPECIAL
    return np.clip(ids, N_SPECIAL, N_TOKENS - 1).astype(np.int16)


def load_channel(udir, prefix, date, file_key, col):
    """Read one raw channel for one day at its own rate. None if absent."""
    path = os.path.join(udir, "%s_%s_%s.csv" % (prefix, date, file_key))
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, usecols=[TIME_COL, col])
    except ValueError:
        return None
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    if df.empty:
        return None
    return df.set_index(TIME_COL)[col]


def build_user(user, udir, preds, methods):
    """Tokenise every window for one user."""
    prefix, dates = None, set()
    for f in glob.glob(os.path.join(udir, "*.csv")):
        m = re.search(r"(.+)_(\d{4}-\d{2}-\d{2})_(\w+)\.csv$",
                      os.path.basename(f))
        if m:
            prefix = m.group(1)
            dates.add(m.group(2))
    if not dates:
        return None, None, None

    out = {m: {c: [] for c in CHANNELS} for m in methods}
    win_rows = []
    stats = {c: {"chunks": 0, "distinct": set(), "empty": 0} for c in CHANNELS}

    for date in sorted(dates):
        series = {c: load_channel(udir, prefix, date, f, col)
                  for c, (f, col, _hz) in CHANNELS.items()}
        present = {c: s for c, s in series.items() if s is not None}
        if not present:
            continue

        t0 = min(s.index.min() for s in present.values())
        t1 = max(s.index.max() for s in present.values())
        print("    %s: %s to %s   have: %s"
              % (date, t0.strftime("%H:%M"), t1.strftime("%H:%M"),
                 ",".join(sorted(present))))

        start = t0
        while start + pd.Timedelta(seconds=WINDOW_SEC) <= t1:
            end = start + pd.Timedelta(seconds=WINDOW_SEC)

            for cname, (_f, _col, hz) in CHANNELS.items():
                s = series.get(cname)
                want_n = WINDOW_SEC * hz
                n_chunks = max(1, want_n // CHUNK)
                want = n_chunks * CHUNK

                seg = None
                if s is not None:
                    seg = s.loc[start:end].to_numpy(dtype=float)
                    if len(seg) == 0:
                        seg = None

                if seg is None:
                    # all-PAD, so the shape stays the same for every window
                    for m in methods:
                        out[m][cname].append(np.zeros((n_chunks, CHUNK),
                                                      dtype=np.int16))
                    stats[cname]["empty"] += n_chunks
                    continue

                # exact length: trim if long, hold the last value if short
                if len(seg) >= want:
                    seg = seg[:want]
                else:
                    seg = np.concatenate([seg,
                                          np.full(want - len(seg), seg[-1])])

                blocks = seg.reshape(n_chunks, CHUNK)
                for m in methods:
                    toks = np.stack([tokenise(b, m) for b in blocks])
                    out[m][cname].append(toks)
                    if m == methods[0]:
                        stats[cname]["chunks"] += n_chunks
                        stats[cname]["distinct"].update(
                            np.unique(toks).tolist())

            win_rows.append({"Subject_ID": user, "date": date,
                             "window_start": start})
            start += pd.Timedelta(seconds=STRIDE_SEC)

    if not win_rows:
        return None, None, None

    windows = pd.DataFrame(win_rows).sort_values("window_start")

    # the model's own stress score, matched on nearest window time
    if preds is not None:
        p = preds[preds["Subject_ID"] == user]
        if len(p):
            right = (p[["mid_time", "stress_score"]]
                     .rename(columns={"mid_time": "window_start",
                                      "stress_score":
                                          "stress_score_predicted"})
                     .sort_values("window_start"))
            windows = pd.merge_asof(
                windows, right, on="window_start", direction="nearest",
                tolerance=pd.Timedelta(seconds=WINDOW_SEC))
    if "stress_score_predicted" not in windows.columns:
        windows["stress_score_predicted"] = np.nan

    packed = {m: {c: (np.concatenate(v, axis=0) if v else
                      np.zeros((0, CHUNK), dtype=np.int16))
                  for c, v in out[m].items()} for m in methods}
    return packed, windows, stats


def main():
    ap = argparse.ArgumentParser(
        description="Generate Chronos time-series tokens from the raw "
                    "EmbracePlus channels.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--preds", default=PRED_CSV)
    ap.add_argument("--users", default=None,
                    help="comma separated, e.g. user20. Default all. Start "
                         "with the smallest to check the design.")
    ap.add_argument("--methods",
                    default="MeanScaleUniformBins,StanNormalizeUniformBins")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    users = sorted(d for d in os.listdir(args.raw_dir)
                   if os.path.isdir(os.path.join(args.raw_dir, d)))
    if args.users:
        want = {u.strip() for u in args.users.split(",")}
        users = [u for u in users if u in want]
    if not users:
        raise SystemExit("No matching user folders in %s" % args.raw_dir)

    preds = None
    if os.path.exists(args.preds):
        preds = pd.read_csv(args.preds, parse_dates=["mid_time"])
        print("stress scores: %s  (%d windows)" % (args.preds, len(preds)))
    else:
        print("no predictions at %s, stress will be left empty" % args.preds)

    print()
    print("=" * 70)
    print("Chronos tokens from the RAW channels")
    print("window  : %d s, stride %d s" % (WINDOW_SEC, STRIDE_SEC))
    print("chunk   : %d values  (Chronos reads at most 512 at once)" % CHUNK)
    print("vocab   : %d tokens, %d reserved for PAD and EOS"
          % (N_TOKENS, N_SPECIAL))
    print("methods : %s" % ", ".join(methods))
    print("users   : %s" % ", ".join(users))
    print("=" * 70)
    print()

    all_stats, summary = {}, []

    for user in users:
        print(user)
        packed, windows, stats = build_user(
            user, os.path.join(args.raw_dir, user), preds, methods)
        if packed is None:
            print("    nothing produced")
            print()
            continue

        for m in methods:
            path = os.path.join(args.out_dir, "%s_%s.npz" % (user, m))
            np.savez_compressed(path, **packed[m])
            print("    saved %-45s %6.1f MB"
                  % (os.path.basename(path), os.path.getsize(path) / 1e6))

        windows.to_csv(os.path.join(args.out_dir, "%s_windows.csv" % user),
                       index=False)
        n_stress = int(windows["stress_score_predicted"].notna().sum())
        print("    windows: %d, with a stress score: %d"
              % (len(windows), n_stress))
        print()

        all_stats[user] = stats
        summary.append({"user": user, "windows": int(len(windows)),
                        "with_stress": n_stress})

    if not all_stats:
        raise SystemExit("Nothing produced.")

    print("=" * 70)
    print("HOW MANY DISTINCT TOKENS DOES EACH CHANNEL USE?")
    print("=" * 70)
    print("A channel using a handful of the 4096 is flat: nothing to read.")
    print("One using hundreds is alive.")
    print()
    print("  %-10s %10s %10s %10s  %s"
          % ("channel", "chunks", "distinct", "empty", "verdict"))
    print("  " + "-" * 60)

    merged = {}
    for c in CHANNELS:
        chunks = sum(all_stats[u][c]["chunks"] for u in all_stats)
        distinct = set()
        for u in all_stats:
            distinct |= all_stats[u][c]["distinct"]
        empty = sum(all_stats[u][c]["empty"] for u in all_stats)
        n = len(distinct)
        verdict = ("DEAD, nothing to read" if n < 20
                   else "weak" if n < 100 else "alive")
        merged[c] = {"chunks": int(chunks), "distinct": int(n),
                     "empty": int(empty), "verdict": verdict}
        print("  %-10s %10d %10d %10d  %s" % (c, chunks, n, empty, verdict))

    print()
    print("For comparison, our 1 Hz averaged BVP used about 10 distinct")
    print("tokens per window, because averaging a pulse over a second cancels")
    print("it. If raw BVP is alive above, using the raw data fixed that.")
    print()

    tot_w = sum(r["windows"] for r in summary)
    tot_s = sum(r["with_stress"] for r in summary)
    print("windows: %d, with a stress score: %d" % (tot_w, tot_s))
    print()

    L = []
    L.append("# Chronos time-series tokens from the raw channels")
    L.append("")
    L.append("Written by `btp/28_chronos_tokens.py`.")
    L.append("")
    L.append("**Not part of AdaStress.** Chronos appears nowhere in that")
    L.append("repository's code, requirements or paper. A new direction,")
    L.append("recorded as one.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| window | %d s, stride %d s |" % (WINDOW_SEC, STRIDE_SEC))
    L.append("| chunk | %d values (Chronos reads at most 512) |" % CHUNK)
    L.append("| vocabulary | %d tokens, %d reserved |" % (N_TOKENS, N_SPECIAL))
    L.append("| methods | %s |" % ", ".join(methods))
    L.append("| stress | the model's own prediction, not a survey answer |")
    L.append("")
    L.append("## Channels")
    L.append("")
    L.append("| channel | source file | rate | values per window | chunks |")
    L.append("|---|---|---:|---:|---:|")
    for c, (f, _col, hz) in CHANNELS.items():
        v = WINDOW_SEC * hz
        L.append("| %s | `%s` | %d Hz | %d | %d |"
                 % (c, f, hz, v, max(1, v // CHUNK)))
    L.append("")
    L.append("`PR` is derived from the heartbeat timestamps by")
    L.append("`btp/27_derive_pr.py`, because EmbracePlus ships no PR channel.")
    L.append("")
    L.append("## Distinct tokens used")
    L.append("")
    L.append("| channel | chunks | distinct tokens | empty | verdict |")
    L.append("|---|---:|---:|---:|---|")
    for c, s in merged.items():
        L.append("| %s | %d | %d | %d | %s |"
                 % (c, s["chunks"], s["distinct"], s["empty"], s["verdict"]))
    L.append("")
    L.append("Our 1 Hz averaged BVP used about 10 distinct tokens per window,")
    L.append("because averaging a pulse waveform over one second cancels it:")
    L.append("roughly 8% of the variation survived. Reading the raw 64 Hz")
    L.append("signal is what using the raw data fixes.")
    L.append("")
    L.append("## Per user")
    L.append("")
    L.append("| user | windows | with a stress score |")
    L.append("|---|---:|---:|")
    for r in summary:
        L.append("| %s | %d | %d |" % (r["user"], r["windows"],
                                       r["with_stress"]))
    L.append("| **total** | **%d** | **%d** |" % (tot_w, tot_s))
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"channels": merged, "users": summary},
              open(os.path.join(args.out_dir, "token_stats.json"), "w"),
              indent=2)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
