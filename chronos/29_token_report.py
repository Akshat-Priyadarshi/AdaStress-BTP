"""
29_token_report.py

WHY THIS FILE EXISTS
    The tokeniser reported how many distinct tokens each channel used, but
    only for the FIRST method. That was a mistake on my part, and it matters,
    because the first run showed two channels looking dead:

        SkinTemp   22 distinct tokens
        PR        413 distinct tokens

    while BVP, the accelerometer and EDA all used thousands.

    The two weak ones are exactly the two signals with a large offset. Skin
    temperature sits around 31 degrees; pulse rate sits around 80 bpm. The
    first method divides by the average size of the signal:

        MeanScaleUniformBins:      scaled = x / mean(|x|)

    For temperature that is 31.5 / 31.5, which is about 1.0 for every single
    sample. The real variation, a fraction of a degree, is swamped by the
    offset and everything falls in the same bin.

    The second method subtracts the mean first:

        StanNormalizeUniformBins:  scaled = (x - mean) / std

    which should recover both. This is very likely why SensorLLM added it;
    upstream Chronos only ships the first.

    Both sets of tokens are already saved, so this can be checked without
    tokenising anything again.

WHAT IT DOES
    Reads the saved .npz files and reports, per channel AND per method:

      distinct tokens   how many of the 4096 are actually used
      spread            how wide a range the tokens cover
      share of the most common token, which shows whether one bin dominates

    Then says which method to use for which channel.

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/29_token_report.py
"""

import os
import re
import glob
import json
import argparse
import numpy as np

TOKEN_DIR = "data/processed/chronos_tokens"
REPORT = "docs/54_token_method_comparison.md"

N_TOKENS = 4096
N_SPECIAL = 2
PAD = 0


def describe(arr):
    """Summarise one channel's tokens under one method."""
    flat = arr.reshape(-1)
    real = flat[flat >= N_SPECIAL]        # ignore the PAD rows
    if real.size == 0:
        return None

    vals, counts = np.unique(real, return_counts=True)
    top_share = float(counts.max() / real.size)

    return {
        "n_values": int(real.size),
        "distinct": int(vals.size),
        "min": int(vals.min()),
        "max": int(vals.max()),
        "spread": int(vals.max() - vals.min()),
        "top_token_share": round(top_share, 4),
        "pad_share": round(float((flat < N_SPECIAL).sum() / flat.size), 4),
    }


def verdict(d):
    """Plain words for whether a channel carries anything."""
    if d is None:
        return "no data"
    if d["distinct"] < 30 or d["top_token_share"] > 0.5:
        return "DEAD, one bin dominates"
    if d["distinct"] < 200:
        return "weak"
    return "alive"


def main():
    ap = argparse.ArgumentParser(
        description="Compare the two Chronos tokenisers on the saved tokens.")
    ap.add_argument("--token-dir", default=TOKEN_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.token_dir, "*.npz")))
    if not files:
        raise SystemExit("No .npz files in %s. Run btp/28_chronos_tokens.py "
                         "first." % args.token_dir)

    # group by method, accumulate across users
    by_method = {}
    for f in files:
        m = re.match(r"(.+)_(MeanScaleUniformBins|StanNormalizeUniformBins)\.npz$",
                     os.path.basename(f))
        if not m:
            continue
        user, method = m.group(1), m.group(2)
        by_method.setdefault(method, {}).setdefault(user, f)

    methods = sorted(by_method)
    print("=" * 74)
    print("DO THE TWO TOKENISERS DISAGREE?")
    print("=" * 74)
    print("Both were saved by btp/28_chronos_tokens.py, so nothing is")
    print("recomputed here.")
    print()
    print("  MeanScaleUniformBins      scaled = x / mean(|x|)      upstream Chronos")
    print("  StanNormalizeUniformBins  scaled = (x - mean) / std   SensorLLM's addition")
    print()

    results = {}
    channels = None

    for method in methods:
        users = by_method[method]
        print("%s   (%d users)" % (method, len(users)))
        per_channel = {}

        for user, path in sorted(users.items()):
            with np.load(path) as z:
                if channels is None:
                    channels = list(z.files)
                for c in z.files:
                    per_channel.setdefault(c, []).append(z[c])

        summary = {}
        for c in channels:
            arr = np.concatenate(per_channel[c], axis=0)
            summary[c] = describe(arr)
        results[method] = summary
        print("    read %d channels" % len(summary))
        print()

    # ---------------- the comparison ----------------
    print("=" * 74)
    print("DISTINCT TOKENS USED, OUT OF 4094 POSSIBLE")
    print("=" * 74)
    print("  %-10s %14s %10s   %14s %10s"
          % ("channel", "MeanScale", "verdict", "StanNormalize", "verdict"))
    print("  " + "-" * 68)

    better = {}
    for c in channels:
        a = results.get("MeanScaleUniformBins", {}).get(c)
        b = results.get("StanNormalizeUniformBins", {}).get(c)
        na = a["distinct"] if a else 0
        nb = b["distinct"] if b else 0
        print("  %-10s %14d %10s   %14d %10s"
              % (c, na, verdict(a), nb, verdict(b)))
        better[c] = ("StanNormalizeUniformBins" if nb > na * 1.2
                     else "MeanScaleUniformBins" if na > nb * 1.2
                     else "either")
    print()

    print("=" * 74)
    print("HOW CONCENTRATED ARE THE TOKENS?")
    print("=" * 74)
    print("Share taken by the single most common token. Above 0.5 means one")
    print("bin swallows the channel and there is nothing left to read.")
    print()
    print("  %-10s %16s %22s" % ("channel", "MeanScale", "StanNormalize"))
    print("  " + "-" * 52)
    for c in channels:
        a = results.get("MeanScaleUniformBins", {}).get(c)
        b = results.get("StanNormalizeUniformBins", {}).get(c)
        print("  %-10s %16.3f %22.3f"
              % (c, a["top_token_share"] if a else float("nan"),
                 b["top_token_share"] if b else float("nan")))
    print()

    print("=" * 74)
    print("WHICH METHOD FOR WHICH CHANNEL")
    print("=" * 74)
    for c in channels:
        print("  %-10s %s" % (c, better[c]))
    print()
    print("The pattern to look for: signals centred on zero (BVP, the")
    print("accelerometer) work under either method, because dividing by the")
    print("average size already spreads them out. Signals with a large offset")
    print("(skin temperature around 31 degrees, pulse rate around 80 bpm)")
    print("need the mean subtracted first, or every value lands in the same")
    print("bin.")
    print()

    # ---------------- report ----------------
    L = []
    L.append("# Which Chronos tokeniser, and for which channel")
    L.append("")
    L.append("Written by `btp/29_token_report.py` from the tokens already")
    L.append("saved by `btp/28_chronos_tokens.py`. Nothing is recomputed.")
    L.append("")
    L.append("SensorLLM ships two tokenisers and they disagree:")
    L.append("")
    L.append("| method | what it does | from |")
    L.append("|---|---|---|")
    L.append("| `MeanScaleUniformBins` | `x / mean(abs(x))` | upstream Chronos |")
    L.append("| `StanNormalizeUniformBins` | `(x - mean) / std` | SensorLLM added it |")
    L.append("")
    L.append("## Distinct tokens used, out of 4094 possible")
    L.append("")
    L.append("| channel | MeanScale | verdict | StanNormalize | verdict |")
    L.append("|---|---:|---|---:|---|")
    for c in channels:
        a = results.get("MeanScaleUniformBins", {}).get(c)
        b = results.get("StanNormalizeUniformBins", {}).get(c)
        L.append("| %s | %d | %s | %d | %s |"
                 % (c, a["distinct"] if a else 0, verdict(a),
                    b["distinct"] if b else 0, verdict(b)))
    L.append("")
    L.append("## Share taken by the single most common token")
    L.append("")
    L.append("Above 0.5 means one bin swallows the channel.")
    L.append("")
    L.append("| channel | MeanScale | StanNormalize |")
    L.append("|---|---:|---:|")
    for c in channels:
        a = results.get("MeanScaleUniformBins", {}).get(c)
        b = results.get("StanNormalizeUniformBins", {}).get(c)
        L.append("| %s | %.3f | %.3f |"
                 % (c, a["top_token_share"] if a else float("nan"),
                    b["top_token_share"] if b else float("nan")))
    L.append("")
    L.append("## Which method for which channel")
    L.append("")
    L.append("| channel | use |")
    L.append("|---|---|")
    for c in channels:
        L.append("| %s | %s |" % (c, better[c]))
    L.append("")
    L.append("## Why")
    L.append("")
    L.append("Signals centred on zero, such as BVP and the accelerometer, work")
    L.append("under either method: dividing by the average size already spreads")
    L.append("them across the bins.")
    L.append("")
    L.append("Signals with a large offset do not. Skin temperature sits around")
    L.append("31 degrees, so `x / mean(abs(x))` is about 1.0 for every sample")
    L.append("and the real variation, a fraction of a degree, is swamped.")
    L.append("Pulse rate has the same problem around 80 bpm. Subtracting the")
    L.append("mean first recovers both.")
    L.append("")
    L.append("This is very likely why SensorLLM added the second tokeniser;")
    L.append("upstream Chronos ships only the first.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")

    json.dump({"per_method": results, "recommended": better},
              open(os.path.join(args.token_dir, "method_comparison.json"), "w"),
              indent=2)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
