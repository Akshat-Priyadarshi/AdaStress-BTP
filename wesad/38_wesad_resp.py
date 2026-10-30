"""
38_wesad_resp.py

WHAT THIS DOES
    Works out a breathing rate for WESAD from the wrist BVP signal, so that
    respiration can become a seventh input channel alongside the six we
    already use.

    It also checks that derivation against WESAD's chest respiration belt,
    which is the one place we can see whether the method works at all. Our
    users have no belt, so this is the only ground truth available anywhere
    in the project.

WHY DERIVE IT WHEN WESAD HAS A REAL BELT
    Because the users do not have one.

    WESAD records respiration directly on a chest strap at 700 Hz. It would
    be easy to use. But if the encoder learned from a measured belt signal
    and were then fed a signal derived from pulse timing, those are two
    different quantities wearing the same column name, and the model would
    have no reason to read the second one correctly.

    So respiration is derived from BVP on the WESAD side too, by the same
    method the users will use. The belt is kept for checking, not for
    training.

HOW BREATHING IS RECOVERED FROM A PULSE
    Breathing is not measured by a wrist device, but it leaves a trace in the
    heart rhythm. The gap between heartbeats shortens as you breathe in and
    lengthens as you breathe out. This is respiratory sinus arrhythmia, and
    it is what makes the method possible.

        BVP at 64 Hz
          -> band-pass 0.5 to 8 Hz        keep the pulse, drop drift and noise
          -> find peaks, at least 0.33 s apart   so at most 180 bpm
          -> gaps between consecutive peaks
          -> resample those gaps onto a steady 4 Hz grid
          -> band-pass 0.1 to 0.5 Hz      6 to 30 breaths a minute
          -> count peaks in each window
          -> breaths per minute

    The same chain the users' RESP comes from, except that their heartbeat
    times are given by the device and here they have to be detected.

WHAT THE CHECK FOUND, WHEN IT WAS RUN
    Mean correlation with the belt across the 15 subjects: 0.185. One
    subject came out negative. The median difference was 2.8 breaths a
    minute, against a resting range of roughly 12 to 20.

    The level is about right - derived 18.9, belt 16.7 - but the
    moment-to-moment tracking is poor.

    This is not a fault in the chain. On a synthetic recording with clean
    beats the same code recovered a breathing rate of 16.0 against a true
    15.0, and tracked a drifting rate at 0.953. The modulation is simply
    weak in real wrist PPG, and movement swamps it. Published work deriving
    respiratory rate this way typically reaches 0.5 to 0.8 against a
    reference in controlled conditions.

    RESP is being used anyway, as a deliberate choice: breathing changes
    under stress, and in free-living data it may carry more than it does in
    a laboratory where everyone sits still. But it is a weak proxy for a
    measured breathing rate, and every report using it should say so and
    quote the 0.185.

WHAT THE CHECK TELLS US
    The script reports, per subject, the correlation between the derived
    breathing rate and the rate measured by the belt, plus the median
    difference in breaths per minute.

    A good agreement means the method is sound and can be trusted on the
    users, where nothing can be checked. A poor one is worth knowing before
    44 GB is downloaded on the strength of it.

    There is reason for caution. On the user data we measured consecutive
    beat intervals jumping by a median of 122 ms, where a resting adult is
    20 to 50. Breathing pulled out of noisy beat timing will be noisy too.
    WESAD's BVP comes from the same kind of wrist sensor, so expect the same.

OUTPUT
    data/processed/wesad_resp/<subject>_resp.csv
        datetime, RESP_derived, RESP_belt        one row per second

    docs/60_wesad_resp.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/38_wesad_resp.py

    A few minutes. Reads the raw .pkl files, which WESAD ships.
"""

import os
import json
import pickle
import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT_DIR = "data/raw/WESAD"
OUT_DIR = "data/processed/wesad_resp"
REPORT = "docs/60_wesad_resp.md"

BVP_FS = 64            # wrist BVP, as WESAD records it
CHEST_FS = 700         # the respiration belt
IBI_GRID_HZ = 4.0      # the gaps are resampled onto this before band-passing
WINDOW_SEC = 1         # one value per second, to match the 1 Hz channels

# A human heart between 40 and 180 bpm. Outside that, the peak finder has
# missed a beat or counted one twice.
MIN_IBI, MAX_IBI = 0.333, 1.5

# 0.1 to 0.5 Hz is 6 to 30 breaths a minute, which covers rest through
# moderate exertion.
RESP_LOW, RESP_HIGH = 0.1, 0.5

MAX_GAP_SEC = 30.0     # longer than this and the stretch is broken


def detect_beats(bvp, fs=BVP_FS):
    """Find heartbeats in a BVP trace.

    The users' device hands us beat timestamps directly. WESAD does not, so
    they have to be found: band-pass to isolate the pulse, then take the
    peaks.
    """
    x = np.asarray(bvp, dtype=float).flatten()
    x = np.nan_to_num(x, nan=0.0)
    if len(x) < fs * 10:
        return np.array([])

    nyq = fs / 2.0
    b, a = butter(3, [0.5 / nyq, min(8.0 / nyq, 0.99)], btype="band")
    filt = filtfilt(b, a, x)

    # at most 180 bpm, and a peak must stand out from the local noise
    peaks, _ = find_peaks(filt, distance=int(fs * MIN_IBI),
                          prominence=np.std(filt) * 0.3)
    return peaks / float(fs)          # seconds from the start


def rate_from_segment(seg, fs, method):
    """Breathing rate from one band-passed segment.

    peaks
        Count the rises and falls. Simple, and what the mentor's notebook
        does, but over 30 seconds it can only return even numbers: 8 peaks
        is 16 breaths a minute, 9 is 18. The resolution is 2 breaths a
        minute against a plausible range of roughly 12 to 22, so both the
        estimate and anything it is compared against get squashed into about
        six possible values. That alone can destroy a correlation.

    spectral
        Take the power spectrum and find the frequency with the most energy
        between 0.1 and 0.5 Hz, then multiply by 60. The usual way of reading
        respiratory rate out of heart-rate variability, and continuous rather
        than stepped.

        We expected this to beat peak counting, because the stepping looked
        like a plausible reason for the poor agreement with WESAD's belt. It
        does not. Tested on a synthetic recording where breathing drifts from
        12 to 22 breaths a minute, peak counting tracked the truth at 0.953
        and the spectrum at 0.565. Peak counting is the default for that
        reason.
    """
    if method == "peaks":
        pk, _ = find_peaks(seg, distance=fs * 1.5)
        return len(pk) * (60.0 / (len(seg) / fs))

    # welch needs a few cycles of the slowest frequency we care about
    nper = min(len(seg), int(fs * 60))
    if nper < fs * 20:
        return np.nan
    f, pxx = welch(seg, fs=fs, nperseg=nper)
    band = (f >= RESP_LOW) & (f <= RESP_HIGH)
    if not band.any() or not np.isfinite(pxx[band]).any():
        return np.nan
    return float(f[band][np.argmax(pxx[band])] * 60.0)


def resp_from_beats(beat_sec, total_sec, window_sec=WINDOW_SEC,
                    method="peaks"):
    """Breathing rate from heartbeat times.

    Returns one value per second, with gaps left as NaN rather than filled.
    Inventing a breathing rate where no beats were detected would be making
    data up.
    """
    out = np.full(int(total_sec / window_sec) + 1, np.nan)
    if len(beat_sec) < 20:
        return out

    ibi = np.diff(beat_sec)
    at = beat_sec[1:]

    ok = (ibi >= MIN_IBI) & (ibi <= MAX_IBI)
    ibi, at = ibi[ok], at[ok]
    if len(ibi) < 20:
        return out

    # split where the sensor stopped reading, so nothing is interpolated
    # across a hole
    breaks = np.where(np.diff(at) > MAX_GAP_SEC)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(at)]

    nyq = IBI_GRID_HZ / 2.0
    b, a = butter(2, [RESP_LOW / nyq, min(RESP_HIGH / nyq, 0.99)],
                  btype="band")

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg_at, seg_ibi = at[lo:hi], ibi[lo:hi]
        if len(seg_at) < 20:
            continue
        g0, g1 = seg_at[0], seg_at[-1]
        if g1 - g0 < 30:                       # too short to see a breath
            continue

        grid = np.arange(g0, g1, 1.0 / IBI_GRID_HZ)
        interp = np.interp(grid, seg_at, seg_ibi)
        filt = filtfilt(b, a, interp)

        n_per_win = int(window_sec * IBI_GRID_HZ) * 30   # 30 s of context
        step = int(window_sec * IBI_GRID_HZ)
        for s in range(0, len(filt) - n_per_win, step):
            seg = filt[s:s + n_per_win]
            rate = rate_from_segment(seg, IBI_GRID_HZ, method)
            idx = int((grid[s] + 15) / window_sec)       # centre of the window
            if 0 <= idx < len(out):
                out[idx] = rate
    return out


def resp_from_belt(resp, total_sec, fs=CHEST_FS, window_sec=WINDOW_SEC,
                   method="peaks"):
    """Breathing rate from the chest belt, for checking only.

    The belt measures chest expansion directly, so the rate is simply how
    often it rises and falls.
    """
    x = np.asarray(resp, dtype=float).flatten()
    x = np.nan_to_num(x, nan=0.0)
    out = np.full(int(total_sec / window_sec) + 1, np.nan)

    nyq = fs / 2.0
    b, a = butter(2, [RESP_LOW / nyq, min(RESP_HIGH / nyq, 0.99)],
                  btype="band")
    filt = filtfilt(b, a, x)

    n_per_win = int(30 * fs)                   # same 30 s of context
    step = int(window_sec * fs)
    for s in range(0, len(filt) - n_per_win, step):
        seg = filt[s:s + n_per_win]
        rate = rate_from_segment(seg, fs, method)
        idx = int((s / fs + 15) / window_sec)
        if 0 <= idx < len(out):
            out[idx] = rate
    return out


def process_subject(folder, method="peaks"):
    subject = os.path.basename(folder)
    pkl = os.path.join(folder, subject + ".pkl")
    if not os.path.exists(pkl):
        return None, None

    with open(pkl, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp = data["signal"]["wrist"]["BVP"]
    belt = data["signal"]["chest"]["Resp"]
    total_sec = len(bvp) / BVP_FS

    beats = detect_beats(bvp)
    derived = resp_from_beats(beats, total_sec, method=method)
    measured = resp_from_belt(belt, total_sec, method=method)

    n = min(len(derived), len(measured))
    derived, measured = derived[:n], measured[:n]

    t0 = pd.Timestamp("2026-01-01")
    df = pd.DataFrame({
        "second": np.arange(n),
        "datetime": [t0 + timedelta(seconds=int(i)) for i in range(n)],
        "RESP_derived": derived,
        "RESP_belt": measured,
    })
    df["Subject_ID"] = subject

    both = df.dropna(subset=["RESP_derived", "RESP_belt"])
    stats = {
        "subject": subject,
        "seconds": int(n),
        "beats_found": int(len(beats)),
        "derived_coverage": round(float(np.mean(~np.isnan(derived))), 3),
        "both_present": int(len(both)),
    }
    if len(both) > 100:
        stats["correlation"] = round(float(
            np.corrcoef(both["RESP_derived"], both["RESP_belt"])[0, 1]), 3)
        stats["median_abs_diff"] = round(float(
            np.median(np.abs(both["RESP_derived"] - both["RESP_belt"]))), 2)
        stats["derived_median"] = round(float(both["RESP_derived"].median()), 1)
        stats["belt_median"] = round(float(both["RESP_belt"].median()), 1)
    else:
        stats["correlation"] = float("nan")
        stats["median_abs_diff"] = float("nan")
        stats["derived_median"] = float("nan")
        stats["belt_median"] = float("nan")

    return df, stats


def main():
    ap = argparse.ArgumentParser(
        description="Derive breathing rate from WESAD's wrist BVP, and check "
                    "it against the chest belt.")
    ap.add_argument("--input-dir", default=INPUT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--method", choices=["peaks", "spectral"],
                    default="peaks",
                    help="how a rate is read out of the band-passed series. "
                         "peaks counts rises and falls, which over 30 s can "
                         "only return even numbers. spectral takes the "
                         "dominant frequency and is continuous.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    subjects = sorted(n for n in os.listdir(args.input_dir)
                      if os.path.isdir(os.path.join(args.input_dir, n))
                      and n.startswith("S"))
    if not subjects:
        raise SystemExit("No subject folders in %s" % args.input_dir)

    print("=" * 74)
    print("Breathing rate from WESAD's wrist BVP")
    print("=" * 74)
    print("Derived the same way the users' will be, so the two match:")
    print("  method: %s" % args.method)
    print("  BVP -> band-pass 0.5-8 Hz -> peaks -> gaps between beats")
    print("      -> 4 Hz grid -> band-pass %.1f-%.1f Hz -> count"
          % (RESP_LOW, RESP_HIGH))
    print()
    print("WESAD's chest belt is kept for CHECKING only, never for training.")
    print("The users have no belt, so a model trained on a measured signal")
    print("would be fed a derived one, and those are different quantities.")
    print("=" * 74)
    print()

    rows = []
    for s in subjects:
        print("  %s ..." % s, end=" ", flush=True)
        df, st = process_subject(os.path.join(args.input_dir, s),
                                 args.method)
        if df is None:
            print("no .pkl")
            continue
        df.to_csv(os.path.join(args.out_dir, "%s_resp.csv" % s), index=False)
        print("%d beats, %.0f%% coverage, correlation %.3f"
              % (st["beats_found"], 100 * st["derived_coverage"],
                 st["correlation"]))
        rows.append(st)

    if not rows:
        raise SystemExit("Nothing processed.")

    r = pd.DataFrame(rows)

    print()
    print("=" * 74)
    print("DOES THE DERIVED RATE MATCH THE BELT?")
    print("=" * 74)
    print("  %-7s %9s %9s %11s %9s %9s %9s"
          % ("subject", "beats", "coverage", "correlation", "med diff",
             "derived", "belt"))
    print("  " + "-" * 68)
    for _, x in r.iterrows():
        print("  %-7s %9d %8.0f%% %11.3f %9.2f %9.1f %9.1f"
              % (x["subject"], x["beats_found"], 100 * x["derived_coverage"],
                 x["correlation"], x["median_abs_diff"],
                 x["derived_median"], x["belt_median"]))
    print("  " + "-" * 68)
    print("  %-7s %9s %8.0f%% %11.3f %9.2f %9.1f %9.1f"
          % ("MEAN", "", 100 * r["derived_coverage"].mean(),
             r["correlation"].mean(), r["median_abs_diff"].mean(),
             r["derived_median"].mean(), r["belt_median"].mean()))
    print()

    c = r["correlation"].mean()
    d = r["median_abs_diff"].mean()
    print("How to read this:")
    print()
    if c > 0.5:
        print("  Correlation %.3f. The derived rate follows the belt." % c)
        print("  The method is sound enough to use on the users, where")
        print("  nothing can be checked.")
    elif c > 0.25:
        print("  Correlation %.3f. Some agreement, not much." % c)
        print("  Usable, but say plainly in any report that this is a weak")
        print("  proxy for a measured breathing rate.")
    else:
        print("  Correlation %.3f. The derived rate does not follow the" % c)
        print("  belt. Adding it as a seventh channel would be adding noise,")
        print("  and the 44 GB of downloading it would require is not")
        print("  justified on this evidence.")
    print()
    print("  Typical difference: %.1f breaths a minute. A resting adult is" % d)
    print("  12 to 20, so a difference of 4 or more is most of the range.")
    print()
    print("  A caution worth carrying: on the user data, consecutive beat")
    print("  intervals jumped by a median of 122 ms, where a resting adult")
    print("  is 20 to 50. Breathing pulled from noisy beat timing is noisy")
    print("  breathing, and WESAD's BVP comes from the same kind of sensor.")
    print()

    L = []
    L.append("# Breathing rate derived from WESAD's wrist BVP")
    L.append("")
    L.append("Written by `btp/38_wesad_resp.py`.")
    L.append("")
    L.append("## Why derive it when WESAD has a real belt")
    L.append("")
    L.append("Because the users do not have one. WESAD records respiration on")
    L.append("a chest strap, but if the encoder learned from a measured belt")
    L.append("signal and were then fed one derived from pulse timing, those")
    L.append("are two different quantities sharing a column name.")
    L.append("")
    L.append("So respiration is derived from BVP on both sides. The belt is")
    L.append("kept for checking only, and this is the only ground truth")
    L.append("available anywhere in the project.")
    L.append("")
    L.append("## The method")
    L.append("")
    L.append("Breathing leaves a trace in the heart rhythm: the gap between")
    L.append("beats shortens on the in-breath and lengthens on the out-breath.")
    L.append("")
    L.append("```")
    L.append("BVP at 64 Hz")
    L.append("  -> band-pass 0.5 to 8 Hz")
    L.append("  -> peaks, at least %.2f s apart" % MIN_IBI)
    L.append("  -> gaps between beats, %.2f to %.2f s kept" % (MIN_IBI, MAX_IBI))
    L.append("  -> resample to %.0f Hz" % IBI_GRID_HZ)
    L.append("  -> band-pass %.1f to %.1f Hz  (%d to %d breaths a minute)"
             % (RESP_LOW, RESP_HIGH, RESP_LOW * 60, RESP_HIGH * 60))
    L.append("  -> count peaks per 30 s, report per second")
    L.append("```")
    L.append("")
    L.append("## Agreement with the belt")
    L.append("")
    L.append("| subject | beats | coverage | correlation | median diff | derived | belt |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, x in r.iterrows():
        L.append("| %s | %d | %.0f%% | %.3f | %.2f | %.1f | %.1f |"
                 % (x["subject"], x["beats_found"],
                    100 * x["derived_coverage"], x["correlation"],
                    x["median_abs_diff"], x["derived_median"],
                    x["belt_median"]))
    L.append("| **mean** | | **%.0f%%** | **%.3f** | **%.2f** | %.1f | %.1f |"
             % (100 * r["derived_coverage"].mean(), r["correlation"].mean(),
                r["median_abs_diff"].mean(), r["derived_median"].mean(),
                r["belt_median"].mean()))
    L.append("")
    L.append("## What this decides")
    L.append("")
    L.append("If the derived rate follows the belt, the method can be trusted")
    L.append("on the users, where there is nothing to check it against, and")
    L.append("the seventh channel is worth the rebuild.")
    L.append("")
    L.append("If it does not, adding it would be adding noise, and the 44 GB")
    L.append("of re-downloading it would need is not justified.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    json.dump({"per_subject": rows,
               "mean_correlation": float(r["correlation"].mean()),
               "mean_abs_diff": float(r["median_abs_diff"].mean()),
               "mean_coverage": float(r["derived_coverage"].mean())},
              open(os.path.join(args.out_dir, "resp_check.json"), "w"),
              indent=2)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
