"""
27_derive_pr.py

WHY THIS FILE EXISTS
    We were asked to feed PR (pulse rate), BVP and skin temperature to
    Chronos. BVP and temperature are in the EmbracePlus download. PR is not.

    But systolicPeaks is, and that is a list of heartbeat timestamps. The gap
    between two beats IS the instantaneous heart rate: a 0.8 second gap means
    60 / 0.8 = 75 beats per minute.

    So PR is not missing, it just has to be worked out. Deriving it this way
    is arguably better than a pre-computed PR column, because we can see and
    control how bad beats are handled instead of trusting someone else's
    smoothing.

WHAT IT DOES
    For each user and each day:

      1. Reads the systolicPeaks file (one row per detected heartbeat).
      2. Works out the gap between consecutive beats.
      3. Turns each gap into a heart rate in beats per minute.
      4. Throws away impossible values.
      5. Resamples onto a steady 4 Hz grid, so PR lines up with EDA.

    It also writes the inter-beat gaps themselves, because those are what
    heart rate variability is computed from, and HRV is the part of the
    cardiac signal that actually tracks stress.

WHAT COUNTS AS A BAD BEAT
    A detected peak can be wrong in two ways. A missed beat makes one gap
    twice as long, which looks like a very slow heart. A doubled beat makes
    one gap half as long, which looks impossibly fast.

    We keep only rates between 40 and 180 bpm, which is gap lengths between
    0.333 and 1.5 seconds. Anything outside that is a detection error, not a
    person. The count of dropped beats is reported, because a high drop rate
    means the sensor was not making good contact and that day's PR should not
    be trusted.

WHY 4 Hz
    A heart beats about once a second, so PR genuinely carries new information
    about once a second. 4 Hz is comfortably above that and matches the EDA
    rate, which means PR and EDA produce the same number of samples per window
    and can be chunked identically for Chronos.

OUTPUT
    data/raw/embraceplus/<user>/<user>_acctN_<date>_PR.csv
        datetime_utc, datetime_ist, PR            4 Hz

    data/raw/embraceplus/<user>/<user>_acctN_<date>_IBI.csv
        datetime_utc, datetime_ist, IBI_ms        one row per beat

    Written next to the other sensors, in the same shape, so everything
    downstream treats PR like any other channel.

    docs/52_derived_pr.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/27_derive_pr.py

    A couple of minutes.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/embraceplus"
REPORT = "docs/52_derived_pr.md"

# A human heart between 40 and 180 bpm. Outside this the peak detector has
# missed a beat or counted one twice.
MIN_BPM, MAX_BPM = 40.0, 180.0

# PR is resampled onto this grid, matching the EDA rate so the two channels
# chunk identically for Chronos.
TARGET_HZ = 4

# If beats stop for longer than this, the sensor lost contact. We do not
# interpolate across the gap, because inventing a heart rate for a period
# where nothing was detected would be making data up.
MAX_GAP_SECONDS = 10.0


def derive_from_peaks(peaks_utc):
    """Turn beat timestamps into heart rate and inter-beat gaps.

    Returns (DataFrame, stats). The frame has one row per usable beat.
    """
    peaks = pd.Series(pd.to_datetime(peaks_utc)).dropna().sort_values()
    n_peaks = len(peaks)
    if n_peaks < 3:
        return None, {"peaks": n_peaks, "kept": 0, "dropped": 0,
                      "dropped_pct": 0.0, "gaps_over_limit": 0}

    gaps = peaks.diff().dt.total_seconds()

    # A beat's rate is credited to the beat that ended the gap.
    bpm = 60.0 / gaps

    ok = (bpm >= MIN_BPM) & (bpm <= MAX_BPM)
    long_gap = gaps > MAX_GAP_SECONDS

    df = pd.DataFrame({
        "datetime_utc": peaks.values,
        "IBI_ms": (gaps * 1000.0).values,
        "PR": bpm.values,
        "ok": ok.values,
        "long_gap": long_gap.values,
    }).iloc[1:]          # the first beat has no gap before it

    kept = df[df["ok"]].copy()
    stats = {
        "peaks": int(n_peaks),
        "kept": int(len(kept)),
        "dropped": int(len(df) - len(kept)),
        "dropped_pct": round(100.0 * (len(df) - len(kept)) / max(1, len(df)), 1),
        "gaps_over_limit": int(df["long_gap"].sum()),
    }
    return kept[["datetime_utc", "IBI_ms", "PR"]], stats


def to_fixed_rate(df, hz):
    """Put PR on a steady grid.

    Beats do not arrive at regular times, so the raw PR series is unevenly
    spaced. Chronos, like any sequence model, wants a fixed step. We take the
    mean of whatever beats fall in each slot and carry the last value forward
    across short gaps, but NOT across the long ones, where the sensor was not
    reading.
    """
    s = df.set_index("datetime_utc")["PR"].sort_index()
    step = "%dms" % int(1000 / hz)
    grid = s.resample(step).mean()

    # carry forward only a short way: 2 seconds at the target rate
    grid = grid.ffill(limit=2 * hz)

    out = grid.dropna().reset_index()
    out.columns = ["datetime_utc", "PR"]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Work out pulse rate from the systolicPeaks heartbeat "
                    "timestamps, since EmbracePlus ships no PR channel.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--hz", type=int, default=TARGET_HZ)
    args = ap.parse_args()

    users = sorted(d for d in os.listdir(args.raw_dir)
                   if os.path.isdir(os.path.join(args.raw_dir, d)))
    if not users:
        raise SystemExit("No user folders in %s" % args.raw_dir)

    print("=" * 68)
    print("Deriving pulse rate from heartbeat timestamps")
    print("raw    : %s" % args.raw_dir)
    print("keep   : %.0f to %.0f bpm (anything else is a detection error)"
          % (MIN_BPM, MAX_BPM))
    print("output : %d Hz, matching the EDA rate" % args.hz)
    print("=" * 68)
    print()

    rows = []

    for user in users:
        udir = os.path.join(args.raw_dir, user)
        peak_files = sorted(glob.glob(os.path.join(udir, "*_systolicPeaks.csv")))
        if not peak_files:
            print("%s: no systolicPeaks files" % user)
            print()
            continue

        print("%s" % user)
        for pf in peak_files:
            base = os.path.basename(pf)
            m = re.search(r"(.+)_(\d{4}-\d{2}-\d{2})_systolicPeaks\.csv$", base)
            if not m:
                continue
            prefix, date = m.group(1), m.group(2)

            try:
                raw = pd.read_csv(pf, usecols=["datetime_utc"])
            except Exception as e:
                print("    %s: could not read (%s)" % (date, e))
                continue

            beats, stats = derive_from_peaks(raw["datetime_utc"])
            if beats is None or beats.empty:
                print("    %s: too few usable beats" % date)
                continue

            grid = to_fixed_rate(beats, args.hz)
            if grid.empty:
                print("    %s: nothing left after resampling" % date)
                continue

            # same two timestamp columns as every other sensor file
            ist = pd.Timedelta(hours=5, minutes=30)
            grid["datetime_ist"] = grid["datetime_utc"] + ist
            beats["datetime_ist"] = beats["datetime_utc"] + ist

            pr_path = os.path.join(udir, "%s_%s_PR.csv" % (prefix, date))
            ibi_path = os.path.join(udir, "%s_%s_IBI.csv" % (prefix, date))
            grid[["datetime_utc", "datetime_ist", "PR"]].to_csv(pr_path,
                                                                index=False)
            beats[["datetime_utc", "datetime_ist", "IBI_ms"]].to_csv(ibi_path,
                                                                     index=False)

            med = float(grid["PR"].median())
            print("    %s: %7d beats, kept %6d (%4.1f%% dropped), "
                  "median %5.1f bpm, %6d rows at %d Hz"
                  % (date, stats["peaks"], stats["kept"], stats["dropped_pct"],
                     med, len(grid), args.hz))

            rows.append({"user": user, "date": date,
                         "peaks": stats["peaks"], "kept": stats["kept"],
                         "dropped_pct": stats["dropped_pct"],
                         "gaps_over_limit": stats["gaps_over_limit"],
                         "median_bpm": round(med, 1),
                         "min_bpm": round(float(grid["PR"].min()), 1),
                         "max_bpm": round(float(grid["PR"].max()), 1),
                         "rows_4hz": len(grid),
                         "hours": round(len(grid) / (args.hz * 3600.0), 2)})
        print()

    if not rows:
        raise SystemExit("No PR could be derived for any user.")

    df = pd.DataFrame(rows)

    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print("  %-8s %6s %9s %9s %9s %9s %8s"
          % ("user", "days", "beats", "kept", "dropped%", "median", "hours"))
    for user, g in df.groupby("user"):
        print("  %-8s %6d %9d %9d %8.1f%% %9.1f %8.2f"
              % (user, len(g), g["peaks"].sum(), g["kept"].sum(),
                 g["dropped_pct"].mean(), g["median_bpm"].median(),
                 g["hours"].sum()))
    print("  " + "-" * 62)
    print("  %-8s %6d %9d %9d %8.1f%% %9.1f %8.2f"
          % ("TOTAL", len(df), df["peaks"].sum(), df["kept"].sum(),
             df["dropped_pct"].mean(), df["median_bpm"].median(),
             df["hours"].sum()))
    print()

    worst = df.sort_values("dropped_pct", ascending=False).head(3)
    print("Days with the most rejected beats:")
    for _, r in worst.iterrows():
        print("  %-8s %s  %.1f%% dropped" % (r["user"], r["date"],
                                             r["dropped_pct"]))
    print("  A high drop rate means the sensor was not making good contact.")
    print("  Those days' PR should be treated with suspicion.")
    print()

    print("Does the result look like a human heart?")
    print("  median across all days : %.1f bpm" % df["median_bpm"].median())
    print("  lowest day median      : %.1f bpm" % df["median_bpm"].min())
    print("  highest day median     : %.1f bpm" % df["median_bpm"].max())
    print("  A resting adult is roughly 55 to 85. Well outside that would")
    print("  mean the peak detection is wrong, not that the person is unwell.")
    print()

    L = []
    L.append("# Pulse rate derived from heartbeat timestamps")
    L.append("")
    L.append("Written by `btp/27_derive_pr.py`.")
    L.append("")
    L.append("We were asked to feed PR, BVP and skin temperature to Chronos.")
    L.append("BVP and temperature are in the EmbracePlus download. **PR is")
    L.append("not.** But `systolicPeaks` is, and it lists the time of every")
    L.append("detected heartbeat, so the gap between two beats gives the")
    L.append("instantaneous rate: a 0.8 second gap is 75 bpm.")
    L.append("")
    L.append("| setting | value | why |")
    L.append("|---|---|---|")
    L.append("| kept range | %.0f to %.0f bpm | outside this the detector missed a beat or counted one twice |" % (MIN_BPM, MAX_BPM))
    L.append("| output rate | %d Hz | matches EDA, so both chunk identically for Chronos |" % args.hz)
    L.append("| gaps not filled beyond | %.0f s | inventing a rate where nothing was detected would be making data up |" % MAX_GAP_SECONDS)
    L.append("")
    L.append("## Per user")
    L.append("")
    L.append("| user | days | beats | kept | dropped | median bpm | hours |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for user, g in df.groupby("user"):
        L.append("| %s | %d | %d | %d | %.1f%% | %.1f | %.2f |"
                 % (user, len(g), g["peaks"].sum(), g["kept"].sum(),
                    g["dropped_pct"].mean(), g["median_bpm"].median(),
                    g["hours"].sum()))
    L.append("")
    L.append("## Per day")
    L.append("")
    L.append("| user | date | beats | kept | dropped | median | min | max | hours |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        L.append("| %s | %s | %d | %d | %.1f%% | %.1f | %.1f | %.1f | %.2f |"
                 % (r["user"], r["date"], r["peaks"], r["kept"],
                    r["dropped_pct"], r["median_bpm"], r["min_bpm"],
                    r["max_bpm"], r["hours"]))
    L.append("")
    L.append("## Also written: inter-beat gaps")
    L.append("")
    L.append("`*_IBI.csv` holds the gap between consecutive beats in")
    L.append("milliseconds, one row per beat. These are what heart rate")
    L.append("variability is computed from, and HRV is the part of the cardiac")
    L.append("signal that actually tracks stress. Averaging BVP to 1 Hz")
    L.append("destroys it, which is why the 1 Hz BVP channel came out dead.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
