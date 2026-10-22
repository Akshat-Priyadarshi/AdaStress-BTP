"""
07_prep_user1.py

WHY THIS FILE EXISTS
    The repo knows nothing about user1, so there is no script in it that can
    read these files. We need our own.

    The job is to put user1 into exactly the same shape as the processed WESAD
    data, so that a classifier trained on WESAD can be applied to it without
    anything being reshaped, padded or guessed at run time.

    "Exactly the same shape" means the same 5 second bins, the same six
    statistics per bin, and the same column names.

WHAT IT DOES
    For each recording session (there are two, one per date):

      1. Reads the four signals both devices record.
      2. Summarises each into 5 second bins: mean, std, min, max, median, skew.
         Same bins and same statistics as WESAD.
      3. Renames them to the WESAD column names.
      4. Works out Activity and Posture from the accelerometer, using the same
         shared rules.
      5. Keeps user1's own context columns as they are.
      6. Marks every row saying whether the accelerometer was recording.

THE FOUR SHARED SIGNALS

    | user1 file      | rate  | becomes             | WESAD source        |
    |-----------------|-------|---------------------|---------------------|
    | accelerometer   | 64 Hz | ACC_Mag_Wrist_*     | wrist ACC, 32 Hz    |
    | bvp             | 64 Hz | BVP_*               | wrist BVP, 64 Hz    |
    | eda             | 4 Hz  | EDA_y_*             | wrist EDA, 4 Hz     |
    | temperature     | 1 Hz  | TEMP_*              | wrist TEMP, 4 Hz    |

    The different raw rates do not matter, because everything is summarised
    into the same 5 second bins first.

    PR and RESP are left out. WESAD has no pulse rate column at all, and its
    respiration comes from a chest sensor, so neither can be matched.
    systolicPeaks is left out because it holds only timestamps, no values.

THE ACCELEROMETER GAP
    On 2026-03-22 the EDA and temperature cover about 1.5 hours but the
    accelerometer stops after about 0.87 hours. So roughly 40% of that session
    has body signals and no movement data.

    Decision D14: those rows are KEPT, with Activity and Posture marked
    "Absent", rather than thrown away.

    Reason: this is the only place in the whole project where zero-padded
    context meets genuinely missing data. On WESAD the padded columns are
    absent by construction, which is a much weaker test. Throwing these rows
    away would throw away the one real test of Experiment A.

    Every count is reported both ways, so results can be given with and
    without them.

OUTPUT
    data/processed/user1/<date>.csv     one per session
    data/processed/user1/user1_all.csv  both sessions together
    docs/30_user1_preprocessing.md      what happened, with numbers

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/07_prep_user1.py

    Takes a minute or two. The accelerometer and bvp files hold a few hundred
    thousand rows each.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew

RAW_DIR = "data/raw/user1"
OUT_DIR = "data/processed/user1"
REPORT = "docs/30_user1_preprocessing.md"

# Same as WESAD. See docs/00_decisions.md D1.
RESAMPLE_WINDOW = "5s"
WINDOW_ROWS = 15
STRIDE_ROWS = 15

# The six statistics WESAD computes per bin, in the same order.
STATS = ["mean", "std", "min", "max", "median", skew]

TIME_COL = "datetime_utc"

# user1 context columns, kept as they are. These are the four that WESAD does
# not have, and that Experiment A fills with zeros on the WESAD side.
USER1_CONTEXT = {
    "steps": "sum",         # a count, so add them up over the bin
    "MET": "mean",
    "ACT_COUNT": "mean",
    "ACT_INT": "mode",      # words, so take the most common in the bin
}


def find_sessions(raw_dir):
    """Group the files by date. Each date is one recording session."""
    sessions = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        name = os.path.basename(path)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        if not m:
            continue
        date = m.group(1)

        # The signal name is everything after the date and before ".csv".
        # It must be taken that way, not by splitting on the last underscore,
        # because names like ACT_COUNT and ACT_INT contain an underscore
        # themselves and would otherwise come out as "COUNT" and "INT".
        signal = name[m.end():]
        signal = signal.lstrip("_")
        if signal.endswith(".csv"):
            signal = signal[:-4]

        sessions.setdefault(date, {})[signal] = path
    return sessions


def read_timed(path, columns):
    """Read a CSV and index it by time. Returns None if it cannot be used."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("      could not read %s: %s" % (os.path.basename(path), e))
        return None

    if TIME_COL not in df.columns:
        print("      %s has no %s column" % (os.path.basename(path), TIME_COL))
        return None

    missing = [c for c in columns if c not in df.columns]
    if missing:
        print("      %s is missing %s" % (os.path.basename(path), missing))
        return None

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).set_index(TIME_COL).sort_index()
    return df[columns]


def summarise(series, out_prefix):
    """Turn one signal into 5 second bins with the six WESAD statistics."""
    frame = series.to_frame(name="v")
    agg = frame.resample(RESAMPLE_WINDOW).agg(STATS)
    agg.columns = ["%s_%s" % (out_prefix, c[1]) for c in agg.columns]
    return agg


def build_session(date, files):
    """Turn one session's files into WESAD-shaped 5 second rows."""
    print("  session %s" % date)
    pieces = []

    # ---- accelerometer, turned into one magnitude then summarised ----
    acc_cols = ["accelerometer_x", "accelerometer_y", "accelerometer_z"]
    acc = read_timed(files.get("accelerometer", ""), acc_cols) \
        if "accelerometer" in files else None
    if acc is not None:
        mag = np.sqrt((acc ** 2).sum(axis=1))
        pieces.append(summarise(mag, "ACC_Mag_Wrist"))
        print("      accelerometer: %d rows, %s to %s"
              % (len(acc), acc.index.min(), acc.index.max()))
    else:
        print("      accelerometer: MISSING for this session")

    # ---- the other three shared signals ----
    for key, column, prefix in [("bvp", "bvp", "BVP"),
                                ("eda", "eda", "EDA_y"),
                                ("temperature", "temperature", "TEMP")]:
        if key not in files:
            print("      %s: MISSING for this session" % key)
            continue
        got = read_timed(files[key], [column])
        if got is None:
            continue
        pieces.append(summarise(got[column], prefix))
        print("      %-12s: %d rows, %s to %s"
              % (key, len(got), got.index.min(), got.index.max()))

    if not pieces:
        print("      nothing usable in this session")
        return None

    # ---- line them all up on the same 5 second grid ----
    df = pieces[0]
    for p in pieces[1:]:
        df = df.join(p, how="outer")

    # ---- user1's own context columns ----
    for key, how in USER1_CONTEXT.items():
        if key not in files:
            print("      %s: MISSING for this session" % key)
            df[key] = np.nan
            continue
        got = read_timed(files[key], [key])
        if got is None:
            df[key] = np.nan
            continue

        if how == "mode":
            # words: take the most common value in each bin, then carry it
            # forward, because it is only recorded every 30 seconds
            r = got[key].resample(RESAMPLE_WINDOW).agg(
                lambda s: s.mode().iloc[0] if len(s.dropna()) else np.nan)
            r = r.ffill()
        elif how == "sum":
            r = got[key].resample(RESAMPLE_WINDOW).sum(min_count=1)
            r = r.fillna(0)      # no step readings in a bin means no steps
        else:
            r = got[key].resample(RESAMPLE_WINDOW).mean().ffill()

        df = df.join(r.rename(key), how="left")

    # ---- keep only bins where at least one body signal was recording ----
    body = [c for c in df.columns
            if c.startswith(("BVP_", "EDA_y_", "TEMP_"))]
    before = len(df)
    if body:
        df = df[df[body].notna().any(axis=1)]
    print("      5 second bins: %d, keeping %d that hold a body signal"
          % (before, len(df)))

    # ---- was the accelerometer recording in this bin? ----
    if "ACC_Mag_Wrist_mean" in df.columns:
        df["acc_present"] = df["ACC_Mag_Wrist_mean"].notna()
    else:
        df["acc_present"] = False

    n_gap = int((~df["acc_present"]).sum())
    print("      bins with no accelerometer: %d of %d (%.1f%%)"
          % (n_gap, len(df), 100.0 * n_gap / max(1, len(df))))

    df["subject_id"] = "user1_%s" % date
    df["datetime_utc"] = df.index
    return df.reset_index(drop=True)


def add_shared_context(df):
    """Work out Activity and Posture from movement, the shared way.

    Activity uses the same rule as WESAD: this recording's own mean and
    standard deviation of movement size. Because it is relative, it carries
    across devices unchanged.

    Posture does NOT use WESAD's fixed cut-offs of 0.85 and 1.05. Those are
    absolute values in g measured on a chest strap, and a wrist device at rest
    reads something different, so they would be meaningless here. Instead we
    take the same three levels from the lower, middle and upper third of this
    dataset's own movement. Recorded as decision D8.

    Rows with no accelerometer get "Absent" for both, which is an all-zero
    one-hot block. See decision D14.
    """
    col = "ACC_Mag_Wrist_mean"
    if col not in df.columns:
        df["Activity"] = "Absent"
        df["Posture"] = "Absent"
        return df, {}

    have = df[col].notna()
    values = df.loc[have, col]

    df["Activity"] = "Absent"
    df["Posture"] = "Absent"

    if len(values) < 10:
        print("  too little movement data to work out Activity and Posture")
        return df, {}

    # Activity: the WESAD rule, relative to this recording
    mean_acc = values.mean()
    std_acc = values.std()
    low = mean_acc + 0.5 * std_acc
    high = mean_acc + std_acc

    df.loc[have, "Activity"] = np.select(
        [values <= low, values > high],
        ["Still", "HeavyMovement"],
        default="SlightMovement")

    # Posture: tertiles within this dataset, not WESAD's fixed cut-offs
    t1, t2 = values.quantile([1 / 3, 2 / 3])
    df.loc[have, "Posture"] = np.select(
        [values <= t1, values > t2],
        ["Sit", "Walk"],
        default="Stand")

    info = {
        "movement_mean": float(mean_acc),
        "movement_std": float(std_acc),
        "activity_low_cut": float(low),
        "activity_high_cut": float(high),
        "posture_tertile_1": float(t1),
        "posture_tertile_2": float(t2),
    }

    print("  Activity cut-offs (this recording's own mean and std): "
          "%.4f and %.4f" % (low, high))
    print("  Posture tertiles (this dataset's own, decision D8): "
          "%.4f and %.4f" % (t1, t2))
    return df, info


def count_windows(df):
    """How many 15 row windows each session gives, and how many hold a gap."""
    rows = []
    for subject, group in df.groupby("subject_id", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        starts = np.arange(0, n - WINDOW_ROWS + 1, STRIDE_ROWS)
        full = partial = none = 0
        for s in starts:
            k = int(group["acc_present"].iloc[s:s + WINDOW_ROWS].sum())
            if k == WINDOW_ROWS:
                full += 1
            elif k == 0:
                none += 1
            else:
                partial += 1
        rows.append({
            "session": subject,
            "rows": n,
            "windows": len(starts),
            "acc_complete": full,
            "acc_partial": partial,
            "acc_absent": none,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Put user1 into the same shape as processed WESAD.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sessions = find_sessions(args.raw_dir)
    if not sessions:
        raise SystemExit("No dated CSV files found in %s" % args.raw_dir)

    print("=" * 62)
    print("sessions found: %d" % len(sessions))
    for date in sorted(sessions):
        print("  %s: %d files" % (date, len(sessions[date])))
    print("=" * 62)
    print()

    frames = []
    cut_info = {}
    for date in sorted(sessions):
        df = build_session(date, sessions[date])
        if df is None:
            continue
        df, info = add_shared_context(df)
        cut_info[date] = info
        out = os.path.join(args.out_dir, "%s.csv" % date)
        df.to_csv(out, index=False)
        print("      saved %s" % out)
        print()
        frames.append(df)

    if not frames:
        raise SystemExit("Nothing could be processed.")

    allf = pd.concat(frames, ignore_index=True)
    all_path = os.path.join(args.out_dir, "user1_all.csv")
    allf.to_csv(all_path, index=False)

    # -------------------------------------------------------------------
    print("=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print("total 5 second rows: %d" % len(allf))
    print("columns: %d" % len(allf.columns))
    print()

    signal_cols = [c for c in allf.columns
                   if c.startswith(("ACC_Mag_Wrist_", "BVP_", "EDA_y_", "TEMP_"))]
    print("shared signal channels produced: %d" % len(signal_cols))
    print("  expected 24 to match the WESAD wrist runs")
    if len(signal_cols) != 24:
        print("  WARNING: not 24. Inference will not line up.")
        groups = {}
        for c in signal_cols:
            groups.setdefault(c.rsplit("_", 1)[0], []).append(c)
        for g, cs in sorted(groups.items()):
            print("    %-16s %d" % (g, len(cs)))
    print()

    n_gap = int((~allf["acc_present"]).sum())
    print("rows with no accelerometer: %d of %d (%.1f%%)"
          % (n_gap, len(allf), 100.0 * n_gap / len(allf)))
    print()

    wrows = count_windows(allf)
    print("Windows of %d rows, counted inside each session:" % WINDOW_ROWS)
    print()
    print("  %-22s %7s %9s %14s %13s %12s"
          % ("session", "rows", "windows", "acc complete", "acc partial",
             "acc absent"))
    for r in wrows:
        print("  %-22s %7d %9d %14d %13d %12d"
              % (r["session"], r["rows"], r["windows"], r["acc_complete"],
                 r["acc_partial"], r["acc_absent"]))
    tw = sum(r["windows"] for r in wrows)
    tc = sum(r["acc_complete"] for r in wrows)
    tp = sum(r["acc_partial"] for r in wrows)
    ta = sum(r["acc_absent"] for r in wrows)
    print("  " + "-" * 82)
    print("  %-22s %7s %9d %14d %13d %12d" % ("TOTAL", "", tw, tc, tp, ta))
    print()

    print("Context values found:")
    for c in ["Activity", "Posture", "ACT_INT"]:
        if c in allf.columns:
            print("  %-10s %s" % (c, dict(allf[c].value_counts(dropna=False))))
    for c in ["steps", "MET", "ACT_COUNT"]:
        if c in allf.columns:
            v = pd.to_numeric(allf[c], errors="coerce")
            print("  %-10s min %.2f, mean %.2f, max %.2f, missing %d"
                  % (c, v.min(), v.mean(), v.max(), int(v.isna().sum())))
    print()
    print("saved %s" % all_path)

    # -------------------------------------------------------------------
    L = []
    L.append("# user1 preprocessing")
    L.append("")
    L.append("Written by `btp/07_prep_user1.py`. There is no equivalent in the")
    L.append("repo; it knows nothing about user1.")
    L.append("")
    L.append("The aim is to put user1 into exactly the shape of the processed")
    L.append("WESAD data, so a WESAD-trained classifier can be applied without")
    L.append("anything being reshaped or guessed at run time.")
    L.append("")
    L.append("| setting | value | same as WESAD? |")
    L.append("|---|---|---|")
    L.append("| bin length | 5 seconds | yes |")
    L.append("| statistics per bin | mean, std, min, max, median, skew | yes |")
    L.append("| window | %d rows, stride %d | yes |" % (WINDOW_ROWS, STRIDE_ROWS))
    L.append("| signal channels | %d | yes, the 24 wrist channels |" % len(signal_cols))
    L.append("")
    L.append("## Sessions")
    L.append("")
    L.append("| session | 5s rows | windows | accelerometer complete | partial | absent |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in wrows:
        L.append("| %s | %d | %d | %d | %d | %d |"
                 % (r["session"], r["rows"], r["windows"], r["acc_complete"],
                    r["acc_partial"], r["acc_absent"]))
    L.append("| **total** | %d | **%d** | %d | %d | %d |"
             % (len(allf), tw, tc, tp, ta))
    L.append("")
    L.append("## Signal mapping")
    L.append("")
    L.append("| user1 file | rate | becomes | WESAD source |")
    L.append("|---|---|---|---|")
    L.append("| `accelerometer` | 64 Hz | `ACC_Mag_Wrist_*` | wrist ACC, 32 Hz |")
    L.append("| `bvp` | 64 Hz | `BVP_*` | wrist BVP, 64 Hz |")
    L.append("| `eda` | 4 Hz | `EDA_y_*` | wrist EDA, 4 Hz |")
    L.append("| `temperature` | 1 Hz | `TEMP_*` | wrist TEMP, 4 Hz |")
    L.append("")
    L.append("The different raw rates do not matter, because both datasets are")
    L.append("summarised into the same 5 second bins before anything else.")
    L.append("")
    L.append("`PR` and `RESP` are left out: WESAD has no pulse rate column, and")
    L.append("its respiration is a chest sensor. `systolicPeaks` is left out")
    L.append("because it holds only timestamps, no values.")
    L.append("")
    L.append("## The accelerometer gap")
    L.append("")
    L.append("%d of %d rows (%.1f%%) have body signals but no movement data."
             % (n_gap, len(allf), 100.0 * n_gap / len(allf)))
    L.append("")
    L.append("Those rows are kept, with `Activity` and `Posture` set to")
    L.append("`Absent`, which becomes an all-zero one-hot block. Decision D14.")
    L.append("")
    L.append("This is the only place in the project where zero-padded context")
    L.append("meets genuinely missing data. On WESAD the padded columns are")
    L.append("absent by construction, which is a weaker test. Results should be")
    L.append("given both with and without these windows, using the")
    L.append("`acc_present` column.")
    L.append("")
    L.append("## Shared context")
    L.append("")
    L.append("| variable | rule | same as WESAD? |")
    L.append("|---|---|---|")
    L.append("| `Activity` | this recording's own mean and standard deviation of movement | yes, the rule is relative so it carries across |")
    L.append("| `Posture` | lower, middle and upper third of this dataset's movement | **no**, WESAD uses fixed cut-offs of 0.85 and 1.05 in g, which are chest values. Decision D8 |")
    L.append("")
    for date, info in cut_info.items():
        if not info:
            continue
        L.append("### %s cut-offs actually used" % date)
        L.append("")
        for k, v in info.items():
            L.append("- `%s`: %.5f" % (k, v))
        L.append("")
    L.append("## Context values found")
    L.append("")
    for c in ["Activity", "Posture", "ACT_INT"]:
        if c in allf.columns:
            L.append("`%s`: %s" % (c, dict(allf[c].value_counts(dropna=False))))
            L.append("")
    L.append("| column | min | mean | max | missing |")
    L.append("|---|---:|---:|---:|---:|")
    for c in ["steps", "MET", "ACT_COUNT"]:
        if c in allf.columns:
            v = pd.to_numeric(allf[c], errors="coerce")
            L.append("| `%s` | %.2f | %.2f | %.2f | %d |"
                     % (c, v.min(), v.mean(), v.max(), int(v.isna().sum())))
    L.append("")
    L.append("## Reminder: no labels")
    L.append("")
    L.append("user1 has no stress label in any of these files (decision D2).")
    L.append("Nothing here can be trained or scored for accuracy. The next")
    L.append("step compares score distributions and gate behaviour between")
    L.append("Experiment A and Experiment B.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("Report written to %s" % REPORT)


if __name__ == "__main__":
    main()
