"""
25_build_user_6ch.py

WHY THIS FILE EXISTS
    Nothing in the repository can read EmbracePlus output, so this has to be
    written. It is the user-side twin of btp/20_build_wesad_13ch.py.

    The job is to make the user data look exactly like the WESAD data, so the
    model trained on WESAD can be pointed at it without anything being
    reshaped or guessed at run time.

WHAT HAS TO MATCH, AND HOW

    1. The sensors.
       WESAD has 13 channels: a chest strap plus a wrist band. EmbracePlus is
       wrist only. Six channels exist in both, and only those six are used:

         accelerometer_x  ->  ACC_X_wrist
         accelerometer_y  ->  ACC_Y_wrist
         accelerometer_z  ->  ACC_Z_wrist
         bvp              ->  BVP
         eda              ->  EDA_wrist
         temperature      ->  Temp_wrist

       The other seven (chest ACC x/y/z, ECG, chest EDA, chest Temp, Resp)
       have no equivalent and are not invented. Filling them with zeros would
       be the same failure we already measured on the context columns, except
       in the signal path where there is no gate to learn around it.

    2. The speed.
       The wrist band records 64 times a second. The model expects one row per
       second, because configs say fs: 1. So every second of readings is
       averaged into one row, exactly as the WESAD converter does.

    3. The names.
       Renamed as above, so adastress/datasets.py reads these files with no
       changes at all. The model has no idea it is looking at a different
       device; it sees six columns with the names it was trained on.

CONTEXT FROM THE CLOCK
    Four columns, worked out from each row's timestamp:

        hour_of_day    0 to 23
        day_of_week    0 Monday to 6 Sunday
        is_weekend     0 or 1
        is_workhour    1 between 09:00 and 18:00 on a weekday

    These are the same four the repository's own In-House pipeline uses. They
    matter because they cannot give the answer away the way Task did on
    WESAD: Task WAS the session the questionnaire asked about, so it predicted
    the label perfectly. A clock reading is not the label under another name.

    Timestamps are read in IST (datetime_ist), because "is this a work hour"
    only means anything in the participant's own local time.

ABOUT THE LABEL
    There is no Stress_Class yet. The survey responses have not arrived, so
    the column is written as empty. Once the survey file exists, one function
    fills it in and nothing else changes.

    That is deliberate: everything except the label can be built, checked and
    counted now.

MEMORY
    The raw folder is about 21 GB, nearly all of it accelerometer and BVP at
    64 Hz. A single day of accelerometer can be five million rows. So each
    file is read in chunks and averaged down as it goes, instead of being
    loaded whole.

OUTPUT
    data/processed/user_6ch/
        user3.csv, user16.csv, user17.csv, user20.csv    one per user
        users_all.csv                                    all four together
    docs/50_user_6ch.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/25_build_user_6ch.py

    Several minutes. It prints progress per user-day.
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

RAW_DIR = "data/raw/embraceplus"
OUT_DIR = "data/processed/user_6ch"
REPORT = "docs/50_user_6ch.md"

# Which EmbracePlus file gives which WESAD column.
SENSOR_MAP = {
    "accelerometer": {"accelerometer_x": "ACC_X_wrist",
                      "accelerometer_y": "ACC_Y_wrist",
                      "accelerometer_z": "ACC_Z_wrist"},
    "bvp":           {"bvp": "BVP"},
    "eda":           {"eda": "EDA_wrist"},
    "temperature":   {"temperature": "Temp_wrist"},
}

SIGNAL_COLUMNS = ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
                  "BVP", "EDA_wrist", "Temp_wrist"]

CONTEXT_COLUMNS = ["hour_of_day", "day_of_week", "is_weekend", "is_workhour"]

# Window settings, straight from the repo's config, so the counts below mean
# the same thing as the WESAD counts.
WINDOW_ROWS = 60
STRIDE_ROWS = 30

CHUNK = 500_000          # rows read at a time
TIME_COL = "datetime_ist"   # local time, so "work hour" means something


def resample_file_to_1hz(path, value_cols):
    """Average one CSV down to one row per second, reading it in chunks.

    Returns a DataFrame indexed by second, or None if the file is unusable.

    Done in chunks because a single day of 64 Hz accelerometer is millions of
    rows. We keep a running sum and count per second, then divide at the end,
    which gives the same answer as loading the whole file and averaging.
    """
    needed = [TIME_COL] + list(value_cols)
    total, count = None, None

    try:
        reader = pd.read_csv(path, usecols=needed, chunksize=CHUNK)
    except ValueError:
        # column names not as expected
        return None

    for chunk in reader:
        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], errors="coerce")
        chunk = chunk.dropna(subset=[TIME_COL])
        if chunk.empty:
            continue
        sec = chunk[TIME_COL].dt.floor("1s")
        g = chunk[list(value_cols)].groupby(sec)
        s, c = g.sum(), g.count()
        total = s if total is None else total.add(s, fill_value=0)
        count = c if count is None else count.add(c, fill_value=0)

    if total is None or len(total) == 0:
        return None

    out = total / count.replace(0, np.nan)
    out.index.name = TIME_COL
    return out


def add_clock_context(df):
    """The four context columns, from the local timestamp.

    Unlike WESAD's Task, none of these can be the label under another name.
    They come from the clock, not from the experiment.
    """
    ts = pd.to_datetime(df[TIME_COL])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0 Monday .. 6 Sunday
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["is_workhour"] = (((ts.dt.hour >= 9) & (ts.dt.hour < 18)) &
                         (ts.dt.dayofweek < 5)).astype(int)
    return df


def build_user(user, user_dir):
    """Turn one user's raw files into 1 Hz rows with the six WESAD columns."""
    files = sorted(glob.glob(os.path.join(user_dir, "*.csv")))

    # group by date, so we handle one day at a time
    by_date = {}
    for f in files:
        m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\w+)\.csv$", os.path.basename(f))
        if not m:
            continue
        date, sensor = m.group(1), m.group(2)
        if sensor in SENSOR_MAP:
            by_date.setdefault(date, {})[sensor] = f

    if not by_date:
        print("    no usable files")
        return None, []

    day_rows = []
    per_day = []

    for date in sorted(by_date):
        sensors = by_date[date]
        pieces = []
        missing = []

        for sensor, mapping in SENSOR_MAP.items():
            if sensor not in sensors:
                missing.append(sensor)
                continue
            got = resample_file_to_1hz(sensors[sensor], list(mapping.keys()))
            if got is None:
                missing.append(sensor)
                continue
            pieces.append(got.rename(columns=mapping))

        if not pieces:
            print("    %s: nothing usable" % date)
            continue

        day = pieces[0]
        for p in pieces[1:]:
            day = day.join(p, how="outer")

        day = day.sort_index().reset_index()

        # keep only seconds where at least one signal was recording
        have = [c for c in SIGNAL_COLUMNS if c in day.columns]
        day = day[day[have].notna().any(axis=1)]

        for c in SIGNAL_COLUMNS:
            if c not in day.columns:
                day[c] = np.nan

        day["date"] = date
        day_rows.append(day)

        hours = len(day) / 3600.0
        per_day.append({"date": date, "seconds": len(day),
                        "hours": round(hours, 2),
                        "missing_sensors": ",".join(missing) if missing else "-"})
        print("    %s: %6d seconds (%.2f h)%s"
              % (date, len(day), hours,
                 "   missing: " + ",".join(missing) if missing else ""))

    if not day_rows:
        return None, per_day

    df = pd.concat(day_rows, ignore_index=True)
    df = add_clock_context(df)
    df["Subject_ID"] = user

    # No labels yet. The survey responses have not arrived. One function fills
    # this in later and nothing else changes.
    df["Stress_Class"] = np.nan

    cols = [TIME_COL, "date"] + SIGNAL_COLUMNS + CONTEXT_COLUMNS + \
           ["Stress_Class", "Subject_ID"]
    return df[cols], per_day


def count_windows(df):
    """How many 60-row windows, cut inside each day so none spans a gap."""
    total = 0
    for (_subj, _date), g in df.groupby(["Subject_ID", "date"], sort=True):
        n = len(g)
        if n >= WINDOW_ROWS:
            total += (n - WINDOW_ROWS) // STRIDE_ROWS + 1
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Put the EmbracePlus user data into the same shape as "
                    "the processed WESAD data.")
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    users = sorted(d for d in os.listdir(args.raw_dir)
                   if os.path.isdir(os.path.join(args.raw_dir, d)))
    if not users:
        raise SystemExit("No user folders in %s" % args.raw_dir)

    print("=" * 66)
    print("Building the 6 channels EmbracePlus and WESAD both record")
    print("raw    : %s" % args.raw_dir)
    print("output : %s" % args.out_dir)
    print("window : %d rows, stride %d  (same as the WESAD runs)"
          % (WINDOW_ROWS, STRIDE_ROWS))
    print("=" * 66)
    print()

    frames, summary, day_tables = [], [], {}

    for user in users:
        print("%s" % user)
        df, per_day = build_user(user, os.path.join(args.raw_dir, user))
        day_tables[user] = per_day
        if df is None or df.empty:
            print("    nothing produced")
            print()
            continue

        path = os.path.join(args.out_dir, "%s.csv" % user)
        df.to_csv(path, index=False)
        frames.append(df)

        w = count_windows(df)
        summary.append({"user": user, "days": len(per_day),
                        "seconds": len(df), "hours": round(len(df) / 3600.0, 2),
                        "windows": w})
        print("    total: %d seconds (%.2f h) -> %d windows"
              % (len(df), len(df) / 3600.0, w))
        print("    saved %s" % path)
        print()

    if not frames:
        raise SystemExit("Nothing produced for any user.")

    allf = pd.concat(frames, ignore_index=True)
    all_path = os.path.join(args.out_dir, "users_all.csv")
    allf.to_csv(all_path, index=False)

    print("=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print("  %-8s %6s %10s %9s %9s" % ("user", "days", "seconds", "hours",
                                       "windows"))
    for r in summary:
        print("  %-8s %6d %10d %9.2f %9d"
              % (r["user"], r["days"], r["seconds"], r["hours"], r["windows"]))
    tw = sum(r["windows"] for r in summary)
    th = sum(r["hours"] for r in summary)
    print("  " + "-" * 46)
    print("  %-8s %6s %10d %9.2f %9d"
          % ("TOTAL", "", len(allf), th, tw))
    print()

    print("signal columns produced: %d" % len(SIGNAL_COLUMNS))
    print("  expected 6, to match runs/W6")
    print()
    print("value ranges, a check that these are real sensor readings:")
    print(allf[SIGNAL_COLUMNS].agg(["min", "max", "mean"]).round(3).to_string())
    print("  Temp_wrist should be roughly 28 to 36 C.")
    print("  EDA_wrist should be small and positive.")
    print()

    miss = allf[SIGNAL_COLUMNS].isna().mean().round(3)
    print("missing values per column (1.0 would mean the sensor never worked):")
    print(miss.to_string())
    print()

    print("context columns:")
    for c in CONTEXT_COLUMNS:
        vals = allf[c].value_counts().sort_index()
        print("  %-12s %s" % (c, dict(vals.head(8))))
    print()

    print("labels: none yet. Stress_Class is empty until the survey file")
    print("        arrives. Everything else is finished.")
    print()
    print("saved %s" % all_path)

    # ---------------- report ----------------
    L = []
    L.append("# user data as the 6 shared channels, 1 Hz")
    L.append("")
    L.append("Written by `btp/25_build_user_6ch.py`. There is no equivalent in")
    L.append("the repository; it cannot read EmbracePlus output.")
    L.append("")
    L.append("The aim is for the user data to look exactly like the processed")
    L.append("WESAD data, so the model trained on WESAD can be pointed at it")
    L.append("with nothing reshaped or guessed at run time.")
    L.append("")
    L.append("## Channel mapping")
    L.append("")
    L.append("| EmbracePlus | becomes | in WESAD |")
    L.append("|---|---|---|")
    L.append("| `accelerometer_x/y/z` | `ACC_X/Y/Z_wrist` | wrist accelerometer |")
    L.append("| `bvp` | `BVP` | wrist BVP |")
    L.append("| `eda` | `EDA_wrist` | wrist EDA |")
    L.append("| `temperature` | `Temp_wrist` | wrist temperature |")
    L.append("")
    L.append("WESAD's other seven channels (chest ACC x/y/z, ECG, chest EDA,")
    L.append("chest Temp, Resp) have no equivalent on a wrist device and are")
    L.append("not invented.")
    L.append("")
    L.append("## What we ended up with")
    L.append("")
    L.append("| user | days | seconds | hours | windows |")
    L.append("|---|---:|---:|---:|---:|")
    for r in summary:
        L.append("| %s | %d | %d | %.2f | %d |"
                 % (r["user"], r["days"], r["seconds"], r["hours"],
                    r["windows"]))
    L.append("| **total** | | **%d** | **%.2f** | **%d** |"
             % (len(allf), th, tw))
    L.append("")
    L.append("Windows are %d rows with a stride of %d, the same as the WESAD"
             % (WINDOW_ROWS, STRIDE_ROWS))
    L.append("runs, and are cut inside each day so none spans a gap between")
    L.append("recordings.")
    L.append("")
    L.append("## Per day")
    L.append("")
    for user in sorted(day_tables):
        rows = day_tables[user]
        if not rows:
            continue
        L.append("### %s" % user)
        L.append("")
        L.append("| date | seconds | hours | sensors missing |")
        L.append("|---|---:|---:|---|")
        for d in rows:
            L.append("| %s | %d | %.2f | %s |"
                     % (d["date"], d["seconds"], d["hours"],
                        d["missing_sensors"]))
        L.append("")
    L.append("## Context from the clock")
    L.append("")
    L.append("| column | meaning |")
    L.append("|---|---|")
    L.append("| `hour_of_day` | 0 to 23, local time |")
    L.append("| `day_of_week` | 0 Monday to 6 Sunday |")
    L.append("| `is_weekend` | 1 on Saturday or Sunday |")
    L.append("| `is_workhour` | 1 between 09:00 and 18:00 on a weekday |")
    L.append("")
    L.append("These are the four the repository's own In-House pipeline uses.")
    L.append("They cannot give the answer away the way `Task` did on WESAD,")
    L.append("where `Task` was the session the questionnaire asked about and")
    L.append("so predicted the label perfectly. A clock reading is not the")
    L.append("label under another name.")
    L.append("")
    L.append("## Labels")
    L.append("")
    L.append("None yet. `Stress_Class` is empty until the survey responses")
    L.append("arrive. Everything else in these files is finished.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
