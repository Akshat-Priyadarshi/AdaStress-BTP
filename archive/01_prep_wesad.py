"""
01_prep_wesad.py

WHY THIS FILE EXISTS
    The preprocessing we need is in the original repo, in Preproc.ipynb,
    cells 26 to 38. Two reasons we cannot just run that notebook:

      1. It does not run. It uses pickle, skew and LabelEncoder and imports
         none of them. It also uses the resample code "5S", which newer pandas
         rejects outright.
      2. vendor/adastress is read-only on purpose, so we cannot edit it there.

    So this file is a copy of those cells, as close to word for word as
    possible. Every line that differs from the notebook is marked with
    "BTP-CHANGE" and a repair number that matches the repair log in
    docs/00_decisions.md. Nothing else was touched. No tidying, no renaming,
    no improvements.

WHAT IT DOES
    For each subject folder in data/raw/WESAD:
      - loads the pickle and the quest file
      - summarises every signal into 5 second rows (mean, std, min, max,
        median, skew)
      - joins chest and wrist signals onto the same time grid
      - works out Activity and Posture from chest accelerometer size
      - works out Time_Sin and Time_Cos from position in the recording
      - works out Task and SessionPhase from the session start/end times
      - throws away all rows outside the five sessions
      - works out Stress_Class from the PANAS answers
      - saves one CSV per subject

OUTPUT
    data/processed/wesad/S2.csv, S3.csv, ... one per subject
    docs/10_wesad_preprocessing.md    what actually happened, with numbers

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/01_prep_wesad.py

    It takes a few minutes. Each subject has about 4 million rows of chest
    signal to summarise.

WHAT TO WATCH FOR IN THE OUTPUT
    Two numbers matter most.

    "labelled rows" - how many 5 second rows end up with a stress label.
    Expect roughly 50 out of roughly 610. The label only reaches 11 rows
    around each of the 5 PANAS answers.

    "columns kept" - the notebook drops any column that has even one missing
    value. If different subjects keep different columns, we cannot simply
    stack them together, and the script will say so at the end.
"""

# =============================================================================
# IMPORTS
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 3
# =============================================================================
import os
import json
import argparse
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# BTP-CHANGE R1: the notebook calls pickle.load but never imports pickle.
import pickle

# BTP-CHANGE R2: the notebook passes skew to .agg() but never imports it.
from scipy.stats import skew

# BTP-CHANGE R3: the notebook uses LabelEncoder but never imports it.
from sklearn.preprocessing import LabelEncoder

# extra, only for printing errors clearly instead of hiding them
import traceback


# =============================================================================
# CONFIGURATION
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 26
# =============================================================================
# BTP-CHANGE: paths point at our folder layout instead of "./data/wesad_raw".
INPUT_DIR = "data/raw/WESAD"

# The task order Preproc.ipynb assumes for every subject, cell 34.
NOTEBOOK_ORDER = ["Base", "TSST", "Medi1", "Fun", "Medi2"]

# Output paths are chosen in main(), based on --task-order, so the two modes
# never write over each other.

# BTP-CHANGE R4: the notebook has "5S". pandas 2.2 and later reject it with
# "Invalid frequency: S. Did you mean s?". Lower case "5s" means exactly the
# same thing: 5 second bins.
RESAMPLE_WINDOW = "5s"


# =============================================================================
# PANAS EXTRACTION
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 28
# Unchanged.
#
# What it does in plain terms: the quest file has one "# PANAS" line per
# session. Each line holds 24 numbers. Ten of them are the positive-feeling
# questions and ten are the negative-feeling questions. It averages each group
# and returns one row per session.
# =============================================================================
def extract_session_order(quest_path):
    """Return the five task names in the order this subject really did them.

    BTP-CHANGE R9: new function. There is nothing like it in the notebook,
    because the notebook never reads the "# ORDER" line.

    The quest file begins with lines like:
        # ORDER;Base;TSST;Medi 1;Fun;Medi 2;;
        # START;7.08;39.55;70.19;81.25;93.38;;
        # END;26.32;50.30;77.10;87.47;100.15;;

    The names on the ORDER line are in the order the subject actually did
    them, and they line up column by column with START and END.

    Returns None if there is no ORDER line, so the caller can decide.
    """
    lines = open(quest_path).readlines()

    order_lines = [
        line for line in lines
        if line.startswith("# ORDER")
    ]

    if not order_lines:
        return None

    tokens = order_lines[0].strip().split(";")[1:]

    # The file writes "Medi 1" with a space, the notebook writes "Medi1".
    # Remove spaces so the two spellings match.
    names = [
        re.sub(r"\s+", "", t).strip()
        for t in tokens
        if re.sub(r"\s+", "", t).strip() != ""
    ]

    return names[:5] if len(names) >= 5 else None


def extract_panas_scores(quest_path, task_names_in_order):

    lines = open(quest_path).readlines()

    panas_lines = [
        line for line in lines
        if line.startswith("# PANAS")
    ]

    scores = []

    for line in panas_lines:

        tokens = line.strip().split(";")[1:25]

        row = [
            int(float(t)) if t.lower() != "nan" else np.nan
            for t in tokens
        ]

        scores.append(row)

    panas = np.array(scores)

    pa_idx = [1, 2, 5, 9, 10, 12, 14, 16, 17, 21]
    na_idx = [0, 3, 6, 7, 11, 13, 15, 18, 19, 20]

    panas_pa = np.nanmean(panas[:, pa_idx], axis=1)
    panas_na = np.nanmean(panas[:, na_idx], axis=1)

    # BTP-CHANGE R9: the names are passed in instead of hard-coded.
    # The notebook's original line was:
    #     task_names = ["Base","TSST","Medi1","Fun","Medi2"][:len(scores)]
    #
    # This matters because the PANAS lines in the file are in the order the
    # subject did the tasks. Naming them from a fixed list attaches each
    # answer to the wrong session for the 9 subjects with a different order.
    task_names = list(task_names_in_order)[:len(scores)]

    return pd.DataFrame({
        "Task": task_names,
        "PANAS_PA": panas_pa,
        "PANAS_NA": panas_na
    })


# =============================================================================
# SESSION TIMING EXTRACTION
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 30
# Unchanged.
#
# The quest file has a "# START" line and an "# END" line, each with five
# numbers. Those numbers are minutes from the start of the recording.
# =============================================================================
def extract_session_timings(quest_path):

    lines = open(quest_path).readlines()

    start_line = [
        line for line in lines
        if line.startswith("# START")
    ][0]

    end_line = [
        line for line in lines
        if line.startswith("# END")
    ][0]

    start_times = list(
        map(float, start_line.strip().split(";")[1:6])
    )

    end_times = list(
        map(float, end_line.strip().split(";")[1:6])
    )

    return start_times, end_times


# =============================================================================
# WRIST SIGNAL RESAMPLING
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 32
# Unchanged.
# =============================================================================
def resample_wrist_signal(data, signal_name, sampling_rate):

    df = pd.DataFrame({
        signal_name: data["signal"]["wrist"][signal_name].flatten()
    })

    df["Time"] = pd.to_datetime(
        np.arange(len(df)) / sampling_rate,
        unit="s",
        origin="unix"
    )

    df.set_index("Time", inplace=True)

    return df.resample(RESAMPLE_WINDOW).agg([
        "mean",
        "std",
        "min",
        "max",
        "median",
        skew
    ])


# =============================================================================
# SUBJECT PROCESSING PIPELINE
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 34
# Unchanged except where marked.
# =============================================================================
def process_subject(subject_folder, task_order):

    subject_id = os.path.basename(subject_folder)

    pkl_path = os.path.join(
        subject_folder,
        f"{subject_id}.pkl"
    )

    quest_xls = os.path.join(
        subject_folder,
        f"{subject_id}_quest.xls"
    )

    quest_csv = os.path.join(
        subject_folder,
        f"{subject_id}_quest.csv"
    )

    if os.path.exists(quest_xls):
        quest_path = quest_xls
    else:
        quest_path = quest_csv

    # =====================================
    # LOAD PICKLE
    # =====================================

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # =====================================
    # CHEST SIGNAL PROCESSING
    # =====================================

    df_chest = pd.DataFrame({

        "ACC_X": data["signal"]["chest"]["ACC"][:, 0],
        "ACC_Y": data["signal"]["chest"]["ACC"][:, 1],
        "ACC_Z": data["signal"]["chest"]["ACC"][:, 2],

        "ECG": data["signal"]["chest"]["ECG"].flatten(),
        "EDA": data["signal"]["chest"]["EDA"].flatten(),
        "Temp": data["signal"]["chest"]["Temp"].flatten(),
        "Resp": data["signal"]["chest"]["Resp"].flatten()
    })

    df_chest["ACC_Mag_Chest"] = np.sqrt(
        df_chest["ACC_X"]**2 +
        df_chest["ACC_Y"]**2 +
        df_chest["ACC_Z"]**2
    )

    # Note, not a change: the three separate axes are thrown away here and only
    # the combined size is kept. This is why the notebook path and the
    # configs/config.yaml path in the repo cannot be mixed. See D1.
    df_chest.drop(
        columns=["ACC_X", "ACC_Y", "ACC_Z"],
        inplace=True
    )

    df_chest["Time"] = pd.to_datetime(
        np.arange(len(df_chest)) / 700,
        unit="s",
        origin="unix"
    )

    df_chest.set_index("Time", inplace=True)

    df_chest = df_chest.resample(RESAMPLE_WINDOW).agg([
        "mean",
        "std",
        "min",
        "max",
        "median",
        skew
    ])

    # =====================================
    # WRIST SIGNAL PROCESSING
    # =====================================

    wrist_acc = data["signal"]["wrist"]["ACC"]

    df_acc = pd.DataFrame({
        "ACC_Mag_Wrist": np.sqrt(
            wrist_acc[:, 0]**2 +
            wrist_acc[:, 1]**2 +
            wrist_acc[:, 2]**2
        )
    })

    df_acc["Time"] = pd.to_datetime(
        np.arange(len(df_acc)) / 32,
        unit="s",
        origin="unix"
    )

    df_acc.set_index("Time", inplace=True)

    df_acc = df_acc.resample(RESAMPLE_WINDOW).agg([
        "mean",
        "std",
        "min",
        "max",
        "median",
        skew
    ])

    df_bvp = resample_wrist_signal(data, "BVP", 64)
    df_eda = resample_wrist_signal(data, "EDA", 4)
    df_temp = resample_wrist_signal(data, "TEMP", 4)

    df_wrist = df_acc.join([
        df_bvp,
        df_eda,
        df_temp
    ])

    df_wrist = df_wrist.ffill()

    # =====================================
    # SENSOR FUSION
    # =====================================

    df_final = pd.merge(
        df_chest,
        df_wrist,
        left_index=True,
        right_index=True,
        how="outer"
    )

    # This one line decides which features survive. It drops any column that
    # has even one missing value anywhere. The last 5 second bin is usually
    # short, so std and skew of that bin are often missing, which takes the
    # whole column out. We report what survived at the end.
    df_final = df_final.dropna(axis=1)

    df_final.columns = [
        "_".join(filter(None, col)).strip()
        for col in df_final.columns.values
    ]

    df_final = df_final.reset_index(drop=True)

    # =====================================
    # ACTIVITY CONTEXT
    # =====================================

    mean_acc = df_final["ACC_Mag_Chest_mean"].mean()
    std_acc = df_final["ACC_Mag_Chest_mean"].std()

    low = mean_acc + 0.5 * std_acc
    high = mean_acc + std_acc

    df_final["Activity"] = np.select(
        [
            df_final["ACC_Mag_Chest_mean"] <= low,
            df_final["ACC_Mag_Chest_mean"] > high
        ],
        [
            "Still",
            "HeavyMovement"
        ],
        default="SlightMovement"
    )

    df_final["Posture"] = np.select(
        [
            df_final["ACC_Mag_Chest_mean"] <= 0.85,
            df_final["ACC_Mag_Chest_mean"] > 1.05
        ],
        [
            "Sit",
            "Walk"
        ],
        default="Stand"
    )

    # =====================================
    # TEMPORAL CONTEXT
    # =====================================
    #
    # Worth knowing: this is not real clock time. It stretches the recording
    # evenly across a pretend 24 hours. Two subjects at the same "hour" are
    # simply at the same point through their own session.

    total_windows = len(df_final)

    time_fraction = np.linspace(
        0,
        24,
        total_windows
    )

    df_final["Time_Sin"] = np.sin(
        2 * np.pi * time_fraction / 24
    )

    df_final["Time_Cos"] = np.cos(
        2 * np.pi * time_fraction / 24
    )

    # =====================================
    # TASK CONTEXT
    # =====================================

    # BTP-CHANGE R9: the order is passed in.
    # The notebook's original line was:
    #     session_labels = ["Base","TSST","Medi1","Fun","Medi2"]
    session_labels = list(task_order)

    starts, ends = extract_session_timings(quest_path)

    # 12 because there are 12 five-second rows in one minute.
    starts_idx = [int(x * 12) for x in starts]
    ends_idx = [int(x * 12) for x in ends]

    df_final["Task"] = "Unknown"
    df_final["SessionPhase"] = "Unknown"

    for label, s, e in zip(
        session_labels,
        starts_idx,
        ends_idx
    ):

        df_final.loc[s:e, "Task"] = label

        phase_len = e - s

        df_final.loc[
            s:s + int(0.3 * phase_len),
            "SessionPhase"
        ] = "Initial"

        df_final.loc[
            s + int(0.3 * phase_len):
            s + int(0.7 * phase_len),
            "SessionPhase"
        ] = "Mid"

        df_final.loc[
            s + int(0.7 * phase_len):e,
            "SessionPhase"
        ] = "End"

    rows_before_session_filter = len(df_final)

    df_final = df_final[
        df_final["Task"] != "Unknown"
    ].copy()

    # =====================================
    # PANAS STRESS LABELS
    # =====================================

    # BTP-CHANGE R9: the same order goes here, so each PANAS answer is named
    # after the session it was actually answered about.
    scores_df = extract_panas_scores(
        quest_path,
        task_order
    )

    df_final["PANAS_PA"] = np.nan
    df_final["PANAS_NA"] = np.nan

    for _, row in scores_df.iterrows():

        task = row["Task"]

        task_indices = df_final.index[
            df_final["Task"] == task
        ]

        if len(task_indices) == 0:
            continue

        # The score goes on the LAST row of the task block only.
        idx = task_indices[-1]

        df_final.at[idx, "PANAS_PA"] = row["PANAS_PA"]
        df_final.at[idx, "PANAS_NA"] = row["PANAS_NA"]

    df_final["stress_raw"] = (
        df_final["PANAS_NA"] -
        df_final["PANAS_PA"]
    ) / 2

    # Cut this subject's 5 scores into three groups: lowest, middle, highest.
    # Worth knowing: this is done per subject, so "HighStress" for one subject
    # is not the same feeling level as "HighStress" for another.
    df_final["Stress_Class"] = pd.qcut(
        df_final["stress_raw"].dropna(),
        q=3,
        labels=[
            "LowStress",
            "MediumStress",
            "HighStress"
        ]
    )

    # Spread each label 5 rows forward and 5 rows back, so 11 rows per score.
    df_final["Stress_Class"] = (
        df_final["Stress_Class"]
        .reindex(df_final.index)
        .ffill(limit=5)
        .bfill(limit=5)
    )

    # BTP-CHANGE R7: keep the raw PANAS numbers instead of dropping them.
    # The notebook deletes these three columns. We keep them so the audit can
    # check how the label was built, and so we can tell whether two subjects'
    # "HighStress" mean the same thing. They are never fed to the model.
    #
    # The notebook's original line was:
    #     df_final.drop(columns=["PANAS_PA","PANAS_NA","stress_raw"],
    #                   inplace=True, errors="ignore")

    return df_final, rows_before_session_filter


# =============================================================================
# ENCODE CATEGORICAL FEATURES
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 36
#
# BTP-CHANGE R5: we do NOT run this on the saved files. The reason is in
# docs/00_decisions.md under D4. Two problems with the original:
#
#   1. It runs on Stress_Class after .astype(str). Rows with no label are
#      NaN, and str(NaN) is the text "nan", so unlabelled rows quietly become
#      a fourth class.
#   2. LabelEncoder sorts words alphabetically, so the numbers come out as
#      HighStress=0, LowStress=1, MediumStress=2. Not low to high. Anyone who
#      later assumes 0 means Low gets wrong accuracy and a wrong stress score.
#
#   3. It is fitted separately for each subject. If one subject never has
#      "HeavyMovement", their Activity numbers mean different things from
#      everyone else's. Stacking subjects then mixes up the meanings.
#
# So we save the words instead of numbers. Turning words into one-hot columns
# later gives exactly the same input to the model, without any of the three
# problems. The function is kept here unchanged so the original behaviour is
# still on record.
# =============================================================================
def encode_categorical_columns(df):

    categorical_cols = [
        "Task",
        "SessionPhase",
        "Activity",
        "Posture",
        "Stress_Class"
    ]

    for col in categorical_cols:

        if col in df.columns:

            encoder = LabelEncoder()

            df[col] = encoder.fit_transform(
                df[col].astype(str)
            )

    return df


# =============================================================================
# MAIN EXECUTION
#
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 38
#
# BTP-CHANGE R6: the notebook wraps each subject in try/except and prints
# "Failed S5: ..." then carries on. That hides problems. We print the full
# error, keep going, and fail loudly at the end with a count.
#
# BTP-CHANGE R8: we add a subject_id column. The notebook does not, but the
# training notebook (Stress.ipynb, cell 2) expects a "subject_id" column to
# exist so it can leave it out of the signal columns. Without it we also
# cannot split by subject or stop windows from crossing between people.
# =============================================================================
def main():

    # BTP-CHANGE R9: two modes, so we can report both the repo's own
    # assumption and the corrected data without one overwriting the other.
    parser = argparse.ArgumentParser(
        description="Preprocess raw WESAD. Ported from Preproc.ipynb."
    )
    parser.add_argument(
        "--task-order",
        choices=["file", "notebook"],
        default="file",
        help=("file: read each subject's real task order from the '# ORDER' "
              "line in the quest file. This is the corrected data. "
              "notebook: use the hard-coded order Base,TSST,Medi1,Fun,Medi2, "
              "exactly as the repo does. Default: file."),
    )
    args = parser.parse_args()
    order_mode = args.task_order

    if order_mode == "file":
        output_dir = "data/processed/wesad"
        report_path = "docs/10_wesad_preprocessing.md"
        mode_text = "real order read from each quest file (corrected)"
    else:
        output_dir = "data/processed/wesad_notebook_order"
        report_path = "docs/10_wesad_preprocessing_notebook_order.md"
        mode_text = "hard-coded order Base,TSST,Medi1,Fun,Medi2, as the repo does"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Task order mode : %s" % order_mode)
    print("  %s" % mode_text)
    print("Output folder   : %s" % output_dir)
    print("=" * 60)
    print()

    subjects = sorted(os.listdir(INPUT_DIR))

    summary = []
    failures = []
    column_sets = {}
    order_used = {}

    for subject in subjects:

        subject_path = os.path.join(
            INPUT_DIR,
            subject
        )

        if not (
            os.path.isdir(subject_path) and
            subject.startswith("S")
        ):
            continue

        print("=" * 60)
        print(f"Processing {subject}...")

        try:

            # BTP-CHANGE R9: pick which order to use for this subject.
            if order_mode == "file":
                quest_xls = os.path.join(subject_path, f"{subject}_quest.xls")
                quest_csv = os.path.join(subject_path, f"{subject}_quest.csv")
                qp = quest_xls if os.path.exists(quest_xls) else quest_csv
                real_order = extract_session_order(qp)
                if real_order is None:
                    print("  no '# ORDER' line found, "
                          "falling back to the notebook order")
                    task_order = NOTEBOOK_ORDER
                else:
                    task_order = real_order
            else:
                task_order = NOTEBOOK_ORDER

            order_used[subject] = list(task_order)
            differs = list(task_order) != NOTEBOOK_ORDER
            print("  task order used : %s%s" % (
                " -> ".join(task_order),
                "   (differs from the notebook)" if differs else ""))

            df, rows_before = process_subject(subject_path, task_order)

            # BTP-CHANGE R8
            df["subject_id"] = subject

            # BTP-CHANGE R5: encode_categorical_columns is deliberately NOT
            # called. Words are saved as words.

            output_path = os.path.join(
                output_dir,
                f"{subject}.csv"
            )

            df.to_csv(output_path, index=False)

            # ---- report the numbers that matter ----
            n_rows = len(df)
            n_labelled = int(df["Stress_Class"].notna().sum())
            pct = (100.0 * n_labelled / n_rows) if n_rows else 0.0

            feature_cols = [
                c for c in df.columns
                if c not in ["Activity", "Posture", "Time_Sin", "Time_Cos",
                             "Task", "SessionPhase", "Stress_Class",
                             "PANAS_PA", "PANAS_NA", "stress_raw",
                             "subject_id"]
            ]
            column_sets[subject] = tuple(sorted(feature_cols))

            print(f"  rows before session filter : {rows_before}")
            print(f"  rows inside the 5 sessions : {n_rows}")
            print(f"  signal feature columns kept: {len(feature_cols)}")
            print(f"  labelled rows              : {n_labelled}  ({pct:.1f}%)")

            label_counts = df["Stress_Class"].value_counts(dropna=True)
            print(f"  label counts               : {dict(label_counts)}")

            task_counts = df["Task"].value_counts()
            print(f"  rows per task              : {dict(task_counts)}")

            print(f"  saved: {output_path}")

            # BTP-CHANGE R9: which stress class ended up on which task.
            # This is the number that changes between the two order modes,
            # so it is the quickest way to see whether the labels make sense.
            lab = df[df["Stress_Class"].notna()]
            per_task = lab.groupby("Task", observed=True)["Stress_Class"].agg(
                lambda col: col.mode().iloc[0] if len(col.mode()) else "-")
            print(f"  label per task             : {dict(per_task)}")

            summary.append({
                "subject": subject,
                "order_differs": "yes" if differs else "no",
                "rows_before_filter": rows_before,
                "rows_in_sessions": n_rows,
                "n_feature_cols": len(feature_cols),
                "labelled_rows": n_labelled,
                "labelled_pct": round(pct, 1),
                "n_low": int(label_counts.get("LowStress", 0)),
                "n_med": int(label_counts.get("MediumStress", 0)),
                "n_high": int(label_counts.get("HighStress", 0)),
                "on_TSST": str(per_task.get("TSST", "-")),
                "on_Fun": str(per_task.get("Fun", "-")),
            })

        except Exception as e:
            print(f"  FAILED {subject}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(subject)

        print()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY   (task order mode: %s)" % order_mode)
    print("=" * 60)

    if summary:
        sdf = pd.DataFrame(summary).sort_values("subject")
        print(sdf.to_string(index=False))
        print()
        print(f"total rows in sessions : {sdf['rows_in_sessions'].sum()}")
        print(f"total labelled rows    : {sdf['labelled_rows'].sum()}")
        print(f"labelled overall       : "
              f"{100.0 * sdf['labelled_rows'].sum() / sdf['rows_in_sessions'].sum():.1f}%")
        print()

        from collections import Counter
        c_tsst = Counter(r["on_TSST"] for r in summary)
        c_fun = Counter(r["on_Fun"] for r in summary)
        n_differ = sum(1 for r in summary if r["order_differs"] == "yes")
        print("subjects whose real order differs from the notebook: %d" % n_differ)
        print()
        print("Which stress class landed on which task, counted over subjects:")
        print("  %-14s %-9s %s" % ("class", "on TSST", "on Fun"))
        for k in sorted(set(list(c_tsst) + list(c_fun))):
            print("  %-14s %-9d %d" % (k, c_tsst.get(k, 0), c_fun.get(k, 0)))
        print()
        print("  TSST is the stress task, Fun is the amusement task. If the")
        print("  labels mean anything, TSST should mostly be HighStress and Fun")
        print("  should mostly not be. Compare this between the two order modes.")
        print()

        print("How many 15-row windows could be fully labelled?")
        print("  A label reaches 11 rows around each PANAS answer.")
        print("  A window is 15 rows. 11 is less than 15.")
        print("  So no window can be fully labelled. This is the main finding")
        print("  from Stage 1 and it goes to the mentor. See D6.")
        print()

    # Do all subjects keep the same columns? If not, we cannot stack them.
    distinct = set(column_sets.values())
    print(f"distinct feature-column sets across subjects: {len(distinct)}")
    if len(distinct) == 1:
        cols = sorted(list(distinct)[0])
        print(f"  good, all subjects kept the same {len(cols)} feature columns:")
        for c in cols:
            print(f"    {c}")
    else:
        print("  WARNING: subjects kept different columns.")
        print("  The assemble step will have to use only the columns that")
        print("  every subject has. Sets found:")
        for cs in distinct:
            who = [s for s, v in column_sets.items() if v == cs]
            print(f"    {len(cs)} cols, subjects {who}")

    print()
    if failures:
        print(f"FAILED SUBJECTS: {failures}")
    else:
        print("all subjects processed with no failures")

    # -------------------------------------------------------------------------
    # WRITE THE REPORT
    # -------------------------------------------------------------------------
    lines = []
    lines.append("# WESAD preprocessing, what actually happened")
    lines.append("")
    lines.append("Written by `btp/01_prep_wesad.py`, which is a copy of")
    lines.append("`Preproc.ipynb` cells 26 to 38 with the repairs listed in")
    lines.append("`docs/00_decisions.md`.")
    lines.append("")
    lines.append("**Task order mode: `%s`** - %s" % (order_mode, mode_text))
    lines.append("")
    lines.append("Output folder: `%s`" % output_dir)
    lines.append("")
    lines.append("### Task order used per subject")
    lines.append("")
    lines.append("| subject | order used | differs from notebook? |")
    lines.append("|---|---|---|")
    for sid in sorted(order_used):
        o = order_used[sid]
        lines.append("| %s | %s | %s |" % (
            sid, " -> ".join(o), "yes" if o != NOTEBOOK_ORDER else "no"))
    lines.append("")
    lines.append("## Settings used")
    lines.append("")
    lines.append("| setting | value | where it comes from |")
    lines.append("|---|---|---|")
    lines.append("| resample bin | 5 seconds | `RESAMPLE_WINDOW`, cell 26 |")
    lines.append("| chest sampling rate assumed | 700 Hz | hard-coded, cell 34 |")
    lines.append("| wrist ACC rate assumed | 32 Hz | hard-coded, cell 34 |")
    lines.append("| wrist BVP rate assumed | 64 Hz | hard-coded, cell 34 |")
    lines.append("| wrist EDA rate assumed | 4 Hz | hard-coded, cell 34 |")
    lines.append("| wrist TEMP rate assumed | 4 Hz | hard-coded, cell 34 |")
    lines.append("| summary stats per bin | mean, std, min, max, median, skew | cell 34 |")
    lines.append("| chest ACC axes | thrown away, replaced by combined size | cell 34 |")
    lines.append("| rows outside the 5 sessions | thrown away | cell 34 |")
    lines.append("| label spread | 5 rows forward, 5 rows back from each PANAS answer | cell 34 |")
    lines.append("| label groups | this subject's 5 scores cut into 3 equal groups | cell 34 |")
    lines.append("")

    if summary:
        lines.append("## Per subject")
        lines.append("")
        lines.append("| subject | rows before filter | rows in sessions | feature cols | labelled rows | labelled % | Low | Med | High |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in summary:
            lines.append("| %s | %d | %d | %d | %d | %.1f | %d | %d | %d |" % (
                r["subject"], r["rows_before_filter"], r["rows_in_sessions"],
                r["n_feature_cols"], r["labelled_rows"], r["labelled_pct"],
                r["n_low"], r["n_med"], r["n_high"]))
        lines.append("")
        tot_rows = sum(r["rows_in_sessions"] for r in summary)
        tot_lab = sum(r["labelled_rows"] for r in summary)
        lines.append("Total rows in sessions: %d" % tot_rows)
        lines.append("")
        lines.append("Total labelled rows: %d (%.1f%%)" % (
            tot_lab, 100.0 * tot_lab / tot_rows if tot_rows else 0))
        lines.append("")
        lines.append("## The labelling problem")
        lines.append("")
        lines.append("Each PANAS answer labels 11 rows: itself, 5 before, 5 after.")
        lines.append("A training window is 15 rows. 11 is smaller than 15, so no")
        lines.append("window can sit entirely inside a labelled block.")
        lines.append("")
        lines.append("This has to be decided before training. See D6 in")
        lines.append("`docs/00_decisions.md`.")
        lines.append("")

    if len(distinct) == 1:
        cols = sorted(list(distinct)[0])
        lines.append("## Feature columns kept (same for every subject)")
        lines.append("")
        lines.append("%d columns:" % len(cols))
        lines.append("")
        for c in cols:
            lines.append("- `%s`" % c)
        lines.append("")
        if any(c.startswith("EDA_x") for c in cols):
            lines.append("Note on `EDA_x` and `EDA_y`: chest EDA and wrist EDA are")
            lines.append("both called `EDA` in the pickle. When the two tables are")
            lines.append("merged, pandas adds `_x` and `_y` to tell them apart.")
            lines.append("`EDA_x` is the chest sensor, `EDA_y` is the wrist sensor.")
            lines.append("This is what the original notebook does too.")
            lines.append("")
    else:
        lines.append("## Feature columns kept")
        lines.append("")
        lines.append("**Subjects kept different columns.** Assemble must use only")
        lines.append("the columns every subject has.")
        lines.append("")
        for cs in distinct:
            who = [s for s, v in column_sets.items() if v == cs]
            lines.append("- %d columns, subjects %s" % (len(cs), who))
        lines.append("")

    lines.append("## Context columns")
    lines.append("")
    lines.append("Saved as words, not numbers. See D4 and repair R5.")
    lines.append("")
    lines.append("| column | possible values | how it is worked out |")
    lines.append("|---|---|---|")
    lines.append("| `Task` | Base, TSST, Medi1, Fun, Medi2 | session start/end times in the quest file |")
    lines.append("| `SessionPhase` | Initial, Mid, End | first 30%, middle 40%, last 30% of each session |")
    lines.append("| `Activity` | Still, SlightMovement, HeavyMovement | chest ACC size against this subject's own mean and std |")
    lines.append("| `Posture` | Sit, Stand, Walk | chest ACC size against fixed cut-offs 0.85 and 1.05 |")
    lines.append("| `Time_Sin`, `Time_Cos` | -1 to 1 | position through the recording, stretched over a pretend 24 hours |")
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("These subjects failed: %s" % failures)
        lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print(f"Report written to {report_path}")
    print(f"Per-subject CSVs written to {output_dir}/")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
