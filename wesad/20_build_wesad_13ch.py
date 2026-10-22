"""
20_build_wesad_13ch.py

WHY THIS FILE EXISTS
    The repository has two pipelines inside it, and they do not fit together.

      1. The notebook path (Preproc.ipynb -> Stress.ipynb). Reads raw WESAD,
         but turns every signal into 5-second summary numbers (mean, std, min,
         max, median, skew).

      2. The library path (adastress/ + configs/config.yaml). This is the real
         package: a proper dataset loader, model file, training loop and
         command line scripts. It wants 13 raw signal columns at 1 Hz.

    We used path 1 so far. That was the wrong choice. Path 2 is what the
    repository is actually built around, and its own config says so.

    The only reason path 2 was never used is that nothing in the repository
    converts raw WESAD into the 13 columns it wants. The file it ships,
    data/sample_data.csv, is only a smoke test: 999 rows, every value squashed
    between 0 and 1, and Task has 7 values when WESAD only has 5 sessions. Its
    own comment says the real dataset is deliberately not included.

    This script is that missing converter. It is the ONLY file we add. After
    it runs, everything else is the repository's own code, unchanged.

WHAT IT DOES
    Reads each raw WESAD subject and writes the exact 18 columns that
    configs/config.yaml expects:

      ACC_X, ACC_Y, ACC_Z, ECG, EDA, Temp, Resp                (chest, 700 Hz)
      ACC_X_wrist, ACC_Y_wrist, ACC_Z_wrist                    (wrist, 32 Hz)
      BVP                                                       (wrist, 64 Hz)
      EDA_wrist                                                 (wrist, 4 Hz)
      Temp_wrist                                                (wrist, 4 Hz)
      Stress_Class, Task, SessionPhase, Activity, Subject_ID

    Every signal is averaged down to 1 Hz, one row per second.

WHAT IS DIFFERENT FROM WHAT WE DID BEFORE
    - The three chest accelerometer axes are KEPT. Our earlier work threw them
      away and replaced them with a single magnitude, because that is what the
      notebook does. The config wants all three.
    - One row is one second of raw signal, not a 5-second summary. The encoder
      can then see the shape of the waveform instead of six numbers.
    - A window is 60 rows, so 60 real seconds. Before it was 15 rows of
      5-second summaries, which is 75 seconds, a different thing entirely.

THREE WAYS OF MAKING THE LABEL  (--label-source)

    This is a real change to the method, not a bug fix, so it is a flag and
    the repository's own way stays the default.

    --label-source panas_per_subject     DEFAULT, what the repo does
        Each subject's five questionnaire scores are sorted and cut into three
        groups, separately for that person. So "HighStress" means "this
        person's worst of their own five sessions".

        The problem, measured: S11's calm baseline session comes out Medium
        and S14's comes out High, while the training people's baseline is Low.
        Nothing about their bodies differs, only how their own five scores
        happened to rank. A model fitted on nine people cannot transfer to
        three others when the target means something different for each.

    --label-source panas_global
        Same PANAS scores, but all 75 of them (15 subjects x 5 sessions) are
        cut into three groups TOGETHER. Now "HighStress" means the same
        feeling level for everybody. This is the smallest possible change that
        tests whether the per-person cut is what caps the accuracy.

    --label-source wesad_condition
        Ignores PANAS. Uses the condition label WESAD itself ships inside the
        .pkl file, which is what most WESAD papers classify:

            1 baseline  -> 0
            3 amusement -> 1
            2 stress    -> 2

        Meditation (4) and the transition markers (0, 5, 6, 7) are dropped, so
        this stays a three class problem like the rest of the pipeline.

        These labels mean the same thing for every subject, and there are far
        more of them, because every second inside a condition is labelled
        rather than eleven seconds around a questionnaire.

TWO MODES, BOTH WORTH RUNNING

    NOTE: --mode only affects the task order and how PANAS answers are spread.
    With --label-source wesad_condition the labels are already dense, so the
    literal / corrected distinction does not apply to them.

    --mode literal
        Exactly what the released code does. The task order is hard-coded and
        each questionnaire answer is spread over 11 rows.

        At 1 Hz that means 11 SECONDS of label, against a 60-second window.
        Expect close to zero usable windows. That is the point of running it:
        it shows, with a number, that the released settings cannot produce a
        trainable labelled set.

    --mode corrected
        Two fixes, both already documented:
          R9  read the real task order from the quest file. 9 of 15 subjects
              do the tasks in a different order than the code assumes, so
              without this the stress and amusement sessions are swapped for
              them, and each questionnaire answer is attached to the wrong
              session.
          D6  label a whole session with that session's own answer, since the
              questionnaire was filled in once per session about that session.

WHAT THE REPOSITORY'S LOADER NEEDS, AND WHY IT SHAPES THIS FILE
    Reading adastress/datasets.py:

      - classes = sorted(df[label].dropna().unique()), then a direct lookup.
        So the labelled file must have NO missing labels anywhere, or it
        crashes. We write a separate labelled file with those rows removed.

      - Context columns are majority-voted per window and then turned into a
        float tensor. So context must be plain whole numbers, not one-hot.
        We write Task, SessionPhase and Activity as integers and save the
        meaning of each number alongside.

      - prepare_labelled_dataset groups by subject, so labelled windows never
        mix two people. Good, and worth noting: our earlier repair R11 was
        only needed for the notebook path.

      - prepare_unlabelled_windows does NOT group by subject, so the
        self-supervised windows WILL cross from one subject to the next at the
        join points. We do not change their code; we just count how many
        windows are affected and report it.

OUTPUT
    data/processed/wesad_13ch_<mode>_<label-source>/
        wesad_all.csv          every second, for self-supervised pretraining
        wesad_labelled.csv     only rows with a label, for the classifier
        label_maps.json        what each context number means
    docs/40_wesad_13ch_<mode>.md

HOW TO RUN
    From the AdaStress_BTP folder:

        # the repo's own way, unchanged
        python3 btp/20_build_wesad_13ch.py --mode corrected
        python3 btp/20_build_wesad_13ch.py --mode literal

        # the two label experiments
        python3 btp/20_build_wesad_13ch.py --mode corrected \
            --label-source panas_global
        python3 btp/20_build_wesad_13ch.py --mode corrected \
            --label-source wesad_condition

    Takes a few minutes per mode. Each subject holds about 4 million rows of
    chest signal to average down.
"""

import os
import re
import json
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd

INPUT_DIR = "data/raw/WESAD"

# One row per second. This is what configs/config.yaml means by fs: 1.
RESAMPLE = "1s"

# The order the released code assumes for every subject.
NOTEBOOK_ORDER = ["Base", "TSST", "Medi1", "Fun", "Medi2"]

# The 13 signal columns, in the exact order configs/config.yaml lists them.
SIGNAL_COLUMNS = [
    "ACC_X", "ACC_Y", "ACC_Z", "ECG", "EDA", "Temp", "Resp",
    "ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
    "BVP", "EDA_wrist", "Temp_wrist",
]

CONTEXT_COLUMNS = ["Task", "SessionPhase", "Activity"]

# Low, Medium, High. Written down once so nothing has to guess later.
# The released code runs LabelEncoder on the words, which sorts them
# alphabetically and gives HighStress = 0. That is not low to high, and it
# also turns unlabelled rows into a fourth class. We map explicitly instead.
LABEL_MAP = {"LowStress": 0, "MediumStress": 1, "HighStress": 2}

# Raw sampling rates, as stated in the WESAD documentation.
CHEST_FS = 700
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

# WESAD's own condition labels, shipped inside the .pkl.
# 0, 5, 6 and 7 are transition and setup markers, not conditions.
# Meditation (4) is dropped so this stays a three class problem.
CONDITION_MAP = {1: 0, 3: 1, 2: 2}      # baseline -> 0, amusement -> 1, stress -> 2
CONDITION_NAMES = {0: "baseline", 1: "amusement", 2: "stress"}


def to_1hz(values, fs, names):
    """Average a raw signal down to one row per second.

    values : array, either [n] or [n, k]
    fs     : how many samples per second it was recorded at
    names  : column name(s) to give the result
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    df = pd.DataFrame(arr, columns=names)
    df["Time"] = pd.to_datetime(np.arange(len(df)) / float(fs),
                                unit="s", origin="unix")
    df = df.set_index("Time")
    return df.resample(RESAMPLE).mean()


def read_quest(quest_path):
    """Pull the task order and the session times out of the quest file."""
    lines = open(quest_path).readlines()

    def find(prefix):
        got = [l for l in lines if l.startswith(prefix)]
        return got[0] if got else None

    order_line = find("# ORDER")
    start_line = find("# START")
    end_line = find("# END")

    order = None
    if order_line:
        tokens = order_line.strip().split(";")[1:]
        names = [re.sub(r"\s+", "", t).strip() for t in tokens]
        names = [n for n in names if n]
        if len(names) >= 5:
            order = names[:5]

    starts = [float(x) for x in start_line.strip().split(";")[1:6]]
    ends = [float(x) for x in end_line.strip().split(";")[1:6]]

    panas = [l for l in lines if l.startswith("# PANAS")]

    return order, starts, ends, panas


def panas_scores(panas_lines, task_order):
    """Turn the questionnaire lines into one positive and one negative score
    per session.

    Ten of the 24 answers are positive-feeling questions and ten are
    negative-feeling. Same index lists the released code uses.
    """
    rows = []
    for line in panas_lines:
        tokens = line.strip().split(";")[1:25]
        rows.append([int(float(t)) if t.lower() != "nan" else np.nan
                     for t in tokens])

    arr = np.array(rows)
    pa_idx = [1, 2, 5, 9, 10, 12, 14, 16, 17, 21]
    na_idx = [0, 3, 6, 7, 11, 13, 15, 18, 19, 20]

    return pd.DataFrame({
        "Task": list(task_order)[:len(rows)],
        # what the repo code does: average the ten items
        "PANAS_PA": np.nanmean(arr[:, pa_idx], axis=1),
        "PANAS_NA": np.nanmean(arr[:, na_idx], axis=1),
        # what the paper describes: sum them, giving 10 to 50
        "PANAS_PA_sum": np.nansum(arr[:, pa_idx], axis=1),
        "PANAS_NA_sum": np.nansum(arr[:, na_idx], axis=1),
    })


def build_subject(folder, mode):
    """Turn one raw subject into 1 Hz rows with the 13 columns plus context."""
    subject = os.path.basename(folder)

    pkl = os.path.join(folder, subject + ".pkl")
    xls = os.path.join(folder, subject + "_quest.xls")
    csv = os.path.join(folder, subject + "_quest.csv")
    quest = xls if os.path.exists(xls) else csv

    with open(pkl, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]

    # ---- chest, 700 Hz, all three accelerometer axes kept this time ----
    parts = [
        to_1hz(chest["ACC"], CHEST_FS, ["ACC_X", "ACC_Y", "ACC_Z"]),
        to_1hz(chest["ECG"].flatten(), CHEST_FS, ["ECG"]),
        to_1hz(chest["EDA"].flatten(), CHEST_FS, ["EDA"]),
        to_1hz(chest["Temp"].flatten(), CHEST_FS, ["Temp"]),
        to_1hz(chest["Resp"].flatten(), CHEST_FS, ["Resp"]),
        # ---- wrist ----
        to_1hz(wrist["ACC"], WRIST_FS["ACC"],
               ["ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist"]),
        to_1hz(wrist["BVP"].flatten(), WRIST_FS["BVP"], ["BVP"]),
        to_1hz(wrist["EDA"].flatten(), WRIST_FS["EDA"], ["EDA_wrist"]),
        to_1hz(wrist["TEMP"].flatten(), WRIST_FS["TEMP"], ["Temp_wrist"]),
    ]

    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="outer")

    # Short gaps come from the slower wrist sensors not lining up exactly with
    # every one-second bin. Filling forward then backward is the same thing the
    # released code does with .ffill() on the wrist table.
    df = df.ffill().bfill()
    df = df[SIGNAL_COLUMNS].reset_index(drop=True)

    n_seconds = len(df)

    # WESAD's own condition label, one per second.
    #
    # The array is at 700 Hz and changes only every few minutes, so taking the
    # value at the start of each second is exact for all but the boundary
    # second of each block.
    if "label" in data:
        raw_lab = np.asarray(data["label"]).flatten()[::CHEST_FS]
        raw_lab = raw_lab[:n_seconds]
        if len(raw_lab) < n_seconds:
            raw_lab = np.concatenate(
                [raw_lab, np.full(n_seconds - len(raw_lab), 0)])
        df["condition_raw"] = raw_lab.astype(int)
    else:
        df["condition_raw"] = 0

    # ---- Activity, worked out from chest movement ----
    # Same rule as the released code: compare against this subject's own
    # average and spread, so it does not depend on the units.
    mag = np.sqrt(chest["ACC"][:, 0] ** 2 +
                  chest["ACC"][:, 1] ** 2 +
                  chest["ACC"][:, 2] ** 2)
    mag_1hz = to_1hz(mag, CHEST_FS, ["m"])["m"].ffill().bfill()
    mag_1hz = mag_1hz.iloc[:n_seconds].reset_index(drop=True)

    m, s = mag_1hz.mean(), mag_1hz.std()
    df["Activity"] = np.select(
        [mag_1hz <= m + 0.5 * s, mag_1hz > m + s],
        ["Still", "HeavyMovement"],
        default="SlightMovement")

    # ---- which task order to use ----
    real_order, starts, ends, panas_lines = read_quest(quest)

    if mode == "corrected" and real_order:
        task_order = real_order
    else:
        task_order = NOTEBOOK_ORDER

    differs = list(task_order) != NOTEBOOK_ORDER

    # ---- Task and SessionPhase ----
    # 60 because there are 60 one-second rows in a minute. The released code
    # uses 12 because its rows are 5 seconds long.
    df["Task"] = "Unknown"
    df["SessionPhase"] = "Unknown"

    for name, st, en in zip(task_order, starts, ends):
        a, b = int(st * 60), int(en * 60)
        if a >= n_seconds:
            continue
        b = min(b, n_seconds - 1)
        df.loc[a:b, "Task"] = name

        span = b - a
        df.loc[a:a + int(0.3 * span), "SessionPhase"] = "Initial"
        df.loc[a + int(0.3 * span):a + int(0.7 * span), "SessionPhase"] = "Mid"
        df.loc[a + int(0.7 * span):b, "SessionPhase"] = "End"

    rows_all = len(df)
    in_session = df[df["Task"] != "Unknown"].copy()

    # ---- stress label from the questionnaire ----
    scores = panas_scores(panas_lines, task_order)
    in_session["stress_raw"] = np.nan
    in_session["stress_raw_sum"] = np.nan

    for _, row in scores.iterrows():
        idx = in_session.index[in_session["Task"] == row["Task"]]
        if len(idx) == 0:
            continue
        # the answer belongs to the end of that session
        in_session.loc[idx[-1], "stress_raw"] = (row["PANAS_NA"] -
                                                 row["PANAS_PA"]) / 2
        # the paper's version: summed items, not averaged
        in_session.loc[idx[-1], "stress_raw_sum"] = (row["PANAS_NA_sum"] -
                                                     row["PANAS_PA_sum"])

    # The label itself is assigned later, in apply_label_source(), because
    # panas_global needs every subject's scores before it can cut them.
    in_session["Subject_ID"] = subject
    df["Subject_ID"] = subject

    info = {
        "subject": subject,
        "order_used": " -> ".join(task_order),
        "order_differs": "yes" if differs else "no",
        "seconds_total": rows_all,
        "seconds_in_sessions": len(in_session),
    }
    return df, in_session, info


def apply_label_source(in_session_all, label_source, mode,
                       panas_threshold=0.5):
    """Fill in Stress_Class for every in-session row.

    in_session_all : all subjects stacked, with stress_raw filled on the last
                     row of each task block, and condition_raw on every row.

    Returns the same frame with Stress_Class as text, plus a short note for
    the report.
    """
    names = ["LowStress", "MediumStress", "HighStress"]
    out = in_session_all.copy()
    out["Stress_Class"] = pd.Series([np.nan] * len(out), index=out.index,
                                    dtype="object")

    # ---------------------------------------------------------------
    if label_source == "wesad_condition":
        # WESAD's own labels. Already one per second, so nothing is spread.
        mapped = out["condition_raw"].map(CONDITION_MAP)
        out["Stress_Class"] = mapped.map(
            {0: names[0], 1: names[1], 2: names[2]})

        kept = int(mapped.notna().sum())
        dropped = int(mapped.isna().sum())
        note = ("WESAD's own condition labels: baseline -> LowStress, "
                "amusement -> MediumStress, stress -> HighStress. "
                "%d rows kept, %d dropped (meditation and transitions)."
                % (kept, dropped))
        return out, note

    # ---------------------------------------------------------------
    # Both PANAS routes start from the same anchors: one score on the last
    # row of each task block.
    anchors = out[out["stress_raw"].notna()][
        ["Subject_ID", "Task", "stress_raw", "stress_raw_sum"]].copy()

    if label_source == "panas_paper":
        # The paper's description, Section VI-A. Summed items, normalized
        # across the whole dataset, cut at fixed distributional thresholds
        # rather than into equal thirds.
        v = anchors["stress_raw_sum"].astype(float)
        z = (v - v.mean()) / (v.std() if v.std() else 1.0)
        lo, hi = -panas_threshold, panas_threshold
        anchors["cls"] = np.select(
            [z <= lo, z > hi], [names[0], names[2]], default=names[1])
        counts = anchors["cls"].value_counts().to_dict()
        note = ("PANAS as the paper describes it: items summed (range 10-50), "
                "normalized across the whole dataset, cut at %+.2f and %+.2f "
                "standard deviations. Unlike equal thirds this keeps the real "
                "shape of the scores. Session counts: %s."
                % (lo, hi, counts))
        out.loc[anchors.index, "Stress_Class"] = anchors["cls"].to_numpy()
        return _spread(out, mode, note)

    if label_source == "panas_global":
        # Cut all 75 scores together, so a class means the same thing for
        # everybody.
        cats = pd.qcut(anchors["stress_raw"], q=3, labels=names)
        anchors["cls"] = cats.astype(object)
        note = ("PANAS cut into three groups across ALL subjects together "
                "(%d scores). A class means the same level for everyone."
                % len(anchors))
    else:
        # The repo's way: cut each subject's five scores separately.
        parts = []
        for subj, g in anchors.groupby("Subject_ID", sort=True):
            cats = pd.qcut(g["stress_raw"], q=3, labels=names)
            g = g.copy()
            g["cls"] = cats.astype(object)
            parts.append(g)
        anchors = pd.concat(parts)
        note = ("PANAS cut into three groups separately for each subject, "
                "as the repository does. A class means 'this person's own "
                "worst / middle / best of five sessions'.")

    # put the class back on its anchor row
    out.loc[anchors.index, "Stress_Class"] = anchors["cls"].to_numpy()

    return _spread(out, mode, note)


def _spread(out, mode, note):
    """Spread each anchor label out, either the repo's way or by session."""
    if mode == "literal":
        # the repo's spread: 5 rows forward, 5 back, so 11 seconds
        parts = []
        for subj, g in out.groupby("Subject_ID", sort=True):
            g = g.copy()
            g["Stress_Class"] = (g["Stress_Class"].ffill(limit=5)
                                 .bfill(limit=5))
            parts.append(g)
        out = pd.concat(parts).sort_index()
        note += " Spread 5 rows each way (11 seconds)."
    else:
        # the whole session gets its own answer
        parts = []
        for (subj, task), g in out.groupby(["Subject_ID", "Task"], sort=False):
            g = g.copy()
            got = g["Stress_Class"].dropna()
            if len(got):
                g["Stress_Class"] = got.iloc[-1]
            parts.append(g)
        out = pd.concat(parts).sort_index()
        note += " Whole session labelled with its own answer."

    return out, note


def main():
    ap = argparse.ArgumentParser(
        description="Convert raw WESAD into the 13-column 1 Hz CSV that "
                    "configs/config.yaml expects. The only file we add.")
    ap.add_argument("--mode", choices=["literal", "corrected"], required=True)
    ap.add_argument("--label-source",
                    choices=["panas_per_subject", "panas_global",
                             "panas_paper", "wesad_condition"],
                    default="panas_per_subject",
                    help="how the stress label is made. The default is what "
                         "the repository does.")
    ap.add_argument("--panas-threshold", type=float, default=0.5,
                    help="only for panas_paper. Where to cut, in standard "
                         "deviations either side of the mean. The paper says "
                         "'distributional thresholds' without giving a "
                         "number, so this is our choice. Default 0.5.")
    ap.add_argument("--input-dir", default=INPUT_DIR)
    args = ap.parse_args()

    mode = args.mode
    lsrc = args.label_source
    tag = mode if lsrc == "panas_per_subject" else "%s_%s" % (mode, lsrc)
    out_dir = "data/processed/wesad_13ch_%s" % tag
    report = "docs/40_wesad_13ch_%s.md" % tag
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("Building the 13-channel 1 Hz dataset the repo's config expects")
    print("mode   : %s" % mode)
    print("labels : %s" % lsrc)
    if mode == "literal":
        print("         hard-coded task order, label spread over 11 seconds")
    else:
        print("         real task order from the quest file (R9),")
        print("         whole session labelled with its own answer (D6)")
    print("output : %s" % out_dir)
    print("=" * 64)
    print()

    subjects = sorted(n for n in os.listdir(args.input_dir)
                      if os.path.isdir(os.path.join(args.input_dir, n))
                      and n.startswith("S"))

    all_rows, lab_rows, infos, failures = [], [], [], []

    for s in subjects:
        print("Processing %s ..." % s)
        try:
            df, in_session, info = build_subject(
                os.path.join(args.input_dir, s), mode)
            all_rows.append(df)
            lab_rows.append(in_session)
            infos.append(info)
            print("  order used   : %s%s" % (
                info["order_used"],
                "   (differs from the released code)"
                if info["order_differs"] == "yes" else ""))
            print("  seconds      : %d total, %d inside sessions"
                  % (info["seconds_total"], info["seconds_in_sessions"]))
        except Exception as e:
            print("  FAILED %s: %s" % (s, e))
            traceback.print_exc()
            failures.append(s)
        print()

    if not all_rows:
        raise SystemExit("Nothing processed.")

    full = pd.concat(all_rows, ignore_index=True)
    labelled = pd.concat(lab_rows, ignore_index=True)

    # Assign the label here, not per subject, because panas_global has to see
    # every subject's scores before it can cut them into three.
    labelled, label_note = apply_label_source(labelled, lsrc, mode,
                                             args.panas_threshold)
    print()
    print("labels: %s" % label_note)
    print()

    labelled = labelled[labelled["Stress_Class"].notna()].copy()

    # the full file needs the column too, even though the SSL step ignores it
    full["Stress_Class"] = np.nan

    # ---- turn words into whole numbers ----
    # The loader majority-votes the context and then makes a float tensor out
    # of it, so context has to be numbers, not one-hot.
    maps = {}
    for col in CONTEXT_COLUMNS:
        for frame in (full, labelled):
            frame[col] = frame[col].astype(str)
        values = sorted(set(full[col].unique()) | set(labelled[col].unique()))
        m = {v: i for i, v in enumerate(values)}
        maps[col] = m
        full[col] = full[col].map(m)
        labelled[col] = labelled[col].map(m)

    labelled["Stress_Class"] = labelled["Stress_Class"].map(LABEL_MAP).astype(int)
    maps["Stress_Class"] = LABEL_MAP

    cols = SIGNAL_COLUMNS + ["Stress_Class"] + CONTEXT_COLUMNS + ["Subject_ID"]
    full_out = full.reindex(columns=cols)
    full_out["Stress_Class"] = 0        # unused by the SSL step, must exist
    labelled_out = labelled.reindex(columns=cols)

    p_all = os.path.join(out_dir, "wesad_all.csv")
    p_lab = os.path.join(out_dir, "wesad_labelled.csv")
    full_out.to_csv(p_all, index=False)
    labelled_out.to_csv(p_lab, index=False)
    json.dump(maps, open(os.path.join(out_dir, "label_maps.json"), "w"),
              indent=2)

    # ---- how many windows will this actually give? ----
    W, S = 60, 30
    ssl_windows = max(0, (len(full_out) - W) // S + 1)

    per_subject = []
    total_lab_windows = 0
    for subj, g in labelled_out.groupby("Subject_ID", sort=True):
        w = max(0, (len(g) - W) // S + 1) if len(g) >= W else 0
        per_subject.append((subj, len(g), w))
        total_lab_windows += w

    print("=" * 64)
    print("SUMMARY   (mode: %s)" % mode)
    print("=" * 64)
    print("subjects processed : %d" % len(infos))
    print("rows, all seconds  : %d" % len(full_out))
    print("rows, labelled     : %d" % len(labelled_out))
    print()
    print("window is %d rows, stride %d, straight from configs/config.yaml"
          % (W, S))
    print("  self-supervised windows : %d" % ssl_windows)
    print("  labelled windows        : %d" % total_lab_windows)
    print()

    print("labelled seconds and windows, per subject:")
    print("  %-8s %10s %10s" % ("subject", "seconds", "windows"))
    for subj, n, w in per_subject:
        print("  %-8s %10d %10d" % (subj, n, w))
    print()

    if total_lab_windows == 0:
        print("  NO LABELLED WINDOWS AT ALL.")
        print("  Each questionnaire answer covers 11 seconds. A window is 60")
        print("  seconds. So no window can be filled. This is what the")
        print("  released settings produce, and it is the reason for the")
        print("  corrected mode.")
        print()

    if len(labelled_out):
        counts = labelled_out["Stress_Class"].value_counts().sort_index()
        print("labelled rows per class (0 Low, 1 Medium, 2 High):")
        print(counts.to_string())
        print()

    # The repo's SSL loader does not split by subject, so its windows run
    # across the joins between people. Count how many.
    boundaries = full_out["Subject_ID"].ne(
        full_out["Subject_ID"].shift()).to_numpy().nonzero()[0][1:]
    crossing = 0
    for st in range(0, len(full_out) - W + 1, S):
        if any(st < b < st + W for b in boundaries):
            crossing += 1
    print("self-supervised windows that run across two subjects: %d of %d"
          % (crossing, ssl_windows))
    print("  prepare_unlabelled_windows in adastress/datasets.py does not")
    print("  split by subject. We leave their code alone and just report it.")
    print()

    print("value ranges, a quick check that these are real sensor readings:")
    print(full_out[["ECG", "EDA", "Temp", "BVP", "Temp_wrist"]]
          .agg(["min", "max"]).round(3).to_string())
    print("  Temp should be about 33 C. If everything sits between 0 and 1,")
    print("  something has scaled the data and that is wrong.")
    print()
    print("context numbers mean:")
    for col in CONTEXT_COLUMNS:
        print("  %-14s %s" % (col, maps[col]))
    print()
    print("written:")
    print("  %s" % p_all)
    print("  %s" % p_lab)
    if failures:
        print()
        print("FAILED: %s" % failures)

    # ---- report ----
    L = []
    L.append("# WESAD as 13 channels at 1 Hz, mode `%s`" % mode)
    L.append("")
    L.append("Written by `btp/20_build_wesad_13ch.py`, the one file we add so")
    L.append("that the repository's own `adastress/` package can be run. There")
    L.append("is no converter in the repository: the file it ships,")
    L.append("`data/sample_data.csv`, is a 999-row smoke test with every value")
    L.append("squashed between 0 and 1 and 7 values for `Task` where WESAD has")
    L.append("5 sessions.")
    L.append("")
    L.append("| setting | value | from |")
    L.append("|---|---|---|")
    L.append("| rows | one per second | `fs: 1` |")
    L.append("| signal columns | 13 | `signal_columns` |")
    L.append("| window | 60 rows, stride 30 | `window_sec`, `stride_sec` |")
    L.append("| context | `Task`, `SessionPhase`, `Activity` as numbers | `context_columns` |")
    L.append("| mode | `%s` | ours |" % mode)
    L.append("| label source | `%s` | ours |" % lsrc)
    L.append("")
    L.append("**How the label was made:** %s" % label_note)
    L.append("")
    L.append("## Counts")
    L.append("")
    L.append("| | |")
    L.append("|---|---:|")
    L.append("| subjects | %d |" % len(infos))
    L.append("| rows, all seconds | %d |" % len(full_out))
    L.append("| rows, labelled | %d |" % len(labelled_out))
    L.append("| self-supervised windows | %d |" % ssl_windows)
    L.append("| labelled windows | **%d** |" % total_lab_windows)
    L.append("")
    L.append("For comparison, the 5-second summary route we used before gave")
    L.append("617 labelled windows from 54 columns.")
    L.append("")
    if total_lab_windows == 0:
        L.append("**No labelled windows at all.** Each questionnaire answer")
        L.append("covers 11 seconds; a window is 60 seconds. Nothing can be")
        L.append("filled. This is what the released settings produce.")
        L.append("")
    L.append("## Per subject")
    L.append("")
    L.append("| subject | order used | differs? | seconds total | in sessions | labelled |")
    L.append("|---|---|---|---:|---:|---:|")
    lab_per = labelled_out.groupby("Subject_ID").size().to_dict()
    for i in infos:
        L.append("| %s | %s | %s | %d | %d | %d |" % (
            i["subject"], i["order_used"], i["order_differs"],
            i["seconds_total"], i["seconds_in_sessions"],
            lab_per.get(i["subject"], 0)))
    L.append("")
    L.append("## Known behaviour we did not change")
    L.append("")
    L.append("`prepare_unlabelled_windows` in `adastress/datasets.py` does not")
    L.append("split by subject, so **%d of %d** self-supervised windows run"
             % (crossing, ssl_windows))
    L.append("across the join between two people. Their code is left alone.")
    L.append("")
    L.append("`prepare_labelled_dataset` **does** split by subject, so labelled")
    L.append("windows never mix two people.")
    L.append("")
    L.append("## What the context numbers mean")
    L.append("")
    for col in CONTEXT_COLUMNS + ["Stress_Class"]:
        L.append("- `%s`: %s" % (col, maps[col]))
    L.append("")

    os.makedirs(os.path.dirname(report), exist_ok=True)
    open(report, "w").write("\n".join(L) + "\n")
    print()
    print("report: %s" % report)


if __name__ == "__main__":
    main()
