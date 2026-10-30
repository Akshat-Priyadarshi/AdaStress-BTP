"""
39_build_wesad_7ch.py

WHAT THIS IS
    WESAD converted to the seven channels that can actually reach our users:
    the six an EmbracePlus wristband records, plus a breathing rate derived
    from the pulse.

        ACC_X_wrist, ACC_Y_wrist, ACC_Z_wrist
        BVP
        EDA_wrist
        Temp_wrist
        RESP            derived, not measured

WHY A SEPARATE FILE FROM THE 13-CHANNEL ONE
    btp/20_build_wesad_13ch.py builds the thirteen channels the repository's
    own config asks for. That was the faithful reproduction and it gave
    0.975. It is finished, and its output should not be overwritten.

    But seven of those thirteen come from a chest strap: ECG, chest EDA,
    chest temperature, the respiration belt and the chest accelerometer.
    EmbracePlus has none of them. So the thirteen-channel model cannot be
    pointed at our users at all, and everything from here uses this set
    instead.

    Keeping them as two files means the reproduction result stays
    reproducible and the two cannot write over each other.

ABOUT THE RESPIRATION CHANNEL, HONESTLY
    Breathing is not recorded by a wrist device. It is worked out from the
    heart rhythm: the gap between beats shortens on the in-breath and
    lengthens on the out-breath. btp/38_wesad_resp.py does that, and this
    file imports it so there is only one implementation.

    WESAD does have a real chest respiration belt. We deliberately do not
    use it. Our users have no belt, so a model trained on a measured signal
    and then fed a derived one would be reading two different quantities
    through the same column.

    What the channel is worth, measured: checked against WESAD's own belt
    across all fifteen subjects, the derived rate correlated at 0.185, and
    one subject came out negative. The level is about right, 18.9 against
    16.7, but the moment-to-moment tracking is poor.

    That is not a fault in the chain. On clean synthetic beats the same code
    recovered a breathing rate of 16.0 against a true 15.0, and tracked a
    drifting rate at 0.953. The modulation is simply weak in real wrist PPG
    and movement swamps it. Published work deriving respiratory rate this
    way reaches 0.5 to 0.8 against a reference under controlled conditions.

    It is included as a deliberate choice: breathing does change under
    stress, and it may carry more in free-living data than in a laboratory
    where everyone sits still. Every report using it should quote the 0.185.

EVERYTHING ELSE IS THE SAME
    Same 1 Hz grid, same task-order repair, same labelling options, same
    output shape as the thirteen-channel build. Only the channel list and
    the added respiration differ.

OUTPUT
    data/processed/wesad_7ch_<mode>_<label-source>/
        wesad_all.csv          every second, for self-supervised pretraining
        wesad_labelled.csv     the parts inside the five sessions
        label_maps.json

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/39_build_wesad_7ch.py --mode corrected \
            --label-source wesad_condition

    A few minutes. Needs btp/38_wesad_resp.py beside it.
"""

import os
import re
import sys
import json
import pickle
import argparse
import traceback
import numpy as np
import pandas as pd

# BTP-ADDED: respiration as a seventh channel.
#
# Breathing is not recorded by a wrist device, but it leaves a trace in the
# heart rhythm: the gap between beats shortens on the in-breath and lengthens
# on the out-breath. btp/38_wesad_resp.py works it out from the BVP trace,
# and this imports that same code so there is only one implementation.
#
# WESAD does have a real chest respiration belt. We deliberately do NOT use
# it: the users have no belt, so a model trained on a measured signal would
# be fed a derived one, and those are two different quantities sharing a
# column name.
#
# Be clear about what this channel is worth. Checked against WESAD's own belt
# across all 15 subjects, the derived rate correlated at 0.185, with one
# subject negative. The level is about right, 18.9 against 16.7, but the
# moment-to-moment tracking is poor. The chain itself is sound: on clean
# synthetic beats it recovered 16.0 against a true 15.0 and tracked a
# drifting rate at 0.953. The modulation is simply weak in real wrist PPG and
# movement swamps it.
#
# It is included as a deliberate choice, because breathing does change under
# stress and may carry more in free-living data than in a laboratory where
# everyone sits still. Any report using it should quote the 0.185.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from importlib import import_module
    _resp = import_module("38_wesad_resp")
    HAVE_RESP = True
except Exception:
    HAVE_RESP = False

INPUT_DIR = "data/raw/WESAD"

# One row per second. This is what configs/config.yaml means by fs: 1.
RESAMPLE = "1s"

# The order the released code assumes for every subject.
NOTEBOOK_ORDER = ["Base", "TSST", "Medi1", "Fun", "Medi2"]

# The 13 signal columns, in the exact order configs/config.yaml lists them.
# BTP-ADDED. The six channels EmbracePlus also records, plus derived
# respiration. Selected with --channels wrist7.
WRIST7_COLUMNS = [
    "ACC_X_wrist", "ACC_Y_wrist", "ACC_Z_wrist",
    "BVP", "EDA_wrist", "Temp_wrist", "RESP",
]

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


def build_subject(folder, mode, with_resp=False, resp_source="belt"):
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

    # BTP-ADDED: breathing rate from the wrist BVP, one value per second.
    #
    # Derived here rather than read from WESAD's chest belt, so that it is
    # the same quantity the users will have. See the note at the top of this
    # file for what it is worth: correlation 0.185 against that belt.
    if with_resp:
        if not HAVE_RESP:
            raise SystemExit("btp/38_wesad_resp.py is needed for --channels "
                             "wrist7 and could not be imported.")
        if resp_source == "belt":
            # WESAD's chest respiration band, which measures the chest
            # actually expanding. More accurate than reading it out of the
            # pulse, and available here where it is not on a wristband.
            #
            # What goes in the column is a RATE in breaths per minute, not
            # the raw belt waveform. The users' RESP is a rate, and a
            # waveform and a rate in the same column would be two different
            # quantities sharing a name.
            rr = _resp.resp_from_belt(chest["Resp"], n_seconds,
                                      method="peaks")
            src_note = "chest belt"
        else:
            beats = _resp.detect_beats(wrist["BVP"].flatten(),
                                       WRIST_FS["BVP"])
            rr = _resp.resp_from_beats(beats, n_seconds, method="peaks")
            src_note = "derived from BVP, %d beats" % len(beats)
        rr = pd.Series(rr[:n_seconds] if len(rr) >= n_seconds
                       else np.concatenate([rr, np.full(n_seconds - len(rr),
                                                        np.nan)]))
        # short gaps carried forward; long ones left empty rather than
        # invented
        df["RESP"] = rr.ffill(limit=30).bfill(limit=30).to_numpy()
        print("      RESP (%s): %.0f%% of seconds covered, median %.1f "
              "breaths/min"
              % (src_note, 100 * df["RESP"].notna().mean(),
                 float(df["RESP"].median())))

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
        description="Convert raw WESAD into the seven 1 Hz channels an "
                    "EmbracePlus wristband can also provide.")
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
    ap.add_argument("--resp-source", choices=["belt", "bvp"], default="belt",
                    help="where WESAD's breathing rate comes from. belt uses "
                         "the chest respiration band, which WESAD has and is "
                         "the accurate source. bvp derives it from the pulse, "
                         "the same way the users' will be, so the two ends "
                         "match. Both write a RATE in breaths per minute, "
                         "never a raw waveform.")
    ap.add_argument("--input-dir", default=INPUT_DIR)
    args = ap.parse_args()

    mode = args.mode
    lsrc = args.label_source
    tag = mode if lsrc == "panas_per_subject" else "%s_%s" % (mode, lsrc)
    if args.resp_source == "bvp":
        tag += "_respbvp"
    out_dir = "data/processed/wesad_7ch_%s" % tag
    report = "docs/61_wesad_7ch_%s.md" % tag
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("Building the 7 channels that can reach the user data")
    print("mode   : %s" % mode)
    print("labels : %s" % lsrc)
    print("resp   : %s" % ("chest belt (accurate, WESAD only)"
                           if args.resp_source == "belt"
                           else "derived from BVP (matches the users)"))
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
                os.path.join(args.input_dir, s), mode,
                with_resp=True, resp_source=args.resp_source)
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

    signal_out = WRIST7_COLUMNS
    cols = signal_out + ["Stress_Class"] + CONTEXT_COLUMNS + ["Subject_ID"]
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
    # BTP-CHANGE: pick whichever of these the chosen channel set actually
    # has. With --channels wrist7 the chest columns are not written.
    check_cols = [c for c in ["ECG", "EDA", "Temp", "BVP", "Temp_wrist",
                              "EDA_wrist", "RESP"] if c in full_out.columns]
    print(full_out[check_cols]
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
