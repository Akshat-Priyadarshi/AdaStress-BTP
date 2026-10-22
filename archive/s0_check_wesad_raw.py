"""
s0_check_wesad_raw.py

WHY THIS FILE EXISTS
    The AdaStress repo has no script that checks the raw WESAD data before
    preprocessing. Its notebook (Preproc.ipynb, cell 38) just loops over
    subject folders inside a try/except and prints "Failed S5: ..." if
    anything is wrong. That hides problems instead of showing them.

    We need to know, before we train anything, that:
      - every subject folder has the two files the notebook reads
      - the pickle has the exact signal keys the notebook asks for
      - the questionnaire file has the lines the notebook parses
      - the session start/end times make sense

WHAT IT DOES
    Reads every subject folder in data/raw/WESAD/ and reports what is there.
    It does not change any data. It only reads and prints.

    It writes a report to:
        docs/WESAD_raw_structure.md

    This report is a deliverable. It is the answer to "what exactly is inside
    raw WESAD".

WHAT THE NOTEBOOK NEEDS (this is what we are checking for)
    data/raw/WESAD/S2/S2.pkl          <- pickle, latin1 encoded
    data/raw/WESAD/S2/S2_quest.csv    <- or S2_quest.xls

    Inside the pickle:
        data["signal"]["chest"]["ACC"]    700 Hz, 3 columns
        data["signal"]["chest"]["ECG"]    700 Hz
        data["signal"]["chest"]["EDA"]    700 Hz
        data["signal"]["chest"]["Temp"]   700 Hz
        data["signal"]["chest"]["Resp"]   700 Hz
        data["signal"]["wrist"]["ACC"]     32 Hz, 3 columns
        data["signal"]["wrist"]["BVP"]     64 Hz
        data["signal"]["wrist"]["EDA"]      4 Hz
        data["signal"]["wrist"]["TEMP"]     4 Hz

    Inside the quest file:
        a line starting with "# START"   -> 5 session start times, in minutes
        a line starting with "# END"     -> 5 session end times, in minutes
        lines starting with "# PANAS"    -> one per session

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/s0_check_wesad_raw.py
"""

import os
import pickle
import numpy as np

# -----------------------------------------------------------------------------
# Paths. Everything is relative to the AdaStress_BTP folder.
# -----------------------------------------------------------------------------
RAW_DIR = "data/raw/WESAD"
REPORT_PATH = "docs/WESAD_raw_structure.md"

# The notebook resamples everything into 5 second bins. We reuse the same
# number here so the window counts we report match what preprocessing will
# actually produce.
RESAMPLE_SECONDS = 5

# Sampling rates the notebook hard-codes.
CHEST_FS = 700
WRIST_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

CHEST_KEYS = ["ACC", "ECG", "EDA", "Temp", "Resp"]
WRIST_KEYS = ["ACC", "BVP", "EDA", "TEMP"]


def find_subject_folders(raw_dir):
    """Return the subject folders, in order. Same rule as the notebook:
    a folder whose name starts with S."""
    if not os.path.isdir(raw_dir):
        return []
    names = sorted(os.listdir(raw_dir))
    out = []
    for name in names:
        path = os.path.join(raw_dir, name)
        if os.path.isdir(path) and name.startswith("S"):
            out.append(path)
    return out


def find_quest_file(folder, subject_id):
    """The notebook prefers the .xls file and falls back to .csv."""
    xls = os.path.join(folder, subject_id + "_quest.xls")
    csv = os.path.join(folder, subject_id + "_quest.csv")
    if os.path.exists(xls):
        return xls
    if os.path.exists(csv):
        return csv
    return None


def read_quest(quest_path):
    """Pull out the numbers the notebook pulls out. Returns a dict.

    The quest file is semicolon separated with comment-style header lines.
    The notebook reads three kinds of line: '# START', '# END', '# PANAS'.
    """
    result = {
        "start_times": None,
        "end_times": None,
        "panas_line_count": 0,
        "problem": None,
    }

    try:
        with open(quest_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        result["problem"] = "could not open file: %s" % e
        return result

    start_lines = [l for l in lines if l.startswith("# START")]
    end_lines = [l for l in lines if l.startswith("# END")]
    panas_lines = [l for l in lines if l.startswith("# PANAS")]

    result["panas_line_count"] = len(panas_lines)

    if not start_lines:
        result["problem"] = "no '# START' line found"
        return result
    if not end_lines:
        result["problem"] = "no '# END' line found"
        return result

    try:
        # The notebook takes columns 1 to 5 (five sessions).
        result["start_times"] = [float(x) for x in start_lines[0].strip().split(";")[1:6]]
        result["end_times"] = [float(x) for x in end_lines[0].strip().split(";")[1:6]]
    except Exception as e:
        result["problem"] = "could not read the start/end numbers: %s" % e

    return result


def read_pickle(pkl_path):
    """Load the pickle and describe the signals. Returns a dict."""
    result = {"chest": {}, "wrist": {}, "extra_keys": [], "problem": None}

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        result["problem"] = "could not load pickle: %s" % e
        return result

    if "signal" not in data:
        result["problem"] = "no 'signal' key in the pickle"
        return result

    result["extra_keys"] = sorted([k for k in data.keys() if k != "signal"])

    for key in CHEST_KEYS:
        try:
            arr = np.asarray(data["signal"]["chest"][key])
            result["chest"][key] = {
                "shape": tuple(arr.shape),
                "seconds": round(arr.shape[0] / float(CHEST_FS), 1),
                "nan_count": int(np.isnan(arr).sum()),
            }
        except Exception as e:
            result["chest"][key] = {"missing": str(e)}

    for key in WRIST_KEYS:
        try:
            arr = np.asarray(data["signal"]["wrist"][key])
            fs = WRIST_RATES[key]
            result["wrist"][key] = {
                "shape": tuple(arr.shape),
                "seconds": round(arr.shape[0] / float(fs), 1),
                "nan_count": int(np.isnan(arr).sum()),
            }
        except Exception as e:
            result["wrist"][key] = {"missing": str(e)}

    # The label array is not used by the AdaStress notebook, but it is the
    # original WESAD condition label. Worth reporting so we can tell the two
    # kinds of label apart later.
    if "label" in data:
        lab = np.asarray(data["label"]).flatten()
        values, counts = np.unique(lab, return_counts=True)
        result["wesad_label_counts"] = dict(
            zip([int(v) for v in values], [int(c) for c in counts])
        )

    return result


def main():
    folders = find_subject_folders(RAW_DIR)

    lines = []
    lines.append("# Raw WESAD structure")
    lines.append("")
    lines.append("Written by `btp/s0_check_wesad_raw.py`. Nothing here was changed;")
    lines.append("this is a read-only report of what is on disk.")
    lines.append("")
    lines.append("Looked in: `%s`" % RAW_DIR)
    lines.append("")

    if not folders:
        msg = "No subject folders found in %s" % RAW_DIR
        print("PROBLEM: " + msg)
        print()
        print("The raw WESAD download should give you one folder per subject:")
        print("  data/raw/WESAD/S2/S2.pkl")
        print("  data/raw/WESAD/S2/S2_quest.csv")
        print("  data/raw/WESAD/S3/S3.pkl")
        print("  ...")
        print()
        print("Copy or link those folders in, then run this script again.")
        lines.append("**No subject folders found.** Raw WESAD is not in place yet.")
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    print("Found %d subject folders" % len(folders))
    print()

    summary_rows = []
    problem_count = 0

    for folder in folders:
        subject_id = os.path.basename(folder)
        print("=" * 60)
        print(subject_id)
        print("=" * 60)

        lines.append("---")
        lines.append("")
        lines.append("## %s" % subject_id)
        lines.append("")

        pkl_path = os.path.join(folder, subject_id + ".pkl")
        quest_path = find_quest_file(folder, subject_id)

        has_pkl = os.path.exists(pkl_path)
        has_quest = quest_path is not None

        print("  %s.pkl        : %s" % (subject_id, "found" if has_pkl else "MISSING"))
        print("  %s_quest      : %s" % (
            subject_id,
            os.path.basename(quest_path) if has_quest else "MISSING"))

        lines.append("- pickle: `%s`" % ("found" if has_pkl else "MISSING"))
        lines.append("- quest file: `%s`" % (
            os.path.basename(quest_path) if has_quest else "MISSING"))

        if not has_pkl or not has_quest:
            problem_count += 1
            print("  -> this subject cannot be processed")
            lines.append("")
            lines.append("**This subject cannot be processed.**")
            lines.append("")
            summary_rows.append((subject_id, "MISSING FILES", "-", "-", "-"))
            print()
            continue

        # ---- pickle ----
        pkl_info = read_pickle(pkl_path)
        if pkl_info["problem"]:
            problem_count += 1
            print("  pickle problem: %s" % pkl_info["problem"])
            lines.append("- pickle problem: %s" % pkl_info["problem"])
            summary_rows.append((subject_id, "PICKLE PROBLEM", "-", "-", "-"))
            print()
            continue

        print("  chest signals (700 Hz):")
        lines.append("")
        lines.append("### Signals")
        lines.append("")
        lines.append("| location | signal | shape | seconds | NaNs |")
        lines.append("|---|---|---|---:|---:|")

        chest_seconds = None
        for key in CHEST_KEYS:
            info = pkl_info["chest"][key]
            if "missing" in info:
                print("    %-6s MISSING" % key)
                lines.append("| chest | %s | MISSING | - | - |" % key)
                problem_count += 1
            else:
                print("    %-6s shape=%-14s  %.1f s  NaNs=%d" % (
                    key, str(info["shape"]), info["seconds"], info["nan_count"]))
                lines.append("| chest | %s | %s | %.1f | %d |" % (
                    key, info["shape"], info["seconds"], info["nan_count"]))
                if key == "ECG":
                    chest_seconds = info["seconds"]

        print("  wrist signals:")
        for key in WRIST_KEYS:
            info = pkl_info["wrist"][key]
            if "missing" in info:
                print("    %-6s MISSING" % key)
                lines.append("| wrist | %s | MISSING | - | - |" % key)
                problem_count += 1
            else:
                print("    %-6s shape=%-14s  %.1f s  NaNs=%d  (%d Hz)" % (
                    key, str(info["shape"]), info["seconds"],
                    info["nan_count"], WRIST_RATES[key]))
                lines.append("| wrist | %s | %s | %.1f | %d |" % (
                    key, info["shape"], info["seconds"], info["nan_count"]))

        if "wesad_label_counts" in pkl_info:
            print("  original WESAD condition labels: %s" % pkl_info["wesad_label_counts"])
            lines.append("")
            lines.append("Original WESAD condition label counts (not used by AdaStress,")
            lines.append("kept here so we do not confuse the two kinds of label):")
            lines.append("")
            lines.append("`%s`" % pkl_info["wesad_label_counts"])

        # ---- quest ----
        quest_info = read_quest(quest_path)
        lines.append("")
        lines.append("### Questionnaire file")
        lines.append("")

        if quest_info["problem"]:
            problem_count += 1
            print("  quest problem: %s" % quest_info["problem"])
            lines.append("- problem: %s" % quest_info["problem"])
            summary_rows.append((subject_id, "QUEST PROBLEM", "-", "-", "-"))
            print()
            continue

        starts = quest_info["start_times"]
        ends = quest_info["end_times"]
        panas_n = quest_info["panas_line_count"]

        print("  PANAS lines found : %d  (the notebook expects 5)" % panas_n)
        print("  session starts (min): %s" % starts)
        print("  session ends   (min): %s" % ends)

        lines.append("- PANAS lines: %d (notebook expects 5)" % panas_n)
        lines.append("- session start times, minutes: `%s`" % starts)
        lines.append("- session end times, minutes: `%s`" % ends)

        # How many 5 second rows will survive?
        #
        # The notebook keeps only rows that fall inside one of the five
        # sessions. Everything else is dropped ("Task != Unknown").
        # 5 second bins means 12 rows per minute.
        session_minutes = sum(e - s for s, e in zip(starts, ends))
        rows_inside_sessions = int(session_minutes * 60 / RESAMPLE_SECONDS)

        total_rows = None
        if chest_seconds is not None:
            total_rows = int(chest_seconds / RESAMPLE_SECONDS)

        print("  minutes inside the 5 sessions : %.1f" % session_minutes)
        print("  5-second rows inside sessions : about %d" % rows_inside_sessions)
        if total_rows:
            print("  5-second rows in whole recording: about %d" % total_rows)
            print("  -> preprocessing will throw away about %d rows outside the sessions"
                  % (total_rows - rows_inside_sessions))

        lines.append("- minutes inside the 5 sessions: %.1f" % session_minutes)
        lines.append("- 5-second rows inside sessions: about %d" % rows_inside_sessions)
        if total_rows:
            lines.append("- 5-second rows in whole recording: about %d" % total_rows)

        # Sanity check on the timings.
        bad_timing = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            if e <= s:
                bad_timing.append("session %d ends before it starts" % (i + 1))
        for i in range(len(starts) - 1):
            if starts[i + 1] < ends[i]:
                bad_timing.append("session %d overlaps session %d" % (i + 1, i + 2))
        if chest_seconds is not None and ends and ends[-1] * 60 > chest_seconds + 60:
            bad_timing.append("last session ends after the recording ends")

        if bad_timing:
            problem_count += 1
            print("  TIMING PROBLEMS: %s" % "; ".join(bad_timing))
            lines.append("- **timing problems:** %s" % "; ".join(bad_timing))
        else:
            print("  timings look consistent")
            lines.append("- timings look consistent")

        lines.append("")

        summary_rows.append((
            subject_id,
            "ok",
            "%d" % panas_n,
            "%.1f" % session_minutes,
            "%d" % rows_inside_sessions,
        ))
        print()

    # ---- overall summary ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("%-8s %-16s %-7s %-14s %s" % (
        "subject", "status", "panas", "session_min", "rows_in_sessions"))
    for row in summary_rows:
        print("%-8s %-16s %-7s %-14s %s" % row)
    print()
    print("subjects found : %d" % len(folders))
    print("problems       : %d" % problem_count)
    print()

    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| subject | status | PANAS lines | session minutes | rows in sessions |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary_rows:
        lines.append("| %s | %s | %s | %s | %s |" % row)
    lines.append("")
    lines.append("Subjects found: %d. Problems: %d." % (len(folders), problem_count))
    lines.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Report written to %s" % REPORT_PATH)
    if problem_count == 0:
        print("Raw WESAD looks complete. Stage 1 can start.")
    else:
        print("Fix the problems above before starting Stage 1.")


if __name__ == "__main__":
    main()
