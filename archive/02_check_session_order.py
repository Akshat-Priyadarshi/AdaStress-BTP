"""
02_check_session_order.py

WHY THIS FILE EXISTS
    Preproc.ipynb decides which session is which by hard-coding the order:

        session_labels = ["Base", "TSST", "Medi1", "Fun", "Medi2"]
        starts, ends = extract_session_timings(quest_path)

    It then pairs that fixed list with the five START/END numbers in the order
    they appear in the file.

    But the WESAD study did not put every subject through the tasks in the same
    order. It was deliberately mixed up between subjects, so that the order
    itself would not affect the results. The quest file records each subject's
    real order on a line starting with "# ORDER". The notebook never reads it.

    If a subject's real order is not Base, TSST, Medi1, Fun, Medi2, then the
    notebook puts the wrong name on the wrong session. That breaks two things
    at once:

      - the Task context feature says "Fun" where the stress task really was
      - the PANAS score for one session gets attached to a different session,
        so the stress label is attached to the wrong part of the recording

    Everything after this point depends on the answer, so we check it before
    building anything else.

WHAT IT DOES
    Reads the "# ORDER" line from every subject's quest file and compares it
    with the order the notebook assumes. Prints one line per subject saying
    match or MISMATCH, then a summary.

    It reads only. It changes nothing.

    It writes:  docs/11_session_order_check.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/02_check_session_order.py

    Takes a couple of seconds.

WHAT THE RESULT MEANS
    All 15 match
        Good. The notebook's assumption happens to hold. Nothing to fix.
        Go straight to assembling the dataset.

    Any mismatch
        The Task labels and the stress labels are wrong for those subjects.
        We add one repair to 01_prep_wesad.py so it reads the real order from
        the file, re-run preprocessing, and record it in the decisions log.
        This is a real finding and it goes to the mentor and the authors.
"""

import os
import re

RAW_DIR = "data/raw/WESAD"
REPORT_PATH = "docs/11_session_order_check.md"

# The order Preproc.ipynb assumes for every subject, cell 34.
NOTEBOOK_ORDER = ["Base", "TSST", "Medi1", "Fun", "Medi2"]


def tidy_name(text):
    """Make names comparable.

    The quest file writes things like 'Medi 1' with a space, and sometimes
    with different capitals. The notebook writes 'Medi1'. We remove spaces
    so the two can be compared.
    """
    return re.sub(r"\s+", "", str(text)).strip()


def find_quest_file(folder, subject_id):
    """Same rule as the notebook: prefer .xls, fall back to .csv."""
    xls = os.path.join(folder, subject_id + "_quest.xls")
    csv = os.path.join(folder, subject_id + "_quest.csv")
    if os.path.exists(xls):
        return xls
    if os.path.exists(csv):
        return csv
    return None


def read_order_and_times(quest_path):
    """Pull the ORDER, START and END lines out of a quest file.

    Returns a dict. 'order' is None if the file has no ORDER line.
    """
    out = {"order": None, "starts": None, "ends": None, "problem": None}

    try:
        with open(quest_path, "r", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        out["problem"] = "could not open: %s" % e
        return out

    order_lines = [l for l in lines if l.startswith("# ORDER")]
    start_lines = [l for l in lines if l.startswith("# START")]
    end_lines = [l for l in lines if l.startswith("# END")]

    if order_lines:
        # drop the "# ORDER" token itself, keep the rest, ignore empty cells
        tokens = order_lines[0].strip().split(";")[1:]
        names = [tidy_name(t) for t in tokens if tidy_name(t) != ""]
        out["order"] = names

    if start_lines and end_lines:
        try:
            out["starts"] = [float(x) for x in start_lines[0].strip().split(";")[1:6]]
            out["ends"] = [float(x) for x in end_lines[0].strip().split(";")[1:6]]
        except Exception as e:
            out["problem"] = "could not read start/end numbers: %s" % e

    return out


def main():
    if not os.path.isdir(RAW_DIR):
        print("PROBLEM: %s does not exist." % RAW_DIR)
        return

    folders = sorted(
        os.path.join(RAW_DIR, n) for n in sorted(os.listdir(RAW_DIR))
        if os.path.isdir(os.path.join(RAW_DIR, n)) and n.startswith("S")
    )

    if not folders:
        print("PROBLEM: no subject folders in %s" % RAW_DIR)
        return

    print("The order Preproc.ipynb assumes for everyone:")
    print("  %s" % " -> ".join(NOTEBOOK_ORDER))
    print()

    rows = []
    matches = 0
    mismatches = []
    no_order_line = []

    for folder in folders:
        subject_id = os.path.basename(folder)
        quest_path = find_quest_file(folder, subject_id)

        if quest_path is None:
            print("%-5s  no quest file" % subject_id)
            rows.append((subject_id, "no quest file", "-", "-"))
            continue

        info = read_order_and_times(quest_path)

        if info["problem"]:
            print("%-5s  %s" % (subject_id, info["problem"]))
            rows.append((subject_id, "problem", "-", "-"))
            continue

        # session lengths in minutes, useful even without an ORDER line
        lengths = []
        if info["starts"] and info["ends"]:
            lengths = [round(e - s, 1) for s, e in zip(info["starts"], info["ends"])]

        if info["order"] is None:
            no_order_line.append(subject_id)
            print("%-5s  NO '# ORDER' LINE   lengths(min): %s" % (subject_id, lengths))
            rows.append((subject_id, "no ORDER line", "-", str(lengths)))
            continue

        real = info["order"]
        assumed = NOTEBOOK_ORDER

        # Compare only the first five, since that is all the notebook uses.
        same = real[:5] == assumed

        if same:
            matches += 1
            print("%-5s  match              lengths(min): %s" % (subject_id, lengths))
            rows.append((subject_id, "match", " -> ".join(real[:5]), str(lengths)))
        else:
            mismatches.append(subject_id)
            print("%-5s  MISMATCH" % subject_id)
            print("        real order : %s" % " -> ".join(real[:5]))
            print("        notebook   : %s" % " -> ".join(assumed))
            print("        lengths(min): %s" % lengths)
            rows.append((subject_id, "MISMATCH", " -> ".join(real[:5]), str(lengths)))

    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("subjects checked : %d" % len(folders))
    print("match            : %d" % matches)
    print("MISMATCH         : %d  %s" % (len(mismatches), mismatches if mismatches else ""))
    print("no ORDER line    : %d  %s" % (len(no_order_line), no_order_line if no_order_line else ""))
    print()

    if mismatches:
        print("WHAT THIS MEANS")
        print()
        print("For the subjects above, Preproc.ipynb puts the wrong task name on")
        print("the wrong session. Two things are then wrong for those subjects:")
        print()
        print("  1. The Task context feature is wrong. It may say 'Fun' where the")
        print("     stress task really happened.")
        print("  2. The stress label is wrong. extract_panas_scores also assumes")
        print("     the same fixed order, so each PANAS answer gets attached to a")
        print("     different session than the one it was answered about.")
        print()
        print("NEXT STEP: add a repair to 01_prep_wesad.py so it reads the real")
        print("order from the quest file, then run preprocessing again.")
        print("Record it in docs/00_decisions.md.")
    elif no_order_line:
        print("WHAT THIS MEANS")
        print()
        print("Some quest files have no '# ORDER' line, so we cannot confirm the")
        print("order from the file. Use the session lengths as a cross-check:")
        print("in WESAD the stress task runs roughly 10 to 12 minutes and")
        print("amusement runs roughly 6 to 7 minutes. If the second session is")
        print("short and the fourth is long, the notebook's order is wrong for")
        print("that subject.")
    else:
        print("WHAT THIS MEANS")
        print()
        print("Every subject's real order matches what the notebook assumes.")
        print("Nothing to fix. The Task labels and the stress labels are")
        print("attached to the right sessions. Move on to assembling the")
        print("dataset.")

    # -------------------------------------------------------------------------
    lines = []
    lines.append("# Session order check")
    lines.append("")
    lines.append("Written by `btp/02_check_session_order.py`. Read-only check.")
    lines.append("")
    lines.append("`Preproc.ipynb` cell 34 hard-codes one task order for every subject:")
    lines.append("")
    lines.append("    %s" % " -> ".join(NOTEBOOK_ORDER))
    lines.append("")
    lines.append("WESAD mixed the order up between subjects on purpose. Each quest")
    lines.append("file records the real order on a `# ORDER` line, which the notebook")
    lines.append("never reads. This check compares the two.")
    lines.append("")
    lines.append("| subject | result | real order | session lengths (min) |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append("| %s | %s | %s | %s |" % r)
    lines.append("")
    lines.append("Checked %d subjects: %d match, %d mismatch, %d with no ORDER line."
                 % (len(folders), matches, len(mismatches), len(no_order_line)))
    lines.append("")

    if mismatches:
        lines.append("## Result: mismatches found")
        lines.append("")
        lines.append("Affected subjects: %s" % mismatches)
        lines.append("")
        lines.append("For these subjects the released preprocessing attaches:")
        lines.append("")
        lines.append("- the wrong task name to each session, so the `Task` context")
        lines.append("  feature is wrong")
        lines.append("- the wrong PANAS answer to each session, so the stress label")
        lines.append("  is attached to the wrong part of the recording")
        lines.append("")
        lines.append("A repair is needed in `btp/01_prep_wesad.py` to read the real")
        lines.append("order from the quest file. Preprocessing must then be re-run.")
        lines.append("")
    elif no_order_line:
        lines.append("## Result: some files have no ORDER line")
        lines.append("")
        lines.append("Subjects: %s" % no_order_line)
        lines.append("")
        lines.append("Use session lengths as a cross-check. In WESAD the stress task")
        lines.append("runs roughly 10 to 12 minutes and amusement roughly 6 to 7.")
        lines.append("")
    else:
        lines.append("## Result: everything matches")
        lines.append("")
        lines.append("The notebook's assumed order is correct for all subjects.")
        lines.append("No repair needed.")
        lines.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print()
    print("Report written to %s" % REPORT_PATH)


if __name__ == "__main__":
    main()
