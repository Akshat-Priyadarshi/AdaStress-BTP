"""
s0_check_user1_context.py

WHY THIS FILE EXISTS
    The AdaStress repo knows nothing about our user1 data, so there is no
    script in it that can look at these files. We need our own.

    Two questions have to be answered before we design anything:

      1. How many real stress labels does user1 actually have?
         This is decision D2 in the plan. If the answer is "four STAI
         questionnaires" then fine-tuning is not possible and both context
         experiments become inference only. Better to find that out now than
         after building a fine-tuning pipeline.

      2. Which columns in user1 could act as context?
         We need the exact column names, units and sampling rates so that
         the context comparison table is based on the files, not on memory.

WHAT IT DOES
    Reads every CSV in data/raw/user1/ and prints, for each one:
      - number of rows and columns
      - column names and types
      - how many values are missing
      - the time range, if it can find a time column
      - a guess at what kind of column each one is
        (physiological / context / label / time / unknown)

    It only reads. It changes nothing.

    It writes:
      docs/user1_raw_structure.md    human readable report
      context/user1_columns.csv      one row per column, for the context table

    The guesses are only guesses, made from column names. You should read the
    report and correct anything it got wrong. That corrected version is what
    goes into context/inventory.md later.

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/s0_check_user1_context.py
"""

import os
import glob
import pandas as pd
import numpy as np

USER1_DIR = "data/raw/user1"
REPORT_PATH = "docs/user1_raw_structure.md"
COLUMNS_CSV = "context/user1_columns.csv"

# How many rows to read from each file when we only need the shape and types.
# Some of these files are long and we do not need all of it to describe them.
PEEK_ROWS = 200000

# Word lists used to guess what a column is. Lower case, matched as substrings.
LABEL_WORDS = ["stai", "stress", "label", "anxiety", "panas", "survey",
               "ema", "selfreport", "self_report", "score", "class"]
CONTEXT_WORDS = ["step", "met", "act_count", "act_int", "activity", "actigraph",
                 "intensity", "calorie", "sleep", "posture", "task"]
PHYSIO_WORDS = ["eda", "bvp", "pr", "pulse", "hr", "heart", "temp", "acc",
                "resp", "ibi", "systolic", "peak", "gsr", "skin"]
TIME_WORDS = ["time", "date", "timestamp", "datetime", "epoch", "unix", "clock"]


def guess_kind(column_name):
    """Guess what a column is, from its name only."""
    name = str(column_name).lower()

    # time first, because "timestamp" also contains "temp"... no it does not,
    # but "datetime" does contain "time", so order matters for clarity.
    for word in TIME_WORDS:
        if word in name:
            return "time"
    for word in LABEL_WORDS:
        if word in name:
            return "label?"
    for word in CONTEXT_WORDS:
        if word in name:
            return "context?"
    for word in PHYSIO_WORDS:
        if word in name:
            return "physio?"
    return "unknown"


def find_time_column(df):
    """Return the name of the most likely time column, or None."""
    for col in df.columns:
        if guess_kind(col) == "time":
            return col
    return None


def describe_time_column(series):
    """Try to read a column as time and report the range and gap.
    Returns a short text description."""
    # try as a normal date string first
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        good = parsed.dropna()
        if len(good) > 1:
            span_seconds = (good.max() - good.min()).total_seconds()
            gaps = good.sort_values().diff().dt.total_seconds().dropna()
            step = gaps.median() if len(gaps) else float("nan")
            rate = (1.0 / step) if step and step > 0 else float("nan")
            return {
                "first": str(good.min()),
                "last": str(good.max()),
                "span_hours": round(span_seconds / 3600.0, 2),
                "median_gap_seconds": round(float(step), 4),
                "implied_hz": round(float(rate), 4),
            }
    except Exception:
        pass

    # try as plain numbers (seconds or milliseconds since epoch)
    try:
        nums = pd.to_numeric(series, errors="coerce").dropna()
        if len(nums) > 1:
            gaps = nums.sort_values().diff().dropna()
            step = gaps.median() if len(gaps) else float("nan")
            return {
                "first": str(nums.min()),
                "last": str(nums.max()),
                "span_hours": None,
                "median_gap_seconds": round(float(step), 4),
                "implied_hz": round(1.0 / float(step), 4) if step and step > 0 else None,
            }
    except Exception:
        pass

    return None


def main():
    if not os.path.isdir(USER1_DIR):
        print("PROBLEM: %s does not exist." % USER1_DIR)
        print("Run setup_stage0.sh first, or make the link yourself:")
        print("  ln -s <path-to-your-user1-folder> data/raw/user1")
        return

    paths = sorted(glob.glob(os.path.join(USER1_DIR, "*.csv")))
    if not paths:
        print("PROBLEM: no CSV files found in %s" % USER1_DIR)
        return

    print("Found %d CSV files in %s" % (len(paths), USER1_DIR))
    print()

    lines = []
    lines.append("# Raw user1 structure")
    lines.append("")
    lines.append("Written by `btp/s0_check_user1_context.py`. Read-only report.")
    lines.append("")
    lines.append("The `kind` column is a **guess based on the column name only**.")
    lines.append("Correct it by hand before using it in `context/inventory.md`.")
    lines.append("")

    column_rows = []          # for the CSV
    label_candidates = []     # files that look like they hold labels

    for path in paths:
        fname = os.path.basename(path)
        print("=" * 60)
        print(fname)
        print("=" * 60)

        try:
            df = pd.read_csv(path, nrows=PEEK_ROWS)
        except Exception as e:
            print("  could not read: %s" % e)
            lines.append("## %s" % fname)
            lines.append("")
            lines.append("Could not read: %s" % e)
            lines.append("")
            continue

        # full row count without loading everything into memory
        try:
            with open(path, "r", errors="replace") as f:
                total_rows = sum(1 for _ in f) - 1
        except Exception:
            total_rows = len(df)

        print("  rows    : %d" % total_rows)
        print("  columns : %d" % len(df.columns))
        if total_rows > PEEK_ROWS:
            print("  (only the first %d rows were read for the details below)" % PEEK_ROWS)
        print()

        lines.append("---")
        lines.append("")
        lines.append("## %s" % fname)
        lines.append("")
        lines.append("- rows: %d" % total_rows)
        lines.append("- columns: %d" % len(df.columns))
        lines.append("")
        lines.append("| column | dtype | kind (guess) | missing | example values |")
        lines.append("|---|---|---|---:|---|")

        print("  %-24s %-10s %-10s %8s  %s" % (
            "column", "dtype", "kind", "missing", "example"))
        print("  " + "-" * 78)

        for col in df.columns:
            series = df[col]
            kind = guess_kind(col)
            missing = int(series.isna().sum())

            examples = series.dropna().unique()[:3]
            example_text = ", ".join(str(x)[:14] for x in examples)

            print("  %-24s %-10s %-10s %8d  %s" % (
                str(col)[:24], str(series.dtype)[:10], kind, missing, example_text[:30]))

            lines.append("| `%s` | %s | %s | %d | %s |" % (
                col, series.dtype, kind, missing, example_text))

            column_rows.append({
                "file": fname,
                "column": col,
                "dtype": str(series.dtype),
                "kind_guess": kind,
                "n_rows": total_rows,
                "n_missing": missing,
                "n_unique": int(series.nunique(dropna=True)),
            })

            if kind == "label?":
                label_candidates.append((fname, col, int(series.nunique(dropna=True)),
                                         int(series.notna().sum())))

        print()

        # time information
        time_col = find_time_column(df)
        if time_col is None:
            print("  no time column found by name")
            lines.append("")
            lines.append("No time column found by name. "
                         "Check whether the rows are just in order.")
        else:
            info = describe_time_column(df[time_col])
            if info is None:
                print("  time column '%s' found but could not be read as time" % time_col)
                lines.append("")
                lines.append("Time column `%s` found but could not be parsed." % time_col)
            else:
                print("  time column   : %s" % time_col)
                print("  first         : %s" % info["first"])
                print("  last          : %s" % info["last"])
                if info["span_hours"] is not None:
                    print("  span (hours)  : %s" % info["span_hours"])
                print("  gap between rows (s): %s" % info["median_gap_seconds"])
                print("  implied rate (Hz)   : %s" % info["implied_hz"])

                lines.append("")
                lines.append("- time column: `%s`" % time_col)
                lines.append("- first: `%s`" % info["first"])
                lines.append("- last: `%s`" % info["last"])
                if info["span_hours"] is not None:
                    lines.append("- span: %s hours" % info["span_hours"])
                lines.append("- median gap between rows: %s seconds" % info["median_gap_seconds"])
                lines.append("- implied sampling rate: %s Hz" % info["implied_hz"])

        lines.append("")
        print()

    # -------------------------------------------------------------------------
    # Decision D2: the label budget
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DECISION D2: how many real stress labels does user1 have?")
    print("=" * 60)
    print()

    lines.append("---")
    lines.append("")
    lines.append("## Decision D2: label budget")
    lines.append("")

    if not label_candidates:
        print("No column name looked like a stress label.")
        print()
        print("This matters. If user1 has no stress labels in these CSVs, then:")
        print("  - fine-tuning cannot be done honestly")
        print("  - Experiment A and Experiment B become inference only")
        print("  - the STAI scores are somewhere else (a notebook, a sheet, on paper)")
        print()
        print("Find where the STAI scores live, and how many there are, before")
        print("Stage 4. Write the number in docs/00_decisions.md under D2.")

        lines.append("No column looked like a stress label.")
        lines.append("")
        lines.append("Find where the STAI scores are kept and how many exist.")
        lines.append("Write the number in `docs/00_decisions.md` under D2.")
    else:
        print("Columns that might hold labels:")
        print()
        print("  %-34s %-18s %10s %10s" % ("file", "column", "n_unique", "n_filled"))
        print("  " + "-" * 76)
        lines.append("Columns that might hold labels:")
        lines.append("")
        lines.append("| file | column | distinct values | filled rows |")
        lines.append("|---|---|---:|---:|")
        for fname, col, n_unique, n_filled in label_candidates:
            print("  %-34s %-18s %10d %10d" % (fname[:34], str(col)[:18], n_unique, n_filled))
            lines.append("| %s | `%s` | %d | %d |" % (fname, col, n_unique, n_filled))
        print()
        print("Read this carefully. A column with thousands of filled rows but only")
        print("3 or 4 distinct values is usually ONE questionnaire answer copied down")
        print("many rows. That is not thousands of labels. It is 3 or 4 labels.")
        print()
        print("The number that matters for D2 is: how many times was the")
        print("questionnaire actually filled in.")

        lines.append("")
        lines.append("**Careful:** a column with many filled rows but only 3 or 4 distinct")
        lines.append("values is usually one questionnaire answer copied down many rows.")
        lines.append("The number that matters is how many times the questionnaire was")
        lines.append("actually answered, not how many rows carry a value.")

    lines.append("")
    lines.append("Decision rule from the plan:")
    lines.append("")
    lines.append("| real labelled windows | what we can do |")
    lines.append("|---|---|")
    lines.append("| 200 or more | fine-tune gate + classifier, Experiments A and B as planned |")
    lines.append("| 30 to 200 | fine-tune classifier head only, wide error bars, no personalisation claim |")
    lines.append("| under 30 | inference only, compare score distributions, no fine-tuning |")
    lines.append("")

    print()
    print("Decision rule:")
    print("  200 or more real labels -> fine-tune as planned")
    print("  30 to 200               -> fine-tune head only, report carefully")
    print("  under 30                -> inference only, no fine-tuning")
    print()

    # -------------------------------------------------------------------------
    # write outputs
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    os.makedirs(os.path.dirname(COLUMNS_CSV), exist_ok=True)
    pd.DataFrame(column_rows).to_csv(COLUMNS_CSV, index=False)

    print("Report written to  %s" % REPORT_PATH)
    print("Column list written to %s" % COLUMNS_CSV)


if __name__ == "__main__":
    main()
