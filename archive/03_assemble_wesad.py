"""
03_assemble_wesad.py

WHY THIS FILE EXISTS
    This step is missing from the repo.

    Preproc.ipynb saves one CSV per subject: S2.csv, S3.csv, ... S17.csv
    Stress.ipynb reads three completely different files:

        all_subjects_unlabelled.csv     for the self-supervised pretraining
        generalized_labelled.csv        for training the classifier
        finetune_labelled.csv           for per-subject fine-tuning

    Nothing in the repo makes those three files. The step that joins the
    subjects together, splits labelled from unlabelled, and decides which
    subjects go where, simply is not there. So we have to write it, which
    means we have to decide it. Every decision is printed and recorded.

WHAT IT DOES
    1. Reads the per-subject CSVs made by 01_prep_wesad.py.
    2. Checks every subject has the same columns.
    3. Splits the 15 subjects into train / validation / test. By subject,
       never by row, so no person appears in two sets.
    4. Turns the stress words into numbers with one fixed mapping.
    5. Writes the three CSVs Stress.ipynb expects.
    6. Counts how many 15-row windows would actually be usable, and prints it.

THREE PROBLEMS IT HAS TO SOLVE
    1. Label leakage through the back door.
       Stress.ipynb cell 2 picks signal channels by dropping any column whose
       name contains stress_class, task, sessionphase, activity or subject_id,
       and keeping every other number. Our PANAS_PA, PANAS_NA and stress_raw
       columns do not match that list, so they would be fed in as signals.
       stress_raw IS the label before it was cut into three. That would hand
       the model the answer.
       So the training files are written WITHOUT those three columns. A
       separate copy keeping them is written for the audit only.

    2. Windows crossing between people.
       Stress.ipynb slides its window over the whole stacked CSV, so at every
       join between two subjects a window would hold rows from two different
       people. We keep subject_id in the files and count windows per subject,
       so the training step can respect the boundary.

    3. Not enough label to fill a window (decision D6).
       Each PANAS answer labels 11 rows. A window is 15 rows. So no window
       sits entirely inside a labelled block. Two ways to handle it, and the
       script does whichever you ask for:

         --label-mode strict
             Keep the repo's 11-row spread exactly as it is. A window gets
             the most common label among whichever of its rows are labelled,
             and is kept only if enough of them are. Faithful, but leaves
             very few windows.

         --label-mode block
             The PANAS questionnaire was answered about a whole session, so
             give every row of that session the label from its own answer.
             Many more windows, and honest about what the questionnaire
             actually measured. But then Task predicts the label perfectly,
             so Task becomes useless as a context feature. The audit
             measures that.

       Run both. Report both. The pair is the finding.

OUTPUT
    data/assembled/<label-mode>/
        all_subjects_unlabelled.csv     train subjects only, no labels needed
        generalized_labelled.csv        train + validation subjects, labelled
        finetune_labelled.csv           test subjects, labelled
        with_panas_for_audit.csv        everything, keeps PANAS. AUDIT ONLY.
        split.json                      which subject went where
    docs/12_assembled_<label-mode>.md

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/03_assemble_wesad.py --label-mode strict
        python3 btp/03_assemble_wesad.py --label-mode block

    Add --input-dir data/processed/wesad_notebook_order to do the same for
    the uncorrected task order.

    Takes a few seconds.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


# =============================================================================
# SETTINGS
# =============================================================================

# The one and only mapping from words to numbers. See D4 in the decisions log.
# The repo's own LabelEncoder gave HighStress=0, which is not low to high and
# would silently break every accuracy number and the stress score.
LABEL_MAP = {
    "LowStress": 0,
    "MediumStress": 1,
    "HighStress": 2,
}

# Window size used by Stress.ipynb cell 1: WINDOW_SEC=15, FS=1, so TW=15 rows.
# Each row is a 5 second summary, so 15 rows is about 75 seconds of real time.
# That mismatch is recorded in D1.
WINDOW_ROWS = 15
STRIDE_ROWS = 15

# Columns that must never reach the model. PANAS_PA and PANAS_NA are the
# questionnaire answers. stress_raw is the label before it was cut into three.
LEAK_COLUMNS = ["PANAS_PA", "PANAS_NA", "stress_raw"]

# Context columns, kept as words. Stress.ipynb one-hot encodes them.
CONTEXT_COLUMNS = ["Task", "SessionPhase", "Activity", "Posture"]

# Which subjects go where.
#
# Decision D7. Chosen once, written down, never changed between experiments,
# so that every result is comparable. 9 train, 3 validation, 3 test.
# The test subjects are held back completely: the self-supervised pretraining
# does not see them either, not even without labels.
#
# S9 is put in train on purpose. It has no MediumStress rows at all, so
# having it in test would make the test set impossible to score fairly.
SPLIT = {
    "train": ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"],
    "val":   ["S11", "S13", "S14"],
    "test":  ["S15", "S16", "S17"],
}


# =============================================================================
# LOADING
# =============================================================================
def load_subjects(input_dir):
    """Read every per-subject CSV. Returns dict of subject -> DataFrame."""
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".csv"))
    if not files:
        raise SystemExit("No CSV files in %s. Run 01_prep_wesad.py first."
                         % input_dir)

    frames = {}
    for fname in files:
        subject = fname[:-4]
        df = pd.read_csv(os.path.join(input_dir, fname))
        if "subject_id" not in df.columns:
            df["subject_id"] = subject
        frames[subject] = df
    return frames


def check_columns(frames):
    """Every subject must have the same columns, or we cannot stack them."""
    sets = {s: tuple(df.columns) for s, df in frames.items()}
    distinct = set(sets.values())
    if len(distinct) == 1:
        return list(list(distinct)[0])

    print("WARNING: subjects do not all have the same columns.")
    common = set.intersection(*[set(c) for c in distinct])
    print("  using only the %d columns every subject has" % len(common))
    for cols in distinct:
        who = [s for s, v in sets.items() if v == cols]
        print("    %d columns, subjects %s" % (len(cols), who))
    return sorted(common)


# =============================================================================
# LABEL MODES  (decision D6)
# =============================================================================
def label_strict(df):
    """Leave the repo's labelling exactly as it is.

    Only about 50 rows per subject carry a label: the 11 rows around each of
    the 5 PANAS answers, minus the 5 that fall off the end.
    """
    return df["Stress_Class"]


def label_block(df):
    """Give every row of a session the label from that session's own answer.

    The PANAS questionnaire was filled in once per session, about that whole
    session. This spreads the answer across the session it belongs to.

    The answer sits on the LAST row of its task block, so we read the label
    there and copy it across the block. We do not use the most common value
    in the block, because the 5-row spread can leak across a boundary into
    the next task.
    """
    out = pd.Series(index=df.index, dtype="object")

    for task, group in df.groupby("Task", sort=False):
        # last row of this task block is where the answer was written
        last_idx = group.index[-1]
        label = df.at[last_idx, "Stress_Class"]

        if pd.isna(label):
            # fall back to any label inside the block
            inside = group["Stress_Class"].dropna()
            label = inside.iloc[-1] if len(inside) else np.nan

        out.loc[group.index] = label

    return out


# =============================================================================
# WINDOW COUNTING
# =============================================================================
def count_windows(df, min_labelled_rows):
    """How many 15-row windows would be usable, counted per subject.

    Windows are counted inside each subject separately, so none crosses from
    one person to another.

    Returns (rows, totals) where rows is one dict per subject.
    """
    rows = []
    totals = {"windows": 0, "usable": 0, "full": 0, "empty": 0}

    for subject, group in df.groupby("subject_id", sort=True):
        labels = group["label_word"].values
        n = len(labels)

        starts = np.arange(0, n - WINDOW_ROWS + 1, STRIDE_ROWS)
        full = usable = empty = 0

        for s in starts:
            seg = pd.Series(labels[s:s + WINDOW_ROWS])
            k = int(seg.notna().sum())
            if k == WINDOW_ROWS:
                full += 1
            if k == 0:
                empty += 1
            if k >= min_labelled_rows:
                usable += 1

        rows.append({
            "subject": subject,
            "rows": n,
            "windows": len(starts),
            "fully_labelled": full,
            "usable": usable,
            "no_label": empty,
        })

        totals["windows"] += len(starts)
        totals["usable"] += usable
        totals["full"] += full
        totals["empty"] += empty

    return rows, totals


def build_windows_table(df, min_labelled_rows):
    """Work out one label per window, per subject.

    Returns a DataFrame with one row per usable window, saying which subject
    it came from, where it starts, and its label. This is not the training
    data itself, it is a record of what the training data will be, so the
    numbers in the report can be checked.
    """
    out = []

    for subject, group in df.groupby("subject_id", sort=True):
        group = group.reset_index(drop=True)
        labels = group["label_word"]
        n = len(group)

        for s in np.arange(0, n - WINDOW_ROWS + 1, STRIDE_ROWS):
            seg = labels.iloc[s:s + WINDOW_ROWS].dropna()
            if len(seg) < min_labelled_rows:
                continue
            winner = seg.mode()
            if not len(winner):
                continue
            out.append({
                "subject_id": subject,
                "window_start_row": int(s),
                "n_labelled_rows": int(len(seg)),
                "label_word": winner.iloc[0],
                "label": LABEL_MAP[winner.iloc[0]],
            })

    return pd.DataFrame(out)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Join the per-subject WESAD CSVs into the three files "
                    "Stress.ipynb expects. This step is missing from the repo."
    )
    parser.add_argument(
        "--label-mode", choices=["strict", "block"], required=True,
        help=("strict: keep the repo's 11-row label spread. "
              "block: give every row of a session its own session's label."))
    parser.add_argument(
        "--input-dir", default="data/processed/wesad",
        help="folder of per-subject CSVs from 01_prep_wesad.py")
    parser.add_argument(
        "--min-labelled-rows", type=int, default=8,
        help=("how many of a window's 15 rows must carry a label for the "
              "window to be used. 8 is a simple majority. Default 8."))
    args = parser.parse_args()

    mode = args.label_mode
    input_dir = args.input_dir
    min_rows = args.min_labelled_rows

    tag = mode
    if "notebook_order" in input_dir:
        tag = mode + "_notebook_order"

    out_dir = os.path.join("data/assembled", tag)
    report_path = "docs/12_assembled_%s.md" % tag
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 62)
    print("Label mode        : %s" % mode)
    print("Input folder      : %s" % input_dir)
    print("Output folder     : %s" % out_dir)
    print("Window            : %d rows, stride %d" % (WINDOW_ROWS, STRIDE_ROWS))
    print("Min labelled rows : %d of %d" % (min_rows, WINDOW_ROWS))
    print("=" * 62)
    print()

    # ---- load ----
    frames = load_subjects(input_dir)
    print("subjects loaded: %d  %s" % (len(frames), sorted(frames)))
    columns = check_columns(frames)
    print()

    df = pd.concat([frames[s] for s in sorted(frames)], ignore_index=True)
    df = df[[c for c in columns if c in df.columns]]
    print("stacked rows: %d, columns: %d" % (len(df), len(df.columns)))
    print()

    # ---- apply the chosen label mode, per subject ----
    print("Applying label mode '%s'..." % mode)
    pieces = []
    for subject, group in df.groupby("subject_id", sort=True):
        group = group.copy()
        if mode == "strict":
            group["label_word"] = label_strict(group)
        else:
            group["label_word"] = label_block(group)
        pieces.append(group)
    df = pd.concat(pieces, ignore_index=True)

    n_lab = int(df["label_word"].notna().sum())
    print("  labelled rows: %d of %d  (%.1f%%)" % (
        n_lab, len(df), 100.0 * n_lab / len(df)))
    print("  label counts : %s" % dict(df["label_word"].value_counts()))
    print()

    # ---- window counts ----
    print("Windows, counted inside each subject so none crosses between people:")
    print()
    wrows, wtot = count_windows(df, min_rows)
    print("  %-9s %7s %9s %16s %9s %10s" % (
        "subject", "rows", "windows", "fully labelled", "usable", "no label"))
    for r in wrows:
        print("  %-9s %7d %9d %16d %9d %10d" % (
            r["subject"], r["rows"], r["windows"],
            r["fully_labelled"], r["usable"], r["no_label"]))
    print("  " + "-" * 66)
    print("  %-9s %7s %9d %16d %9d %10d" % (
        "TOTAL", "", wtot["windows"], wtot["full"], wtot["usable"], wtot["empty"]))
    print()

    if wtot["usable"] == 0:
        print("  NO USABLE WINDOWS. Lower --min-labelled-rows or use")
        print("  --label-mode block.")
        print()

    # ---- the window table ----
    wtable = build_windows_table(df, min_rows)
    if len(wtable):
        print("Usable windows by class:")
        print(wtable["label_word"].value_counts().to_string())
        print()
        biggest = wtable["label_word"].value_counts().iloc[0]
        print("  guessing the most common class every time would score %.1f%%"
              % (100.0 * biggest / len(wtable)))
        print("  any model must beat this to be worth anything")
        print()

    # ---- drop the leaky columns ----
    present_leaks = [c for c in LEAK_COLUMNS if c in df.columns]
    print("Removing columns that would leak the answer into the signals:")
    for c in present_leaks:
        print("  %s" % c)
    if not present_leaks:
        print("  none found")
    print("  (Stress.ipynb keeps every numeric column whose name does not")
    print("   contain stress_class, task, sessionphase, activity or")
    print("   subject_id, so these would have been used as signals)")
    print()

    # audit copy keeps everything
    audit_path = os.path.join(out_dir, "with_panas_for_audit.csv")
    df.to_csv(audit_path, index=False)

    df_clean = df.drop(columns=present_leaks)

    # ---- numeric label with our fixed mapping ----
    df_clean["Stress_Class"] = df_clean["label_word"].map(LABEL_MAP)
    df_clean = df_clean.drop(columns=["label_word"])

    print("Label mapping used (see D4):")
    for word, num in LABEL_MAP.items():
        print("  %-13s -> %d" % (word, num))
    print()

    # ---- subject-wise split ----
    known = set(df_clean["subject_id"].unique())
    for name, subs in SPLIT.items():
        missing = [s for s in subs if s not in known]
        if missing:
            print("WARNING: %s subjects not found in the data: %s" % (name, missing))

    train_df = df_clean[df_clean["subject_id"].isin(SPLIT["train"])]
    val_df = df_clean[df_clean["subject_id"].isin(SPLIT["val"])]
    test_df = df_clean[df_clean["subject_id"].isin(SPLIT["test"])]

    unassigned = known - set(SPLIT["train"]) - set(SPLIT["val"]) - set(SPLIT["test"])
    if unassigned:
        print("WARNING: these subjects are in no split and will be unused: %s"
              % sorted(unassigned))
        print()

    print("Split by subject (decision D7). No person is in two sets.")
    print("  train : %2d subjects, %5d rows  %s" % (
        len(SPLIT["train"]), len(train_df), SPLIT["train"]))
    print("  val   : %2d subjects, %5d rows  %s" % (
        len(SPLIT["val"]), len(val_df), SPLIT["val"]))
    print("  test  : %2d subjects, %5d rows  %s" % (
        len(SPLIT["test"]), len(test_df), SPLIT["test"]))
    print()

    # ---- write the three files ----
    #
    # all_subjects_unlabelled.csv : train subjects only. The self-supervised
    #   step must not see the test subjects, not even unlabelled, or the
    #   comparison is contaminated in a way that is hard to explain later.
    #
    # generalized_labelled.csv : labelled rows from train and validation.
    #
    # finetune_labelled.csv : labelled rows from the held-out test subjects.

    unlab = train_df.drop(columns=["Stress_Class"])
    gen = pd.concat([train_df, val_df]).dropna(subset=["Stress_Class"]).copy()
    fine = test_df.dropna(subset=["Stress_Class"]).copy()

    gen["Stress_Class"] = gen["Stress_Class"].astype(int)
    fine["Stress_Class"] = fine["Stress_Class"].astype(int)

    paths = {
        "all_subjects_unlabelled.csv": unlab,
        "generalized_labelled.csv": gen,
        "finetune_labelled.csv": fine,
    }

    print("Files written:")
    for name, frame in paths.items():
        p = os.path.join(out_dir, name)
        frame.to_csv(p, index=False)
        print("  %-32s %6d rows, %3d cols" % (name, len(frame), len(frame.columns)))
    print("  %-32s %6d rows  (AUDIT ONLY, keeps PANAS)" % (
        "with_panas_for_audit.csv", len(df)))
    if len(wtable):
        wp = os.path.join(out_dir, "window_index.csv")
        wtable.to_csv(wp, index=False)
        print("  %-32s %6d rows  (one row per usable window)" % (
            "window_index.csv", len(wtable)))
    print()

    print("Class counts in generalized_labelled.csv (rows, not windows):")
    print(gen["Stress_Class"].value_counts().sort_index().to_string())
    print()
    print("Class counts in finetune_labelled.csv:")
    print(fine["Stress_Class"].value_counts().sort_index().to_string())
    print()

    # ---- record the split ----
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({
            "label_mode": mode,
            "input_dir": input_dir,
            "min_labelled_rows": min_rows,
            "window_rows": WINDOW_ROWS,
            "stride_rows": STRIDE_ROWS,
            "label_map": LABEL_MAP,
            "split": SPLIT,
            "leak_columns_removed": present_leaks,
        }, f, indent=2)

    # ---- report ----
    lines = []
    lines.append("# Assembled WESAD dataset, label mode `%s`" % mode)
    lines.append("")
    lines.append("Written by `btp/03_assemble_wesad.py`. This step does not")
    lines.append("exist in the repo. `Preproc.ipynb` saves one CSV per subject,")
    lines.append("`Stress.ipynb` reads three combined files, and nothing joins")
    lines.append("the two. Every choice below is ours and is recorded here.")
    lines.append("")
    lines.append("| setting | value |")
    lines.append("|---|---|")
    lines.append("| input folder | `%s` |" % input_dir)
    lines.append("| label mode | `%s` |" % mode)
    lines.append("| window | %d rows, stride %d |" % (WINDOW_ROWS, STRIDE_ROWS))
    lines.append("| min labelled rows per window | %d of %d |" % (min_rows, WINDOW_ROWS))
    lines.append("| label mapping | LowStress=0, MediumStress=1, HighStress=2 |")
    lines.append("")
    lines.append("## Label mode")
    lines.append("")
    if mode == "strict":
        lines.append("`strict` keeps the repo's labelling untouched: 11 rows around")
        lines.append("each of the 5 PANAS answers, about 50 rows per subject. A")
        lines.append("window is 15 rows, so no window is fully labelled. Windows are")
        lines.append("kept when at least %d of their 15 rows carry a label, and take" % min_rows)
        lines.append("the most common label among those rows.")
    else:
        lines.append("`block` gives every row of a session the label from that")
        lines.append("session's own PANAS answer, because the questionnaire was")
        lines.append("filled in once per session about that whole session. This")
        lines.append("gives many more usable windows.")
        lines.append("")
        lines.append("**Cost:** `Task` now predicts the label perfectly, by")
        lines.append("construction. Any accuracy from a model that can see `Task`")
        lines.append("is meaningless in this mode. The audit measures this.")
    lines.append("")
    lines.append("## Labelled rows")
    lines.append("")
    lines.append("%d of %d rows carry a label (%.1f%%)." % (
        n_lab, len(df), 100.0 * n_lab / len(df)))
    lines.append("")
    lines.append("## Windows per subject")
    lines.append("")
    lines.append("Counted inside each subject, so none crosses between people.")
    lines.append("")
    lines.append("| subject | rows | windows | fully labelled | usable | no label |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in wrows:
        lines.append("| %s | %d | %d | %d | %d | %d |" % (
            r["subject"], r["rows"], r["windows"],
            r["fully_labelled"], r["usable"], r["no_label"]))
    lines.append("| **TOTAL** | | **%d** | **%d** | **%d** | **%d** |" % (
        wtot["windows"], wtot["full"], wtot["usable"], wtot["empty"]))
    lines.append("")
    if len(wtable):
        lines.append("## Usable windows by class")
        lines.append("")
        lines.append("| class | windows |")
        lines.append("|---|---:|")
        for k, v in wtable["label_word"].value_counts().items():
            lines.append("| %s | %d |" % (k, v))
        biggest = wtable["label_word"].value_counts().iloc[0]
        lines.append("")
        lines.append("Always guessing the most common class scores **%.1f%%**."
                     % (100.0 * biggest / len(wtable)))
        lines.append("Any model has to beat that to mean anything.")
        lines.append("")
    lines.append("## Columns removed to stop the answer leaking")
    lines.append("")
    lines.append("`Stress.ipynb` cell 2 keeps every numeric column whose name")
    lines.append("does not contain stress_class, task, sessionphase, activity or")
    lines.append("subject_id. These would have gone in as signal channels:")
    lines.append("")
    for c in present_leaks:
        lines.append("- `%s`" % c)
    if not present_leaks:
        lines.append("- none present")
    lines.append("")
    lines.append("`stress_raw` is the label before it was cut into three groups,")
    lines.append("so feeding it in would hand the model the answer.")
    lines.append("")
    lines.append("They are kept in `with_panas_for_audit.csv`, which is never")
    lines.append("used for training.")
    lines.append("")
    lines.append("## Note on Time_Sin and Time_Cos")
    lines.append("")
    lines.append("These do not match the drop list either, so in the original they")
    lines.append("are signal channels, not context. Time of day never reaches the")
    lines.append("context gate. Kept as the original has it, recorded here.")
    lines.append("")
    lines.append("## Split by subject (D7)")
    lines.append("")
    lines.append("| set | subjects | rows |")
    lines.append("|---|---|---:|")
    lines.append("| train | %s | %d |" % (", ".join(SPLIT["train"]), len(train_df)))
    lines.append("| val | %s | %d |" % (", ".join(SPLIT["val"]), len(val_df)))
    lines.append("| test | %s | %d |" % (", ".join(SPLIT["test"]), len(test_df)))
    lines.append("")
    lines.append("Chosen once and never changed, so all experiments compare fairly.")
    lines.append("")
    lines.append("The self-supervised file holds train subjects only. The test")
    lines.append("subjects are not seen at that stage either, not even without")
    lines.append("labels.")
    lines.append("")
    lines.append("S9 is in train on purpose: it has no MediumStress rows at all,")
    lines.append("so putting it in test would make the test set unscoreable.")
    lines.append("")
    lines.append("## Files written")
    lines.append("")
    lines.append("| file | rows | what it is for |")
    lines.append("|---|---:|---|")
    lines.append("| `all_subjects_unlabelled.csv` | %d | self-supervised pretraining, train subjects only |" % len(unlab))
    lines.append("| `generalized_labelled.csv` | %d | classifier training, train + validation |" % len(gen))
    lines.append("| `finetune_labelled.csv` | %d | held-out test subjects |" % len(fine))
    lines.append("| `with_panas_for_audit.csv` | %d | audit only, never trained on |" % len(df))
    if len(wtable):
        lines.append("| `window_index.csv` | %d | one row per usable window |" % len(wtable))
    lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Report written to %s" % report_path)
    print("Split recorded in %s" % os.path.join(out_dir, "split.json"))


if __name__ == "__main__":
    main()
