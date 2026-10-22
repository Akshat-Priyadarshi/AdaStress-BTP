"""
04_audit_wesad.py

WHY THIS FILE EXISTS
    The repo has no data checks at all. Before training anything we need to
    answer three questions, because each one can make a good-looking accuracy
    number meaningless.

    Question 1: does the context give away the answer?
        AdaStress feeds Task, SessionPhase and Activity into a context gate.
        In WESAD the stress label comes from a PANAS questionnaire answered
        once per session, and Task IS the session. So a model that can see
        Task might score well by recognising which session it is looking at,
        not by reading the body. We measure exactly how much of the label
        Task alone can predict.

    Question 2: are the _skew columns usable?
        Preprocessing printed dozens of warnings about precision loss. That
        is scipy computing skew on 5 second bins where the signal barely
        moves. 9 of the 54 feature columns are _skew columns. We check what
        actually ended up in them.

    Question 3: is there any signal at all?
        Before building a neural network we fit a plain random forest on
        simple window statistics, with a subject-held-out split. If that
        cannot beat guessing the most common class, a neural network will not
        either, and tuning one would be tuning noise. This is a floor check,
        NOT an AdaStress result, and it is labelled that way everywhere.

WHAT IT REPORTS
    C1  class balance: overall, per subject, and which subjects miss a class
    C2  how much each context column gives away, in rows and in windows
    C3  how much subject identity alone gives away
    C4  health of the _skew columns
    C5  the random forest floor check

    It reads only. It changes no data.

OUTPUT
    docs/13_audit_<tag>.md

HOW TO RUN
    From the AdaStress_BTP folder, once per assembled folder:

        python3 btp/04_audit_wesad.py --assembled-dir data/assembled/strict
        python3 btp/04_audit_wesad.py --assembled-dir data/assembled/block
        python3 btp/04_audit_wesad.py --assembled-dir data/assembled/strict_notebook_order
        python3 btp/04_audit_wesad.py --assembled-dir data/assembled/block_notebook_order

    The block ones take about a minute because of C5. The strict ones are
    instant, and C5 will say there is too little data, which is the answer.

HOW TO READ C2
    Two numbers per context column.

    "lookup accuracy" - learn the most common label for each value of that
    column from the training subjects, then apply it to the test subjects.
    This is the score a model gets for free just by reading that column.

    "Cramer's V" - how strongly the column and the label move together, from
    0 (unrelated) to 1 (one decides the other).

    If Task's lookup accuracy is close to the accuracy a finished model
    reports, the model has learned to read Task, not to read the body.
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

# The precision-loss warnings are what C4 measures, so we do not want them
# filling the screen again while we measure them.
warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_ROWS = 15
STRIDE_ROWS = 15

LABEL_MAP = {"LowStress": 0, "MediumStress": 1, "HighStress": 2}
LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]
CONTEXT_COLUMNS = ["Task", "SessionPhase", "Activity", "Posture"]

# Columns that are not physiological features.
NON_FEATURE = {"Activity", "Posture", "Task", "SessionPhase", "Stress_Class",
               "PANAS_PA", "PANAS_NA", "stress_raw", "subject_id",
               "label_word", "Time_Sin", "Time_Cos"}


def cramers_v(a, b):
    """How strongly two label-like columns move together. 0 to 1."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    smaller = min(table.shape) - 1
    if n == 0 or smaller == 0:
        return float("nan")
    return float(np.sqrt((chi2 / n) / smaller))


def lookup_accuracy(train_df, test_df, column, label_col="label_word"):
    """Score you get for free by reading one column.

    Learn the most common label for each value of the column from the
    training subjects, apply it to the test subjects. Values not seen in
    training fall back to the most common label overall.
    """
    tr = train_df.dropna(subset=[label_col])
    te = test_df.dropna(subset=[label_col])
    if len(tr) == 0 or len(te) == 0:
        return float("nan")

    table = {}
    for value, group in tr.groupby(column, observed=True):
        winner = group[label_col].mode()
        if len(winner):
            table[value] = winner.iloc[0]

    fallback = tr[label_col].mode()
    fallback = fallback.iloc[0] if len(fallback) else None

    pred = te[column].map(table).fillna(fallback)
    return float(accuracy_score(te[label_col], pred))


def make_windows(df, min_labelled_rows):
    """Turn rows into windows, inside each subject, as assemble does.

    Returns per-window information plus mean and standard deviation of every
    feature column, which is what the floor check in C5 uses.
    """
    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE
                    and pd.api.types.is_numeric_dtype(df[c])]

    rows, X = [], []

    for subject, group in df.groupby("subject_id", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        feats = group[feature_cols].to_numpy(dtype=float)

        for s in np.arange(0, n - WINDOW_ROWS + 1, STRIDE_ROWS):
            seg = group["label_word"].iloc[s:s + WINDOW_ROWS].dropna()
            if len(seg) < min_labelled_rows:
                continue
            winner = seg.mode()
            if not len(winner):
                continue

            block = feats[s:s + WINDOW_ROWS]
            X.append(np.concatenate([block.mean(axis=0), block.std(axis=0)]))

            entry = {"subject_id": subject, "label_word": winner.iloc[0]}
            for col in CONTEXT_COLUMNS:
                if col in group.columns:
                    m = group[col].iloc[s:s + WINDOW_ROWS].mode()
                    entry[col] = m.iloc[0] if len(m) else None
            rows.append(entry)

    meta = pd.DataFrame(rows)
    Xarr = np.array(X) if X else np.zeros((0, 2 * len(feature_cols)))
    return meta, Xarr, feature_cols


def main():
    parser = argparse.ArgumentParser(
        description="Check the assembled WESAD data before any training.")
    parser.add_argument("--assembled-dir", required=True,
                        help="e.g. data/assembled/block")
    parser.add_argument("--min-labelled-rows", type=int, default=8)
    args = parser.parse_args()

    adir = args.assembled_dir
    min_rows = args.min_labelled_rows
    tag = os.path.basename(adir.rstrip("/"))
    report_path = "docs/13_audit_%s.md" % tag

    audit_csv = os.path.join(adir, "with_panas_for_audit.csv")
    if not os.path.exists(audit_csv):
        raise SystemExit("Not found: %s\nRun 03_assemble_wesad.py first."
                         % audit_csv)

    with open(os.path.join(adir, "split.json")) as f:
        split = json.load(f)

    df = pd.read_csv(audit_csv)
    train_subs = split["split"]["train"]
    val_subs = split["split"]["val"]
    test_subs = split["split"]["test"]

    L = []

    print("=" * 62)
    print("AUDIT: %s" % tag)
    print("=" * 62)
    print("rows: %d, columns: %d" % (len(df), len(df.columns)))
    print("label mode: %s" % split.get("label_mode"))
    print("preprocessing input: %s" % split.get("input_dir"))
    print()

    L.append("# Audit of assembled WESAD data: `%s`" % tag)
    L.append("")
    L.append("Written by `btp/04_audit_wesad.py`. Read-only. The repo has no")
    L.append("data checks, so all of this is ours.")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| assembled folder | `%s` |" % adir)
    L.append("| label mode | `%s` |" % split.get("label_mode"))
    L.append("| preprocessing input | `%s` |" % split.get("input_dir"))
    L.append("| rows | %d |" % len(df))
    L.append("| window | %d rows, stride %d |" % (WINDOW_ROWS, STRIDE_ROWS))
    L.append("")

    # ---------------- C1 ----------------
    print("-" * 62)
    print("C1  CLASS BALANCE")
    print("-" * 62)
    L.append("## C1 Class balance")
    L.append("")

    counts = df["label_word"].value_counts()
    n_lab = int(df["label_word"].notna().sum())
    print("labelled rows: %d of %d (%.1f%%)"
          % (n_lab, len(df), 100.0 * n_lab / len(df)))
    print(counts.to_string())
    print()

    L.append("Labelled rows: %d of %d (%.1f%%)."
             % (n_lab, len(df), 100.0 * n_lab / len(df)))
    L.append("")
    L.append("| class | rows |")
    L.append("|---|---:|")
    for k in LABEL_NAMES:
        L.append("| %s | %d |" % (k, int(counts.get(k, 0))))
    L.append("")

    lab_only = df.dropna(subset=["label_word"])
    per_sub = lab_only.pivot_table(index="subject_id", columns="label_word",
                                   values="Task", aggfunc="count",
                                   fill_value=0)
    for k in LABEL_NAMES:
        if k not in per_sub.columns:
            per_sub[k] = 0
    per_sub = per_sub[LABEL_NAMES]

    print("per subject:")
    print(per_sub.to_string())
    print()

    L.append("### Per subject")
    L.append("")
    L.append("| subject | Low | Medium | High | split |")
    L.append("|---|---:|---:|---:|---|")
    for sid, row in per_sub.iterrows():
        where = ("train" if sid in train_subs else
                 "val" if sid in val_subs else
                 "test" if sid in test_subs else "unused")
        L.append("| %s | %d | %d | %d | %s |"
                 % (sid, row["LowStress"], row["MediumStress"],
                    row["HighStress"], where))
    L.append("")

    missing = per_sub[(per_sub == 0).any(axis=1)]
    if len(missing):
        print("SUBJECTS MISSING A WHOLE CLASS:")
        for sid, row in missing.iterrows():
            gone = [c for c in LABEL_NAMES if row[c] == 0]
            print("  %-5s has no %s" % (sid, ", ".join(gone)))
        print()
        print("  Cause: pd.qcut cuts each subject's 5 PANAS scores into three")
        print("  groups. When two scores tie, one group disappears.")
        print()
        L.append("**Subjects missing a whole class:**")
        L.append("")
        for sid, row in missing.iterrows():
            gone = [c for c in LABEL_NAMES if row[c] == 0]
            L.append("- %s has no %s" % (sid, ", ".join(gone)))
        L.append("")
        L.append("`pd.qcut` cuts each subject's 5 PANAS scores into three groups.")
        L.append("When two scores tie, one group disappears. The cut is also done")
        L.append("per subject, so HighStress for one subject is not the same")
        L.append("feeling level as HighStress for another.")
        L.append("")

    # ---------------- windows ----------------
    meta, X, feature_cols = make_windows(df, min_rows)
    print("windows built: %d, numbers per window: %d" % (len(meta), X.shape[1]))
    print()

    L.append("## Windows")
    L.append("")
    L.append("%d usable windows from %d feature columns (mean and standard"
             % (len(meta), len(feature_cols)))
    L.append("deviation each, so %d numbers per window)." % X.shape[1])
    L.append("")

    if len(meta) == 0:
        L.append("**No usable windows. Nothing else can be measured.**")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        open(report_path, "w").write("\n".join(L) + "\n")
        print("No usable windows. Report written to %s" % report_path)
        return

    win_train = meta["subject_id"].isin(train_subs)
    win_val = meta["subject_id"].isin(val_subs)
    win_test = meta["subject_id"].isin(test_subs)

    print("windows: train %d, val %d, test %d"
          % (int(win_train.sum()), int(win_val.sum()), int(win_test.sum())))
    L.append("Windows by split: train %d, validation %d, test %d."
             % (int(win_train.sum()), int(win_val.sum()), int(win_test.sum())))
    L.append("")

    tr_mode = meta.loc[win_train, "label_word"].mode()
    tr_mode = tr_mode.iloc[0] if len(tr_mode) else LABEL_NAMES[0]
    base_test = (float((meta.loc[win_test, "label_word"] == tr_mode).mean())
                 if win_test.sum() else float("nan"))

    print()
    print("Guessing '%s' every time scores %.1f%% on the test subjects."
          % (tr_mode, 100 * base_test))
    print("EVERY accuracy we report must be shown next to this number.")
    print()

    L.append("### The number every result must beat")
    L.append("")
    L.append("Always guessing `%s`, the most common class in training, scores"
             % tr_mode)
    L.append("**%.1f%%** on the test subjects. Any accuracy we report has to be"
             % (100 * base_test))
    L.append("shown next to this, or it means nothing.")
    L.append("")

    # ---------------- C2 ----------------
    print("-" * 62)
    print("C2  DOES THE CONTEXT GIVE AWAY THE ANSWER?")
    print("-" * 62)
    L.append("## C2 Does the context give away the answer?")
    L.append("")
    L.append("For each context column: the score a model gets for free just by")
    L.append("reading it, and how strongly it moves with the label (0 to 1).")
    L.append("")

    row_train = df[df["subject_id"].isin(train_subs)]
    row_test = df[df["subject_id"].isin(test_subs)]

    print("%-14s %10s %9s %10s %9s"
          % ("column", "rows_acc", "rows_V", "win_acc", "win_V"))
    print("-" * 56)
    L.append("| context column | lookup accuracy, rows | Cramer's V, rows "
             "| lookup accuracy, windows | Cramer's V, windows |")
    L.append("|---|---:|---:|---:|---:|")

    c2 = {}
    for col in CONTEXT_COLUMNS:
        if col not in df.columns:
            continue
        r_acc = lookup_accuracy(row_train, row_test, col)
        r_v = cramers_v(lab_only[col], lab_only["label_word"])
        w_acc = lookup_accuracy(meta[win_train], meta[win_test], col)
        w_v = cramers_v(meta[col], meta["label_word"])
        c2[col] = (r_acc, r_v, w_acc, w_v)
        print("%-14s %10.3f %9.3f %10.3f %9.3f" % (col, r_acc, r_v, w_acc, w_v))
        L.append("| `%s` | %.3f | %.3f | %.3f | %.3f |"
                 % (col, r_acc, r_v, w_acc, w_v))
    print()
    L.append("")

    task_acc = c2.get("Task", (np.nan,) * 4)[2]
    if not np.isnan(task_acc):
        print("Reading Task alone scores %.1f%% on the test subjects."
              % (100 * task_acc))
        print("Guessing the most common class scores %.1f%%." % (100 * base_test))
        print()
        L.append("**Reading `Task` alone scores %.1f%%** on the test subjects,"
                 % (100 * task_acc))
        L.append("against %.1f%% for guessing the most common class."
                 % (100 * base_test))
        L.append("")
        if task_acc >= 0.95:
            print("  Task decides the label almost completely. In this mode any")
            print("  accuracy from a model that can see Task is meaningless: it")
            print("  only has to recognise which session it is looking at. The")
            print("  honest number here is the no-context model.")
            L.append("`Task` decides the label almost completely. In this mode any")
            L.append("accuracy from a model that can see `Task` is meaningless: the")
            L.append("model only has to work out which session it is looking at,")
            L.append("not read the body. The honest number here is the no-context")
            L.append("model.")
        elif task_acc > base_test + 0.10:
            print("  Task gives a real head start over guessing. Report the")
            print("  no-context model beside every context-aware one, so the")
            print("  reader can see how much came from physiology.")
            L.append("`Task` gives a real head start over guessing. Every")
            L.append("context-aware result must be reported beside the no-context")
            L.append("one, so the reader can see how much came from physiology.")
        else:
            print("  Task does not give much away here. Context-aware accuracy")
            print("  can be read closer to face value.")
            L.append("`Task` does not give much away in this mode, so")
            L.append("context-aware accuracy can be read closer to face value.")
        L.append("")

    ct = pd.crosstab(meta["Task"], meta["label_word"])
    for k in LABEL_NAMES:
        if k not in ct.columns:
            ct[k] = 0
    ct = ct[LABEL_NAMES]
    print("Task by label, windows:")
    print(ct.to_string())
    print()

    L.append("### Task by label, counted in windows")
    L.append("")
    L.append("| Task | Low | Medium | High |")
    L.append("|---|---:|---:|---:|")
    for t, row in ct.iterrows():
        L.append("| %s | %d | %d | %d |"
                 % (t, row["LowStress"], row["MediumStress"], row["HighStress"]))
    L.append("")

    # ---------------- C3 ----------------
    print("-" * 62)
    print("C3  DOES SUBJECT IDENTITY GIVE AWAY THE ANSWER?")
    print("-" * 62)
    v_sub = cramers_v(meta["subject_id"], meta["label_word"])
    print("Cramer's V between subject and label, windows: %.3f" % v_sub)
    print()
    print("The label is cut into three groups separately for each subject, so")
    print("if this is high a model could score well just by recognising the")
    print("person. This is why we split by subject. A random row split would")
    print("put the same person on both sides and flatter the model.")
    print()

    L.append("## C3 Does subject identity give away the answer?")
    L.append("")
    L.append("Cramer's V between subject and label, windows: **%.3f**" % v_sub)
    L.append("")
    L.append("The label is cut into three groups separately for each subject, so")
    L.append("if this is high a model could score well just by recognising the")
    L.append("person. This is why we split by subject. A random row split would")
    L.append("put the same person on both sides and flatter the model.")
    L.append("")

    # ---------------- C4 ----------------
    print("-" * 62)
    print("C4  ARE THE _skew COLUMNS USABLE?")
    print("-" * 62)
    L.append("## C4 Are the `_skew` columns usable?")
    L.append("")
    L.append("Preprocessing printed many warnings about precision loss. That is")
    L.append("`scipy` computing skew on 5 second bins where the signal barely")
    L.append("moves. Here is what actually ended up in those columns.")
    L.append("")

    skew_cols = [c for c in feature_cols if c.endswith("_skew")]
    print("%d _skew columns out of %d feature columns"
          % (len(skew_cols), len(feature_cols)))
    print()
    print("%-26s %7s %6s %11s %11s %8s"
          % ("column", "NaN", "inf", "min", "max", "|x|>10"))
    print("-" * 74)

    L.append("%d `_skew` columns out of %d feature columns."
             % (len(skew_cols), len(feature_cols)))
    L.append("")
    L.append("| column | NaN | inf | min | max | size over 10 |")
    L.append("|---|---:|---:|---:|---:|---:|")

    skew_bad = 0
    for col in skew_cols:
        v = pd.to_numeric(df[col], errors="coerce")
        arr = v.to_numpy(dtype=float)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        vmin = float(finite.min()) if finite.size else float("nan")
        vmax = float(finite.max()) if finite.size else float("nan")
        n_big = int((np.abs(finite) > 10).sum())
        if n_nan or n_inf or n_big:
            skew_bad += 1
        print("%-26s %7d %6d %11.2f %11.2f %8d"
              % (col, n_nan, n_inf, vmin, vmax, n_big))
        L.append("| `%s` | %d | %d | %.2f | %.2f | %d |"
                 % (col, n_nan, n_inf, vmin, vmax, n_big))
    print()
    L.append("")

    if skew_bad == 0:
        print("All _skew columns look sane. The warnings were noisy but did no")
        print("damage. Keep all %d features." % len(feature_cols))
        L.append("All `_skew` columns look sane. The warnings were noisy but did")
        L.append("no damage. **Keep all %d features.**" % len(feature_cols))
    else:
        print("%d of %d _skew columns hold missing, infinite or very large"
              % (skew_bad, len(skew_cols)))
        print("values, from bins where the signal was nearly flat.")
        print("Recommendation: run one experiment with them and one without,")
        print("and record which was used.")
        L.append("%d of %d `_skew` columns hold missing, infinite or very large"
                 % (skew_bad, len(skew_cols)))
        L.append("values, from bins where the signal was nearly flat.")
        L.append("")
        L.append("**Recommendation:** run one experiment with these columns and")
        L.append("one without, and record which was used in each result.")
    L.append("")

    # ---------------- C5 ----------------
    print("-" * 62)
    print("C5  IS THERE ANY SIGNAL AT ALL?")
    print("    (floor check, NOT an AdaStress result)")
    print("-" * 62)
    L.append("## C5 Is there any signal at all?")
    L.append("")
    L.append("**A floor check, not an AdaStress result.** A plain random forest")
    L.append("on simple window statistics: no self-supervision, no context gate.")
    L.append("Trained on the train subjects, scored on the test subjects. If")
    L.append("this cannot beat guessing, a neural network will not either, and")
    L.append("tuning one would be tuning noise.")
    L.append("")

    n_tr, n_te = int(win_train.sum()), int(win_test.sum())

    if n_tr < 30 or n_te < 10:
        msg = ("Too little data to check: %d training windows, %d test windows."
               % (n_tr, n_te))
        print(msg)
        print()
        print("That is itself the finding. In this label mode the repo's own")
        print("labelling does not produce enough labelled windows to train a")
        print("three-class model, or to measure one.")
        print()
        L.append(msg)
        L.append("")
        L.append("**That is itself the finding.** In this label mode the repo's")
        L.append("labelling does not produce enough labelled windows to train a")
        L.append("three-class model, or to measure one.")
        L.append("")
    else:
        y = meta["label_word"].map(LABEL_MAP).to_numpy()
        m_tr, m_te = win_train.to_numpy(), win_test.to_numpy()
        Xtr = np.nan_to_num(X[m_tr], nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(X[m_te], nan=0.0, posinf=0.0, neginf=0.0)
        ytr, yte = y[m_tr], y[m_te]

        clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                     n_jobs=-1, class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        acc = accuracy_score(yte, pred)
        bal = balanced_accuracy_score(yte, pred)
        mf1 = f1_score(yte, pred, average="macro")

        print("trained on %d windows, tested on %d" % (n_tr, n_te))
        print("  accuracy           : %.3f" % acc)
        print("  balanced accuracy  : %.3f" % bal)
        print("  macro F1           : %.3f" % mf1)
        print("  guessing baseline  : %.3f" % base_test)
        print("  reading Task alone : %.3f" % task_acc)
        print()

        cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
        print("confusion matrix, rows are truth:")
        print("%-18s %9s %9s %9s" % ("", "pred Low", "pred Med", "pred High"))
        for i, name in enumerate(LABEL_NAMES):
            print("%-18s %9d %9d %9d"
                  % ("true " + name, cm[i][0], cm[i][1], cm[i][2]))
        print()

        L.append("Trained on %d windows, tested on %d." % (n_tr, n_te))
        L.append("")
        L.append("| measure | value |")
        L.append("|---|---:|")
        L.append("| accuracy | %.3f |" % acc)
        L.append("| balanced accuracy | %.3f |" % bal)
        L.append("| macro F1 | %.3f |" % mf1)
        L.append("| guessing the most common class | %.3f |" % base_test)
        L.append("| reading `Task` alone | %.3f |" % task_acc)
        L.append("")
        L.append("Confusion matrix, rows are truth:")
        L.append("")
        L.append("| | predicted Low | predicted Medium | predicted High |")
        L.append("|---|---:|---:|---:|")
        for i, name in enumerate(LABEL_NAMES):
            L.append("| true %s | %d | %d | %d |"
                     % (name, cm[i][0], cm[i][1], cm[i][2]))
        L.append("")

        if acc <= base_test + 0.02:
            print("VERDICT: no usable signal. The physiology in these windows")
            print("does not beat guessing. If a model later reports high")
            print("accuracy on this data, that has to be explained.")
            L.append("**Verdict: no usable signal.** The physiology in these")
            L.append("windows does not beat guessing. If a model later reports high")
            L.append("accuracy on this data, that has to be explained, not")
            L.append("celebrated.")
        elif not np.isnan(task_acc) and acc < task_acc:
            print("VERDICT: some signal, but reading Task alone does better.")
            print("Context, not physiology, carries most of the accuracy here.")
            L.append("**Verdict: some signal, but reading `Task` alone does")
            L.append("better.** Context, not physiology, carries most of the")
            L.append("accuracy here.")
        else:
            print("VERDICT: real signal in the physiology, above both guessing")
            print("and reading Task. Worth building the model.")
            L.append("**Verdict: real signal in the physiology**, above both")
            L.append("guessing and reading `Task`. Worth building the model.")
        L.append("")

        imp = pd.Series(clf.feature_importances_,
                        index=[c + "__mean" for c in feature_cols] +
                              [c + "__sd" for c in feature_cols])
        top = imp.sort_values(ascending=False).head(12)
        print("top 12 features by importance:")
        print(top.to_string())
        print()
        L.append("Top 12 features by importance:")
        L.append("")
        L.append("| feature | importance |")
        L.append("|---|---:|")
        for k, v in top.items():
            L.append("| `%s` | %.4f |" % (k, v))
        L.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    open(report_path, "w").write("\n".join(L) + "\n")

    print("=" * 62)
    print("Report written to %s" % report_path)


if __name__ == "__main__":
    main()
