"""
32_user_leak_check.py

WHY THIS EXISTS
    Before any context column is fed to the model, we ask one question of it:

        how well does this column predict the label ON ITS OWN,
        with no model at all?

    On WESAD this test found that `Task` predicted the stress label with an
    accuracy of exactly 1.000 on held-out people. Not 0.9 - exactly 1.000, in
    every rotation. The reason was structural: the label came from one
    questionnaire per session, and `Task` WAS the session, so they were the
    same thing under two names. Every accuracy from a model that could see
    `Task` was really measuring its ability to read a column.

    The same thing can happen here, and the most likely candidate is HOUR.
    If people tended to answer the survey at similar times of day, then the
    clock quietly carries the answer.

HOW THE TEST WORKS
    Learn the most common label for each value of one column, using some
    users. Then apply that lookup to a different user. Whatever score it
    gets, a model gets for free just by reading that column.

    Continuous columns are put into 10 bins first, so the same lookup idea
    works for pulse rate as for posture.

    It is always tested on a HELD-OUT user, never the same people it learned
    from, because that is how the model will be used.

HOW TO READ THE RESULT
    near 1.0            the column is the label under another name. Any
                        accuracy from a model that sees it is meaningless.

    well above guessing real information, and worth having. Report the
                        no-context model beside the context one so the reader
                        can see where the accuracy came from.

    near guessing       carries nothing. Including it adds width without
                        information, and a little noise.

A NOTE ON WHAT COUNTS AS A LEAK
    A physiological column predicting stress is not a leak. If pulse rate
    predicts the label, that is the finding, not a problem: the body really
    does respond to stress.

    A leak is when a column predicts the label for a reason that has nothing
    to do with the person's state. `Task` on WESAD was a leak because it
    encoded which part of the experiment was running. HOUR here would be a
    leak if it only encoded when the survey happened to be filled in.

    This script reports the numbers. Which ones are leaks is a judgement, and
    the report says which is which and why.

OUTPUT
    docs/57_user_leak_check.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/32_user_leak_check.py
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

LABELLED = "data/processed/labels/labelled_windows.csv"
REPORT = "docs/57_user_leak_check.md"

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

# Columns that are not context: identifiers, times, or the label itself.
NOT_CONTEXT = {"Subject_ID", "datetime_utc", "datetime_ist", "date",
               "response_time", "STAI6", "STAI6_RAW", "LABEL", "LABEL_NAME"}

N_BINS = 10          # continuous columns are binned before the lookup


def to_categories(series, n_bins=N_BINS):
    """Turn any column into something a lookup table can use.

    Words stay as they are. Numbers are cut into equal-sized groups, so the
    same lookup works for pulse rate as for posture.
    """
    if series.dtype == object or str(series.dtype).startswith("str"):
        return series.astype(str)
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop").astype(str)
    except Exception:
        return series.astype(str)


def lookup_accuracy(train_vals, train_y, test_vals, test_y):
    """The score a model gets for free by reading one column.

    Learn the most common label for each value from the training users, then
    apply it to the held-out user. A value never seen falls back to the most
    common label overall.
    """
    table = {}
    for v in pd.unique(train_vals):
        sel = train_vals == v
        lab, cnt = np.unique(train_y[sel], return_counts=True)
        table[v] = lab[cnt.argmax()]

    lab, cnt = np.unique(train_y, return_counts=True)
    fallback = lab[cnt.argmax()]

    pred = np.array([table.get(v, fallback) for v in test_vals])
    return float((pred == test_y).mean())


def cramers_v(a, b):
    """How strongly two label-like columns move together, 0 to 1."""
    t = pd.crosstab(pd.Series(a), pd.Series(b))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan")
    chi2 = chi2_contingency(t)[0]
    n = t.to_numpy().sum()
    smaller = min(t.shape) - 1
    if n == 0 or smaller == 0:
        return float("nan")
    return float(np.sqrt((chi2 / n) / smaller))


def main():
    ap = argparse.ArgumentParser(
        description="Measure how much each context column gives the label "
                    "away, before any of it reaches the model.")
    ap.add_argument("--labelled", default=LABELLED)
    ap.add_argument("--bins", type=int, default=N_BINS)
    args = ap.parse_args()

    if not os.path.exists(args.labelled):
        raise SystemExit("Not found: %s\nRun btp/31_build_labels.py first."
                         % args.labelled)

    df = pd.read_csv(args.labelled)
    users = sorted(df["Subject_ID"].unique())
    cols = [c for c in df.columns if c not in NOT_CONTEXT]

    print("=" * 74)
    print("LEAK CHECK ON THE USER CONTEXT")
    print("=" * 74)
    print("windows : %d" % len(df))
    print("users   : %s" % ", ".join(users))
    print("columns : %d" % len(cols))
    print()
    print("For each column: learn the most common label for each of its")
    print("values from four users, then apply that to the fifth. Whatever it")
    print("scores, a model gets for free just by reading that column.")
    print()

    # one column of categories per context column
    cat = pd.DataFrame({c: to_categories(df[c], args.bins) for c in cols})
    cat["Subject_ID"] = df["Subject_ID"].values
    y = df["LABEL"].to_numpy()

    # leave one user out, so the test is always on someone unseen
    per_col = {c: [] for c in cols}
    baselines = []

    for held in users:
        te = (cat["Subject_ID"] == held).to_numpy()
        tr = ~te
        if te.sum() < 20 or tr.sum() < 50:
            continue

        lab, cnt = np.unique(y[tr], return_counts=True)
        baselines.append(float((y[te] == lab[cnt.argmax()]).mean()))

        for c in cols:
            v = cat[c].to_numpy()
            per_col[c].append(lookup_accuracy(v[tr], y[tr], v[te], y[te]))

    base = float(np.mean(baselines))

    print("%-20s %8s %8s %8s %8s %8s %9s"
          % ("column", *[u.replace("user", "u") for u in users], "MEAN"))
    print("-" * 74)

    rows = []
    for c in cols:
        vals = per_col[c]
        m = float(np.mean(vals))
        rows.append({"column": c, "mean": m, "per_user": vals,
                     "cramers_v": cramers_v(cat[c], y)})
        print("%-20s %s %9.3f"
              % (c, " ".join("%8.3f" % v for v in vals), m))

    print("%-20s %s %9.3f"
          % ("guessing", " ".join("%8.3f" % b for b in baselines), base))
    print()

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("guessing baseline: %.3f" % base)
    print()

    rows.sort(key=lambda r: -r["mean"])
    for r in rows:
        m, c = r["mean"], r["column"]
        if m >= 0.95:
            v = "GIVES THE ANSWER AWAY, do not use"
        elif m > base + 0.15:
            v = "strong, check it is physiology and not an artefact"
        elif m > base + 0.05:
            v = "real information (+%.3f over guessing)" % (m - base)
        elif m > base + 0.02:
            v = "a little information (+%.3f)" % (m - base)
        else:
            v = "carries nothing, at or below guessing"
        r["verdict"] = v
        print("  %-20s %.3f   %s" % (c, m, v))
    print()

    worst = max(r["mean"] for r in rows)
    if worst >= 0.95:
        print("At least one column is effectively the label. That is the same")
        print("shape of problem as Task on WESAD, and it has to be excluded")
        print("rather than explained.")
    else:
        print("No column reaches 0.95, so nothing here is the label wearing a")
        print("different name. That is already better than WESAD, where Task")
        print("scored exactly 1.000 in every rotation.")
    print()
    print("Two things to keep separate when reading the table:")
    print()
    print("  A body signal predicting stress is the FINDING, not a problem.")
    print("  If pulse rate predicts the label, that is what we hoped for.")
    print()
    print("  A clock or a bookkeeping column predicting stress is a PROBLEM,")
    print("  because it says more about when the survey was filled in than")
    print("  about how the person felt. HOUR is the one to watch here.")
    print()

    hour = next((r for r in rows if r["column"] == "HOUR"), None)
    if hour and hour["mean"] > base + 0.05:
        print("  HOUR scores %.3f against %.3f for guessing. People seem to"
              % (hour["mean"], base))
        print("  have answered at similar times of day, so the clock carries")
        print("  some of the answer. Worth reporting the model with and")
        print("  without it.")
        print()

    # ---------------- report ----------------
    L = []
    L.append("# Leak check on the user context")
    L.append("")
    L.append("Written by `btp/32_user_leak_check.py`. No model is trained.")
    L.append("")
    L.append("For each context column: learn the most common label for each of")
    L.append("its values from four users, then apply that lookup to the fifth.")
    L.append("Whatever it scores, a model gets for free just by reading the")
    L.append("column.")
    L.append("")
    L.append("This is the test that found `Task` predicting the WESAD label")
    L.append("with an accuracy of exactly 1.000 in every rotation, because the")
    L.append("label came from one questionnaire per session and `Task` was the")
    L.append("session.")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| windows | %d |" % len(df))
    L.append("| users | %s |" % ", ".join(users))
    L.append("| guessing baseline | %.3f |" % base)
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| column | " + " | ".join(users) + " | mean | Cramer's V | verdict |")
    L.append("|---|" + "---:|" * (len(users) + 2) + "---|")
    for r in rows:
        L.append("| `%s` | %s | **%.3f** | %.3f | %s |"
                 % (r["column"], " | ".join("%.3f" % v for v in r["per_user"]),
                    r["mean"], r["cramers_v"], r["verdict"]))
    L.append("| guessing | %s | **%.3f** | | |"
             % (" | ".join("%.3f" % b for b in baselines), base))
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("Two different things can make a column score well, and they")
    L.append("should not be confused.")
    L.append("")
    L.append("**A body signal predicting stress is the finding.** If pulse")
    L.append("rate or EDA predicts the label, that is what the whole project")
    L.append("is looking for.")
    L.append("")
    L.append("**A clock or bookkeeping column predicting stress is a")
    L.append("problem**, because it says more about when the survey was")
    L.append("answered than about how the person felt. `HOUR` is the one to")
    L.append("watch.")
    L.append("")
    L.append("## Note on the survey's own context")
    L.append("")
    L.append("`responses_scored.csv` also holds what people reported about")
    L.append("themselves: activity, situation, environment, tiredness, and the")
    L.append("main source of stress in the last hour.")
    L.append("")
    L.append("**Main source of stress must not become a feature.** It is close")
    L.append("to asking the label directly, and would be a `Task` of its own.")
    L.append("It is useful for explaining results, not for making them.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")

    json.dump({"baseline": base, "users": users,
               "columns": {r["column"]: {"mean": r["mean"],
                                         "per_user": r["per_user"],
                                         "cramers_v": r["cramers_v"],
                                         "verdict": r["verdict"]}
                           for r in rows}},
              open("data/processed/labels/leak_check.json", "w"), indent=2)

    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
