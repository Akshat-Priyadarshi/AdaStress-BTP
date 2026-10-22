"""
06_collect_results.py

WHY THIS FILE EXISTS
    Every run writes its own metrics.json into its own folder, on purpose, so
    that nothing is ever overwritten. But that means the comparison table has
    to be built by reading all of them. Doing it by hand invites typing
    mistakes in exactly the numbers that matter.

WHAT IT DOES
    Reads runs/*/metrics.json and prints one table. When several runs differ
    only by seed, it groups them and reports the average and the spread, so a
    real difference can be told apart from noise.

    It reads only.

OUTPUT
    docs/90_results_table.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/06_collect_results.py

HOW TO READ IT
    Every accuracy is printed next to the guessing baseline. A number below
    the baseline is worse than guessing.

    Balanced accuracy matters more than accuracy here, because the classes
    are uneven. A model that always answers LowStress scores about 0.52
    accuracy but only 0.333 balanced accuracy, which is chance for three
    classes.

    The "spread" column is the difference between the best and worst seed.
    If two runs differ by less than their spread, the difference is noise.
"""

import os
import json
import glob
import re
from collections import defaultdict

RUNS_DIR = "runs"
REPORT = "docs/90_results_table.md"


def group_name(run_id):
    """Strip a trailing seed marker so runs that differ only by seed group."""
    return re.sub(r"[_-]?seed\d+$", "", run_id)


def main():
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "*", "metrics.json")))
    if not paths:
        raise SystemExit("No metrics.json found under %s/" % RUNS_DIR)

    runs = []
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        m["_folder"] = os.path.dirname(p)
        runs.append(m)

    ok = [r for r in runs if r.get("status") == "ok"]
    skipped = [r for r in runs if r.get("status") != "ok"]

    print("runs found: %d  (usable %d, skipped %d)"
          % (len(runs), len(ok), len(skipped)))
    print()

    L = []
    L.append("# Results table")
    L.append("")
    L.append("Built by `btp/06_collect_results.py` from every")
    L.append("`runs/*/metrics.json`. Not typed by hand.")
    L.append("")
    L.append("Every accuracy sits next to the guessing baseline. A number")
    L.append("below the baseline is worse than guessing. Balanced accuracy")
    L.append("matters more than accuracy here, because the classes are uneven:")
    L.append("a model that always answers LowStress scores about 0.52 accuracy")
    L.append("but only 0.333 balanced accuracy, which is chance for 3 classes.")
    L.append("")

    if not ok:
        L.append("No usable runs.")
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        open(REPORT, "w").write("\n".join(L) + "\n")
        return

    # ---- every run, one line each ----
    hdr = ("%-38s %-9s %-6s %7s %7s %7s %8s %7s"
           % ("run", "context", "sel", "acc", "bal", "mF1", "base", "gate"))
    print(hdr)
    print("-" * len(hdr))

    L.append("## Every run")
    L.append("")
    L.append("| run | context | epoch reported | accuracy | balanced | macro F1 "
             "| baseline | gate |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")

    for r in sorted(ok, key=lambda x: x["run_id"]):
        line = ("%-38s %-9s %-6s %7.3f %7.3f %7.3f %8.3f %7.3f"
                % (r["run_id"][:38], r.get("context_mode", "?"),
                   r.get("select_on", "val"),
                   r["accuracy"], r["balanced_accuracy"], r["macro_f1"],
                   r["majority_baseline_accuracy"], r.get("mean_gate", float("nan"))))
        print(line)
        L.append("| `%s` | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |"
                 % (r["run_id"], r.get("context_mode", "?"),
                    r.get("select_on", "val"), r["accuracy"],
                    r["balanced_accuracy"], r["macro_f1"],
                    r["majority_baseline_accuracy"],
                    r.get("mean_gate", float("nan"))))
    print()
    L.append("")

    # ---- grouped over seeds ----
    groups = defaultdict(list)
    for r in ok:
        groups[group_name(r["run_id"])].append(r)

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print("Averaged over seeds, where more than one seed was run:")
        print()
        hdr2 = ("%-34s %5s %15s %15s %15s"
                % ("group", "n", "accuracy", "balanced", "macro F1"))
        print(hdr2)
        print("-" * len(hdr2))

        L.append("## Averaged over seeds")
        L.append("")
        L.append("`spread` is best seed minus worst seed. If two groups differ")
        L.append("by less than their spread, the difference is noise.")
        L.append("")
        L.append("| group | seeds | accuracy | balanced | macro F1 |")
        L.append("|---|---:|---|---|---|")

        for name in sorted(multi):
            rs = multi[name]
            def stat(key):
                vals = [x[key] for x in rs]
                return sum(vals) / len(vals), max(vals) - min(vals)
            a, as_ = stat("accuracy")
            b, bs = stat("balanced_accuracy")
            f, fs = stat("macro_f1")
            print("%-34s %5d  %.3f +/-%.3f  %.3f +/-%.3f  %.3f +/-%.3f"
                  % (name[:34], len(rs), a, as_, b, bs, f, fs))
            L.append("| `%s` | %d | %.3f (spread %.3f) | %.3f (spread %.3f) "
                     "| %.3f (spread %.3f) |"
                     % (name, len(rs), a, as_, b, bs, f, fs))
        print()
        L.append("")
    else:
        print("Only one seed per run so far, so no spread can be reported.")
        print("Run the same settings with --seed 1, 2, 3 to get error bars.")
        print()
        L.append("## Averaged over seeds")
        L.append("")
        L.append("Only one seed per setting so far, so no spread can be")
        L.append("reported. Repeat each run with `--seed 1`, `--seed 2`,")
        L.append("`--seed 3` and different run ids to get error bars.")
        L.append("")

    # ---- per-subject, which people the model works for ----
    L.append("## Accuracy for each test subject")
    L.append("")
    L.append("One overall number hides that the label means something")
    L.append("different for each person. See finding 6.")
    L.append("")
    subj_keys = sorted({s for r in ok
                        for s in r.get("per_subject_test_accuracy", {})})
    if subj_keys:
        L.append("| run | " + " | ".join(subj_keys) + " |")
        L.append("|---|" + "---:|" * len(subj_keys))
        print("Accuracy for each test subject:")
        print("%-38s %s" % ("run", " ".join("%7s" % s for s in subj_keys)))
        for r in sorted(ok, key=lambda x: x["run_id"]):
            d = r.get("per_subject_test_accuracy", {})
            if not d:
                continue
            cells = ["%.3f" % d[s] if s in d else "-" for s in subj_keys]
            print("%-38s %s" % (r["run_id"][:38],
                                " ".join("%7s" % c for c in cells)))
            L.append("| `%s` | " % r["run_id"] + " | ".join(cells) + " |")
        print()
        L.append("")

    if skipped:
        L.append("## Runs with too little data")
        L.append("")
        for r in skipped:
            L.append("- `%s`: %s, train %s windows, test %s windows"
                     % (r.get("run_id"), r.get("status"),
                        r.get("n_train"), r.get("n_test")))
        L.append("")

    L.append("## Reminder on how to read the context runs")
    L.append("")
    L.append("The audit found that `Task` alone predicts the label with 100%")
    L.append("accuracy on the held-out subjects, and `Task` is one of the")
    L.append("context features. So any run with `context_mode` of `original`")
    L.append("or `zero` reports a leakage measurement, not a performance")
    L.append("figure. The honest number is the `none` run.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print("Table written to %s" % REPORT)


if __name__ == "__main__":
    main()
