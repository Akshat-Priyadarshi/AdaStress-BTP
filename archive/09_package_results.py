"""
09_package_results.py

WHY THIS FILE EXISTS
    The work is spread across four places on purpose: runs/ holds one folder
    per experiment so nothing is overwritten, docs/ holds the written
    decisions, data/ holds the processed files, and vendor/ holds the
    untouched original code. That is right for doing the work and wrong for
    showing it.

    This script gathers one dataset's results into a single folder that can
    be handed over or zipped, without moving or changing anything.

WHAT IT DOES
    Builds results/<dataset>/ containing:

      README.md            one page: the headline table and how to redo it
      summary.csv          the same table, machine readable
      decisions.md         copy of docs/00_decisions.md
      findings.md          copy of the findings document
      results.md           copy of the results document
      reports/             copies of the numbered docs for this dataset
      runs/<id>/           metrics.json, predictions.csv, env.txt, config.json
                           per run. Checkpoints are left behind, because they
                           are large and are not evidence.

    It only reads and copies. Nothing in runs/, docs/ or data/ is changed.

NOTE ON WINDOWS DRIVES
    We copy file contents only, with shutil.copyfile, and deliberately do not
    copy timestamps or permissions. On WSL, a Windows drive mounted at /mnt/e
    can take seconds per file just to set a timestamp, which makes the script
    look frozen. Contents are all we need.

HOW TO RUN
    From the AdaStress_BTP folder:

        python3 btp/09_package_results.py --dataset wesad

    Run it again any time. It rebuilds the folder from scratch.
"""

import os
import re
import json
import glob
import shutil
import argparse
from collections import defaultdict

BASELINE_NOTE = (
    "Always guessing the most common class scores 0.516 on the test "
    "subjects. Every accuracy below must be read next to that number."
)

# Which docs belong to which dataset.
DOC_PATTERNS = {
    "wesad": ["10_wesad_*.md", "11_session_order*.md", "12_assembled_*.md",
              "13_audit_*.md", "WESAD_raw_structure.md"],
    "user1": ["user1_*.md", "3*_user1*.md", "4*_user1*.md"],
}

# Which findings and results docs to copy, per dataset, first match wins.
FINDINGS = {
    "wesad": ["docs/90_findings_stage1.md", "docs/90_findings.md"],
    "user1": ["docs/90_findings_stage1.md", "docs/90_findings.md"],
}
RESULTS = {
    "wesad": ["docs/91_results_stage2.md"],
    "user1": ["docs/32_results_stage4.md", "docs/91_results_user1.md"],
}


def group_name(run_id):
    """Strip a trailing seed marker so runs differing only by seed group."""
    return re.sub(r"[_-]?seed\d+$", "", run_id)


def load_runs(dataset):
    """Read the run summaries for this dataset.

    The two stages record their results in different files, because they
    measure different things. WESAD training writes metrics.json with an
    accuracy. user1 inference writes metrics_user1.json with no accuracy at
    all, because user1 has no labels (decision D2).
    """
    out = []

    if dataset == "user1":
        # inference runs: no accuracy, so a different shape
        for p in sorted(glob.glob("runs/*/metrics_user1.json")):
            try:
                m = json.load(open(p))
            except Exception:
                continue
            m["_folder"] = os.path.dirname(p)
            m["run_id"] = m.get("out_id", os.path.basename(m["_folder"]))
            m["status"] = "ok"
            out.append(m)
        return out

    for p in sorted(glob.glob("runs/*/metrics.json")):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        m["_folder"] = os.path.dirname(p)
        where = str(m.get("assembled_dir", "")) + " " + str(m.get("run_id", ""))
        if "user1" in where.lower():
            continue
        out.append(m)
    return out


def mean_spread(values):
    if not values:
        return float("nan"), float("nan")
    return sum(values) / len(values), max(values) - min(values)


def main():
    ap = argparse.ArgumentParser(
        description="Gather one dataset's results into a single folder.")
    ap.add_argument("--dataset", default="wesad", choices=["wesad", "user1"])
    args = ap.parse_args()
    ds = args.dataset

    out_dir = os.path.join("results", ds)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "reports"))
    os.makedirs(os.path.join(out_dir, "runs"))

    runs = load_runs(ds)
    ok = [r for r in runs if r.get("status") == "ok"]
    print("runs for %s: %d found, %d usable" % (ds, len(runs), len(ok)))

    # ---- copy the per-run evidence, but not the checkpoints ----
    keep = ["metrics.json", "metrics_user1.json", "predictions.csv",
            "env.txt", "config.json", "log.txt", "sig_cols.json",
            "context_names.json"]
    for r in runs:
        src = r["_folder"]
        dst = os.path.join(out_dir, "runs", os.path.basename(src))
        os.makedirs(dst, exist_ok=True)
        for name in keep:
            p = os.path.join(src, name)
            if os.path.exists(p):
                # contents only, no timestamps: see the note at the top
                shutil.copyfile(p, os.path.join(dst, name))
    print("copied evidence for %d run folders (checkpoints left behind)"
          % len(runs))
    print()

    # ---- copy the written documents ----
    copied = []
    for src, name in [(FINDINGS.get(ds, []), "findings.md"),
                      (RESULTS.get(ds, []), "results.md")]:
        for candidate in src:
            if os.path.exists(candidate):
                print("  copying %s" % candidate, flush=True)
                shutil.copyfile(candidate, os.path.join(out_dir, name))
                copied.append(candidate)
                break

    if os.path.exists("docs/00_decisions.md"):
        print("  copying docs/00_decisions.md", flush=True)
        shutil.copyfile("docs/00_decisions.md",
                        os.path.join(out_dir, "decisions.md"))
        copied.append("docs/00_decisions.md")

    for pattern in DOC_PATTERNS.get(ds, []):
        for p in sorted(glob.glob(os.path.join("docs", pattern))):
            print("  copying %s" % p, flush=True)
            shutil.copyfile(p, os.path.join(out_dir, "reports",
                                            os.path.basename(p)))
            copied.append(p)

    print()
    print("copied %d documents" % len(copied))

    if os.path.exists("vendor/adastress_commit.txt"):
        shutil.copyfile("vendor/adastress_commit.txt",
                        os.path.join(out_dir, "original_repo_commit.txt"))

    # ---- build the summary table ----
    #
    # user1 has no labels, so there is no accuracy to average. Instead we
    # report the things that show whether a model has broken: how lopsided
    # its predictions are, the stress score, the gate, and how far its
    # context landed outside the training range.
    if ds == "user1":
        rows = []
        for r in sorted(ok, key=lambda x: x["run_id"]):
            oor = r.get("context_columns_out_of_range") or []
            rows.append({
                "run": r["run_id"],
                "source_model": r.get("source_run", ""),
                "context_mode": r.get("context_mode", "?"),
                "context_scale": r.get("context_scale", ""),
                "derived_scale": r.get("derived_scale", ""),
                "n_windows": r.get("n_windows", 0),
                "stress_score_mean": round(r.get("stress_score_mean", 0), 3),
                "gate_mean": round(r.get("gate_mean", 0), 3),
                "gate_on_wesad": round(r.get("wesad_gate_mean") or 0, 3),
                "uncertainty_mean": round(r.get("entropy_mean", 0), 3),
                "largest_class_share": round(r.get("largest_class_share", 0), 3),
                "predicted_counts": str(r.get("predicted_counts", {})),
                "context_out_of_range": ";".join(oor) if oor else "none",
                "wesad_accuracy_of_source": round(r.get("wesad_accuracy") or 0, 3),
            })

        if rows:
            import csv
            with open(os.path.join(out_dir, "summary.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for row in rows:
                    w.writerow(row)

        L = []
        L.append("# USER1 results")
        L.append("")
        L.append("Built by `btp/09_package_results.py --dataset user1`.")
        L.append("Everything here is copied, nothing was moved or changed.")
        L.append("")
        L.append("## user1 has no stress label")
        L.append("")
        L.append("Not one, in any of the 22 files (decision D2). Two recording")
        L.append("sessions would give about two questionnaire answers even if")
        L.append("they were found on paper.")
        L.append("")
        L.append("**So there is no accuracy anywhere in this folder.** These are")
        L.append("WESAD-trained models applied to user1 unchanged, compared by")
        L.append("how they behave.")
        L.append("")
        L.append("## What broke, and what did not")
        L.append("")
        L.append("| run | context | scaling | predictions | stress score | gate | on WESAD | uncertainty | context out of range |")
        L.append("|---|---|---|---|---:|---:|---:|---:|---|")
        for row in rows:
            scale = row["context_scale"]
            if row["context_mode"] == "derived":
                scale = row["derived_scale"]
            L.append("| `%s` | %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %s |"
                     % (row["run"], row["context_mode"], scale,
                        row["predicted_counts"], row["stress_score_mean"],
                        row["gate_mean"], row["gate_on_wesad"],
                        row["uncertainty_mean"], row["context_out_of_range"]))
        L.append("")
        L.append("## How to read it")
        L.append("")
        L.append("There is no accuracy, so these are the signs of breakage:")
        L.append("")
        L.append("- **one class taking nearly every window**: the model has")
        L.append("  stopped telling windows apart")
        L.append("- **uncertainty of 0.000**: a three-class softmax cannot reach")
        L.append("  zero entropy unless it is fully saturated. The output is")
        L.append("  identical for every window")
        L.append("- **context out of range**: a column landed far outside")
        L.append("  anything the model saw in training, which is the cause, not")
        L.append("  a symptom")
        L.append("- **a gate far from its WESAD value**: the model is weighting")
        L.append("  context differently from how it was trained")
        L.append("")
        L.append("## The result in one line")
        L.append("")
        L.append("Experiment A, the zero-filled context, gives **100% LowStress**")
        L.append("with the real numbers and **88% HighStress** with the same")
        L.append("numbers divided by a constant. Same model, same weights, same")
        L.append("windows. The answer is decided by an undocumented scaling")
        L.append("choice, not by the person's body.")
        L.append("")
        L.append("Experiment B, the shared movement context, pads nothing and is")
        L.append("the right construction, but is degenerate under one scaling")
        L.append("and reasonable under the other. Full reasoning in `results.md`.")
        L.append("")
        L.append("## What is in this folder")
        L.append("")
        L.append("| file | what it is |")
        L.append("|---|---|")
        L.append("| `README.md` | this page |")
        L.append("| `summary.csv` | the table above, machine readable |")
        L.append("| `results.md` | the full Stage 4 write-up |")
        L.append("| `findings.md` | what the WESAD data showed beforehand |")
        L.append("| `decisions.md` | every choice made, with reasons |")
        L.append("| `reports/` | how user1 was inspected and prepared |")
        L.append("| `runs/<id>/` | per-window predictions and summary |")
        L.append("")
        L.append("Each `runs/<id>/predictions.csv` holds one row per window with")
        L.append("the three class probabilities, the stress score, the")
        L.append("uncertainty and the gate value.")
        L.append("")
        open(os.path.join(out_dir, "README.md"), "w").write("\n".join(L) + "\n")

        print()
        print("Packaged into %s/" % out_dir)
        print("  README.md      the one page to open first")
        print("  summary.csv    %d inference runs" % len(rows))
        print("  results.md, findings.md, decisions.md")
        print("  reports/, runs/")
        print()
        print("To hand it over:  zip -r user1_results.zip %s" % out_dir)
        return

    groups = defaultdict(list)
    for r in ok:
        key = (r.get("context_mode", "?"), r.get("select_on", "val"))
        groups[key].append(r)

    rows = []
    for (mode, sel), rs in sorted(groups.items()):
        acc, acc_s = mean_spread([x["accuracy"] for x in rs])
        bal, bal_s = mean_spread([x["balanced_accuracy"] for x in rs])
        f1, f1_s = mean_spread([x["macro_f1"] for x in rs])
        gate, _ = mean_spread([x.get("mean_gate", float("nan")) for x in rs])
        base, _ = mean_spread([x["majority_baseline_accuracy"] for x in rs])
        rows.append({
            "context_mode": mode,
            "epoch_reported": sel,
            "n_runs": len(rs),
            "accuracy": round(acc, 3),
            "accuracy_spread": round(acc_s, 3),
            "balanced_accuracy": round(bal, 3),
            "balanced_spread": round(bal_s, 3),
            "macro_f1": round(f1, 3),
            "macro_f1_spread": round(f1_s, 3),
            "mean_gate": round(gate, 3),
            "guessing_baseline": round(base, 3),
            "run_ids": ";".join(sorted(x["run_id"] for x in rs)),
        })

    # summary.csv
    if rows:
        import csv
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)

    # ---- one page README ----
    L = []
    L.append("# %s results" % ds.upper())
    L.append("")
    L.append("Built by `btp/09_package_results.py --dataset %s`." % ds)
    L.append("Everything here is copied, nothing was moved or changed.")
    L.append("")
    if os.path.exists(os.path.join(out_dir, "original_repo_commit.txt")):
        commit = open(os.path.join(out_dir,
                                   "original_repo_commit.txt")).read().strip()
        L.append("Original repository reproduced at commit `%s`." % commit)
        L.append("")
    L.append("## Headline table")
    L.append("")
    L.append(BASELINE_NOTE)
    L.append("")
    L.append("| context input | runs | accuracy | balanced accuracy "
             "| macro F1 | gate |")
    L.append("|---|---:|---:|---:|---:|---:|")

    label = {
        "none": "none (the honest number)",
        "original": "Task, SessionPhase, Activity (as released)",
        "zero": "Case 1: the above plus zero-filled columns",
        "derived": "Case 2: accelerometer movement",
    }
    # Show the rows in a logical order, not alphabetical: the honest number
    # first, then the two leaky ones, then the transferable one.
    order = {"none": 0, "original": 1, "zero": 2, "derived": 3}
    for row in sorted(rows, key=lambda r: order.get(r["context_mode"], 9)):
        if row["epoch_reported"] != "last":
            continue
        L.append("| %s | %d | %.3f (spread %.3f) | %.3f | %.3f | %.3f |"
                 % (label.get(row["context_mode"], row["context_mode"]),
                    row["n_runs"], row["accuracy"], row["accuracy_spread"],
                    row["balanced_accuracy"], row["macro_f1"],
                    row["mean_gate"]))
    L.append("")
    L.append("Runs reporting the best-validation epoch instead of the last one "
             "are in `summary.csv`. They are kept because the difference is "
             "itself a finding: see finding 6 in `results.md`.")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("Balanced accuracy matters more than accuracy, because the")
    L.append("classes are uneven. A model that always answers LowStress scores")
    L.append("about 0.52 accuracy but only 0.333 balanced accuracy, which is")
    L.append("chance for three classes.")
    L.append("")
    L.append("The runs whose context includes `Task` report a **leakage**")
    L.append("measurement, not a performance figure. `Task` predicts the label")
    L.append("with 100% accuracy on held-out subjects, because the label comes")
    L.append("from one questionnaire per session and `Task` is the session.")
    L.append("The honest number is the no-context row.")
    L.append("")
    L.append("## What is in this folder")
    L.append("")
    L.append("| file | what it is |")
    L.append("|---|---|")
    L.append("| `README.md` | this page |")
    L.append("| `summary.csv` | the table above, machine readable |")
    L.append("| `findings.md` | what was found in the data before training |")
    L.append("| `results.md` | what the trained models did |")
    L.append("| `decisions.md` | every choice made, with reasons |")
    L.append("| `reports/` | the per-step reports the scripts wrote |")
    L.append("| `runs/<id>/` | per-run metrics, predictions, environment, log |")
    L.append("| `original_repo_commit.txt` | the exact commit reproduced |")
    L.append("")
    L.append("Model checkpoints are deliberately left out. They are large and")
    L.append("they are not evidence. They stay in `runs/` in the working tree.")
    L.append("")
    L.append("## How to reproduce it")
    L.append("")
    L.append("```bash")
    L.append("bash setup_stage0.sh")
    L.append("python3 btp/s0_check_wesad_raw.py")
    L.append("python3 btp/02_check_session_order.py")
    L.append("python3 btp/01_prep_wesad.py --task-order file")
    L.append("python3 btp/03_assemble_wesad.py --label-mode block")
    L.append("python3 btp/04_audit_wesad.py --assembled-dir data/assembled/block")
    L.append("python3 btp/05_train.py --run-id R01_wesad_nocontext \\")
    L.append("    --assembled-dir data/assembled/block --context-mode none")
    L.append("python3 btp/06_collect_results.py")
    L.append("python3 btp/09_package_results.py --dataset wesad")
    L.append("```")
    L.append("")
    L.append("## Every run on record")
    L.append("")
    L.append("| run id | context | epoch | accuracy | balanced | macro F1 |")
    L.append("|---|---|---|---:|---:|---:|")
    for r in sorted(ok, key=lambda x: x["run_id"]):
        L.append("| `%s` | %s | %s | %.3f | %.3f | %.3f |"
                 % (r["run_id"], r.get("context_mode", "?"),
                    r.get("select_on", "val"), r["accuracy"],
                    r["balanced_accuracy"], r["macro_f1"]))
    L.append("")
    skipped = [r for r in runs if r.get("status") != "ok"]
    if skipped:
        L.append("Runs that stopped for lack of data:")
        L.append("")
        for r in skipped:
            L.append("- `%s`: train %s windows, test %s windows"
                     % (r.get("run_id"), r.get("n_train"), r.get("n_test")))
        L.append("")

    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(L) + "\n")

    print()
    print("Packaged into %s/" % out_dir)
    print("  README.md      the one page to open first")
    print("  summary.csv    the table, machine readable")
    print("  findings.md, results.md, decisions.md")
    print("  reports/, runs/")
    print()
    print("To hand it over:  zip -r %s_results.zip %s" % (ds, out_dir))


if __name__ == "__main__":
    main()
