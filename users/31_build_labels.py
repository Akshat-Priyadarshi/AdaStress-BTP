"""
31_build_labels.py

WHAT THIS DOES
    Turns the 91 survey responses into stress labels attached to context
    windows, so the model has something to learn from.

    Three jobs:
      1. score each response with the STAI-6 rule
      2. cut the scores into three classes
      3. attach each response to the windows around the moment it was given

THE SURVEY IS STAI-6
    Six items: calm, tense, upset, relaxed, content, worried. That is the
    validated six-item short form of the State-Trait Anxiety Inventory, so it
    has a published scoring rule rather than something we invent.

        reverse the positive items   calm, relaxed, content   (4->1, 3->2, ...)
        add all six                                           ->  6 to 24
        multiply by 20/6                                      -> 20 to 80

    The multiplication puts it on the same 20 to 80 scale as the full
    20-item STAI, which is what makes the numbers comparable to published
    work.

CUTTING INTO THREE CLASSES, AND WHY ACROSS EVERYONE
    The classes are cut across all users together, not per person.

    This is the single thing that went wrong with WESAD. There, each
    subject's five questionnaire scores were cut into their own thirds, so
    "HighStress" meant "the worst of this person's own five sessions". S11's
    calm baseline came out Medium and S14's came out High, with nothing
    different about their bodies. A model fitted on nine people could not
    transfer to three others, because the target meant a different thing for
    each of them.

    Cutting across everyone means a score of 60 is the same level of anxiety
    whoever reported it. We can do this here and could not there, because
    these people answered 6 to 31 times each rather than 5, and the STAI-6
    scale is absolute rather than relative.

ATTACHING A RESPONSE TO WINDOWS
    A response is one moment; a window is 30 seconds. The survey asks about
    "the last hour", so the answer describes a stretch of time, not an
    instant.

    We attach each response to the windows within 15 minutes either side.
    That gives about 60 windows per response.

    Why 15 and not more: the wider the reach, the more windows share a single
    answer, and the more they behave like copies of each other rather than
    independent examples. 15 minutes either side is enough to train on
    without claiming one answer describes two hours.

    Where two responses both reach the same window, the nearer one wins.

OUTPUT
    data/processed/labels/responses_scored.csv
        one row per survey response, with its STAI-6 score and class

    data/processed/labels/labelled_windows.csv
        one row per window that a response reaches, with every context
        column and the label

    docs/56_labels.md

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/31_build_labels.py

    Seconds.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd

SURVEY = "mentor_files/Stress_and_STAI_Survey_Responses.xlsx"
CONTEXT_DIR = "data/processed/context"
OUT_DIR = "data/processed/labels"
REPORT = "docs/56_labels.md"

# STAI-6. The three positive items are reversed before adding.
POSITIVE = ["I feel calm.", "I am relaxed.", "I feel content."]
NEGATIVE = ["I am tense.", "I feel upset.", "I am worried."]
ITEMS = POSITIVE + NEGATIVE

# Context the survey itself records, kept as reported.
SURVEY_CONTEXT = {
    "1. Current Activity": "SURVEY_ACTIVITY",
    "2. Current Situation": "SURVEY_SITUATION",
    "3. Current Environment": "SURVEY_ENVIRONMENT",
    "4. Tiredness Right Now": "SURVEY_TIREDNESS",
    "5. Main Source of Stress in the Last Hour": "SURVEY_STRESS_SOURCE",
}

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]
REACH_MIN = 15          # minutes either side of a response


def score_stai6(df):
    """Apply the published STAI-6 rule."""
    missing = [c for c in ITEMS if c not in df.columns]
    if missing:
        raise SystemExit("Survey is missing these items: %s" % missing)

    bad = df[ITEMS].apply(lambda s: ~s.isin([1, 2, 3, 4])).any(axis=1)
    if bad.any():
        print("  dropping %d responses with an item outside 1 to 4"
              % int(bad.sum()))
        df = df[~bad].copy()

    raw = df[NEGATIVE].sum(axis=1) + (5 - df[POSITIVE]).sum(axis=1)
    df["STAI6_RAW"] = raw                       # 6 to 24
    df["STAI6"] = raw * 20.0 / 6.0              # 20 to 80
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Score the survey and attach the labels to context "
                    "windows.")
    ap.add_argument("--survey", default=SURVEY)
    ap.add_argument("--context-dir", default=CONTEXT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--reach-min", type=int, default=REACH_MIN,
                    help="minutes either side of a response that it labels")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.survey):
        raise SystemExit("Survey not found at %s" % args.survey)

    print("=" * 70)
    print("Building labels from the survey")
    print("survey : %s" % args.survey)
    print("reach  : %d minutes either side of each response" % args.reach_min)
    print("=" * 70)
    print()

    # ---------------- the responses ----------------
    sv = pd.read_excel(args.survey)
    sv["t"] = pd.to_datetime(sv["Timestamp"], errors="coerce")

    before = len(sv)
    sv = sv[sv["t"].notna()]
    sv = sv[sv["User"].astype(str).str.match(r"^user\d+$", na=False)]
    print("responses: %d of %d usable (rest had no user id or no timestamp)"
          % (len(sv), before))

    sv = score_stai6(sv)

    # ---------------- three classes, across everyone ----------------
    lo, hi = sv["STAI6"].quantile([1 / 3, 2 / 3])
    sv["LABEL"] = np.where(sv["STAI6"] <= lo, 0,
                           np.where(sv["STAI6"] >= hi, 2, 1))
    sv["LABEL_NAME"] = [LABEL_NAMES[i] for i in sv["LABEL"]]

    print()
    print("STAI-6 score: min %.1f  median %.1f  max %.1f"
          % (sv["STAI6"].min(), sv["STAI6"].median(), sv["STAI6"].max()))
    print("cut points  : %.1f and %.1f, applied to everyone alike" % (lo, hi))
    print()
    print("class counts:")
    print(sv["LABEL_NAME"].value_counts().reindex(LABEL_NAMES).to_string())
    print()

    per_user = pd.crosstab(sv["User"], sv["LABEL_NAME"]) \
        .reindex(columns=LABEL_NAMES, fill_value=0)
    print("per user:")
    print(per_user.to_string())
    print()

    single = per_user[(per_user > 0).sum(axis=1) == 1]
    if len(single):
        print("  NOTE: %s answered with only one class."
              % ", ".join(single.index))
        print("  Holding such a user out makes a fold where always guessing")
        print("  that class scores 100%. Balanced accuracy is the number to")
        print("  read there, and the fold should be flagged in any report.")
        print()

    # keep the context the survey itself recorded
    keep = ["User", "t", "STAI6_RAW", "STAI6", "LABEL", "LABEL_NAME"]
    for src, dst in SURVEY_CONTEXT.items():
        if src in sv.columns:
            sv[dst] = sv[src]
            keep.append(dst)
    scored = sv[keep].rename(columns={"User": "Subject_ID",
                                      "t": "response_time"})
    scored.to_csv(os.path.join(args.out_dir, "responses_scored.csv"),
                  index=False)

    # ---------------- attach to windows ----------------
    print("=" * 70)
    print("ATTACHING RESPONSES TO WINDOWS")
    print("=" * 70)

    reach = pd.Timedelta(minutes=args.reach_min)
    rows, summary = [], []

    for user, g in scored.groupby("Subject_ID"):
        path = os.path.join(args.context_dir, "%s_context.csv" % user)
        if not os.path.exists(path):
            print("  %-8s no context file, skipped" % user)
            continue

        ctx = pd.read_csv(path)
        ctx["datetime_ist"] = pd.to_datetime(ctx["datetime_ist"],
                                             errors="coerce")
        ctx = ctx.dropna(subset=["datetime_ist"]).sort_values("datetime_ist")

        g = g.sort_values("response_time")

        # nearest response within reach; merge_asof does the work and picks
        # the closer one where two could reach the same window
        joined = pd.merge_asof(
            ctx, g[["response_time", "STAI6", "LABEL", "LABEL_NAME"]],
            left_on="datetime_ist", right_on="response_time",
            direction="nearest", tolerance=reach)

        hit = joined[joined["LABEL"].notna()].copy()
        hit["Subject_ID"] = user
        rows.append(hit)

        n_resp_used = hit["response_time"].nunique()
        print("  %-8s %5d windows from %2d of %2d responses  (%.1f h)"
              % (user, len(hit), n_resp_used, len(g),
                 len(hit) * 30 / 3600.0))
        summary.append({"user": user, "responses": len(g),
                        "responses_matched": n_resp_used,
                        "windows": len(hit)})

    if not rows:
        raise SystemExit("No windows could be labelled.")

    lab = pd.concat(rows, ignore_index=True)

    # Two columns were dropped from the context list after we measured them:
    #
    #   PRV_RMSSD        the beat timings jump by a median of 122 ms from one
    #                    beat to the next, so beat-to-beat variability came
    #                    out around 290 ms where a resting adult is 20 to 50
    #   AROUSAL_AT_REST  0.90 with PR, so it was pulse rate again
    #
    # user17's context file was written before they went on the drop list,
    # and the other four were written after, which is why these show about
    # 10 percent coverage: only one user in five has them. Re-deriving user17
    # would mean re-downloading 8.8 GB, so they come out here instead.
    stale = [c for c in ("PRV_RMSSD", "PRV_SDNN", "AROUSAL_AT_REST",
                         "MOVE_SD", "EDA_PHASIC_AMP") if c in lab.columns]
    if stale:
        lab = lab.drop(columns=stale)
        print()
        print("dropped columns left over from an earlier derivation run:")
        print("  %s" % ", ".join(stale))
        print("  (only user17 had them, which is why coverage looked low)")

    lab.to_csv(os.path.join(args.out_dir, "labelled_windows.csv"), index=False)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print("labelled windows: %d  (%.1f hours)"
          % (len(lab), len(lab) * 30 / 3600.0))
    print()
    print("by class:")
    print(lab["LABEL_NAME"].value_counts().reindex(LABEL_NAMES).to_string())
    print()
    print("by user and class:")
    print(pd.crosstab(lab["Subject_ID"], lab["LABEL_NAME"])
          .reindex(columns=LABEL_NAMES, fill_value=0).to_string())
    print()

    biggest = lab["LABEL_NAME"].value_counts().iloc[0]
    print("always answering the most common class scores %.1f%%"
          % (100.0 * biggest / len(lab)))
    print("every accuracy we report has to be read next to that number")
    print()

    ctx_cols = [c for c in lab.columns
                if c not in ("Subject_ID", "datetime_utc", "datetime_ist",
                             "date", "response_time", "STAI6", "LABEL",
                             "LABEL_NAME")]
    print("context columns carried through: %d" % len(ctx_cols))
    print("  %s" % ", ".join(ctx_cols))
    print()
    cov = lab[ctx_cols].notna().mean().sort_values()
    print("coverage within the labelled windows, lowest first:")
    print((cov * 100).round(1).head(6).to_string())
    print()

    # ---------------- report ----------------
    L = []
    L.append("# Labels from the survey")
    L.append("")
    L.append("Written by `btp/31_build_labels.py`.")
    L.append("")
    L.append("## The instrument")
    L.append("")
    L.append("The six items are the STAI-6, the validated short form of the")
    L.append("State-Trait Anxiety Inventory, so the scoring rule is published")
    L.append("rather than invented:")
    L.append("")
    L.append("    reverse calm, relaxed and content   (4->1, 3->2, 2->3, 1->4)")
    L.append("    add all six items                   ->  6 to 24")
    L.append("    multiply by 20/6                    -> 20 to 80")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append("| responses used | %d |" % len(scored))
    L.append("| score range | %.1f to %.1f |" % (scored["STAI6"].min(),
                                                 scored["STAI6"].max()))
    L.append("| cut points | %.1f and %.1f |" % (lo, hi))
    L.append("| reach | %d minutes either side |" % args.reach_min)
    L.append("")
    L.append("## Why the classes are cut across everyone")
    L.append("")
    L.append("Not per person. That is the single thing that went wrong with")
    L.append("WESAD: each subject's five scores were cut into their own")
    L.append("thirds, so HighStress meant \"the worst of this person's own")
    L.append("five sessions\". S11's calm baseline came out Medium and S14's")
    L.append("came out High, with nothing different about their bodies.")
    L.append("")
    L.append("Here a score of 60 means the same level of anxiety whoever")
    L.append("reported it. We can do that because these people answered 6 to")
    L.append("31 times each rather than 5, and because the STAI-6 scale is")
    L.append("absolute rather than relative.")
    L.append("")
    L.append("## Responses")
    L.append("")
    L.append("| user | responses | matched to windows | windows |")
    L.append("|---|---:|---:|---:|")
    for r in summary:
        L.append("| %s | %d | %d | %d |" % (r["user"], r["responses"],
                                            r["responses_matched"],
                                            r["windows"]))
    L.append("| **total** | **%d** | | **%d** |"
             % (sum(r["responses"] for r in summary), len(lab)))
    L.append("")
    L.append("## Classes")
    L.append("")
    L.append("| | responses | windows |")
    L.append("|---|---:|---:|")
    for n in LABEL_NAMES:
        L.append("| %s | %d | %d |"
                 % (n, int((scored["LABEL_NAME"] == n).sum()),
                    int((lab["LABEL_NAME"] == n).sum())))
    L.append("")
    L.append("Always answering the most common class scores **%.1f%%**."
             % (100.0 * biggest / len(lab)))
    L.append("Every accuracy must be read next to that.")
    L.append("")
    L.append("### Per user")
    L.append("")
    L.append(pd.crosstab(lab["Subject_ID"], lab["LABEL_NAME"])
             .reindex(columns=LABEL_NAMES, fill_value=0).to_markdown())
    L.append("")
    if len(single):
        L.append("**%s answered with only one class.** Holding such a user"
                 % ", ".join(single.index))
        L.append("out gives a fold where always guessing that class scores")
        L.append("100%. Read balanced accuracy there, and flag the fold.")
        L.append("")
    L.append("## Window reach")
    L.append("")
    L.append("Each response labels the windows within %d minutes either side,"
             % args.reach_min)
    L.append("about %d windows per response. The survey asks about \"the last"
             % (args.reach_min * 4))
    L.append("hour\", so an answer describes a stretch of time rather than an")
    L.append("instant. Reaching further would make more windows share one")
    L.append("answer, and they would start behaving like copies of each other")
    L.append("rather than independent examples. Where two responses both")
    L.append("reach a window, the nearer one wins.")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")

    json.dump({"reach_min": args.reach_min, "cut_points": [float(lo), float(hi)],
               "responses": int(len(scored)), "windows": int(len(lab)),
               "per_user": summary},
              open(os.path.join(args.out_dir, "label_summary.json"), "w"),
              indent=2)

    print("saved %s/labelled_windows.csv" % args.out_dir)
    print("report: %s" % REPORT)


if __name__ == "__main__":
    main()
