# """
# 42_build_context8.py

# WHAT THIS DOES
#     Rebuilds the labelled-window file so its context is exactly the eight
#     columns asked for, and nothing else.

#         HOUR             time of day        from the timestamp
#         IS_WEEKEND       weekend or not     from the timestamp
#         POSTURE          sit/stand/walk     derived by us from the wrist
#         ACT_COUNT        movement           derived by us from the wrist
#         ACT_CLASS        activity class     Empatica biomarker
#         BODY_POSITION    body position      Empatica biomarker
#         SLEEP            sleep stage        Empatica biomarker
#         INDOOR_OUTDOOR   environment        survey answer, see the warning

#     It writes a new file and leaves the earlier ones untouched, so the
#     twelve-column and seven-column runs stay reproducible.

# THE THREE BIOMARKER COLUMNS, AND THE UPSAMPLING
#     Empatica's biomarkers arrive at one value a minute. Our windows move
#     every 30 seconds, so each minute covers two windows.

#     The value is HELD, not interpolated. These are words - "walking",
#     "standing", "asleep" - and there is no halfway point between two words.

# TWO THINGS TO KNOW BEFORE TRUSTING THESE COLUMNS

#     Coverage. Across the five users the biomarkers are present for 22 to 42
#     percent of minutes. The rest are marked device_not_recording or
#     device_not_worn_correctly. Survey answers tend to come during waking
#     hours when the watch was on, so the overlap with labelled windows may be
#     better than that, and this script reports the real figure rather than
#     guessing.

#     Agreement. Checked against a 97 minute recording with activities logged
#     by hand, these three disagreed with what the person was actually doing:

#         body_position_left      called sitting "standing" 33 times of 56
#         activity_intensity      called sitting "vigorous" 10 times
#         sleep_detection_stage   stayed 0 throughout the logged sleep

#     They are included because they were asked for. The disagreement is
#     measured, not suspected, and belongs beside any result that uses them.

# INDOOR_OUTDOOR IS NOT A SENSOR READING
#     It is the survey's "Current Environment" answer, so it exists only where
#     a response exists and carries the same reach as the label.

#     It is also answered in the same sitting, in the same form, as the six
#     STAI items the label comes from. That puts it in the same category as
#     "main source of stress in the last hour", which we excluded for exactly
#     this reason.

#     There is already a sign of trouble: 73 percent of Outdoor windows are
#     Low, but most Outdoor windows come from one user who answered Low every
#     time. Run btp/32_user_leak_check.py on this file before using the column.

#     ambientLight from the raw files would be a real measurement, but the raw
#     data was deleted. Use --no-indoor to leave the column out.

# HOW TO RUN
#     python3 btp/42_build_context8.py

#     then, to check it before training:
#         python3 btp/32_user_leak_check.py \\
#             --labelled data/processed/labels/labelled_windows_ctx8.csv
# """

# import os
# import re
# import sys
# import argparse
# import numpy as np
# import pandas as pd

# LABELLED = "data/processed/labels/labelled_windows.csv"
# BIOMARKER_DIR = "data/processed/biomarkers"
# SURVEY = "mentor_files/Stress_and_STAI_Survey_Responses.xlsx"
# OUT = "data/processed/labels/labelled_windows_ctx8.csv"
# REPORT = "docs/63_context8.md"

# # What btp/34_train_users.py will not treat as context. Must match its
# # NOT_CONTEXT set, or a meta column would be fed to the gate.
# META = ["Subject_ID", "datetime_utc", "datetime_ist", "date",
#         "response_time", "STAI6", "LABEL", "LABEL_NAME"]

# # Already in the labelled file, derived by btp/30_derive_context.py
# FROM_CONTEXT = ["HOUR", "IS_WEEKEND", "POSTURE", "ACT_COUNT"]

# # From the Empatica biomarkers, one value a minute
# FROM_BIOMARKER = {
#     "activity_class": "ACT_CLASS",
#     "body_position_left": "BODY_POSITION",
#     "sleep_detection_stage": "SLEEP",
# }

# CONTEXT8 = FROM_CONTEXT + list(FROM_BIOMARKER.values()) + ["INDOOR_OUTDOOR"]

# # Anything here reaching the output is a bug. A signal must never be context.
# SIGNALS = {"PR", "RESP", "MET", "EDA_TONIC", "EDA_PHASIC_N", "EDA_PHASIC_AMP",
#            "SKIN_TEMP", "PRV_RMSSD", "PRV_SDNN", "AROUSAL_AT_REST", "MOVE_SD",
#            "BVP", "EDA_wrist", "Temp_wrist", "ACC_X_wrist", "ACC_Y_wrist",
#            "ACC_Z_wrist", "STAI6_RAW", "TILT", "ACT_INT"}


# def env_name(x):
#     s = str(x).strip().lower()
#     if s.startswith("indoor"):
#         return "Indoor"
#     if s.startswith("outdoor"):
#         return "Outdoor"
#     if s.startswith("travel"):
#         return "Travelling"
#     return "Unknown"


# def user_name(x):
#     m = re.findall(r"\d+", str(x))
#     return "user%d" % int(m[0]) if m else None


# def attach_biomarkers(lab, biomarker_dir):
#     """Carry each minute's value onto the two 30-second windows it covers."""
#     out, stats = [], []

#     for user, g in lab.groupby("Subject_ID", sort=True):
#         path = os.path.join(biomarker_dir, "%s_biomarkers.csv" % user)
#         g = g.copy()

#         if not os.path.exists(path):
#             print("  %-8s no biomarker file" % user)
#             for c in FROM_BIOMARKER.values():
#                 g[c] = np.nan
#             out.append(g)
#             stats.append({"user": user, "windows": len(g), "matched": 0})
#             continue

#         bio = pd.read_csv(path, usecols=lambda c: c in
#                           ({"timestamp_iso"} | set(FROM_BIOMARKER)))
#         bio["t"] = pd.to_datetime(bio["timestamp_iso"], errors="coerce",
#                                   utc=True).dt.tz_localize(None)
#         bio = bio.dropna(subset=["t"]).sort_values("t")
#         bio = bio.rename(columns=FROM_BIOMARKER)

#         g["_t"] = pd.to_datetime(g["datetime_utc"], errors="coerce")
#         g = g.sort_values("_t")

#         cols = [c for c in FROM_BIOMARKER.values() if c in bio.columns]
#         # backward with a 60 second tolerance: a minute's value covers the
#         # 60 seconds that follow it, which is two of our windows
#         merged = pd.merge_asof(g, bio[["t"] + cols], left_on="_t",
#                                right_on="t", direction="backward",
#                                tolerance=pd.Timedelta(seconds=60))
#         merged = merged.drop(columns=["_t", "t"])

#         n_match = int(merged[cols].notna().any(axis=1).sum()) if cols else 0
#         print("  %-8s %5d windows, %5d got a biomarker value (%.0f%%)"
#               % (user, len(merged), n_match, 100 * n_match / max(len(g), 1)))

#         out.append(merged)
#         stats.append({"user": user, "windows": int(len(merged)),
#                       "matched": n_match})

#     return pd.concat(out, ignore_index=True), stats


# def attach_environment(d, survey_path):
#     """The survey's environment answer, matched to each response."""
#     if not os.path.exists(survey_path):
#         print("  survey not found at %s" % survey_path)
#         return d, 0

#     sv = pd.read_excel(survey_path)
#     col = [c for c in sv.columns if "Current Environment" in str(c)]
#     if not col or "User" not in sv.columns or "Timestamp" not in sv.columns:
#         print("  survey has no usable environment question")
#         return d, 0

#     sv["ENV"] = sv[col[0]].map(env_name)
#     sv["User_"] = sv["User"].map(user_name)
#     sv["Timestamp"] = pd.to_datetime(sv["Timestamp"], errors="coerce")
#     sv = sv.dropna(subset=["Timestamp", "User_"])

#     resp = d[["Subject_ID", "response_time"]].drop_duplicates().copy()
#     resp["response_time"] = pd.to_datetime(resp["response_time"],
#                                            errors="coerce")

#     # the survey clock may sit 5.5 hours from response_time; try each and
#     # keep whichever matches most, rather than assuming
#     best = None
#     for hours in (0.0, -5.5, 5.5):
#         s = sv.copy()
#         s["_t"] = s["Timestamp"] + pd.Timedelta(hours=hours)
#         m = pd.merge_asof(resp.sort_values("response_time"),
#                           s[["User_", "_t", "ENV"]].sort_values("_t"),
#                           left_on="response_time", right_on="_t",
#                           left_by="Subject_ID", right_by="User_",
#                           tolerance=pd.Timedelta("2min"),
#                           direction="nearest")
#         n = int(m["ENV"].notna().sum())
#         if best is None or n > best[0]:
#             best = (n, hours, m)

#     n, hours, m = best
#     print("  responses %d, matched %d (survey clock %+.1f h)"
#           % (len(resp), n, hours))

#     d["response_time"] = pd.to_datetime(d["response_time"], errors="coerce")
#     d = d.merge(m[["Subject_ID", "response_time", "ENV"]],
#                 on=["Subject_ID", "response_time"], how="left")
#     d = d.rename(columns={"ENV": "INDOOR_OUTDOOR"})
#     d["INDOOR_OUTDOOR"] = d["INDOOR_OUTDOOR"].fillna("Unknown")
#     return d, n


# def main():
#     ap = argparse.ArgumentParser(
#         description="Build the eight-column context: four derived, three "
#                     "from the Empatica biomarkers, one from the survey.")
#     ap.add_argument("--labelled", default=LABELLED)
#     ap.add_argument("--biomarker-dir", default=BIOMARKER_DIR)
#     ap.add_argument("--survey", default=SURVEY)
#     ap.add_argument("--out", default=OUT)
#     ap.add_argument("--no-indoor", action="store_true",
#                     help="leave INDOOR_OUTDOOR out. It is a survey answer "
#                          "given alongside the STAI items the label comes "
#                          "from, so it carries a leak risk the other seven "
#                          "do not.")
#     args = ap.parse_args()

#     d = pd.read_csv(args.labelled)
#     print("=" * 70)
#     print("Building the eight-column context")
#     print("=" * 70)
#     print("read %s: %d rows, %d columns" % (args.labelled, len(d),
#                                             d.shape[1]))
#     print()

#     missing = [c for c in META + FROM_CONTEXT if c not in d.columns]
#     if missing:
#         sys.exit("the labelled file lacks: %s" % missing)

#     print("attaching the biomarkers, one value a minute held across two")
#     print("thirty-second windows:")
#     d, bstats = attach_biomarkers(d, args.biomarker_dir)
#     print()

#     want = list(CONTEXT8)
#     if args.no_indoor:
#         want.remove("INDOOR_OUTDOOR")
#         print("INDOOR_OUTDOOR left out, as asked")
#     else:
#         print("attaching the survey's environment answer:")
#         d, _ = attach_environment(d, args.survey)
#     print()

#     for c in want:
#         if c not in d.columns:
#             d[c] = np.nan

#     out = d[META + want].copy()

#     leaked = SIGNALS & set(out.columns)
#     assert not leaked, "signal columns reached the output: %s" % sorted(leaked)
#     seen = set(out.columns) - set(META)
#     assert seen == set(want), "context is %s" % sorted(seen)

#     os.makedirs(os.path.dirname(args.out), exist_ok=True)
#     out.to_csv(args.out, index=False)

#     print("=" * 70)
#     print("WHAT CAME OUT")
#     print("=" * 70)
#     print("wrote %s: %d rows" % (args.out, len(out)))
#     print("context columns (%d): %s" % (len(want), ", ".join(want)))
#     print()

#     print("coverage within the labelled windows, which is the figure that")
#     print("matters rather than coverage across all minutes:")
#     print()
#     for c in want:
#         n = out[c].notna().sum()
#         print("  %-16s %6d of %d  (%.1f%%)"
#               % (c, n, len(out), 100 * n / len(out)))
#     print()

#     bio_cols = [c for c in FROM_BIOMARKER.values() if c in out.columns]
#     if bio_cols:
#         any_bio = out[bio_cols].notna().any(axis=1).mean()
#         print("  windows with at least one biomarker value: %.1f%%"
#               % (100 * any_bio))
#         if any_bio > 0.45:
#             print("  Better than the 22 to 42 percent across all minutes, so")
#             print("  survey answers do cluster when the watch was worn.")
#         else:
#             print("  About the same as across all minutes, so the survey")
#             print("  answers are not concentrated in the worn periods.")
#         print()

#     for c in want:
#         if out[c].dtype == object:
#             vc = out[c].value_counts(dropna=False).head(6)
#             print("  %-16s %s" % (c, dict(vc)))
#     print()

#     if "INDOOR_OUTDOOR" in out.columns:
#         print("INDOOR_OUTDOOR against the label, the pairing to be careful")
#         print("about:")
#         print(pd.crosstab(out["INDOOR_OUTDOOR"],
#                           out["LABEL_NAME"]).to_string())
#         print()
#         print("and by user, because one user answering a single class can")
#         print("make an environment look predictive when it is not:")
#         print(pd.crosstab(out["Subject_ID"],
#                           out["INDOOR_OUTDOOR"]).to_string())
#         print()

#     print("Before training on this, run:")
#     print("  python3 btp/32_user_leak_check.py --labelled %s" % args.out)
#     print()

#     L = []
#     L.append("# The eight-column context")
#     L.append("")
#     L.append("Written by `btp/42_build_context8.py`.")
#     L.append("")
#     L.append("| column | source | rate |")
#     L.append("|---|---|---|")
#     L.append("| `HOUR` | timestamp | 30 s |")
#     L.append("| `IS_WEEKEND` | timestamp | 30 s |")
#     L.append("| `POSTURE` | our derivation from the wrist | 30 s |")
#     L.append("| `ACT_COUNT` | our derivation from the wrist | 30 s |")
#     L.append("| `ACT_CLASS` | Empatica biomarker | 1 min, held |")
#     L.append("| `BODY_POSITION` | Empatica biomarker | 1 min, held |")
#     L.append("| `SLEEP` | Empatica biomarker | 1 min, held |")
#     L.append("| `INDOOR_OUTDOOR` | survey answer | per response |")
#     L.append("")
#     L.append("## Coverage inside the labelled windows")
#     L.append("")
#     L.append("| column | present | percent |")
#     L.append("|---|---:|---:|")
#     for c in want:
#         n = int(out[c].notna().sum())
#         L.append("| `%s` | %d | %.1f%% |" % (c, n, 100 * n / len(out)))
#     L.append("")
#     L.append("## Two cautions, both measured")
#     L.append("")
#     L.append("**The three biomarkers disagreed with reality.** Checked")
#     L.append("against a 97 minute recording with activities logged by hand,")
#     L.append("`body_position_left` called sitting \"standing\" 33 times of")
#     L.append("56, `activity_intensity` called sitting \"vigorous\" 10 times,")
#     L.append("and `sleep_detection_stage` stayed 0 throughout the logged")
#     L.append("sleep. They are here because they were asked for.")
#     L.append("")
#     L.append("**`INDOOR_OUTDOOR` is a survey answer, not a sensor reading.**")
#     L.append("It is given in the same sitting and the same form as the six")
#     L.append("STAI items the label comes from, which puts it in the same")
#     L.append("category as \"main source of stress in the last hour\", a")
#     L.append("column excluded for exactly that reason. Run the leak check")
#     L.append("before using it.")
#     L.append("")

#     os.makedirs(os.path.dirname(REPORT), exist_ok=True)
#     open(REPORT, "w").write("\n".join(L) + "\n")
#     print("report: %s" % REPORT)


# if __name__ == "__main__":
#     main()




"""
42_build_context8.py

WHAT THIS DOES
    Builds the labelled-window file whose context is exactly the eight columns
    the mentor asked for, and nothing that is a signal:

        HOUR, IS_WEEKEND          the clock
        POSTURE                   from wrist tilt (btp/30_derive_context.py)
        ACT_COUNT                 from the accelerometer (same script)
        ACT_CLASS                 Empatica's activity_class
        BODY_POSITION             Empatica's body_position (left or right wrist)
        SLEEP                     Empatica's sleep_detection_stage
        INDOOR_OUTDOOR            the survey's "Current Environment" answer

    It starts from labelled_windows_ctx7.csv (written by 41_build_context7.py),
    which already holds the clock, POSTURE, ACT_COUNT and INDOOR_OUTDOOR for
    every labelled window, and adds the three Empatica columns to it.
    TILT and ACT_INT are dropped because the new list does not contain them.

HOW THE EMPATICA COLUMNS ARE JOINED
    The biomarker export is one row per minute (UTC). Each labelled window
    takes the row of the minute its start time falls in, so one value covers
    both 30 second slots of that minute. It is held, not interpolated: a value
    halfway between two activity classes means nothing.

    Where Empatica left the minute blank (watch not worn, or not touching the
    skin) the cell becomes the word "Unknown". That is a real state, but it is
    also information about the wearer, so this script prints whether it lines
    up with the label. Read that table before training.

WHAT IS HONEST ABOUT THEM
    * Coverage on all minutes is only 22 to 42 percent per user, because the
      watch was off or loose the rest of the time. What matters is the
      coverage over the labelled windows, printed below.
    * INDOOR_OUTDOOR is the survey answer, not a sensor reading. It exists
      only where a survey answer exists.
    * On the calibration recording, Empatica's body_position called sitting
      "standing" 33 times out of 56. Look at the value counts before trusting
      it.

HOW TO RUN
    python3 btp/42_build_context8.py 2>&1 | tee docs/logs/42_context8.log
"""
import argparse
import os
import sys

import pandas as pd

BASE = "data/processed/labels/labelled_windows_ctx7.csv"
BIO_DIR = "data/processed/biomarkers"
OUT = "data/processed/labels/labelled_windows_ctx8.csv"

META = ["Subject_ID", "datetime_utc", "datetime_ist", "date",
        "response_time", "STAI6", "LABEL", "LABEL_NAME"]
CONTEXT = ["HOUR", "IS_WEEKEND", "POSTURE", "ACT_COUNT", "ACT_CLASS",
           "BODY_POSITION", "SLEEP", "INDOOR_OUTDOOR"]

SIGNALS = {"PR", "RESP", "MET", "EDA_TONIC", "EDA_PHASIC_N", "EDA_PHASIC_AMP",
           "SKIN_TEMP", "PRV_RMSSD", "PRV_SDNN", "AROUSAL_AT_REST", "MOVE_SD",
           "BVP", "EDA_wrist", "Temp_wrist", "ACC_X_wrist", "ACC_Y_wrist",
           "ACC_Z_wrist", "STAI6_RAW", "pulse_rate_bpm", "respiratory_rate_brpm",
           "eda_scl_usiemens", "temperature_celsius", "prv_rmssd_ms", "met"}

NEEDED = ["activity_class", "body_position_left", "body_position_right",
          "sleep_detection_stage"]


def minute_key(s):
    return pd.to_datetime(s).dt.floor("min")


def load_bio(user):
    path = os.path.join(BIO_DIR, "%s_biomarkers.csv" % user)
    if not os.path.exists(path):
        sys.exit("missing %s" % path)
    b = pd.read_csv(path)
    miss = [c for c in NEEDED if c not in b.columns]
    if miss:
        sys.exit("%s lacks %s\nit has: %s" % (path, miss, list(b.columns)))
    if "timestamp_iso" in b.columns:
        t = pd.to_datetime(b["timestamp_iso"], utc=True).dt.tz_localize(None)
    elif "timestamp_unix" in b.columns:
        t = pd.to_datetime(b["timestamp_unix"], unit="ms")
    elif "datetime_utc" in b.columns:
        t = pd.to_datetime(b["datetime_utc"])
    else:
        sys.exit("no time column in %s" % path)
    out = pd.DataFrame({"Subject_ID": user, "_min": t.dt.floor("min")})
    out["ACT_CLASS"] = b["activity_class"]
    left, right = b["body_position_left"], b["body_position_right"]
    out["BODY_POSITION"] = left.where(left.notna(), right)
    out["_right_only"] = left.isna() & right.notna()
    out["SLEEP"] = b["sleep_detection_stage"]
    return out.drop_duplicates(["Subject_ID", "_min"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = pd.read_csv(args.base)
    print("read %s: %d rows" % (args.base, len(d)))
    d["_min"] = minute_key(d["datetime_utc"])

    bio = pd.concat([load_bio(u) for u in sorted(d["Subject_ID"].unique())])
    print("biomarker minutes loaded: %d (rows where body position came from "
          "the right-wrist column: %d)" % (len(bio), int(bio["_right_only"].sum())))
    d = d.merge(bio.drop(columns="_right_only"), on=["Subject_ID", "_min"],
                how="left")

    # ---- what we got, before blanks become "Unknown" ----
    print("\nCOVERAGE OVER THE %d LABELLED WINDOWS (value present, not blank)"
          % len(d))
    cov = d.groupby("Subject_ID")[["ACT_CLASS", "BODY_POSITION", "SLEEP"]] \
           .agg(lambda s: s.notna().mean()).round(2)
    cov["all"] = d[["ACT_CLASS", "BODY_POSITION", "SLEEP"]].notna().all(axis=1) \
                  .groupby(d["Subject_ID"]).mean().round(2)
    print(cov.to_string())
    print("overall: %.0f%% of labelled windows have ACT_CLASS"
          % (100 * d["ACT_CLASS"].notna().mean()))

    print("\nVALUES")
    for c in ["ACT_CLASS", "BODY_POSITION", "SLEEP"]:
        print("%s: %s" % (c, d[c].value_counts(dropna=False).to_dict()))

    print("\nIS A BLANK TELLING US THE LABEL? (windows with ACT_CLASS blank)")
    d["_blank"] = d["ACT_CLASS"].isna()
    print(pd.crosstab(d["_blank"], d["LABEL_NAME"], margins=True).to_string())

    for c in ["ACT_CLASS", "BODY_POSITION", "SLEEP"]:
        d[c] = d[c].astype(object).where(d[c].notna(), "Unknown")

    missing = [c for c in META + CONTEXT if c not in d.columns]
    if missing:
        sys.exit("lacking columns: %s" % missing)
    out = d[META + CONTEXT].copy()

    leaked = SIGNALS & set(out.columns)
    assert not leaked, "signal columns in output: %s" % sorted(leaked)
    assert set(out.columns) - set(META) == set(CONTEXT)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print("\nwrote %s: %d rows" % (args.out, len(out)))
    print("context columns (%d): %s" % (len(CONTEXT), ", ".join(CONTEXT)))


if __name__ == "__main__":
    main()