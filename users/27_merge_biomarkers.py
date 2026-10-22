"""
Merge EmbracePlus per-metric digital-biomarker CSVs (one file per metric)
into ONE wide CSV: one row per minute, one column per signal.

Usage:
    python merge_biomarkers.py <input_folder> <output_csv>
Works for any number of participants/days: files are grouped by
(participant, date), merged side-by-side, then stacked.
"""
import re, sys
from pathlib import Path
import pandas as pd

KEYS = ["timestamp_unix", "timestamp_iso", "participant_full_id"]
# filename pattern: <participant>_<YYYY-MM-DD>_<metric>.csv  (any prefix before is ignored)
PAT = re.compile(r"(\d+-\d+-\d+)_(\d{4}-\d{2}-\d{2})_([a-z0-9-]+)\.csv$")

def main(in_dir, out_csv):
    groups = {}
    for f in sorted(Path(in_dir).glob("*.csv")):
        m = PAT.search(f.name)
        if not m:
            print(f"skip (name not recognised): {f.name}")
            continue
        pid, day, metric = m.groups()
        groups.setdefault((pid, day), []).append((metric, f))

    merged_days = []
    for (pid, day), files in groups.items():
        merged = None
        for metric, f in files:
            df = pd.read_csv(f)
            # keep each file's reason column, renamed so they don't collide
            df = df.rename(columns={"missing_value_reason": f"{metric}__missing_reason"})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=KEYS, how="outer", validate="one_to_one")
        merged_days.append(merged)
        print(f"{pid} {day}: {len(files)} metrics -> {merged.shape}")

    out = pd.concat(merged_days, ignore_index=True)
    out = out.sort_values(["participant_full_id", "timestamp_unix"]).reset_index(drop=True)

    # move reason columns to the end so signals sit together
    reason_cols = [c for c in out.columns if c.endswith("__missing_reason")]
    out = out[[c for c in out.columns if c not in reason_cols] + reason_cols]

    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv}: {out.shape[0]} rows x {out.shape[1]} cols")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
