"""
30_derive_context.py

WHAT THIS IS
    Builds every contextual signal we need for one user, from their raw
    EmbracePlus files, at 30 second resolution.

    30 seconds is deliberate: it matches the stride of our 60 second windows,
    so each window gets exactly two context values and nothing has to be
    stretched or guessed.

WHERE THE CODE COMES FROM
    PORTED, unchanged, from the mentor's notebook
    `mentor_files/embrace_derived_modalities__2_.ipynb`:

        compute_pr            60 / IBI, mean per window
        compute_resp          RIFV: IBI resampled, band-passed 0.1-0.5 Hz,
                              peaks counted per window
        compute_act_count     agcounts if installed, else band-pass 0.25-2.5 Hz
                              then rectify and sum
        compute_act_int_met   ENMO, Hildebrand wrist cut-points, MET regression
        COLUMN_MAP, make_output_df, get_xyz, parse_dt

    Everything marked BTP-ADDED below is ours, and each one says why it was
    needed. In short: we checked Empatica's own biomarker export against a
    97 minute calibration recording, and several of its columns were either
    empty or disagreed with what the person was actually doing.

WHAT WE ADDED, AND WHY

    PRV (RMSSD and SDNN)
        How much the gap between heartbeats varies. This is the clearest
        cardiac stress signal there is: heart rate up, variability down.
        The biomarker export had prv_rmssd_ms for 22 minutes out of 2880,
        which is 0.8 percent. Derived from the beat timings it is dense.

    EDA tonic and phasic
        The biomarker column eda_scl_usiemens had 2 non-zero values in 2880
        minutes. But the raw EDA file is 99.7 percent non-zero and varies
        properly, and the Chronos tokeniser found 3336 distinct tokens in it,
        so the sensor is fine. Their computed column is the problem, not the
        signal.

        Deriving it ourselves also gets us something their column never had:
        the phasic part, meaning the fast spikes. The slow tonic level is a
        baseline; the spikes are the actual arousal response.

    Posture
        Empatica's body_position called the calibration sitting "standing"
        33 times out of 56. We work it out from the wrist tilt instead, using
        the direction of gravity, which is what actually distinguishes an arm
        hanging down from one resting on a desk.

    Arousal at rest
        Pulse rate divided by movement. During the calibration sleep period
        movement was 51 and pulse was 86; during sitting movement was 66 and
        pulse was 79. So the person moved LESS while asleep but their heart
        ran FASTER than while sitting.

        Neither channel alone can see that. This one can, and it is exactly
        the distinction the context gate exists for: a racing heart because
        of exercise versus a racing heart because of stress.

    Clock
        Hour of day and whether it is a weekend. Free from the timestamps,
        and it cannot secretly be the label the way WESAD's Task column was.

OUTPUT
    data/processed/context/<user>_context.csv
        one row per 30 seconds, every context column

    docs/55_context_<user>.md
        coverage per channel and the correlation table

HOW TO RUN
    One user at a time, so only one user's raw files need to be on disk:

        python3 btp/30_derive_context.py --user user17

    Then check the report, delete that user's raw folder, and move to the
    next one.
"""

import os
import re
import glob
import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)

RAW_DIR = "data/raw/embraceplus"
OUT_DIR = "data/processed/context"

# ---------------------------------------------------------------------------
# PORTED from the mentor's notebook, cells 2 and 6
# ---------------------------------------------------------------------------
IST_OFFSET = timedelta(hours=5, minutes=30)

COLUMN_MAP = {
    "accelerometer": {"time": "datetime_utc", "x": "accelerometer_x",
                      "y": "accelerometer_y", "z": "accelerometer_z"},
    "bvp":           {"time": "datetime_utc", "value": "bvp"},
    "eda":           {"time": "datetime_utc", "value": "eda"},
    "steps":         {"time": "datetime_utc", "value": "steps"},
    "systolicPeaks": {"time": "datetime_utc"},
    "temperature":   {"time": "datetime_utc", "value": "temperature"},
}

ACCEL_FS = 64
EDA_FS = 4
WINDOW_SEC = 30          # matches our window stride, so nothing is stretched


def parse_dt(series):
    """PORTED. Turn a datetime_utc column into real timestamps."""
    return pd.to_datetime(series, errors="coerce")


def make_output_df(window_start_utc, value_col_name, values):
    """PORTED. Build a frame matching the existing file schema."""
    dt_utc = pd.to_datetime(pd.Series(window_start_utc))
    out = pd.DataFrame({
        "datetime_utc": dt_utc,
        value_col_name: values,
    })
    return out


def get_xyz(accel_df):
    """PORTED."""
    c = COLUMN_MAP["accelerometer"]
    missing = [x for x in (c["x"], c["y"], c["z"]) if x not in accel_df.columns]
    if missing:
        raise KeyError("Accelerometer columns %s not found. Have: %s"
                       % (missing, list(accel_df.columns)))
    return accel_df[[c["x"], c["y"], c["z"]]].values


# ---------------------------------------------------------------------------
# PORTED from the mentor's notebook, cell 10
# ---------------------------------------------------------------------------
# BTP-ADDED. Two guards the ported functions need on real data.
#
# 1. A corrupt timestamp. One file for user17 held a beat time that was years
#    away from the rest, so the span from first beat to last came out as about
#    4.8 billion seconds. compute_resp then tried to build a 4 Hz grid across
#    it and asked for 142 GiB of memory.
#
# 2. Gaps when the watch was off. Even with clean timestamps, building one
#    grid from the first beat to the last means interpolating straight through
#    hours of nothing. That is inventing data. Real recordings come in
#    stretches, and each stretch should be handled on its own.
# BTP-ADDED. A beat gap outside this is a detection failure, not a heartbeat.
# 40 to 180 bpm is 0.333 to 1.5 seconds between beats.
#
# Why this is needed: on user17, PR came out as low as 0.8 bpm, which is one
# beat every 75 seconds, and PRV_RMSSD averaged 291 ms where a resting adult
# is 20 to 50 ms. Both are the same artefact. During movement the optical
# sensor loses clean contact and misses beats, so two real beats are read as
# one long gap, which looks like a very slow heart.
#
# The mentor's compute_pr has no such filter. Our compute_prv already had
# one, and btp/27_derive_pr.py used the same range, so this makes PR
# consistent with both.
MIN_IBI_SEC, MAX_IBI_SEC = 0.333, 1.5

MAX_GAP_SEC = 60.0          # longer than this and the sensor was not reading
MIN_STRETCH_BEATS = 20      # too short to say anything about breathing


def clean_peak_times(systolic_df):
    """Return sorted beat times with impossible ones removed.

    Anything more than a day away from the median is a broken timestamp, not
    a heartbeat.
    """
    if systolic_df is None or len(systolic_df) < 2:
        return None, 0

    t = parse_dt(systolic_df[COLUMN_MAP["systolicPeaks"]["time"]]).dropna()
    if len(t) < 2:
        return None, 0

    t = t.sort_values().reset_index(drop=True)
    centre = t.median()
    ok = (t - centre).abs() <= pd.Timedelta(days=1)
    dropped = int((~ok).sum())
    t = t[ok].reset_index(drop=True)
    return (t if len(t) >= 2 else None), dropped


def split_stretches(t_dt, max_gap=MAX_GAP_SEC):
    """Break the beat times into continuous stretches.

    A gap longer than max_gap means the watch stopped reading, so the two
    sides are separate recordings and must not be joined.
    """
    gaps = t_dt.diff().dt.total_seconds()
    breaks = list(np.where(gaps > max_gap)[0])
    bounds = [0] + breaks + [len(t_dt)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 2:
            out.append(t_dt.iloc[a:b].reset_index(drop=True))
    return out


def compute_pr(systolic_df, window_sec=WINDOW_SEC):
    """PORTED, unchanged. PR = 60 / IBI, averaged per window."""
    if systolic_df is None or len(systolic_df) < 2:
        return make_output_df([], "PR", [])

    # BTP-ADDED: drop broken timestamps first
    t_dt, _ = clean_peak_times(systolic_df)
    if t_dt is None:
        return make_output_df([], "PR", [])

    t0 = t_dt.iloc[0]
    t_sec = (t_dt - t0).dt.total_seconds().values

    ibi = np.diff(t_sec)
    # BTP-ADDED: was `valid = ibi > 0`, which let a 75 second gap through as
    # a 0.8 bpm heartbeat.
    valid = (ibi >= MIN_IBI_SEC) & (ibi <= MAX_IBI_SEC)
    ibi = ibi[valid]
    if len(ibi) == 0:
        return make_output_df([], "PR", [])

    inst_pr = 60.0 / ibi
    peak_sec = t_sec[1:][valid]

    win = (peak_sec // window_sec).astype(int)
    # BTP-ADDED: a 30 second window should hold at least 15 beats at any
    # plausible heart rate. Fewer means detection failed, so report nothing
    # rather than a wrong number.
    tmp = pd.DataFrame({"win": win, "PR": inst_pr})
    counts = tmp.groupby("win")["PR"].size()
    tmp = tmp[tmp["win"].isin(counts[counts >= 15].index)]
    if tmp.empty:
        return make_output_df([], "PR", [])
    agg = tmp.groupby("win")["PR"].mean().reset_index()
    starts = [t0 + timedelta(seconds=int(w) * window_sec) for w in agg["win"]]
    return make_output_df(starts, "PR", agg["PR"].values)


# ---------------------------------------------------------------------------
# PORTED from the mentor's notebook, cell 12
# ---------------------------------------------------------------------------
def compute_resp(systolic_df, window_sec=WINDOW_SEC, fs_target=4.0):
    """PORTED, unchanged. Breathing pulled out of the beat timing.

    Breathing is not measured by the device, but it is hidden inside the heart
    rhythm: the gap between beats shortens on the in-breath and lengthens on
    the out-breath. Band-pass the beat gaps to 0.1-0.5 Hz, which is 6 to 30
    breaths a minute, and count the peaks.
    """
    # BTP-ADDED: clean the timestamps, then handle each continuous stretch
    # separately. The original built one grid from the first beat to the last,
    # which breaks in two ways on real data: a single corrupt timestamp asks
    # for an impossible amount of memory, and gaps when the watch was off get
    # interpolated straight through. The filtering below is the mentor's,
    # unchanged; only the looping around it is new.
    t_all, _ = clean_peak_times(systolic_df)
    if t_all is None or len(t_all) < MIN_STRETCH_BEATS:
        return make_output_df([], "RESP", [])

    nyq = fs_target / 2.0
    b, a = butter(2, [0.1 / nyq, min(0.5 / nyq, 0.99)], btype="band")
    n_per_win = int(window_sec * fs_target)

    starts, vals = [], []

    for t_dt in split_stretches(t_all):
        if len(t_dt) < MIN_STRETCH_BEATS:
            continue

        t0 = t_dt.iloc[0]
        t_sec = (t_dt - t0).dt.total_seconds().values
        ibi = np.diff(t_sec)
        peak_sec = t_sec[1:]
        if len(ibi) < 4:
            continue

        g0, g1 = peak_sec[0], peak_sec[-1]
        if g1 <= g0:
            continue

        grid = np.arange(g0, g1, 1.0 / fs_target)
        if len(grid) < n_per_win + 2:
            continue

        ibi_interp = np.interp(grid, peak_sec, ibi)
        filtered = filtfilt(b, a, ibi_interp)

        for s in range(0, len(filtered) - n_per_win, n_per_win):
            seg = filtered[s:s + n_per_win]
            peaks, _ = find_peaks(seg, distance=fs_target * 1.5)
            vals.append(len(peaks) * (60.0 / window_sec))
            starts.append(t0 + timedelta(seconds=float(grid[s])))

    return make_output_df(starts, "RESP", vals)


# ---------------------------------------------------------------------------
# PORTED from the mentor's notebook, cell 14
# ---------------------------------------------------------------------------
try:
    from agcounts.extract import get_counts
    HAVE_AGCOUNTS = True
except ImportError:
    HAVE_AGCOUNTS = False


def compute_act_count(accel_df, window_sec=WINDOW_SEC, fs=ACCEL_FS):
    """PORTED, unchanged. ActiGraph-style activity counts."""
    if accel_df is None or len(accel_df) < fs:
        return make_output_df([], "ACT_COUNT", [])

    t_dt = parse_dt(accel_df[COLUMN_MAP["accelerometer"]["time"]]) \
        .reset_index(drop=True)
    t0 = t_dt.iloc[0]
    xyz = get_xyz(accel_df)

    if HAVE_AGCOUNTS:
        counts = get_counts(xyz, freq=fs, epoch=window_sec)
        vm = np.sqrt((counts ** 2).sum(axis=1))
        starts = [t0 + timedelta(seconds=i * window_sec) for i in range(len(vm))]
        return make_output_df(starts, "ACT_COUNT", vm)

    nyq = fs / 2.0
    b, a = butter(4, [0.25 / nyq, min(2.5 / nyq, 0.99)], btype="band")
    vm = np.sqrt((xyz ** 2).sum(axis=1))
    rect = np.abs(filtfilt(b, a, vm))

    n_per_win = int(window_sec * fs)
    starts, vals = [], []
    for s in range(0, len(rect) - n_per_win, n_per_win):
        vals.append(rect[s:s + n_per_win].sum())
        starts.append(t0 + timedelta(seconds=s / fs))
    return make_output_df(starts, "ACT_COUNT", vals)


# ---------------------------------------------------------------------------
# PORTED from the mentor's notebook, cell 16
# ---------------------------------------------------------------------------
def compute_act_int_met(accel_df, window_sec=WINDOW_SEC, fs=ACCEL_FS):
    """PORTED, unchanged. ENMO, Hildebrand cut-points, MET.

    Note from the mentor's own notebook: the MET coefficients 0.972 + 12.24
    are an approximation of the Hildebrand form and should be checked against
    the published equation before being cited.
    """
    empty = (make_output_df([], "ACT_INT", []), make_output_df([], "MET", []))
    if accel_df is None or len(accel_df) < fs:
        return empty

    t_dt = parse_dt(accel_df[COLUMN_MAP["accelerometer"]["time"]]) \
        .reset_index(drop=True)
    t0 = t_dt.iloc[0]
    xyz = get_xyz(accel_df)
    vm = np.sqrt((xyz ** 2).sum(axis=1))
    enmo = np.maximum(vm - 1.0, 0.0)          # assumes units of g

    n_per_win = int(window_sec * fs)
    starts, vals = [], []
    for s in range(0, len(enmo) - n_per_win, n_per_win):
        vals.append(enmo[s:s + n_per_win].mean())
        starts.append(t0 + timedelta(seconds=s / fs))

    if not vals:
        return empty

    enmo_mg = np.array(vals) * 1000

    def classify(x):
        if x < 47.4:
            return "sedentary"
        if x < 69.1:
            return "light"
        if x < 258.7:
            return "moderate"
        return "vigorous"

    act_int = [classify(x) for x in enmo_mg]
    met = 0.972 + 12.24 * np.array(vals)
    return (make_output_df(starts, "ACT_INT", act_int),
            make_output_df(starts, "MET", met))


# ---------------------------------------------------------------------------
# BTP-ADDED. Not in the mentor's notebook.
# ---------------------------------------------------------------------------
def compute_prv(systolic_df, window_sec=WINDOW_SEC):
    """BTP-ADDED. How much the gap between heartbeats varies.

    Why: this is the clearest cardiac stress signal. Under stress the heart
    speeds up AND becomes more regular, so variability falls. The biomarker
    export gave prv_rmssd_ms for 22 minutes out of 2880, which is unusable.
    From the beat timings it is dense.

    RMSSD: the typical size of the change from one beat to the next.
    SDNN:  the overall spread of the gaps in the window.
    """
    if systolic_df is None or len(systolic_df) < 3:
        return make_output_df([], "PRV_RMSSD", [])

    # BTP-ADDED: drop broken timestamps first
    t_dt, _ = clean_peak_times(systolic_df)
    if t_dt is None or len(t_dt) < 3:
        return make_output_df([], "PRV_RMSSD", [])

    t0 = t_dt.iloc[0]
    t_sec = (t_dt - t0).dt.total_seconds().values
    ibi_ms = np.diff(t_sec) * 1000.0
    peak_sec = t_sec[1:]

    # same range as PR above
    ok = (ibi_ms >= MIN_IBI_SEC * 1000) & (ibi_ms <= MAX_IBI_SEC * 1000)
    ibi_ms, peak_sec = ibi_ms[ok], peak_sec[ok]
    if len(ibi_ms) < 3:
        return make_output_df([], "PRV_RMSSD", [])

    win = (peak_sec // window_sec).astype(int)
    df = pd.DataFrame({"win": win, "ibi": ibi_ms})

    rows = []
    for w, g in df.groupby("win"):
        # BTP-ADDED: same 15 beat gate as PR. RMSSD computed from 3 shaky
        # beats is noise, and that is what inflated it to 291 ms.
        if len(g) < 15:
            continue
        d = np.diff(g["ibi"].values)
        rows.append({
            "win": int(w),
            "PRV_RMSSD": float(np.sqrt(np.mean(d ** 2))),
            "PRV_SDNN": float(np.std(g["ibi"].values)),
        })
    if not rows:
        return make_output_df([], "PRV_RMSSD", [])

    r = pd.DataFrame(rows)
    out = pd.DataFrame({
        "datetime_utc": [t0 + timedelta(seconds=int(w) * window_sec)
                         for w in r["win"]],
        "PRV_RMSSD": r["PRV_RMSSD"].values,
        "PRV_SDNN": r["PRV_SDNN"].values,
    })
    return out


def compute_eda_context(eda_df, window_sec=WINDOW_SEC, fs=EDA_FS):
    """BTP-ADDED. Split EDA into its slow and fast parts.

    Why: the biomarker column eda_scl_usiemens had 2 non-zero values in 2880
    minutes, but the raw EDA file is 99.7 percent non-zero and the Chronos
    tokeniser found 3336 distinct tokens in it. The sensor is fine; their
    computed column is not.

    Deriving it ourselves also gets the fast part, which their column never
    had:

      tonic  the slow baseline, how sweaty the person is overall
      phasic the fast spikes, which are the actual arousal responses

    The spikes are the stress-relevant part.
    """
    cols = ["EDA_TONIC", "EDA_PHASIC_N", "EDA_PHASIC_AMP"]
    if eda_df is None or len(eda_df) < fs * 4:
        return pd.DataFrame(columns=["datetime_utc"] + cols)

    t_dt = parse_dt(eda_df[COLUMN_MAP["eda"]["time"]]).reset_index(drop=True)
    v = pd.to_numeric(eda_df[COLUMN_MAP["eda"]["value"]],
                      errors="coerce").to_numpy(dtype=float)
    t0 = t_dt.iloc[0]

    # tonic: slow baseline, a rolling median over 30 seconds
    tonic = pd.Series(v).rolling(int(fs * window_sec), min_periods=1,
                                 center=True).median().to_numpy()
    # phasic: what is left once the baseline is removed
    phasic = v - tonic

    n_per_win = int(window_sec * fs)
    starts, rows = [], []
    for s in range(0, len(v) - n_per_win + 1, n_per_win):
        seg_t = tonic[s:s + n_per_win]
        seg_p = phasic[s:s + n_per_win]
        if np.isnan(seg_t).all():
            continue
        # a response is a spike above a small threshold
        pk, props = find_peaks(np.nan_to_num(seg_p), height=0.01,
                               distance=int(fs * 1.0))
        rows.append({
            "EDA_TONIC": float(np.nanmean(seg_t)),
            "EDA_PHASIC_N": int(len(pk)),
            "EDA_PHASIC_AMP": float(props["peak_heights"].mean())
                              if len(pk) else 0.0,
        })
        starts.append(t0 + timedelta(seconds=s / fs))

    if not rows:
        return pd.DataFrame(columns=["datetime_utc"] + cols)

    out = pd.DataFrame(rows)
    out.insert(0, "datetime_utc", pd.to_datetime(pd.Series(starts)))
    return out


def compute_posture(accel_df, window_sec=WINDOW_SEC, fs=ACCEL_FS):
    """BTP-ADDED. Sitting, standing or walking, from wrist tilt.

    Why: Empatica's body_position called the calibration sitting "standing"
    33 times out of 56, so it cannot be trusted.

    How: gravity always pulls down, so the average direction of the
    accelerometer over a window tells us how the wrist is held. An arm hanging
    by the side points one way; a forearm resting on a desk points another.
    Movement is measured separately as the spread around that average.

    The tilt cut-offs are a starting point from the calibration recording and
    should be refined with more logged activity. The movement cut-off matters
    more and is better grounded.
    """
    cols = ["POSTURE", "TILT", "MOVE_SD"]
    if accel_df is None or len(accel_df) < fs:
        return pd.DataFrame(columns=["datetime_utc"] + cols)

    t_dt = parse_dt(accel_df[COLUMN_MAP["accelerometer"]["time"]]) \
        .reset_index(drop=True)
    t0 = t_dt.iloc[0]
    xyz = get_xyz(accel_df).astype(float)
    vm = np.sqrt((xyz ** 2).sum(axis=1))

    n_per_win = int(window_sec * fs)
    starts, rows = [], []
    for s in range(0, len(xyz) - n_per_win + 1, n_per_win):
        seg = xyz[s:s + n_per_win]
        g = seg.mean(axis=0)                     # direction of gravity
        norm = np.linalg.norm(g)
        # angle of the forearm away from vertical, 0 = hanging down
        tilt = float(np.degrees(np.arccos(np.clip(abs(g[2]) / (norm + 1e-9),
                                                  0, 1))))
        move = float(np.std(vm[s:s + n_per_win]))

        if move > 0.12:
            posture = "walking"
        elif tilt < 40:
            posture = "standing"
        else:
            posture = "sitting"

        rows.append({"POSTURE": posture, "TILT": round(tilt, 1),
                     "MOVE_SD": round(move, 4)})
        starts.append(t0 + timedelta(seconds=s / fs))

    if not rows:
        return pd.DataFrame(columns=["datetime_utc"] + cols)

    out = pd.DataFrame(rows)
    out.insert(0, "datetime_utc", pd.to_datetime(pd.Series(starts)))
    return out


# ---------------------------------------------------------------------------
def load_modality(udir, prefix, date, modality):
    path = os.path.join(udir, "%s_%s_%s.csv" % (prefix, date, modality))
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def process_date(udir, prefix, date):
    """Everything for one user-day, merged onto one 30 second grid."""
    accel = load_modality(udir, prefix, date, "accelerometer")
    syst = load_modality(udir, prefix, date, "systolicPeaks")

    # BTP-ADDED: say so when a file holds broken timestamps
    if syst is not None:
        _, n_bad = clean_peak_times(syst)
        if n_bad:
            print("      %s: dropped %d beat timestamps more than a day from "
                  "the rest" % (date, n_bad))
    eda = load_modality(udir, prefix, date, "eda")
    temp = load_modality(udir, prefix, date, "temperature")

    pieces = []

    # ---- ported ----
    pieces.append(compute_pr(syst))
    pieces.append(compute_resp(syst))
    if accel is not None:
        pieces.append(compute_act_count(accel))
        ai, met = compute_act_int_met(accel)
        pieces += [ai, met]

    # ---- BTP-ADDED ----
    pieces.append(compute_prv(syst))
    pieces.append(compute_eda_context(eda))
    if accel is not None:
        pieces.append(compute_posture(accel))

    # temperature: plain mean per window
    if temp is not None:
        t = parse_dt(temp[COLUMN_MAP["temperature"]["time"]])
        v = pd.to_numeric(temp[COLUMN_MAP["temperature"]["value"]],
                          errors="coerce")
        s = pd.Series(v.values, index=t).dropna()
        if len(s):
            r = s.resample("%ds" % WINDOW_SEC).mean().dropna()
            pieces.append(pd.DataFrame({"datetime_utc": r.index,
                                        "SKIN_TEMP": r.values}))

    pieces = [p for p in pieces if p is not None and len(p)]
    if not pieces:
        return None

    # snap every piece to the same 30 second grid before merging, so small
    # differences in start time do not create duplicate rows
    out = None
    for p in pieces:
        p = p.copy()
        p["datetime_utc"] = pd.to_datetime(p["datetime_utc"]) \
            .dt.floor("%ds" % WINDOW_SEC)
        p = p.groupby("datetime_utc", as_index=False).first()
        out = p if out is None else pd.merge(out, p, on="datetime_utc",
                                             how="outer")

    out = out.sort_values("datetime_utc").reset_index(drop=True)
    out["date"] = date
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Derive every context signal for one user, at 30 second "
                    "resolution.")
    ap.add_argument("--user", required=True)
    ap.add_argument("--raw-dir", default=RAW_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--drop-redundant", action="store_true",
                    help="drop the channels that measured as near-duplicates "
                         "on user17, so every user gets the same final set")
    args = ap.parse_args()

    udir = os.path.join(args.raw_dir, args.user)
    if not os.path.isdir(udir):
        raise SystemExit("No folder: %s" % udir)
    os.makedirs(args.out_dir, exist_ok=True)

    prefix, dates = None, set()
    for f in glob.glob(os.path.join(udir, "*.csv")):
        m = re.search(r"(.+)_(\d{4}-\d{2}-\d{2})_(\w+)\.csv$",
                      os.path.basename(f))
        if m:
            prefix = m.group(1)
            dates.add(m.group(2))
    if not dates:
        raise SystemExit("No dated files in %s" % udir)

    print("=" * 70)
    print("Deriving context for %s" % args.user)
    print("window   : %d s  (matches our window stride)" % WINDOW_SEC)
    print("agcounts : %s" % ("installed, using the real ActiGraph algorithm"
                             if HAVE_AGCOUNTS else
                             "NOT installed, using the band-pass "
                             "approximation (pip install agcounts)"))
    print("=" * 70)
    print()

    frames = []
    for date in sorted(dates):
        got = process_date(udir, prefix, date)
        if got is None or got.empty:
            print("  %s: nothing usable" % date)
            continue
        print("  %s: %5d rows (%.1f h)" % (date, len(got),
                                           len(got) * WINDOW_SEC / 3600.0))
        frames.append(got)

    if not frames:
        raise SystemExit("Nothing produced.")

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "Subject_ID", args.user)

    # ---- BTP-ADDED: arousal at rest, and the clock ----
    #
    # In the calibration recording, sleep had movement 51 and pulse 86, while
    # sitting had movement 66 and pulse 79. So the person moved LESS asleep
    # but their heart ran FASTER than sitting. Neither channel alone sees
    # that. This one does, and it is exactly what separates a racing heart
    # from exercise and a racing heart from stress.
    if "PR" in df.columns and "ACT_COUNT" in df.columns:
        # First attempt was PR / (ACT_COUNT + 1). On user17 that came out at
        # -0.92 against ACT_COUNT, because dividing by movement makes the
        # movement term dominate: it was inverse movement under another name.
        #
        # The idea is still the right one. We want the part of the pulse that
        # movement does NOT explain, so that a fast heart while sitting still
        # looks different from a fast heart while walking.
        #
        # So: fit a straight line predicting pulse from movement, then keep
        # what is left over. Positive means the heart is faster than this
        # person's movement would predict. Being a residual, it cannot
        # correlate with movement by construction.
        m = df["ACT_COUNT"].notna() & df["PR"].notna()
        if m.sum() > 50:
            x = np.log1p(df.loc[m, "ACT_COUNT"].to_numpy(dtype=float))
            y = df.loc[m, "PR"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            df.loc[m, "AROUSAL_AT_REST"] = y - (slope * x + intercept)
            print("  arousal-at-rest: pulse minus the part movement explains")
            print("    fit  PR = %.2f * log1p(ACT_COUNT) + %.2f"
                  % (slope, intercept))
        else:
            df["AROUSAL_AT_REST"] = np.nan

    ist = df["datetime_utc"] + pd.Timedelta(IST_OFFSET)
    df["HOUR"] = ist.dt.hour
    df["IS_WEEKEND"] = (ist.dt.dayofweek >= 5).astype(int)
    df["datetime_ist"] = ist

    # Channels that measured above 0.85 against something we keep, on
    # user17's 22,831 rows. Dropping them here means every user ends up with
    # the same final column set.
    REDUNDANT = {
        "MOVE_SD":        "0.97 with ACT_COUNT",
        "PRV_SDNN":       "0.91 with PRV_RMSSD",
        "EDA_PHASIC_AMP": "0.98 with EDA_PHASIC_N",
        # Dropped after looking at the raw beat timings for user17.
        #
        # PRV_RMSSD came out at about 290 ms everywhere, where a resting
        # adult is 20 to 50 ms. It is not an aggregation problem: the beat
        # intervals themselves jump by a median of 122 ms from one beat to
        # the next, and a quarter of them jump by more than 300 ms. The
        # average over a window is fine, which is why PR works, but the
        # individual timings are not precise enough for beat-to-beat
        # variability.
        #
        # Empatica appear to know this. Their own prv_rmssd_ms column was
        # populated for 0.8% of minutes, presumably only the windows that
        # passed their quality check.
        "PRV_RMSSD":      "beat timings too noisy, ~290 ms vs 20-50 normal",
        # 0.90 with PR. The residual idea fixed the movement problem but
        # left something that is essentially pulse rate again.
        "AROUSAL_AT_REST": "0.90 with PR",
    }
    if args.drop_redundant:
        gone = [c for c in REDUNDANT if c in df.columns]
        df = df.drop(columns=gone)
        print()
        print("dropped as near-duplicates (measured on user17):")
        for c in gone:
            print("  %-16s %s" % (c, REDUNDANT[c]))

    path = os.path.join(args.out_dir, "%s_context.csv" % args.user)
    df.to_csv(path, index=False)

    print()
    print("=" * 70)
    print("COVERAGE  (how often each channel actually has a value)")
    print("=" * 70)
    print("  rows: %d  (%.1f hours at %d s each)"
          % (len(df), len(df) * WINDOW_SEC / 3600.0, WINDOW_SEC))
    print()
    chans = [c for c in df.columns
             if c not in ("Subject_ID", "datetime_utc", "datetime_ist",
                          "date", "HOUR", "IS_WEEKEND")]
    print("  %-18s %9s %9s  %s" % ("channel", "present", "percent", "source"))
    print("  " + "-" * 56)
    ported = {"PR", "RESP", "ACT_COUNT", "ACT_INT", "MET"}
    for c in chans:
        n = df[c].notna().sum()
        src = "ported" if c in ported else "BTP-ADDED"
        print("  %-18s %9d %8.1f%%  %s" % (c, n, 100.0 * n / len(df), src))
    print()

    num = df[chans].select_dtypes(include=[np.number])
    num = num.loc[:, num.notna().sum() > 20]
    if len(num.columns) > 1:
        print("=" * 70)
        print("CORRELATION  (Spearman, pairs where both are present)")
        print("=" * 70)
        print("Channels above about 0.85 are near-duplicates: keep one.")
        print()
        cm = num.corr(method="spearman").round(2)
        print(cm.to_string())
        print()
        high = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns)
                for b in cm.columns[i + 1:] if abs(cm.loc[a, b]) > 0.85]
        if high:
            print("  near-duplicate pairs:")
            for a, b, v in high:
                print("    %-18s %-18s %5.2f" % (a, b, v))
        else:
            print("  no pair above 0.85, all channels carry something of")
            print("  their own")
        print()

    if "POSTURE" in df.columns:
        print("posture counts:", dict(df["POSTURE"].value_counts()))
    if "ACT_INT" in df.columns:
        print("activity intensity:", dict(df["ACT_INT"].value_counts()))
    print()
    print("saved %s" % path)


if __name__ == "__main__":
    main()
