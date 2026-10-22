"""
05_train.py

WHY THIS FILE EXISTS
    The training code is in the repo, in Stress.ipynb cells 4 to 8. We cannot
    run it as it is, for three reasons:

      1. vendor/adastress is read-only on purpose.
      2. It slides its window over the whole stacked CSV, so at every join
         between two subjects a window holds rows from two different people.
      3. It splits train from validation randomly with train_test_split, so
         the same person lands on both sides. Our audit measured Cramer's V
         of 0.277 between subject and label, so a random split flatters the
         model. It also has no test set: it reports the same validation
         numbers it selected on.

    So this file copies those cells as closely as possible and fixes only
    those things. Every difference is marked BTP-CHANGE with a repair number.

WHAT IT DOES
    1. Trains the encoder with self-supervised contrastive learning, on the
       train subjects only, no labels used.
    2. Freezes the encoder and trains the gated context classifier.
    3. Scores it on the held-out test subjects.
    4. Writes everything into one self-contained run folder.

CONTEXT MODES
    --context-mode none
        No context at all. The context vector is a single zero. This is the
        honest number, because the audit found Task predicts the label with
        100% accuracy.

    --context-mode original
        Task, SessionPhase and Activity, one-hot, exactly as the repo does.
        Expect near 100% accuracy. That is a measurement of leakage, not
        performance, and must be reported as such.

    --context-mode zero
        Experiment A. The context schema is widened to include the columns
        user1 has and WESAD does not (steps, MET, ACT_COUNT, ACT_INT). For
        WESAD those are all zero. One-hot is used so that an all-zero block
        is not any category, which is what makes "absent" unambiguous.

    --context-mode derived
        Experiment B. Movement statistics worked out from the wrist
        accelerometer: the average and the spread of movement size across the
        window. The same calculation can run on user1, which is why this is
        the only context that can transfer.

OUTPUT
    runs/<run-id>/
        config.json        every setting used
        env.txt            versions, seed, commit, encoder hash
        encoder.pth        the pretrained encoder
        scaler.pkl         the feature scaler, fitted on train subjects only
        sig_cols.json      which columns were used as signals
        classifier.pth     the gated context classifier
        metrics.json       all scores, in a fixed shape for every run
        predictions.csv    per-window probabilities and stress score
        log.txt            everything printed

HOW TO RUN
    R01, no context. This one also trains the encoder that the others reuse:

        python3 btp/05_train.py --run-id R01_wesad_nocontext \
            --assembled-dir data/assembled/block --context-mode none

    R02 and R03 reuse R01's encoder, so only the context branch differs:

        python3 btp/05_train.py --run-id R02_wesad_context_original \
            --assembled-dir data/assembled/block --context-mode original \
            --encoder-from runs/R01_wesad_nocontext

        python3 btp/05_train.py --run-id R03_wesad_context_derived \
            --assembled-dir data/assembled/block --context-mode derived \
            --encoder-from runs/R01_wesad_nocontext

    Add --drop-skew to leave out the 9 _skew columns (see decision D10).

    A run folder is never reused. Change a setting, use a new run id.
"""

import os
import sys
import json
import argparse
import hashlib
import platform
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix,
                             classification_report)
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# SETTINGS
# PORTED FROM: vendor/adastress/Stress.ipynb, cell 1
# =============================================================================
WINDOW_SEC = 15
STRIDE_SEC = 15
FS = 1
TW = WINDOW_SEC * FS          # 15 rows per window
SH = STRIDE_SEC * FS

BATCH_SIZE_PRETRAIN = 128
EPOCHS_PRETRAIN = 15
LR_PRETRAIN = 1e-3
TEMPERATURE = 0.1
BATCH_SIZE_CLASSIF = 64
EPOCHS_CLASSIF = 30
LR_CLASSIF = 1e-3

LABEL_NAMES = ["LowStress", "MediumStress", "HighStress"]

# Columns user1 has and WESAD does not. Used by --context-mode zero.
USER1_ONLY_CONTEXT = ["steps", "MET", "ACT_COUNT", "ACT_INT"]

# BTP-CHANGE R19: the signals that exist in BOTH datasets.
#
# The encoder trained on all 56 WESAD channels cannot ever be used on user1.
# 32 of those channels come from sensors user1 does not have: ECG, chest EDA
# (EDA_x), chest temperature (Temp), chest respiration (Resp) and the chest
# accelerometer. user1 is a wrist device.
#
# Padding 32 of 56 channels with zeros would be the same corruption the
# --context-mode zero experiment is designed to avoid, except in the signal
# path where there is no gate to learn around it.
#
# So for anything that has to reach user1, we restrict to the four signals
# both devices record, six statistics each, 24 channels:
#
#   ACC_Mag_Wrist   WESAD wrist ACC 32 Hz   user1 accelerometer 64 Hz
#   BVP             WESAD wrist BVP 64 Hz   user1 bvp 64 Hz
#   EDA_y           WESAD wrist EDA 4 Hz    user1 eda 4 Hz
#   TEMP            WESAD wrist TEMP 4 Hz   user1 temperature 1 Hz
#
# Both are summarised into the same 5 second bins, so the different raw
# sampling rates do not matter after preprocessing.
#
# Time_Sin and Time_Cos are left out on purpose. WESAD has no wall-clock
# time, so the notebook invents them by stretching each recording over a
# pretend 24 hours. user1 has real timestamps. The two are not the same
# quantity and must not be matched up.
WRIST_SHARED_PREFIXES = ["ACC_Mag_Wrist_", "BVP_", "EDA_y_", "TEMP_"]


class Tee:
    """Print to the screen and to log.txt at the same time."""
    def __init__(self, path):
        self.f = open(path, "w")
    def write(self, text):
        sys.__stdout__.write(text)
        self.f.write(text)
    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


# =============================================================================
# PICKING SIGNAL COLUMNS
# PORTED FROM: vendor/adastress/Stress.ipynb, cell 2
# =============================================================================
def pick_numeric_columns(df):
    drop_like = {"stress_class", "stressclass", "task", "sessionphase",
                 "activity", "subject_id"}
    cols = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in drop_like):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# =============================================================================
# WINDOWING
# PORTED FROM: vendor/adastress/Stress.ipynb, cell 2
#
# BTP-CHANGE R11: the original slides over the whole stacked CSV:
#     def windows_from_array(arr, win=TW, stride=SH):
#         starts = np.arange(0, len(arr)-win+1, stride)
#         Xw = np.stack([arr[s:s+win].T for s in starts])
# With every subject stacked into one file, a window at each join holds rows
# from two different people. We window inside each subject instead. The
# window maths is unchanged.
# =============================================================================
def windows_per_subject(df, feature_matrix, label_col="Stress_Class"):
    """Cut windows inside each subject. Returns arrays and per-window info."""
    Xw, rows = [], []
    pos = 0
    for subject, group in df.groupby("subject_id", sort=True):
        n = len(group)
        block = feature_matrix[pos:pos + n]
        gi = group.reset_index(drop=True)

        for s in np.arange(0, n - TW + 1, SH):
            seg_y = gi[label_col].iloc[s:s + TW].dropna()
            if len(seg_y) == 0:
                continue
            label = int(seg_y.astype(int).mode().iloc[0])

            Xw.append(block[s:s + TW].T)          # [channels, time]
            rows.append({
                "subject_id": subject,
                "window_start_row": int(s),
                "global_row": int(pos + s),
                "label": label,
            })
        pos += n

    if not Xw:
        return np.zeros((0, feature_matrix.shape[1], TW)), pd.DataFrame()
    return np.stack(Xw).astype(np.float32), pd.DataFrame(rows)


# =============================================================================
# AUGMENTATIONS
# PORTED FROM: vendor/adastress/Stress.ipynb, cell 2. Unchanged.
# =============================================================================
def time_warp(x, max_warp=0.1):
    C, T = x.shape
    alpha = np.random.uniform(-max_warp, max_warp)
    scale = 1.0 + alpha
    new_idx = np.linspace(0, T - 1, int(np.round(T * scale)))
    warped = np.zeros_like(x)
    for c in range(C):
        vals = np.interp(new_idx, np.arange(T), x[c])
        warped[c] = np.interp(np.linspace(0, len(vals) - 1, T),
                              np.arange(len(vals)), vals)
    return warped


def augment_view(x):
    x = x.copy().astype(np.float32)
    C, T = x.shape
    x *= np.random.uniform(0.9, 1.1)
    x += np.random.normal(0, 0.1, x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        x = time_warp(x)
    mask = np.random.rand(*x.shape) < 0.15
    x[mask] = 0
    if np.random.rand() < 0.5:
        drop_ch = np.random.choice(C, 1)
        x[drop_ch] = 0
    return x


class SimCLRWindows(Dataset):
    """PORTED FROM: Stress.ipynb cell 3. Unchanged."""
    def __init__(self, X):
        self.X = X.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return (torch.from_numpy(augment_view(self.X[i])),
                torch.from_numpy(augment_view(self.X[i])))


class EmbCtxDataset(Dataset):
    """PORTED FROM: Stress.ipynb cell 3. Unchanged."""
    def __init__(self, H, C, y):
        self.H = torch.tensor(H, dtype=torch.float32)
        self.C = torch.tensor(C, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.H[i], self.C[i], self.y[i]


# =============================================================================
# MODELS
# PORTED FROM: vendor/adastress/Stress.ipynb, cells 4 and 7. Unchanged.
#
# Worth recording: the encoder returns the output of its projection head as
# the embedding. The paper says the projection head is thrown away after
# pretraining and the layer before it is used. The notebook keeps it. We keep
# the notebook's version and note the difference. See D1.
# =============================================================================
class EncoderCNNLSTM(nn.Module):
    def __init__(self, in_channels, proj_dim=128, lstm_hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU()
        )
        # BTP-CHANGE R15: dropout removed from this call. With one LSTM layer
        # it does nothing at all and PyTorch prints a warning saying so.
        # Original line:
        #   self.lstm=nn.LSTM(64,lstm_hidden,batch_first=True,
        #                     bidirectional=True,dropout=dropout)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True,
                            bidirectional=True)
        self.proj = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128), nn.ReLU(),
            nn.Linear(128, proj_dim)
        )

    def forward(self, x):
        h = self.net(x)
        h = h.permute(0, 2, 1)
        h, _ = self.lstm(h)
        h = h.mean(1)
        z = self.proj(h)
        return F.normalize(z, dim=1)


class GatedFusionClassifier(nn.Module):
    def __init__(self, emb_dim, ctx_dim, num_classes):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, emb_dim)
        self.gate = nn.Linear(emb_dim * 2, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, h, c, return_gate=False):
        cproj = self.ctx_proj(c)
        g = torch.sigmoid(self.gate(torch.cat([h, cproj], dim=1)))
        fused = g * h + (1 - g) * cproj
        out = self.classifier(fused)
        if return_gate:
            return out, g
        return out


def nt_xent_loss(z1, z2, temp=TEMPERATURE):
    """PORTED FROM: Stress.ipynb cell 5. Unchanged."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / temp
    sim.fill_diagonal_(-9e15)
    targets = (torch.arange(2 * N) + N) % (2 * N)
    return F.cross_entropy(sim, targets.to(z1.device))


# =============================================================================
# CONTEXT
# PORTED FROM: vendor/adastress/Stress.ipynb, cell 8, context section
#
# BTP-CHANGE R16: turned into a function with four modes instead of one fixed
# list. The original line was:
#   context_cols = [c for c in ["Task","SessionPhase","Activity"]
#                   if c in df_lab.columns]
# =============================================================================
def build_context(df, win_info, mode, fitted=None):
    """Return the context matrix, one row per window, plus what it was made of.

    Note, kept from the original: the context is read from the FIRST row of
    each window (Cw = np.stack([C[s] for s in starts])), not voted across the
    window. Kept as the repo has it.
    """
    n = len(win_info)

    if mode == "none":
        return np.zeros((n, 1), dtype=np.float32), {"kind": "none", "names": []}

    if mode in ("original", "zero"):
        cols = [c for c in ["Task", "SessionPhase", "Activity"]
                if c in df.columns]
        raw = df[cols].astype(str).to_numpy()

        if fitted is None:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe.fit(raw)
        else:
            ohe = fitted["ohe"]

        C_all = ohe.transform(raw).astype(np.float32)
        C = C_all[win_info["global_row"].to_numpy()]

        names = list(ohe.get_feature_names_out(cols))

        if mode == "zero":
            # Experiment A. Widen the schema to hold the columns user1 has and
            # WESAD does not, and fill them with zeros for WESAD.
            #
            # One-hot matters here. An all-zero block is not any category, so
            # "absent" is unambiguous. With plain whole numbers, 0 would be a
            # real category and the model would learn "stress happens at
            # exactly zero steps", which is the failure this experiment is
            # meant to avoid.
            pad = np.zeros((len(C), len(USER1_ONLY_CONTEXT)), dtype=np.float32)
            C = np.concatenate([C, pad], axis=1)
            names = names + ["%s_absent" % c for c in USER1_ONLY_CONTEXT]

        return C, {"kind": mode, "names": names, "ohe": ohe}

    if mode == "derived":
        # Experiment B. Movement worked out from the wrist accelerometer,
        # because user1 wears a wrist device and the same sum can be computed
        # there.
        src = "ACC_Mag_Wrist_mean"
        if src not in df.columns:
            raise SystemExit("Cannot build derived context: %s is missing." % src)

        values = df[src].to_numpy(dtype=float)
        feats = []
        pos_lookup = {}
        p = 0
        for subject, group in df.groupby("subject_id", sort=True):
            pos_lookup[subject] = p
            p += len(group)

        for _, r in win_info.iterrows():
            g = int(r["global_row"])
            block = values[g:g + TW]
            feats.append([np.nanmean(block), np.nanstd(block)])

        C = np.array(feats, dtype=np.float32)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

        # Put both numbers on the same scale. This is essential for the
        # transfer to user1: a chest or wrist sensor on a different device
        # reports movement in different units, so raw values do not compare.
        # The scaler is fitted on the training windows only and saved.
        if fitted is None:
            csc = StandardScaler().fit(C)
        else:
            csc = fitted["ctx_scaler"]
        C = csc.transform(C).astype(np.float32)

        return C, {"kind": "derived",
                   "names": ["acc_mag_mean", "acc_mag_std"],
                   "ctx_scaler": csc}

    raise SystemExit("Unknown context mode: %s" % mode)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Train the AdaStress encoder and gated context classifier "
                    "on assembled WESAD data. Ported from Stress.ipynb.")
    ap.add_argument("--run-id", required=True,
                    help="e.g. R01_wesad_nocontext. Never reuse one.")
    ap.add_argument("--assembled-dir", required=True,
                    help="e.g. data/assembled/block")
    ap.add_argument("--context-mode", required=True,
                    choices=["none", "original", "zero", "derived"])
    ap.add_argument("--encoder-from", default=None,
                    help="reuse the encoder from another run folder, so only "
                         "the context branch differs between runs")
    ap.add_argument("--drop-skew", action="store_true",
                    help="leave out the 9 _skew columns, see decision D10")
    # BTP-CHANGE R19
    ap.add_argument("--channels", choices=["all", "wrist_shared"],
                    default="all",
                    help="all: every WESAD channel, 56. wrist_shared: only "
                         "the 4 signals user1 also records, 24 channels. Use "
                         "wrist_shared for anything that has to reach user1.")
    ap.add_argument("--epochs-pretrain", type=int, default=EPOCHS_PRETRAIN)
    ap.add_argument("--epochs-classif", type=int, default=EPOCHS_CLASSIF)
    ap.add_argument("--seed", type=int, default=42)
    # BTP-CHANGE R17: which epoch to report.
    #
    # "val" keeps the epoch that scored best on the validation subjects.
    # That is normally the right thing to do, but here it can pick an
    # untrained model. The label is cut into thirds separately for each
    # subject, so the validation subjects' labels do not mean the same thing
    # as the training subjects'. S11's Base session is Medium, S14's is High,
    # and the training subjects' is Low. Validation therefore never rewards
    # learning, and the best epoch is often epoch 1.
    #
    # "last" keeps the final epoch instead. Use it to show what the model
    # actually learned on the training subjects, and how much of that
    # transfers. Report which one was used.
    ap.add_argument("--select-on", choices=["val", "last"], default="val",
                    help="which epoch to report: best on validation "
                         "subjects, or the last one. Default val.")
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run_id)
    if os.path.exists(run_dir):
        raise SystemExit(
            "Run folder already exists: %s\n"
            "Runs are never overwritten. Use a new --run-id." % run_dir)
    os.makedirs(run_dir)

    sys.stdout = Tee(os.path.join(run_dir, "log.txt"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adir = args.assembled_dir
    with open(os.path.join(adir, "split.json")) as f:
        split = json.load(f)
    train_subs = split["split"]["train"]
    val_subs = split["split"]["val"]
    test_subs = split["split"]["test"]

    print("=" * 64)
    print("RUN          : %s" % args.run_id)
    print("context mode : %s" % args.context_mode)
    print("assembled    : %s" % adir)
    print("label mode   : %s" % split.get("label_mode"))
    print("drop _skew   : %s" % args.drop_skew)
    print("device       : %s" % device)
    print("seed         : %d" % args.seed)
    print("=" * 64)
    print()

    # -------------------------------------------------------------------
    # STAGE 1: self-supervised pretraining
    # PORTED FROM: Stress.ipynb cell 6
    # -------------------------------------------------------------------
    df_un = pd.read_csv(os.path.join(adir, "all_subjects_unlabelled.csv"))
    print("unlabelled rows: %d  (train subjects only)" % len(df_un))
    print("  subjects: %s" % sorted(df_un["subject_id"].unique()))

    if args.encoder_from:
        enc_dir = args.encoder_from
        sig_cols = json.load(open(os.path.join(enc_dir, "sig_cols.json")))
        scaler = joblib.load(os.path.join(enc_dir, "scaler.pkl"))
        enc_path = os.path.join(enc_dir, "encoder.pth")
        print()
        print("Reusing the encoder from %s" % enc_dir)
        print("  so the only difference between runs is the context branch")
        # BTP-CHANGE R20: copy them in, so this run folder alone is enough to
        # run the model on new data later.
        import shutil as _sh
        _sh.copyfile(enc_path, os.path.join(run_dir, "encoder.pth"))
        _sh.copyfile(os.path.join(enc_dir, "scaler.pkl"),
                     os.path.join(run_dir, "scaler.pkl"))
        json.dump(sig_cols, open(os.path.join(run_dir, "sig_cols.json"), "w"))
    else:
        sig_cols = pick_numeric_columns(df_un)

        # BTP-CHANGE R19: restrict to the signals both devices record.
        if args.channels == "wrist_shared":
            before = len(sig_cols)
            sig_cols = [c for c in sig_cols
                        if any(c.startswith(p) for p in WRIST_SHARED_PREFIXES)]
            print()
            print("restricted to the signals user1 also records: %d of %d "
                  "channels" % (len(sig_cols), before))
            print("  kept: %s" % ", ".join(WRIST_SHARED_PREFIXES))
            print("  left out: ECG, chest EDA, chest temperature, chest")
            print("            respiration, chest accelerometer, Time_Sin,")
            print("            Time_Cos. user1 has none of these.")
            if not sig_cols:
                raise SystemExit("No channels left. Check the column names.")

        if args.drop_skew:
            dropped = [c for c in sig_cols if c.endswith("_skew")]
            sig_cols = [c for c in sig_cols if not c.endswith("_skew")]
            print()
            print("dropped %d _skew columns (decision D10): %s"
                  % (len(dropped), dropped))
        print()
        print("signal channels: %d" % len(sig_cols))

        scaler = StandardScaler()
        X = scaler.fit_transform(df_un[sig_cols].astype(float))
        X = pd.DataFrame(X).interpolate(limit_direction="both").fillna(0).values

        df_un2 = df_un.copy()
        df_un2["Stress_Class"] = 0        # labels unused here
        Xw, _ = windows_per_subject(df_un2, X)
        print("unlabelled windows: %s  (cut inside each subject, R11)"
              % (Xw.shape,))
        print()

        ds = SimCLRWindows(Xw)
        dl = DataLoader(ds, batch_size=min(BATCH_SIZE_PRETRAIN, max(2, len(ds))),
                        shuffle=True, drop_last=True)

        encoder = EncoderCNNLSTM(in_channels=Xw.shape[1]).to(device)
        opt = torch.optim.Adam(encoder.parameters(), lr=LR_PRETRAIN)

        print("Self-supervised pretraining, %d epochs, no labels used:"
              % args.epochs_pretrain)
        for ep in range(1, args.epochs_pretrain + 1):
            losses = []
            for v1, v2 in dl:
                v1, v2 = v1.to(device), v2.to(device)
                loss = nt_xent_loss(encoder(v1), encoder(v2))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print("  epoch %2d: loss %.4f" % (ep, float(np.mean(losses))))

        enc_path = os.path.join(run_dir, "encoder.pth")
        torch.save(encoder.state_dict(), enc_path)
        joblib.dump(scaler, os.path.join(run_dir, "scaler.pkl"))
        json.dump(sig_cols, open(os.path.join(run_dir, "sig_cols.json"), "w"))
        print()
        print("encoder saved to %s" % enc_path)

    print()

    # -------------------------------------------------------------------
    # STAGE 2: labelled data, windows, embeddings
    # PORTED FROM: Stress.ipynb cell 8
    # -------------------------------------------------------------------
    df_gen = pd.read_csv(os.path.join(adir, "generalized_labelled.csv"))
    df_test = pd.read_csv(os.path.join(adir, "finetune_labelled.csv"))

    # BTP-CHANGE R12: split by subject, not with train_test_split.
    # The original line was:
    #   train_idx, val_idx = train_test_split(np.arange(len(ds)),
    #       test_size=0.2, stratify=y_enc, random_state=SEED)
    # That puts the same person on both sides. The audit measured Cramer's V
    # of 0.277 between subject and label, so a random split flatters the
    # model. It also left no test set at all: the numbers reported were the
    # validation numbers the model was selected on.
    df_all = pd.concat([df_gen, df_test], ignore_index=True)

    def prepare(frame):
        Xs = scaler.transform(frame[sig_cols].astype(float))
        Xs = pd.DataFrame(Xs).interpolate(limit_direction="both").fillna(0).values
        return windows_per_subject(frame, Xs)

    Xw_all, info_all = prepare(df_all)
    print("labelled windows: %s" % (Xw_all.shape,))

    C_all, ctx_meta = build_context(df_all, info_all, args.context_mode, None)
    # BTP-CHANGE R20: save the fitted context encoder and the exact column
    # order.
    #
    # Without this, the classifier cannot be used on another dataset. The
    # one-hot encoder decides how many columns there are and in what order,
    # and it is fitted here on WESAD. To run this same classifier on user1 we
    # must build user1's context with THIS encoder, not a fresh one, or the
    # columns will not line up and the numbers will be meaningless while
    # looking perfectly normal.
    ctx_to_save = {k: v for k, v in ctx_meta.items() if k != "names"}
    joblib.dump(ctx_to_save, os.path.join(run_dir, "context_fit.pkl"))
    json.dump({"mode": ctx_meta["kind"],
               "names": ctx_meta["names"],
               "width": int(C_all.shape[1])},
              open(os.path.join(run_dir, "context_names.json"), "w"), indent=2)

    print("context: mode=%s, width=%d" % (ctx_meta["kind"], C_all.shape[1]))
    if ctx_meta["names"]:
        show = ctx_meta["names"][:10]
        print("  columns: %s%s" % (show,
                                   " ..." if len(ctx_meta["names"]) > 10 else ""))
    print()

    # frozen encoder embeddings
    encoder = EncoderCNNLSTM(in_channels=Xw_all.shape[1]).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder.eval()
    with torch.no_grad():
        H = torch.cat([
            encoder(torch.tensor(Xw_all[i:i + 64], dtype=torch.float32).to(device))
            for i in range(0, len(Xw_all), 64)
        ]).cpu().numpy()
    print("embeddings: %s  (encoder frozen)" % (H.shape,))

    subs = info_all["subject_id"].to_numpy()
    y = info_all["label"].to_numpy()

    m_tr = np.isin(subs, train_subs)
    m_va = np.isin(subs, val_subs)
    m_te = np.isin(subs, test_subs)

    print("windows by split: train %d, val %d, test %d"
          % (m_tr.sum(), m_va.sum(), m_te.sum()))
    print("  train subjects: %s" % train_subs)
    print("  val subjects  : %s" % val_subs)
    print("  test subjects : %s" % test_subs)
    print()

    if m_tr.sum() < 20 or m_te.sum() < 5:
        print("STOP: too few windows to train or measure.")
        print("  train %d, test %d" % (m_tr.sum(), m_te.sum()))
        print("  This is the strict label mode problem. See D6.")
        json.dump({"run_id": args.run_id, "status": "insufficient_data",
                   "n_train": int(m_tr.sum()), "n_test": int(m_te.sum())},
                  open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
        return

    # the number every result must beat
    vals, cnts = np.unique(y[m_tr], return_counts=True)
    majority = int(vals[cnts.argmax()])
    base_test = float((y[m_te] == majority).mean())
    print("Always guessing %s scores %.3f on the test subjects."
          % (LABEL_NAMES[majority], base_test))
    print("Every accuracy below must be read next to that number.")
    print()

    # -------------------------------------------------------------------
    # STAGE 3: train the gated context classifier
    # PORTED FROM: Stress.ipynb cell 8
    # -------------------------------------------------------------------
    ds = EmbCtxDataset(H, C_all, y)
    tr_dl = DataLoader(Subset(ds, np.where(m_tr)[0]),
                       batch_size=BATCH_SIZE_CLASSIF, shuffle=True)
    va_dl = DataLoader(Subset(ds, np.where(m_va)[0]),
                       batch_size=BATCH_SIZE_CLASSIF)

    model = GatedFusionClassifier(H.shape[1], C_all.shape[1], 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_CLASSIF)
    crit = nn.CrossEntropyLoss()

    print("Training the classifier, %d epochs. Encoder stays frozen."
          % args.epochs_classif)
    best_val, best_state, best_epoch = -1.0, None, 0
    for ep in range(1, args.epochs_classif + 1):
        model.train(); tacc = []
        for Ht, Ct, yt in tr_dl:
            Ht, Ct, yt = Ht.to(device), Ct.to(device), yt.to(device)
            out = model(Ht, Ct)
            loss = crit(out, yt)
            opt.zero_grad(); loss.backward(); opt.step()
            tacc.append((out.argmax(1) == yt).float().mean().item())

        model.eval(); vacc = []
        with torch.no_grad():
            for Hv, Cv, yv in va_dl:
                Hv, Cv, yv = Hv.to(device), Cv.to(device), yv.to(device)
                vacc.append((model(Hv, Cv).argmax(1) == yv).float().mean().item())
        v = float(np.mean(vacc)) if vacc else 0.0

        if ep % 5 == 0 or ep == 1:
            print("  epoch %2d: train_acc %.3f  val_acc %.3f"
                  % (ep, float(np.mean(tacc)), v))

        # BTP-CHANGE R14: the best epoch is chosen on the validation
        # subjects, and the reported score comes from the test subjects,
        # which are never used for choosing. The original reported the same
        # validation numbers it selected on.
        if v > best_val:
            best_val, best_epoch = v, ep
            best_state = {k: t.clone() for k, t in model.state_dict().items()}

    # BTP-CHANGE R17
    if args.select_on == "val" and best_state:
        model.load_state_dict(best_state)
        print()
        print("reporting the best validation epoch: %d (val_acc %.3f)"
              % (best_epoch, best_val))
        if best_epoch <= 3:
            print("  WARNING: the best validation epoch is very early, so this")
            print("  model has barely trained. That happens because the label")
            print("  is cut into thirds separately for each subject, so the")
            print("  validation subjects' labels do not mean the same thing as")
            print("  the training subjects'. Run again with --select-on last to")
            print("  see what the model actually learned.")
    else:
        print()
        print("reporting the LAST epoch, not the best validation one")
        print("  best validation was %.3f at epoch %d" % (best_val, best_epoch))
    print()

    # -------------------------------------------------------------------
    # STAGE 4: score on the held-out test subjects
    # -------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        Ht = torch.tensor(H[m_te], dtype=torch.float32).to(device)
        Ct = torch.tensor(C_all[m_te], dtype=torch.float32).to(device)
        logits, gates = model(Ht, Ct, return_gate=True)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        gate_mean = float(gates.mean().item())

    pred = probs.argmax(1)
    yte = y[m_te]

    acc = float(accuracy_score(yte, pred))
    bal = float(balanced_accuracy_score(yte, pred))
    mf1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist()

    # stress score, 0 to 2, as the paper describes it
    score = probs[:, 1] * 1.0 + probs[:, 2] * 2.0
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)

    print("=" * 64)
    print("RESULTS on the held-out test subjects %s" % test_subs)
    print("=" * 64)
    print("  windows            : %d" % len(yte))
    print("  accuracy           : %.3f" % acc)
    print("  balanced accuracy  : %.3f" % bal)
    print("  macro F1           : %.3f" % mf1)
    print("  guessing baseline  : %.3f" % base_test)
    print()
    print("  mean stress score  : %.3f  (0 to 2)" % float(score.mean()))
    print("  mean uncertainty   : %.3f  (higher means a flatter guess)"
          % float(entropy.mean()))
    print("  mean gate value    : %.3f" % gate_mean)
    print("     the gate is how much the model leans on physiology rather")
    print("     than context. Near 1 means it is ignoring the context.")
    print()
    # BTP-CHANGE R18: accuracy for each test subject on its own.
    # One overall number hides the fact that the label means something
    # different for each person.
    print("accuracy for each test subject on its own:")
    te_subs = info_all[m_te]["subject_id"].to_numpy()
    per_subject_acc = {}
    for sid in sorted(set(te_subs)):
        sel = te_subs == sid
        a = float((pred[sel] == yte[sel]).mean())
        true_counts = {LABEL_NAMES[c]: int((yte[sel] == c).sum())
                       for c in [0, 1, 2]}
        pred_counts = {LABEL_NAMES[c]: int((pred[sel] == c).sum())
                       for c in [0, 1, 2]}
        per_subject_acc[sid] = a
        print("  %-5s acc %.3f   true %s   predicted %s"
              % (sid, a, true_counts, pred_counts))
    print()

    print("confusion matrix, rows are truth:")
    print("  %-18s %9s %9s %9s" % ("", "pred Low", "pred Med", "pred High"))
    for i, name in enumerate(LABEL_NAMES):
        print("  %-18s %9d %9d %9d" % ("true " + name, cm[i][0], cm[i][1], cm[i][2]))
    print()
    print(classification_report(yte, pred, labels=[0, 1, 2],
                                target_names=LABEL_NAMES, zero_division=0))

    if acc <= base_test + 0.02:
        print("VERDICT: no better than guessing.")
    elif args.context_mode in ("original", "zero"):
        print("VERDICT: read with care. The audit found Task predicts the")
        print("label with 100% accuracy, and this run can see Task. Treat")
        print("this as a leakage measurement, not a performance figure.")
        print("Compare against the no-context run.")
    else:
        print("VERDICT: above guessing, and this run cannot see Task.")
    print()

    # -------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(run_dir, "classifier.pth"))

    per_class = classification_report(yte, pred, labels=[0, 1, 2],
                                      target_names=LABEL_NAMES,
                                      output_dict=True, zero_division=0)

    metrics = {
        "run_id": args.run_id,
        "status": "ok",
        "assembled_dir": adir,
        "label_mode": split.get("label_mode"),
        "context_mode": args.context_mode,
        "context_width": int(C_all.shape[1]),
        "drop_skew": bool(args.drop_skew),
        "n_signal_channels": len(sig_cols),
        "channels": args.channels,
        "split": "subject_held_out",
        "subjects_train": train_subs,
        "subjects_val": val_subs,
        "subjects_test": test_subs,
        "n_train_windows": int(m_tr.sum()),
        "n_val_windows": int(m_va.sum()),
        "n_test_windows": int(m_te.sum()),
        "majority_baseline_accuracy": base_test,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": mf1,
        "best_val_accuracy": float(best_val),
        "best_val_epoch": int(best_epoch),
        "select_on": args.select_on,
        "per_subject_test_accuracy": per_subject_acc,
        "confusion_matrix": cm,
        "per_class": per_class,
        "mean_stress_score": float(score.mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_gate": gate_mean,
        "seed": args.seed,
    }
    json.dump(metrics, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)

    out = info_all[m_te].copy().reset_index(drop=True)
    out["P_Low"] = probs[:, 0]
    out["P_Medium"] = probs[:, 1]
    out["P_High"] = probs[:, 2]
    out["predicted_class"] = pred
    out["true_class"] = yte
    out["stress_score"] = score
    out["uncertainty"] = entropy
    out.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"),
              indent=2)

    commit = "unknown"
    cpath = "vendor/adastress_commit.txt"
    if os.path.exists(cpath):
        commit = open(cpath).read().strip()

    with open(os.path.join(run_dir, "env.txt"), "w") as f:
        f.write("run_id:          %s\n" % args.run_id)
        f.write("context_mode:    %s\n" % args.context_mode)
        f.write("assembled_dir:   %s\n" % adir)
        f.write("vendor_commit:   %s\n" % commit)
        f.write("encoder_path:    %s\n" % enc_path)
        f.write("encoder_sha256:  %s\n" % sha256_of(enc_path))
        f.write("seed:            %d\n" % args.seed)
        f.write("python:          %s\n" % platform.python_version())
        f.write("torch:           %s\n" % torch.__version__)
        f.write("numpy:           %s\n" % np.__version__)
        f.write("pandas:          %s\n" % pd.__version__)
        f.write("split:           subject held out, train %s / val %s / test %s\n"
                % (train_subs, val_subs, test_subs))
        f.write("channels_mode:   %s\n" % args.channels)
        f.write("signal_channels: %d\n" % len(sig_cols))
        f.write("context_width:   %d\n" % C_all.shape[1])

    print("Everything saved in %s/" % run_dir)
    print("  metrics.json, predictions.csv, classifier.pth, env.txt, log.txt")


if __name__ == "__main__":
    main()
