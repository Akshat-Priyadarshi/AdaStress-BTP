# AdaStress on WESAD and on wrist wearable data

A reproduction of **AdaStress** (IEEE Transactions on Consumer Electronics)
on the WESAD dataset, then an attempt to carry the trained model over to
EmbracePlus wrist recordings from five people who also reported their own
stress.

The model code is the authors', unmodified, in `vendor/adastress`. Everything
else here is data handling, evaluation, and the things we found along the way.

---

## What came out of it

| what was tested | balanced accuracy | chance |
|---|---:|---:|
| WESAD, held-out subjects, 6 wrist channels | **0.935** | 0.531 |
| Our users, pooled random 70/10/20 | 0.889 | 0.333 |
| Our users, same split with each survey answer kept whole | **0.468** | 0.333 |
| Our users, leave one person out | **0.310** | 0.333 |

The model works on WESAD. It does not transfer between people. Why, and how
we know, is below and in [`docs/`](docs/).

Two findings worth the click:

- On WESAD the `Task` context column predicts the stress label with accuracy
  **exactly 1.000** on held-out subjects. It is one of the three context
  columns the paper feeds to its gate.
- Averaging the pulse waveform to 1 Hz, as the released config does, keeps
  about **8%** of its variation.

---

## Pipeline

```
┌───────────────────── WESAD ──────────────────────┐
│ 15 people, lab, chest strap + wrist band         │
│                                                  │
│   raw .pkl                                       │
│      │ wesad/20_build_wesad_13ch.py              │
│      │ average every channel to 1 Hz             │
│      ▼                                           │
│   13 channels, 86,868 rows                       │
│      │ vendor/.../pretrain_encoder.py            │
│      │ contrastive, NO labels                    │
│      ▼                                           │
│   ENCODER ─────────── frozen from here ──────┐   │
│      │                                       │   │
│      │ wesad/21_train_subjectwise.py         │   │
│      │ gate + classifier, labels used here   │   │
│      ▼                                       │   │
│   0.935 balanced                             │   │
└──────────────────────────────────────────────┼───┘
                                               │
                   the encoder carries over ───┤
                                               │
┌───────────────── OUR 5 USERS ────────────────┼───┐
│ EmbracePlus, wrist only, 884 hours           │   │
│                                              │   │
│   S3 avro                                    │   │
│      │ users/24b_download_remaining_users    │   │
│      ▼                                       │   │
│   raw CSVs                                   │   │
│      ├── users/25_build_user_6ch.py ─────────┤   │
│      │   6 channels at 1 Hz                  │   │
│      │                                       │   │
│      └── users/27_derive_pr.py               │   │
│          users/30_derive_context.py          │   │
│          12 context columns at 30 s          │   │
│                   │                          │   │
│   91 survey answers                          │   │
│      │ users/31_build_labels.py              │   │
│      │ STAI-6 → three classes                │   │
│      ▼            │                          │   │
│   labelled windows│                          │   │
│                   ▼                          ▼   │
│          users/32_user_leak_check.py             │
│          users/34_train_users.py                 │
│                   │                              │
│          encoder (frozen) ──┐                    │
│                             ├─ GATE ─→ classifier│
│          12 contexts ───────┘                    │
└──────────────────────────────────────────────────┘
```

---

## Setup

```bash
git clone https://github.com/<you>/adastress-btp.git
cd adastress-btp
pip install -r requirements.txt
```

**numpy must stay below 2.** torch, pandas and scipy here were built against
numpy 1.x, and upgrading breaks all three at once with a message about dtype
sizes that never mentions numpy. If that happens:

```bash
pip install --force-reinstall "numpy==1.26.4"
```

Check it:

```bash
python3 -c "
import numpy, pandas, torch, numpy as np
print(numpy.__version__, pandas.__version__, torch.__version__)
print('torch reads numpy:', torch.from_numpy(np.zeros(3)).sum().item() == 0)
"
```

Known good: numpy 1.26.4, pandas 2.3.3, scipy 1.16.3, torch 2.1.0.

Run everything from the repository root.

---

## Part 1: WESAD

### Get the data

WESAD is public, from the University of Siegen. Request it, unpack it, and
lay it out as:

```
data/raw/WESAD/S2/S2.pkl, S2_quest.csv
               S3/...          up to S17
```

### Preprocess

```bash
python3 wesad/20_build_wesad_13ch.py --mode corrected --label-source wesad_condition
```

This is the only file here that reads raw WESAD, and it had to be written.
The repository ships a 999-row synthetic file for smoke testing and nothing
that turns the real thing into the 13 columns its own config asks for.

Three repairs are applied, each marked in the source:

- three missing imports, and a pandas call newer versions reject outright
- **the task order, wrong for 9 of the 15 subjects.** WESAD varied the order
  deliberately and records it on a `# ORDER` line the released code never
  reads. Without the fix the stress session is labelled as amusement for
  those nine
- the labelling. Each questionnaire answer covers 11 rows and a window is 15,
  so the released settings produce **no fully labelled window at all**

`--mode literal` reproduces the released behaviour and reports zero usable
windows, which is the point of keeping it.

### Stage 1: pretrain the encoder

```bash
python3 vendor/adastress/scripts/pretrain_encoder.py --config wesad/configs/w6.yaml
```

The authors' script, unchanged. Contrastive learning on unlabelled windows:
two altered copies of the same window are taught to match. No labels are used
at any point. The encoder is frozen afterwards and never updated again.

`w13.yaml` uses all 13 channels. `w6.yaml` uses the six a wrist device can
also record, and is the one that transfers.

### Stage 2: train the classifier

```bash
python3 wesad/21_train_subjectwise.py --config wesad/configs/w6.yaml \
    --run-id W_pooled --split pooled --ratios 70/10/20 --seeds 5

python3 wesad/21_train_subjectwise.py --config wesad/configs/w6.yaml \
    --run-id W_rotation --rotation all --context off
```

Only the gate and classifier are trained, which is the paper's design:
*"only the gating and classification parameters are updated, while the
pretrained encoder remains fixed."*

The released `train_classifier.py` splits at random with no test set and
reports the validation numbers it selected on, so this script adds a real
held-out set and the option to hold subjects out entirely.

### Check the data before believing any of it

```bash
python3 wesad/23_context_leak_check.py --config wesad/configs/w6.yaml
python3 wesad/22_floor_check.py --config wesad/configs/w6.yaml
```

The first learns the commonest label for each value of a context column from
the training subjects and applies it to held-out ones, with no model at all.
`Task` scores **1.000**. The label comes from one questionnaire per session,
and `Task` **is** the session.

The second fits a random forest on nothing but the window mean and spread. On
WESAD's own condition labels it reaches **0.733**, against 0.567 for the full
pipeline on the same windows.

### WESAD results

| split | 13 channels | 6 channels |
|---|---:|---:|
| pooled random 70/10/20 | 0.975 | **0.935** |
| pooled, overlapping windows kept together | 0.833 | 0.800 |
| leave three subjects out | 0.567 | — |
| guessing | 0.531 | 0.531 |

14 points of that gap come from overlapping windows, 27 from the same person
appearing on both sides.

---

## Part 2: wrist data from five people

### Download

```bash
jupyter notebook users/24b_download_remaining_users.ipynb
```

Pulls raw sensor data from the Empatica S3 bucket. Three cells need filling
in: your credentials, the session table saying who wore which device on which
dates, and the output path. It is resumable: if it stops, run the last cell
again and it skips what it already has.

Credentials and participant details are stripped from the committed version.

### Preprocess

```bash
python3 users/25_build_user_6ch.py
python3 users/27_derive_pr.py
python3 users/30_derive_context.py --user <name> --drop-redundant
python3 users/33_check_outputs.py
```

**`25`** builds the six channels the encoder reads, averaged to 1 Hz, in
chunks so a day of 64 Hz accelerometer does not exhaust memory. Six and not
thirteen because EmbracePlus is wrist-only: ECG, chest EDA, respiration and
chest accelerometer are simply not there, and padding them with zeros would
feed the model numbers it never saw in training.

**`27`** works out pulse rate from the heartbeat timestamps, because
EmbracePlus ships no pulse rate channel. The gap between two beats is the
instantaneous rate. Only 40 to 180 bpm is kept; outside that the detector has
missed a beat or counted one twice.

**`30`** builds twelve context columns at 30 seconds. Five functions are
ported from the study team's own notebook with their citations. The rest,
marked `BTP-ADDED`, exist because Empatica's own computed columns did not
survive checking against a 97-minute calibration recording with activities
logged by hand:

| their column | what was wrong |
|---|---|
| `eda_scl_usiemens` | 2 non-zero values in 2,880 minutes |
| `respiratory_rate_brpm` | 1.1% coverage |
| `prv_rmssd_ms` | 0.8% coverage |
| `body_position_left` | called sitting "standing" 33 times of 56 |
| `activity_intensity` | called sitting "vigorous" 10 times |

Deriving them ourselves gave **91% coverage** against their 20%.

**`33`** lists what exists per person and whether the raw folder is safe to
delete. Worth running: we once deleted 35 GB after deriving context, without
noticing the signal channels had not been built, and had to download it all
again.

### Labels

```bash
python3 users/31_build_labels.py --survey <your survey file>.xlsx
```

The survey is the **STAI-6**, the validated short form of the State-Trait
Anxiety Inventory, so the scoring rule is published rather than invented:
reverse the three positive items, sum all six, multiply by 20/6 for a 20 to
80 score.

Classes are cut **across everyone**, not per person. That is the single thing
that went wrong with WESAD, where each subject's five scores were cut into
their own thirds, so "HighStress" meant "the worst of this person's own five
sessions" and a calm baseline session came out Medium for one subject and
High for another.

Each answer labels the windows within 15 minutes either side.

### Check, then train

```bash
python3 users/32_user_leak_check.py

python3 users/34_train_users.py --config wesad/configs/w6.yaml \
    --run-id U_loo --context on --split user

python3 users/34_train_users.py --config wesad/configs/w6.yaml \
    --run-id U_response --context on --split response --seeds 5
```

Three ways to split, answering different questions:

| `--split` | what it asks |
|---|---|
| `user` | does this work on somebody never seen before? |
| `random` | pooled 70/10/20, the same way WESAD was split |
| `response` | pooled, but each survey answer stays on one side |

### Score new data

```bash
python3 users/26_apply_to_users.py \
    --model checkpoints/P06b_w6_pooled_noctx/model.pth \
    --users data/processed/user_6ch/users_all.csv --scale user
```

One row per window: three class probabilities, the predicted class, an
uncertainty, and

```
stress_score = 1 × P(Moderate) + 2 × P(High)        range 0–2
```

**That formula is ours.** AdaStress outputs the three probabilities and
nothing more.

---

## Part 3: Chronos tokens, a side branch

```bash
python3 chronos/28_chronos_tokens.py --users <name>
python3 chronos/29_token_report.py
```

Not part of AdaStress, which has no tokenisation anywhere in its code,
requirements or paper. Explored because Chronos normalises each window by its
own statistics, which removes the unit mismatch between two different
devices.

Two things it measured:

**Averaging destroys the pulse.** Raw 64 Hz BVP uses **4,058** of 4,094
possible tokens. The same signal averaged to 1 Hz uses about **10**. A pulse
swings above and below zero with every beat, so a one-second mean cancels it.

**Upstream Chronos destroys signals with a baseline offset.** Skin
temperature sits around 31 degrees, so dividing by the average size gives
about 1.0 for every sample and **98.2% of values land in one bin**.
SensorLLM's alternative tokeniser, which subtracts the mean first, gives
3,341 bins. That is very likely why they added it.

---

## Reading the user results

| split | no context | with context |
|---|---:|---:|
| pooled random 70/10/20 | 0.550 | **0.889** |
| pooled, answers kept whole | 0.468 | **0.333** |
| leave one person out | 0.310 | 0.253 |
| chance | 0.333 | 0.333 |

Context is worth **+0.34** under one split and **−0.13** under another. A
feature carrying real information does not flip sign.

The mechanism is measurable. Windows overlap by 30 seconds and each survey
answer labels about 60 consecutive windows, so under random splitting **all
100 survey answers** have some windows in training and some in test. Under
answer-grouping, none do. Context values half a minute apart are nearly
identical, so the model learns each moment's context fingerprint and
recognises it again in the test set. That is recall, not generalisation.

| number | what it answers |
|---|---|
| 0.889 | can it recognise moments it has already seen? yes |
| 0.468 | can it generalise to new moments from the same people? a little |
| 0.310 | can it transfer to someone new? no |

### Why transfer fails

Two folds scored near zero with a guessing baseline of 0.000, which is the
clue: the majority class learned from four people **does not occur** in the
fifth. One person reported low stress on all 18 occasions; another never
reported low at all. Five people used a self-report scale in five different
parts of its range, over five different weeks.

That is not a fault in the model. More people, covering more of the range, is
what would change it.

---

## Repository layout

```
wesad/        WESAD preprocessing, training, checks
users/        download, preprocessing, context, labels, training, scoring
chronos/      tokenisation experiments
vendor/       the AdaStress authors' code, MIT, unmodified
checkpoints/  trained encoder and classifier, 1.6 MB
docs/         findings, decisions, results
archive/      superseded scripts, kept because findings came from them
```

No data is included. WESAD must be requested from its authors. The wrist
recordings and survey answers are human subjects data and stay out.

---

## Whose code is whose

| | |
|---|---|
| encoder, gate, classifier, SSL, training loop | AdaStress, unmodified, in `vendor/` |
| EmbracePlus download | the study team's notebook |
| pulse, breathing, activity derivation | the study team's notebook, ported and marked |
| WESAD converter, user preprocessing, context, labels, evaluation | ours |
| the 0–2 score formula | ours |

`vendor/adastress` is MIT licensed, its `LICENSE` is kept, and nothing in it
was edited. Every line changed in ported code is marked `BTP-CHANGE` or
`BTP-ADDED` with the reason.

### Citing

AdaStress: Ara, T. and Mitra, B., *Adaptive Stress Monitoring on Wearable
Devices for Long-Term Consumer Health*, IEEE Transactions on Consumer
Electronics.

WESAD: Schmidt, P. et al., *Introducing WESAD, a Multimodal Dataset for
Wearable Stress and Affect Detection*, ICMI 2018.

---

## Known limitations

**No context in the shared checkpoint.** `context_dim: 0`, so the gate is
absent and scores come from physiology alone.

**The gate never learned to prefer either input.** Across every run and every
fold it sat between 0.46 and 0.53, the halfway point.

**Overnight scores read as stress.** The score peaks between 4 and 8 in the
morning. At night the body is still and skin temperature shifts, and in WESAD
those same patterns meant the stress task.

**Pulse rate falls during movement**, on all five users. The optical sensor
loses contact and misses beats, so the apparent rate drops.

**Beat-to-beat variability is unusable.** Consecutive intervals jump by a
median of 122 ms, where a resting adult is 20 to 50 ms.
