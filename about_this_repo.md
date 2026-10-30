# AdaStress-BTP: what each file does

Paths are relative to the repository root. Scripts are numbered in the order they were written, which is roughly the order they run: 20s for WESAD, 24-34 for the wrist-device users, 40s for the seven-channel and eight-context versions, 50s for deployment.

The AdaStress authors' code lives in `vendor/adastress/` and is not modified. Everything outside `vendor/` is ours.

---

## Three things called "score"

The stress score is not STAI. They are different things:

| name | what it is |
|---|---|
| **STAI-6** | the **label**: what people reported, scored 20-80 by the published rule |
| three probabilities | the **model output**: P(Low), P(Medium), P(High). AdaStress produces nothing else |
| `stress_score` | ours: `P(Medium) + 2 x P(High)`, range 0-2 |
| `stai_equiv` | ours: the same probabilities mapped onto the 20-80 STAI scale so they can be compared with the label |

The 0-2 score and `stai_equiv` are our formulas, not part of AdaStress, and should be labelled that way wherever they are quoted.

---

## Root

```
README.md            project overview, results, how to run
FILE_GUIDE.md        this file
requirements.txt     Python packages for the scripts (numpy must stay below 2)
.gitignore           keeps data/, runs/, mentor_files/, *.xlsx, *.avro, *.pkl and __pycache__ out of git
```

---

## `wesad/`: WESAD preprocessing, training and checks

```
20_build_wesad_13ch.py       raw WESAD -> the 13 channels the AdaStress config asks for, 1 Hz,
                             plus context columns. Repairs the wrong task order (9 of 15 subjects).
                             --mode corrected | literal. The faithful reproduction (0.975 / 0.935)
39_build_wesad_7ch.py        raw WESAD -> the 7 channels a wristband can supply
                             (6 wrist channels + RESP). --resp-source belt | bvp,
                             --label-source wesad_condition. Windows: 60 rows, stride 30
38_wesad_resp.py             breathing rate from heartbeat timing, and the check against the chest
                             belt (mean correlation 0.185). 39 imports it
21_train_subjectwise.py      trains the gate + classifier on the frozen encoder. Adds a real held-out
                             test set. --split pooled (random or blocked) or --rotation (held-out
                             subjects), --context on | off, --seeds
22_floor_check.py            random-forest baseline on window mean and spread. A floor, never
                             reported as an AdaStress number
23_context_leak_check.py     lookup test per context column. Found Task = 1.000
```

```
configs/
  w6.yaml        6-channel config (wrist channels only)
  w7.yaml        7-channel config (wrist + RESP)
  w13.yaml       13-channel config (the faithful reproduction)
```

---

## `users/`: EmbracePlus download, signals, context, labels, training, scoring

Download and signals:

```
24b_download_remaining_users.ipynb   S3 -> raw EmbracePlus files. Resumable.
                                     Credentials come from environment variables
25_build_user_6ch.py                 raw -> 6 channels at 1 Hz, with the WESAD channel names
40_build_user_7ch.py                 adds RESP (carried forward from the 30 s context) -> 7 channels
27_derive_pr.py                      pulse rate and beat intervals from heartbeat timestamps
                                     (keeps 40-180 bpm, reports dropped beats)
27_merge_biomarkers.py               Empatica's one-file-per-metric biomarker CSVs -> one wide CSV
```

Context and labels:

```
30_derive_context.py                 derived context at 30 s: PR, RESP, ACT_COUNT, ACT_INT, MET,
                                     EDA tonic/phasic, POSTURE, TILT, SKIN_TEMP, clock.
                                     Ported functions from the study team's notebook, plus our additions
42_build_context8.py                 the final 8-column context set (HOUR, IS_WEEKEND, POSTURE,
                                     ACT_COUNT, ACT_CLASS, BODY_POSITION, SLEEP, INDOOR_OUTDOOR);
                                     biomarkers are held across each minute; --no-indoor drops the survey column
31_build_labels.py                   survey -> STAI-6 score -> 3 classes (cut across everyone) ->
                                     labels attached to windows within 15 min of each answer
```

Checks, training and scoring:

```
32_user_leak_check.py                leave-one-user-out lookup per context column
33_check_outputs.py                  lists what exists per person; run before deleting any raw folder
34_train_users.py                    stage 2: trains the gate + classifier on the frozen encoder.
                                     --split random | response | user, --context on | off, --seeds
26_apply_to_users.py                 applies a saved WESAD-trained model to user data, no labels needed.
                                     Reports input ranges, class spread, confidence, gate position,
                                     hour-of-day profile. --scale wesad | user
```

---

## `chronos/`: Chronos tokenisation, a side experiment

Not part of AdaStress, and not part of the results.

```
28_chronos_tokens.py        turns raw signal chunks into Chronos tokens (PR, BVP, skin temperature, ...)
29_token_report.py          compares tokenisers per channel: how many distinct tokens each uses.
                            Showed that mean-scaling collapses offset signals like skin temperature
```

---

## `deploy/`: edge-device inference and demo

```
50_inference.py             streaming inference: read 60 s, predict, repeat. Rebuilds the encoder,
                            gate and classifier from the checkpoint, so it adapts to 6 or 7 channels
51_make_sample.py           packs one person's day into a self-contained folder
52_inference_demo.py        runs a packed sample like a device would, and measures how much each
                            context column changed the answer (turn one off, re-score).
                            Also reports latency, CPU and memory
53_power_measure.py         power cost of inference on the Jetson: idle before, running, idle after
command_script.txt          the commands used for the Jetson demo and power run
con.txt                     one-line note about the Jetson Orin Nano deployment
requirements.txt            packages needed on the device
```

```
adastress/                  copy of the AdaStress package, so the demo runs without vendor/
  __init__.py, adaptation_loop.py, config.py, continual_learning.py, datasets.py,
  drift_detection.py, inference.py, models.py, ssl_pretrain.py, stats_tests.py,
  train_classifier.py, utils.py
  __pycache__/              compiled Python files that were committed by accident

sample_data/user25_2026-09-16/     one person's day, packed by 51_make_sample.py
  signals.csv       7 channels, one row per second
  context.csv       8 context columns, every 30 seconds
  truth.csv         what the person reported
  model.pth         trained model (gate + classifier)
  encoder.pth       frozen encoder trained on WESAD
  scaler.pkl        the scaling the model was trained with
  README.md         what each file is and how to run it
```

---

## `checkpoints/`: trained models

The files directly in `checkpoints/` are the 7-channel set (7 input channels, 256-dimensional embedding, 60 s windows, stride 30).

```
encoder.pth                 frozen 7-channel encoder, pretrained on WESAD with no labels
model.pth                   classifier head + gate + context transformers (the 8-column demo model).
                            Needs encoder.pth beside it
scaler.pkl                  the WESAD scaling used for the 7 channels
meta.json                   channel names, window length, stride, input channels, embedding size
```

```
W6/                         6-channel set (wrist channels only, no RESP)
  encoder.pth, scaler.pkl, meta.json
P06b_w6_pooled_noctx/       6-channel classifier trained without context, pooled random 70/10/20 split
  model.pth                 the classifier
  metrics.json              accuracy, balanced accuracy, macro F1 over 5 seeds
```

---

## `docs/`: findings, decisions, results

```
00_decisions.md                 every choice that could change a result, with the reason. Open this first
90_findings_stage1.md           what the WESAD data showed before any model was trained
91_results_stage2.md            WESAD training results (3 seeds, subject-held-out split)
54_token_method_comparison.md   which Chronos tokeniser suits which channel
56_labels.md                    how the survey became labels (STAI-6, class cuts)
57_user_leak_check.md           leak-check results on the user context columns
60_wesad_resp.md                the breathing-rate derivation and the 0.185 check
42_context8.log                 output of 42_build_context8.py: coverage of the biomarker columns
```

---

## `archive/`: superseded scripts, kept for provenance

Findings in `docs/` came from these, so they stay.

```
01_prep_wesad.py             the original notebook's preprocessing, repaired and marked BTP-CHANGE
02_check_session_order.py    found that the task order differs between WESAD subjects
03_assemble_wesad.py         builds the three CSVs the original training notebook expects
04_audit_wesad.py            checks whether context gives away the label
05_train.py                  the original training cells with subject-level splits and a test set
06_collect_results.py        gathers every runs/*/metrics.json into one table
07_prep_user1.py             first attempt at putting one user into the WESAD shape
08_infer_user1.py            applies WESAD-trained models to that one user, no labels
09_package_results.py        gathers one dataset's results into a folder to hand over
24_download_embraceplus.ipynb   the first S3 downloader (replaced by users/24b_...)
s0_check_user1_context.py    inspects user files: label count, possible context columns
s0_check_wesad_raw.py        checks raw WESAD folders before preprocessing

configs/
  btp_13ch.yaml, btp_13ch_cond.yaml, btp_13ch_global.yaml,
  btp_13ch_noTask.yaml, btp_13ch_paper.yaml, btp_13ch_taskonly.yaml
                             13-channel configs for the earlier context experiments
```

---

## `vendor/adastress/`: the authors' code, unmodified

MIT licensed. The `LICENSE` file is kept. Nothing in here is edited.

```
README.md, LICENSE, .gitignore, requirements.txt
Preproc.ipynb                the original WESAD preprocessing notebook
Stress.ipynb                 the original training notebook

adastress/                   the package
  config.py                  config loader
  datasets.py                windowing and dataset builder
  models.py                  CNN-BiLSTM encoder, context gate, MLP classifier
  ssl_pretrain.py            contrastive pretraining
  train_classifier.py        classifier training
  inference.py               inference
  drift_detection.py         KL-divergence drift detector
  adaptation_loop.py         retraining on drift
  continual_learning.py      continual-learning helpers (EWC)
  stats_tests.py             statistical tests
  utils.py                   seeding and helpers

scripts/
  pretrain_encoder.py        stage 1: pretrain the encoder, no labels
  train_classifier.py        stage 2: train the gate and classifier
  run_multi_seed_training.py repeat training across seeds
  run_inference.py           run inference on a file
  run_adaptation_demo.py     demo of the drift-and-adapt cycle
  run_stats_tests.py         run the statistical tests

configs/config.yaml          the authors' 13-channel config
checkpoints/                 the authors' released encoder, classifier, drift reference, meta.json
assets/                      architecture.png, adaptation_cycle.png
tests/test_smoke.py          smoke test
```

---

## Not in the repository

```
data/, runs/                 raw and processed recordings, training outputs
mentor_files/                survey responses with user IDs, raw watch data
*.xlsx, *.avro, *.pkl        (except checkpoint scalers)
AWS keys, session table, RESPONSE_DATES
```

These are human-subjects data or credentials and are supplied separately. Per-person numbers in the docs use P1-P5.
