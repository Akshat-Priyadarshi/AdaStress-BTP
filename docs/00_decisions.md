# Decisions log

This is the file to open first. Every choice that could change a result is
written here, with the reason. If a number in a report cannot be traced back to
a decision in this file, the number is not finished.

Written at Stage 0. Updated as the work goes on.

---

## D1. Which part of the repo counts as "the original pipeline"

**Status: decided.**

**Decision:** the notebook path. `Preproc.ipynb` for preprocessing, then
`Stress.ipynb` for training.

**Why:** the repo actually holds two different pipelines that do not fit
together, and only one of them can read raw WESAD.

| | notebook path (chosen) | library path (`adastress/` + `configs/config.yaml`) |
|---|---|---|
| what it reads | raw WESAD `S2.pkl` + `S2_quest.csv` | a ready-made 13-column CSV |
| one row is | a 5 second summary: mean, std, min, max, median, skew | one raw sample |
| chest ACC | 3 axes thrown away, replaced by one magnitude column | all 3 axes kept |
| window | 15 rows | 60 rows, moving 30 at a time |
| context | one-hot | whole numbers 0, 1, 2, ... |
| encoder | conv 32/64/64, BiLSTM 64, output 128 | conv 64/128/128, BiLSTM 128, output 256 |
| temperature in the loss | 0.1 | 0.5 |

The library path needs a CSV that nothing in the repo makes from raw WESAD. The
only such CSV in the repo is `data/sample_data.csv`: 999 rows of made-up numbers
between 0 and 1. So the library path cannot be the WESAD pipeline.

**This note must appear next to every WESAD result we report:**

> This follows the repo's notebook pipeline. It does not match the model
> described in the paper. The paper says 15 second windows at 1 Hz, one LSTM
> layer going in one direction with 64 units, and a 64-number embedding. The
> notebook uses 15 rows of 5-second summaries, an LSTM going both directions,
> and a 128-number embedding. These are different models.

Also worth knowing: one notebook window is 15 rows of 5-second summaries, so it
covers about 75 seconds of real time, not 15 seconds.

---

## D2. How many real stress labels does user1 have

**Status: TO FILL IN.** Run `python3 btp/s0_check_user1_context.py`, read the
report it writes, then fill this table in.

| question | answer |
|---|---|
| How many times was the STAI questionnaire actually answered? | |
| What time period does each answer cover? | |
| How many 1 Hz windows can one answer fairly be spread over? | |
| **Total real labelled windows (N)** | |
| Where do the STAI scores live (which file, sheet, or notebook)? | |

**Rule agreed now, before we know the answer:**

| N | what we are allowed to do |
|---|---|
| 200 or more | fine-tune the gate and the classifier, run Experiment A and B as planned |
| 30 to 200 | fine-tune the classifier head only, report wide error bars, claim no personalisation |
| under 30 | inference only, compare score distributions, do not fine-tune |

Why decide the rule first: if we build the fine-tuning code and count labels
afterwards, there is a strong pull to use whatever labels exist, however few.
Setting the rule in advance removes that pull.




## D2. How many real stress labels does user1 have

**Status: answered. N = 0 in the data files.**

| question | answer |
|---|---|
| STAI answers found in the 22 CSVs | none. No column holds a stress or STAI value. |
| What the 22 files are | 11 signal types x 2 recording days |
| Day 1 (2026-03-22) | about 1.5 hours |
| Day 2 (2026-07-13) | about 1.0 hours |
| **Total real labelled windows (N)** | **0** |
| Where the STAI scores live | NOT FOUND YET - to be tracked down |

**Consequence, by the rule agreed in advance (N under 30):**
Stage 4 is inference only. We do not fine-tune on user1 and we make no
personalisation claim. We compare stress score distributions between
Case 1 and Case 2, and we look at what the context gate does.

If STAI scores turn up later, count them and revisit this entry. Two
recording sessions will give at most 2 questionnaire answers, which is
still far under 30.

---

## D3. Keeping the original code apart from ours

**Status: done by `setup_stage0.sh`.**

- The clone sits at `vendor/adastress/`.
- Its commit is saved in `vendor/adastress_commit.txt`.
- The whole folder is set to read-only with `chmod -R a-w`.
- Nothing inside `vendor/` is ever edited.

All of our code sits in `btp/`. When we copy a function out of the repo, the copy
carries a header saying where it came from:

```python
# PORTED FROM: vendor/adastress/Preproc.ipynb, cell 34 (process_subject)
# BTP-CHANGE: added the missing imports, see the repair log
```

That is the whole method. No wrappers, no extra framework. If someone asks what
we changed, the answer is: every line marked `BTP-CHANGE`, and nothing else.

If you ever really need to write inside `vendor/`, stop and ask why first. If it
is still needed: `chmod -R u+w vendor/adastress`.

---

## D4. How the stress class numbers are read

**Status: decided.** This is easy to get wrong, and getting it wrong would
quietly spoil every number we report.

`Preproc.ipynb` cell 36 runs this on the `Stress_Class` column:

```python
encoder = LabelEncoder()
df[col] = encoder.fit_transform(df[col].astype(str))
```

Two things happen there:

1. Rows with no label are `NaN`. `str(NaN)` gives the text `"nan"`, so unlabelled
   rows turn into a fourth class instead of being left out.
2. `LabelEncoder` sorts the text alphabetically, so the numbers come out as:

```
HighStress   -> 0
LowStress    -> 1
MediumStress -> 2
nan          -> 3     (means "no label", not a stress level)
```

So in the saved CSVs the numbers are **not** in low-to-high order, and 3 is not a
stress level at all. Anyone reading that CSV and assuming 0 = Low, 1 = Moderate,
2 = High will get wrong accuracy and a wrong stress score.

**Decision:** we do not use those numbers. In `btp/` we will:

- keep `Stress_Class` as text when we save processed WESAD,
- drop rows where it is missing, in a step that prints how many were dropped,
- turn text into numbers in one place only, with this fixed mapping:

```
LowStress    -> 0
MediumStress -> 1
HighStress   -> 2
```

- so the stress score means what we say it means:
  `score = 1 * P(Medium) + 2 * P(High)`, giving a number between 0 and 2.

Every result table must say which mapping it used.

---

## D5. Earlier results we are not reporting

**Status: withdrawn.** Kept here so the reason is on record.

The earlier work ran `vendor/adastress/scripts/run_adaptation_demo.py` as it
comes. That script contains:

```python
def label_oracle(windows_batch):
    idx = np.random.choice(len(y), size=n, replace=True)
    return y[idx]
```

It picks labels **at random** instead of returning the labels for the windows it
was handed. So the EWC step was trained on random labels.

The same script also builds its drift reference from the first 300 windows of the
same stream it then plays back, and compares a 15-window buffer against it. Drift
was almost certain to fire whatever the data did.

So these earlier results are withdrawn and must not be shown as results:

- "10 distinct concept drift events"
- the stress score moving from about 1.30 down to about 1.01 after adaptation
- any claim that EWC personalisation worked

Not much is lost. Drift is out of scope for this phase anyway.

---


## D6. How the labels are turned into windows

**Status: decided, after measuring both.**

A window is 15 rows. The repo spreads each questionnaire answer over 11 rows.
11 is less than 15, so no window is ever fully labelled.

| option | usable windows | verdict |
|---|---:|---|
| `strict` — keep the repo's 11-row spread | 39 (23 train / 8 test) | cannot train or measure a 3-class model |
| `block` — give each session its own answer | 617 (369 / 124) | usable |

**Decision:** all training uses `block`. `strict` is reported as a finding:
the released labelling cannot fill the window size the released training
code asks for.

**Cost of `block`, measured:** `Task` then predicts the label with 100%
accuracy on held-out subjects. So in `block` mode, any accuracy from a model
that can see `Task` is a leakage measurement, not a performance figure.

## D7. Subject split

**Status: fixed, never to be changed.**

- train: S2, S3, S4, S5, S6, S7, S8, S9, S10
- validation: S11, S13, S14
- test: S15, S16, S17

Recorded in every `data/assembled/*/split.json`. The self-supervised step
sees train subjects only, so test subjects are unseen even unlabelled.
S9 is in train on purpose: it has no MediumStress rows and would make a test
set unscoreable.

## D10. The `_skew` columns

**Status: decided.**

`ACC_Mag_Wrist_skew` runs exactly -12.53 to +12.53. For 160 samples
(32 Hz over a 5-second bin) the largest skew arithmetically possible is
exactly 12.53. So 103 values are pinned at the limit, which happens when a
bin is constant apart from one sample. `TEMP_skew` runs +/-4.13, and the limit
for 20 samples (4 Hz over 5 seconds) is exactly 4.13.

These columns hold an arithmetic bound, not a measurement. This is what the
hundreds of "precision loss" warnings were reporting.

**Decision:** drop all 9 `_skew` columns, so 45 features instead of 54.
Run R01 both ways to show the conclusion does not depend on it.




## D9. Session order

**Status: found and repaired.**

`btp/02_check_session_order.py` found 9 of 15 subjects whose real task order
differs from the one `Preproc.ipynb` assumes:

- 8 subjects did Base, Fun, Medi1, TSST, Medi2. The notebook calls session 2
  "TSST" and session 4 "Fun", so for these the stress task is labelled as
  amusement and the amusement task as stress.
- S14 did Base, TSST, Medi2, Fun, Medi1. Stress and amusement are right;
  the two meditation sessions are swapped.

Session lengths confirm it on their own: the stress task runs 11 to 12
minutes and amusement runs 6 to 7. For the 8 mismatched subjects the long
block is fourth, not second.

Both the `Task` context feature and the stress label are affected, because
`extract_panas_scores` uses the same fixed list.

**Decision:** all reported results use `--task-order file`. The
`--task-order notebook` version is kept alongside it so the effect of the
repo's assumption can be shown.

**For the authors:** `Preproc.ipynb` cell 34 hard-codes the task order and
never reads the `# ORDER` line, but 9 of 15 WESAD subjects have a different
order. Was this known?



**Effect of the repair, measured.** Which stress class landed on which
task, counted over the 15 subjects:

| | on TSST (stress) | on Fun (amusement) |
|---|---|---|
| corrected order | High 9, Med 3, Low 3 | High 2, Med 4, Low 9 |
| repo's order | High 5, Med 2, Low 8 | High 6, Med 5, Low 4 |

With the repo's hard-coded order, the stress task is labelled HighStress
5 times in 15, which is chance level for three classes, and the amusement
task is labelled HighStress more often (6) than the stress task (5). The
label carries no usable signal.

With the order read from the quest file, the stress task is HighStress
9 times in 15 and amusement only 2. Base is LowStress for 13 of 15
subjects.

All reported results therefore use `--task-order file`.




## Repair log

`Preproc.ipynb` does not run as it is. It uses `pickle`, `skew` and
`LabelEncoder` and imports none of them. Every fix needed to make it run goes in
this table. Fixes only. No improvements, no tidying, no renaming.

| # | file | change | why it was needed |
|---|---|---|---|
| R1 | `btp/prep_wesad.py` | add `import pickle` | the notebook calls `pickle.load` but never imports it |
| R2 | `btp/prep_wesad.py` | add `from scipy.stats import skew` | the notebook passes `skew` to `.agg()` but never imports it |
| R3 | `btp/prep_wesad.py` | add `from sklearn.preprocessing import LabelEncoder` | the notebook uses it but never imports it |
| R4 | | | |

Fill in R4 onward as you hit them. One you will probably hit: newer pandas wants
`"5s"` instead of `"5S"` in the resample step.


| R9 | `btp/01_prep_wesad.py` | read the real task order from the quest file's `# ORDER` line instead of hard-coding it | 9 of 15 subjects did the tasks in a different order. The hard-coded list puts the wrong name on each session and attaches each PANAS answer to the wrong session. |
| R10 | `btp/01_prep_wesad.py` | added `--task-order {file,notebook}` | so the repo's own assumption and the corrected version can both be reported, without either overwriting the other |

---

## The step the repo is missing

`Preproc.ipynb` saves one CSV per subject: `S2.csv`, `S3.csv`, and so on.

`Stress.ipynb` reads three different files:

```
all_subjects_unlabelled.csv
generalized_labelled.csv
finetune_labelled.csv
```

Nothing in the repo makes those three files. The step that joins the subjects
together, separates labelled from unlabelled rows, and decides what to do with
the unlabelled ones, is simply not there.

We have to write it, which means we have to decide it. It goes in
`btp/assemble_wesad.py` at Stage 1, and the choice gets recorded here as D6.

---

## Still open

| # | question | needed before |
|---|---|---|
| D2 | user1 label budget | Stage 4 |
| D6 | how the labelled / unlabelled split is defined | Stage 1 |
| D7 | which subjects go in train, validation and test | Stage 2 |
| D8 | how ACC-derived context is put on the same scale across the two datasets | Stage 3 |

---

## Questions to ask the AdaStress authors

Short and answerable. Each one links to a decision above.

1. Is `Stress_Class` meant to be worked out from `Task`? PANAS is filled in once
   per session, the score is written to the last row of that `Task` block, cut
   into thirds per subject, then filled 5 rows forward and back. `Task` is then
   given to the model as a context feature. (We will measure this in Stage 1.)
2. Where the notebook and the paper disagree, which one is correct? Window
   length, LSTM direction, embedding size. (D1)
3. Which script makes `all_subjects_unlabelled.csv`, `generalized_labelled.csv`
   and `finetune_labelled.csv`?
4. Is the alphabetical label order on `Stress_Class` intended? It makes 0 mean
   HighStress. (D4)
5. Is working out context from the accelerometer stream in line with what the
   context gate was meant to do? (This is the Experiment B question.)
