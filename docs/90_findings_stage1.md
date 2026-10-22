# WESAD reproduction: what we found before training anything

Interim findings, Stage 1 complete. No model has been trained yet. Every
number below comes from the data itself and can be re-checked by running the
scripts in `btp/`.

Repository reproduced: `giTan7/Adaptive-Stress-Monitoring-on-Wearable-Devices`
at commit `15d1ca7`. Dataset: WESAD, all 15 subjects, raw `.pkl` and `_quest`
files.

---

## Short version

We set out to reproduce the WESAD pipeline end to end. It does not run as
released, and once repaired the labels turn out to carry very little
information about stress. Five separate problems were found and measured.

The most important one: **the context feature `Task` predicts the stress
label with 100% accuracy.** Any accuracy reported by a model that can see
`Task` measures the model's ability to recognise which part of the experiment
it is looking at, not to read the body.

---

## What was reproduced

| stage | result |
|---|---|
| raw WESAD to 5-second summary rows | works, after 4 repairs |
| join subjects, split labelled from unlabelled | **missing from the repo**, written by us |
| self-supervised pretraining | not started |
| classifier and context gate | not started |

Numbers from preprocessing:

| | |
|---|---|
| subjects | 15 |
| rows inside the five sessions | 9,359 |
| feature columns, same for every subject | 54 |
| rows carrying a stress label | 750 (8.0%) |

---

## Finding 1: the released preprocessing does not run

`Preproc.ipynb` uses `pickle`, `scipy.stats.skew` and `LabelEncoder`, and
imports none of them. It also calls `.resample("5S")`, which pandas 2.2 and
later reject outright rather than warn about.

Four repairs, all logged in `docs/00_decisions.md`. None changes any
calculation.

---

## Finding 2: a step is missing between the two notebooks

`Preproc.ipynb` writes one CSV per subject (`S2.csv` … `S17.csv`).
`Stress.ipynb` reads three different files:

```
all_subjects_unlabelled.csv
generalized_labelled.csv
finetune_labelled.csv
```

Nothing in the repository produces those three. The step that joins the
subjects, separates labelled from unlabelled rows, and decides which subjects
go where, is absent. We wrote it as `btp/03_assemble_wesad.py` and recorded
every choice.

---

## Finding 3: the task order is wrong for 9 of 15 subjects

`Preproc.ipynb` hard-codes one order for everybody:

```python
session_labels = ["Base", "TSST", "Medi1", "Fun", "Medi2"]
```

WESAD deliberately varied the order between subjects and records each
subject's real order on a `# ORDER` line in the quest file. The notebook
never reads it.

Nine subjects are affected:

| pattern | subjects | effect |
|---|---|---|
| `Base, Fun, Medi1, TSST, Medi2` | S4, S5, S7, S8, S10, S13, S15, S17 | stress task labelled as amusement and vice versa |
| `Base, TSST, Medi2, Fun, Medi1` | S14 | the two meditation sessions swapped |

Session lengths confirm it independently: the stress task runs 11 to 12
minutes and amusement 6 to 7. For the eight affected subjects the long block
is fourth, not second.

Both the `Task` feature and the stress label are affected, because
`extract_panas_scores` uses the same fixed list, so each questionnaire answer
is attached to the wrong session.

**Measured effect.** Which stress class lands on which task, over 15 subjects:

| | on TSST (stress) | on Fun (amusement) |
|---|---|---|
| order read from the file | High 9, Med 3, Low 3 | High 2, Med 4, Low 9 |
| order hard-coded, as released | High 5, Med 2, Low 8 | High 6, Med 5, Low 4 |

With the released code the stress task is labelled HighStress 5 times in 15,
which is chance level for three classes, and amusement scores as HighStress
more often than the stress task. After the repair the stress task is
HighStress 9 times in 15, amusement only 2, and `Base` is LowStress for 13 of
15 subjects.

Repair: read the order from the quest file. One function, `extract_session_order`.

---

## Finding 4: the labels cannot fill a training window

`Stress.ipynb` uses windows of 15 rows. `Preproc.ipynb` spreads each
questionnaire answer over 11 rows (the answer, five rows before, five after).

11 is smaller than 15, so **no window can sit entirely inside a labelled
block.** Not one, for any subject.

Keeping the released labelling exactly as it is and requiring a majority of a
window to be labelled:

| | |
|---|---|
| windows in total | 617 |
| fully labelled | **0** |
| usable (8 or more of 15 rows labelled) | **39** |
| after splitting by subject | 23 train, 8 validation, 8 test |

Twenty-three windows cannot train a three-class model and eight cannot
measure one. This is very likely why an earlier attempt on this pipeline
reported a macro F1 of 0.3212, which is chance for three classes.

To get a usable dataset we added a second labelling option: since the
questionnaire was answered once per session about that whole session, give
every row of a session its own session's label. That yields 617 usable
windows. Both options are kept and both are reported.

---

## Finding 5: `Task` gives away the answer completely

This is the one that matters most.

Method: learn the most common stress class for each `Task` value using the
training subjects, then apply that lookup to the held-out test subjects. This
is the score a model gets for free just by reading `Task`.

| context column | accuracy from that column alone | Cramer's V |
|---|---:|---:|
| **`Task`** | **1.000** | 0.541 |
| `Posture` | 0.516 | 0.041 |
| `Activity` | 0.516 | 0.100 |
| `SessionPhase` | 0.516 | 0.025 |
| always guess the most common class | 0.516 | |

`Task` by label, in windows:

| Task | Low | Medium | High |
|---|---:|---:|---:|
| Base | 207 | 16 | 16 |
| Fun | 48 | 20 | 10 |
| Medi1 | 16 | 48 | 18 |
| Medi2 | 16 | 5 | 58 |
| TSST | 27 | 27 | 85 |

The cause is structural, not accidental. The label comes from one
questionnaire per session, and `Task` *is* the session. So `Task` and the
label are two names for the same thing.

`Task` is one of the three context features AdaStress feeds into its context
gate. So on WESAD, the context gate has direct access to the answer.

### A number worth noting

Reading `Task` alone, using the **released** task order rather than the
corrected one, scores **87.9%**.

The paper reports **88.4%** accuracy for AdaStress on WESAD.

This is a coincidence we cannot resolve from the outside, and we are not
claiming the paper's number is only leakage. But the two figures are close
enough that the question should be asked, and it is easy to settle: report
the WESAD accuracy of the same model with the context inputs removed.

---

## Is there any signal in the physiology at all?

A floor check, not an AdaStress result. A plain random forest on the mean and
standard deviation of each feature per window, no self-supervision, no context
gate, trained on 9 subjects and scored on 3 held-out subjects.

| measure | value |
|---|---:|
| accuracy | 0.589 |
| balanced accuracy | 0.443 |
| macro F1 | 0.429 |
| always guess the most common class | 0.516 |
| read `Task` alone | 1.000 |

Confusion matrix, rows are truth:

| | predicted Low | predicted Medium | predicted High |
|---|---:|---:|---:|
| true Low | 58 | 0 | 6 |
| true Medium | 7 | 2 | 6 |
| true High | 23 | 9 | 13 |

So the physiology does carry something: 58.9% against 51.6% for guessing, and
balanced accuracy of 44.3% against 33.3% for chance. But it is weak. The model
predicts Low for most windows, finds Medium almost never (2 of 15), and calls
half of the genuinely high-stress windows Low.

The point of this check: a model reporting far more than about 60% on this
data is almost certainly reading the context, not the body.

---

## Two smaller problems

**Subject S9 has no MediumStress rows at all.** `pd.qcut` cuts each subject's
five questionnaire scores into three groups, and when two scores tie a group
vanishes. Class counts also swing widely between subjects: 22/11/17 for most,
but 22/22/6 for S2 and 33/0/17 for S9. Because the cut is per subject,
HighStress for one subject is not the same feeling level as HighStress for
another.

**Two `_skew` columns are mathematically saturated.** `ACC_Mag_Wrist_skew`
runs exactly −12.53 to +12.53. For 160 samples, which is 32 Hz over a 5-second
bin, the largest skew arithmetically possible is exactly 12.53. So 103 of
those values are pinned at the limit, which happens when a bin is constant
apart from a single sample. `TEMP_skew` runs ±4.13, and the limit for 20
samples (4 Hz over 5 seconds) is exactly 4.13. Computing skew on 20 samples of
near-constant skin temperature produces a number at its arithmetic bound, not
information. This is what the hundreds of "precision loss" warnings during
preprocessing were reporting.

---

## What this means for the experiments

The two context strategies still get compared, but the comparison has to be
read with Finding 5 in view.

| run | context input | what it will show |
|---|---|---|
| no context | none | the honest number. Compare against 51.6% guessing and 58.9% floor. |
| released context | `Task`, `SessionPhase`, `Activity` | expected near 100%. A demonstration of leakage, not of performance. |
| shared accelerometer context | movement statistics from both datasets | the only context that can transfer to `user1`. `Activity` already scores 0.516, the same as guessing, so expect little gain. |

Reporting the second run as a performance figure would be misleading. It will
be reported as a measurement of the leakage.

---

## Notes on `user1`

Its 22 files hold no stress label. They are 11 signal types across two
recording sessions, about 1.5 hours and 1.0 hours. Even if the STAI scores are
found on paper, two sessions give about two labels.

So fine-tuning on `user1` was never possible. Stage 4 will be inference only:
compare the distribution of stress scores between the two context strategies
and look at what the context gate does. This was decided before the count was
known, and recorded in `docs/00_decisions.md` under D2.

One useful discovery: `user1` has an `ACT_INT` column holding the words
`sedentary`, `light` and `moderate`. WESAD's `Activity` is `Still`,
`SlightMovement`, `HeavyMovement`. Those are the same idea in three levels, and
may be a better shared context than raw accelerometer statistics, which have
a scale problem between a chest sensor and a wrist sensor.

---

## Withdrawn

Earlier results from this project are withdrawn. The adaptation script used,
`scripts/run_adaptation_demo.py`, contains:

```python
def label_oracle(windows_batch):
    idx = np.random.choice(len(y), size=n, replace=True)
    return y[idx]
```

It draws labels at random rather than returning the labels for the windows it
was given, so the continual-learning step trained on random labels. The
"10 concept drift events" and the stress score moving from 1.30 to 1.01 are
therefore not reportable. Recorded in `docs/00_decisions.md` under D5.

---

## Questions for the AdaStress authors

1. `Preproc.ipynb` cell 34 hard-codes the task order and never reads the
   `# ORDER` line, but 9 of 15 WESAD subjects have a different order. Was this
   known?
2. Windows are 15 rows but each label reaches only 11, so no window is fully
   labelled. How were the labelled windows built for the reported WESAD result?
3. Which script produces `all_subjects_unlabelled.csv`,
   `generalized_labelled.csv` and `finetune_labelled.csv`?
4. On WESAD, `Task` predicts the stress label perfectly and is also fed to the
   context gate. What is the WESAD accuracy with the context inputs removed?
5. Is deriving context from the shared accelerometer stream in line with what
   the context gate was intended to do?
