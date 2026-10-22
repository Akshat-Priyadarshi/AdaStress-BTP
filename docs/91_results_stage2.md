# Stage 2 results: WESAD training runs

Companion to `docs/90_findings_stage1.md`, which covers what was found in the
data before any model was trained. This one covers the models.

Every number is the average of three seeds (42, 1, 2) on the same
subject-held-out split. Built from `runs/*/metrics.json` by
`btp/06_collect_results.py`, not typed by hand.

---

## Setup

| | |
|---|---|
| labelled windows | 617 |
| split | by subject: train 9, validation 3, test 3 |
| train subjects | S2 S3 S4 S5 S6 S7 S8 S9 S10 |
| validation subjects | S11 S13 S14 |
| test subjects | S15 S16 S17 |
| test windows | 124 |
| signal channels | 56 |
| encoder | trained once on the train subjects, reused by every run |
| seeds | 42, 1, 2 |

Only the context branch differs between runs. Same encoder file, same split,
same epochs. So any difference between the rows below is caused by the context
input and nothing else.

---

## Results

| context input | accuracy | balanced accuracy | macro F1 | gate |
|---|---:|---:|---:|---:|
| always guess the most common class | 0.516 | 0.333 | - | - |
| **none** | **0.529** (spread 0.008) | 0.428 | 0.335 | 0.649 |
| **`Task`, `SessionPhase`, `Activity`** | **0.710** (spread 0.057) | 0.600 | 0.604 | 0.455 |
| **Case 1: the above plus four zero columns** | **0.699** (spread 0.089) | 0.594 | 0.597 | 0.465 |
| **Case 2: accelerometer movement** | **0.481** (spread 0.024) | 0.440 | 0.413 | 0.549 |
| a plain random forest, no context | 0.589 | 0.443 | 0.429 | - |
| reading `Task` alone, no model at all | **1.000** | - | - | - |

"gate" is the model's own weighting. Near 1 means it is leaning on the body
signals; lower means it is leaning on the context.

---

## What each row means

### No context: 0.529 against a baseline of 0.516

This is the honest reproduction number. The full pipeline, self-supervised
pretraining and all, beats guessing by about one point.

It also predicts almost nothing but LowStress. Across seeds it assigned
HighStress to between 0 and 1 of the 45 genuinely high-stress test windows.
Balanced accuracy of 0.428 against 0.333 for chance shows there is a trace of
signal, but only a trace.

The paper reports 88.4% on WESAD. We get 52.9%.

### The pipeline is beaten by a random forest

A random forest on nothing but the mean and standard deviation of each feature
per window scores 0.589, against 0.529 for the encoder plus gated classifier.
Same split, same windows, same features.

So on this data the contrastive pretraining and the gated fusion are not
buying anything over a simple baseline. Worth stating plainly, because it is
easy to assume the deep pipeline must be doing something.

### Adding `Task` adds 18 points, and `Task` is the label

0.529 with no context, 0.710 with it. Nothing else changed between those runs,
not even the encoder file.

Stage 1 measured why: `Task` predicts the label with 100% accuracy on the
held-out subjects, because the stress label comes from one questionnaire per
session and `Task` **is** the session. They are two names for the same thing.

So the 18 points are leakage, not context modelling. Read as a leakage
measurement, this run is useful. Read as a performance figure, it is
misleading.

The gate agrees. It sits at 0.649 with no context and drops to 0.455 when
`Task` is available: the model measurably moves its weight onto context
exactly when context holds the answer.

### Case 1, missing context filled with zeros: 0.699

Case 1 widens the context to include the four columns `user1` has and WESAD
does not (`steps`, `MET`, `ACT_COUNT`, `ACT_INT`), and fills them with zeros
for WESAD.

**Result: 0.699, against 0.710 for the same context without the padding.** The
spreads are 0.089 and 0.057, so those two numbers cannot be told apart.

That is the useful finding: **on the source dataset, zero-padding the missing
context is harmless.** It neither helps nor hurts. One-hot encoding is what
makes this work. An all-zero block is not any category, so "absent" is
unambiguous and the gate can ignore it. With plain whole numbers, 0 would be a
real category and the model would learn "stress happens at exactly zero steps".

**But this number must not be used to prefer Case 1 over Case 2.** Case 1 is
`Task` plus four zeros, so it carries the whole leak. Its apparent 22-point
advantage over Case 2 is the leak, not the padding strategy. On `user1` the
four columns stop being zero and `Task` does not exist at all, so none of that
advantage can transfer. WESAD cannot tell us which case works on `user1`.

### Case 2, shared accelerometer context: 0.481

Worse than guessing. Per subject it is uneven and sometimes bad: S17 averages
0.285 across seeds, well below the 0.516 baseline.

Balanced accuracy is slightly higher than the no-context run, 0.440 against
0.428, because the model stops collapsing onto LowStress. But the extra
predictions are mostly wrong, so raw accuracy falls.

This matches Stage 1, where `Activity` scored 0.516, exactly the guessing
baseline. Movement in a seated laboratory protocol carries no information
about stress, so a context feature built from movement has nothing to add.

**This does not mean Case 2 is the wrong choice for `user1`.** It means WESAD
is the wrong place to test it. In a lab protocol everyone sits still, so
movement is nearly constant. In `user1`, recorded during ordinary activity,
movement varies and may genuinely separate physical exertion from stress. That
is the whole reason the AdaStress paper wants a context gate.

---

## Accuracy for each test subject

One number hides that the label means something different for each person.

| context | S15 | S16 | S17 |
|---|---:|---:|---:|
| none | 0.480 | 0.571 | 0.537 |
| `Task` etc. | 0.650 | 0.698 | 0.780 |
| Case 1, zeros | 0.707 | 0.684 | 0.772 |
| Case 2, accelerometer | 0.537 | 0.619 | 0.285 |

Case 2 works about as well as no context for S15 and S16, then fails badly for
S17. A single average would hide that completely.

---

## Finding 6: the labels are not comparable between subjects

New from Stage 2, and it explains the collapse.

`pd.qcut` cuts each subject's five questionnaire scores into thirds
**separately for each subject.** So "HighStress" means "the highest of this
person's five scores", not a shared level of stress.

The validation subjects show it directly:

| subject | Low | Medium | High | which class dominates |
|---|---:|---:|---:|---|
| training subjects, typical | ~320 | ~130 | ~190 | Low |
| S11 (validation) | 169 | **317** | 143 | **Medium** |
| S13 (validation) | 324 | 85 | 221 | Low |
| S14 (validation) | 170 | 77 | **378** | **High** |

S11's baseline session is labelled Medium. S14's is High. The training
subjects' is Low. Same session, same protocol, three different answers.

The consequence showed up as a flat line. Validation accuracy sat at 0.342 in
every run at every epoch while training accuracy climbed to 0.79. Choosing the
best epoch on validation therefore picked epoch 1, an untrained model that
answers LowStress to everything. That is why the first pass of these runs all
landed exactly on the guessing baseline.

We report the final epoch instead, and record that we did
(`--select-on last`). Both versions are kept in `runs/`.

A model fitted on 9 subjects cannot transfer to 3 others when the target means
something different for each person. This is a property of the labelling, not
of the model.

---

## Finding 7: the deep pipeline underperforms a random forest

Stated above, repeated here because it belongs on the findings list. 0.589 for
a random forest on window statistics, 0.529 for the self-supervised encoder
with gated context fusion. Same data, same split.

---

## Where the reproduction stands

| | accuracy |
|---|---:|
| paper, AdaStress on WESAD | 0.884 |
| ours, no context, subject-held-out | **0.529** |
| ours, with the released context | **0.710** |
| reading `Task` alone, no model | **1.000** |
| guessing the most common class | 0.516 |

We did not reproduce 0.884, and there are now seven measured reasons why. The
whole gap between 0.529 and 0.710 is caused by one context feature that is a
deterministic function of the label.

---

## What Stage 2 settles, and what it does not

Settled:

- Zero-padding missing context is harmless on the source dataset, provided
  one-hot encoding is used. Case 1 is a valid construction.
- The context gate visibly shifts its weight onto context when context is
  informative, so the mechanism works as described.
- Physiology alone, under a subject-held-out split, barely beats guessing on
  WESAD as this pipeline labels it.

Not settled, and not settleable on WESAD:

- Whether Case 1 or Case 2 is better for `user1`. Case 1 wins on WESAD only
  because it carries the leak, and Case 2 loses on WESAD only because everyone
  sits still in a laboratory. Neither reason applies to `user1`.

That question needs `user1` itself, which is Stage 3 and Stage 4. With no
labels there (decision D2), the comparison will be between score distributions
and gate behaviour rather than accuracy.

---

## Runs on disk

| run id | context | epoch reported | seeds |
|---|---|---|---|
| `R01_wesad_nocontext` | none | best validation | 42 |
| `R01b_wesad_nocontext_noskew` | none, `_skew` dropped | best validation | 42 |
| `R02_wesad_context_original` | original | best validation | 42 |
| `R03_wesad_context_derived` | derived | best validation | 42 |
| `R01L_wesad_nocontext_last` | none | last | 42 |
| `R02L_wesad_context_original_last` | original | last | 42 |
| `R03L_wesad_context_derived_last` | derived | last | 42 |
| `R04L_wesad_context_zero_last` | zero (Case 1) | last | 42 |
| `R0L_{none,original,derived,zero}_seed{1,2}` | all four | last | 1, 2 |

Sixteen runs, none overwritten. Each folder holds its own config, encoder
hash, metrics, per-window predictions and log.

`R01b` shows dropping the nine saturated `_skew` columns changes nothing:
0.516 accuracy either way, the same confusion matrix. Decision D10 stands but
does not affect any conclusion.
