# Leak check on the user context

Written by `btp/32_user_leak_check.py`. No model is trained.

For each context column: learn the most common label for each of
its values from four users, then apply that lookup to the fifth.
Whatever it scores, a model gets for free just by reading the
column.

This is the test that found `Task` predicting the WESAD label
with an accuracy of exactly 1.000 in every rotation, because the
label came from one questionnaire per session and `Task` was the
session.

| | |
|---|---|
| windows | 5119 |
| users | user17, user18, user25, user26, user27 |
| guessing baseline | 0.257 |

## Results

| column | user17 | user18 | user25 | user26 | user27 | mean | Cramer's V | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `IS_WEEKEND` | 0.557 | 0.000 | 0.226 | 0.000 | 0.500 | **0.257** | nan | carries nothing, at or below guessing |
| `RESP` | 0.481 | 0.036 | 0.226 | 0.000 | 0.483 | **0.245** | 0.110 | carries nothing, at or below guessing |
| `ACT_INT` | 0.513 | 0.000 | 0.222 | 0.025 | 0.444 | **0.241** | 0.060 | carries nothing, at or below guessing |
| `SKIN_TEMP` | 0.397 | 0.000 | 0.241 | 0.033 | 0.500 | **0.234** | 0.307 | carries nothing, at or below guessing |
| `TILT` | 0.363 | 0.098 | 0.208 | 0.141 | 0.297 | **0.222** | 0.198 | carries nothing, at or below guessing |
| `PR` | 0.393 | 0.075 | 0.208 | 0.000 | 0.408 | **0.217** | 0.222 | carries nothing, at or below guessing |
| `EDA_PHASIC_N` | 0.342 | 0.000 | 0.226 | 0.010 | 0.500 | **0.216** | 0.094 | carries nothing, at or below guessing |
| `POSTURE` | 0.406 | 0.061 | 0.160 | 0.000 | 0.422 | **0.210** | 0.119 | carries nothing, at or below guessing |
| `MET` | 0.362 | 0.030 | 0.260 | 0.000 | 0.356 | **0.201** | 0.177 | carries nothing, at or below guessing |
| `ACT_COUNT` | 0.281 | 0.000 | 0.235 | 0.000 | 0.447 | **0.193** | 0.111 | carries nothing, at or below guessing |
| `EDA_TONIC` | 0.148 | 0.098 | 0.367 | 0.000 | 0.342 | **0.191** | 0.342 | carries nothing, at or below guessing |
| `HOUR` | 0.244 | 0.118 | 0.072 | 0.000 | 0.447 | **0.176** | 0.253 | carries nothing, at or below guessing |
| guessing | 0.557 | 0.000 | 0.226 | 0.000 | 0.500 | **0.257** | | |

## How to read this

Two different things can make a column score well, and they
should not be confused.

**A body signal predicting stress is the finding.** If pulse
rate or EDA predicts the label, that is what the whole project
is looking for.

**A clock or bookkeeping column predicting stress is a
problem**, because it says more about when the survey was
answered than about how the person felt. `HOUR` is the one to
watch.

## Note on the survey's own context

`responses_scored.csv` also holds what people reported about
themselves: activity, situation, environment, tiredness, and the
main source of stress in the last hour.

**Main source of stress must not become a feature.** It is close
to asking the label directly, and would be a `Task` of its own.
It is useful for explaining results, not for making them.

