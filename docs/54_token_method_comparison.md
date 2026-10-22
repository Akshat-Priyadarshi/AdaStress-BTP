# Which Chronos tokeniser, and for which channel

Written by `btp/29_token_report.py` from the tokens already
saved by `btp/28_chronos_tokens.py`. Nothing is recomputed.

SensorLLM ships two tokenisers and they disagree:

| method | what it does | from |
|---|---|---|
| `MeanScaleUniformBins` | `x / mean(abs(x))` | upstream Chronos |
| `StanNormalizeUniformBins` | `(x - mean) / std` | SensorLLM added it |

## Distinct tokens used, out of 4094 possible

| channel | MeanScale | verdict | StanNormalize | verdict |
|---|---:|---|---:|---|
| BVP | 4058 | alive | 2416 | alive |
| ACC_X | 3735 | alive | 3459 | alive |
| ACC_Y | 3286 | alive | 3522 | alive |
| ACC_Z | 3409 | alive | 3511 | alive |
| EDA | 3336 | alive | 3550 | alive |
| PR | 413 | alive | 2999 | alive |
| SkinTemp | 22 | DEAD, one bin dominates | 3341 | alive |

## Share taken by the single most common token

Above 0.5 means one bin swallows the channel.

| channel | MeanScale | StanNormalize |
|---|---:|---:|
| BVP | 0.036 | 0.036 |
| ACC_X | 0.112 | 0.034 |
| ACC_Y | 0.077 | 0.029 |
| ACC_Z | 0.146 | 0.052 |
| EDA | 0.090 | 0.004 |
| PR | 0.044 | 0.005 |
| SkinTemp | 0.982 | 0.026 |

## Which method for which channel

| channel | use |
|---|---|
| BVP | MeanScaleUniformBins |
| ACC_X | either |
| ACC_Y | either |
| ACC_Z | either |
| EDA | either |
| PR | StanNormalizeUniformBins |
| SkinTemp | StanNormalizeUniformBins |

## Why

Signals centred on zero, such as BVP and the accelerometer, work
under either method: dividing by the average size already spreads
them across the bins.

Signals with a large offset do not. Skin temperature sits around
31 degrees, so `x / mean(abs(x))` is about 1.0 for every sample
and the real variation, a fraction of a degree, is swamped.
Pulse rate has the same problem around 80 bpm. Subtracting the
mean first recovers both.

This is very likely why SensorLLM added the second tokeniser;
upstream Chronos ships only the first.

