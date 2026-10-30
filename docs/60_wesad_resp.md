# Breathing rate derived from WESAD's wrist BVP

Written by `btp/38_wesad_resp.py`.

## Why derive it when WESAD has a real belt

Because the users do not have one. WESAD records respiration on
a chest strap, but if the encoder learned from a measured belt
signal and were then fed one derived from pulse timing, those
are two different quantities sharing a column name.

So respiration is derived from BVP on both sides. The belt is
kept for checking only, and this is the only ground truth
available anywhere in the project.

## The method

Breathing leaves a trace in the heart rhythm: the gap between
beats shortens on the in-breath and lengthens on the out-breath.

```
BVP at 64 Hz
  -> band-pass 0.5 to 8 Hz
  -> peaks, at least 0.33 s apart
  -> gaps between beats, 0.33 to 1.50 s kept
  -> resample to 4 Hz
  -> band-pass 0.1 to 0.5 Hz  (6 to 30 breaths a minute)
  -> count peaks per 30 s, report per second
```

## Agreement with the belt

| subject | beats | coverage | correlation | median diff | derived | belt |
|---|---:|---:|---:|---:|---:|---:|
| S10 | 8040 | 99% | 0.156 | 2.00 | 18.0 | 16.0 |
| S11 | 6204 | 98% | 0.191 | 4.00 | 20.0 | 16.0 |
| S13 | 7365 | 99% | 0.453 | 2.00 | 18.0 | 16.0 |
| S14 | 7856 | 99% | -0.142 | 4.00 | 20.0 | 16.0 |
| S15 | 6081 | 93% | 0.148 | 4.00 | 18.0 | 16.0 |
| S16 | 7321 | 99% | 0.093 | 4.00 | 20.0 | 18.0 |
| S17 | 7264 | 99% | 0.132 | 2.00 | 18.0 | 16.0 |
| S2 | 7603 | 100% | 0.198 | 2.00 | 20.0 | 20.0 |
| S3 | 6948 | 98% | 0.354 | 2.00 | 18.0 | 16.0 |
| S4 | 7308 | 100% | 0.155 | 2.00 | 20.0 | 18.0 |
| S5 | 7046 | 93% | 0.090 | 2.00 | 18.0 | 16.0 |
| S6 | 8321 | 100% | 0.202 | 2.00 | 20.0 | 18.0 |
| S7 | 6302 | 99% | 0.449 | 2.00 | 18.0 | 18.0 |
| S8 | 6690 | 99% | 0.004 | 4.00 | 20.0 | 16.0 |
| S9 | 6996 | 99% | 0.293 | 4.00 | 18.0 | 14.0 |
| **mean** | | **98%** | **0.185** | **2.80** | 18.9 | 16.7 |

## What this decides

If the derived rate follows the belt, the method can be trusted
on the users, where there is nothing to check it against, and
the seventh channel is worth the rebuild.

If it does not, adding it would be adding noise, and the 44 GB
of re-downloading it would need is not justified.

