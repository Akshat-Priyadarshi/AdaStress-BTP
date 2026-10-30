# Sample: user25, 2026-09-16

One person's day, packed so it runs on another machine with
nothing else from the project.

| file | what it is |
|---|---|
| `signals.csv` | 7 channels, one row per second |
| `context.csv` | 8 context columns, every 30 seconds |
| `truth.csv` | what the person reported, for checking |
| `model.pth` | the trained model |
| `encoder.pth` | the frozen encoder, trained on WESAD |
| `scaler.pkl` | the scaling the encoder was trained with |

## Running it

```
python3 52_edge_demo.py --sample sample_data/user25_2026-09-16
```

Needs `numpy`, `pandas`, `torch`, `scikit-learn`, `joblib`, and
the `vendor/adastress` package for the model classes.

**numpy must stay below 2.** torch and pandas here were built
against 1.x and all of them break together if it is upgraded.

## What the numbers mean

Three probabilities per window, turned into one score:

```
stress_score = 1 x P(Medium) + 2 x P(High)     0 to 2
```

That formula is ours. AdaStress outputs the three
probabilities and nothing more.

## How much to trust this

This is one day of one person. It shows the pipeline working,
not how well the model works.

The numbers that carry weight, measured across five people
against their own reported stress:

| test | balanced accuracy | chance |
|---|---:|---:|
| held-out WESAD subjects | 0.935 | 0.531 |
| pooled random split | 0.890 | 0.333 |
| survey answers kept whole | 0.380 | 0.333 |
| a person never seen before | 0.286 | 0.333 |

The random split is high because windows overlap by 30 seconds
and each survey answer labels about 60 of them, so near
identical windows land on both sides. All 100 survey answers
had windows in both training and test. The model recognises
moments it has already seen.

**Transfer to a new person does not work on this data.** Five
people used a self-report scale in five different parts of its
range. One reported low stress on all 18 occasions; another
never reported low at all.

One more thing worth saying before anyone notices it: the score
tends to peak between four and eight in the morning. At night
the body is still and skin temperature shifts, and in WESAD
those same patterns meant the stress task.

