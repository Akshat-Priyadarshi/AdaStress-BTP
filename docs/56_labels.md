# Labels from the survey

Written by `btp/31_build_labels.py`.

## The instrument

The six items are the STAI-6, the validated short form of the
State-Trait Anxiety Inventory, so the scoring rule is published
rather than invented:

    reverse calm, relaxed and content   (4->1, 3->2, 2->3, 1->4)
    add all six items                   ->  6 to 24
    multiply by 20/6                    -> 20 to 80

| | |
|---|---|
| responses used | 91 |
| score range | 20.0 to 80.0 |
| cut points | 46.7 and 56.7 |
| reach | 15 minutes either side |

## Why the classes are cut across everyone

Not per person. That is the single thing that went wrong with
WESAD: each subject's five scores were cut into their own
thirds, so HighStress meant "the worst of this person's own
five sessions". S11's calm baseline came out Medium and S14's
came out High, with nothing different about their bodies.

Here a score of 60 means the same level of anxiety whoever
reported it. We can do that because these people answered 6 to
31 times each rather than 5, and because the STAI-6 scale is
absolute rather than relative.

## Responses

| user | responses | matched to windows | windows |
|---|---:|---:|---:|
| user17 | 13 | 13 | 705 |
| user18 | 31 | 28 | 1649 |
| user25 | 23 | 23 | 1329 |
| user26 | 18 | 18 | 1076 |
| user27 | 6 | 6 | 360 |
| **total** | **91** | | **5119** |

## Classes

| | responses | windows |
|---|---:|---:|
| LowStress | 33 | 1949 |
| MediumStress | 27 | 1567 |
| HighStress | 31 | 1603 |

Always answering the most common class scores **38.1%**.
Every accuracy must be read next to that.

### Per user

| Subject_ID   |   LowStress |   MediumStress |   HighStress |
|:-------------|------------:|---------------:|-------------:|
| user17       |         393 |             98 |          214 |
| user18       |           0 |           1229 |          420 |
| user25       |         300 |            240 |          789 |
| user26       |        1076 |              0 |            0 |
| user27       |         180 |              0 |          180 |

**user26 answered with only one class.** Holding such a user
out gives a fold where always guessing that class scores
100%. Read balanced accuracy there, and flag the fold.

## Window reach

Each response labels the windows within 15 minutes either side,
about 60 windows per response. The survey asks about "the last
hour", so an answer describes a stretch of time rather than an
instant. Reaching further would make more windows share one
answer, and they would start behaving like copies of each other
rather than independent examples. Where two responses both
reach a window, the nearer one wins.

