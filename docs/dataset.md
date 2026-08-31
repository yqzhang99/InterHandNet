# Preparing the data

The paper evaluates on two public datasets. Neither is redistributed here;
download them from the original authors and build a skeleton archive locally.

## Datasets used in the paper

| Dataset | Content | Used for |
| --- | --- | --- |
| [Lulla et al. (2021)](https://doi.org/10.3390/data6040038) | 3,185 hand-washing videos, over 23 hours, six camera settings (100..105), annotated with the six WHO steps | Table II, Table III, Table IV |
| [Xie et al. (2022)](https://doi.org/10.1016/j.bspc.2022.103651) | 656 videos annotated with the six WHO steps | Table V, comparison against RGB-based methods |

## Label convention

The Lulla et al. annotations give every frame a movement code:

| Code | Meaning |
| --- | --- |
| 0 | other movement (an incorrectly performed or undefined motion) |
| 1..6 | the six WHO steps of Fig. 1, in order |
| 7 | closing the faucet with a paper towel |

The dataset authors fold code 7 into code 0 for classification, which leaves a
7-way problem. This repository uses the movement code directly as the class
index, so `model.num_classes` defaults to 7 and class 0 means "other". That extra
class is what lets a deployed system report "no step in progress" instead of
being forced to pick one of the six steps. Set `model.num_classes: 6` and drop
the code-0 windows if you only want the WHO steps.

## Building the skeleton archive

`tools/extract_skeletons.py` runs MediaPipe Hands over the videos, cuts the
result into windows and writes a single `.npz` file:

```bash
pip install -e ".[extraction]"

python tools/extract_skeletons.py \
    --videos data/raw/lulla \
    --output data/handwashing_mediapipe.npz \
    --window-size 30
```

Labels and camera settings are parsed from the file paths. The defaults assume

```
data/raw/lulla/<camera>/<name>_<code>.mp4
```

where `<code>` is the movement code of the table above, used as the class index
as-is. Pass `--label-regex`, `--camera-regex` and `--label-offset` for a
different naming scheme; `--label-offset 1` suits file names that encode 1-based
steps with no "other" class.

By default the extractor stores MediaPipe's **world landmarks**, which are metric
3D coordinates. This matters: the distance matrix `D` of Eq. (2) is only
meaningful when the coordinates share a metric scale. Use `--image-landmarks`
only for ablations that do not rely on inter-hand distance.

## Archive layout

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `skeletons` | `(S, T, 42, 3)` float32 | one window per sample; joints `0..20` left hand, `21..41` right hand |
| `labels` | `(S,)` int64 | movement code, `0` = other, `1..6` = WHO steps |
| `cameras` | `(S,)` | camera setting, used to group the cross-validation |
| `session_ids` | `(S,)` | source recording of the window |
| `start_frames` | `(S,)` int64 | window position in the global recording order |

Windows where a hand was not detected contain exact zeros for that hand. Keep
them: this missing data is what InterHand Temporal Fusion reconstructs, and
removing it changes what the ablations measure.

`start_frames` accumulates across videos so that sorting by it reproduces the
recording order. The temporal-order split of Section IV-B relies on that
ordering.

## Using a different skeleton extractor

Table IV of the paper uses [InterWild](https://github.com/facebookresearch/InterWild)
instead of MediaPipe Hands to isolate the Interaction Graph and Interaction
Attention from the effect of missing keypoints. Any extractor works as long as
it writes the archive layout above with 21 joints per hand in the documented
order.

## Sanity checks

```python
import numpy as np

with np.load("data/handwashing_mediapipe.npz", allow_pickle=True) as archive:
    print({key: archive[key].shape for key in archive})
    skeletons = archive["skeletons"]
    missing = np.all(skeletons == 0, axis=-1)
    print("fraction of undetected joints:", missing.mean())
    print("label distribution:", np.bincount(archive["labels"]))
```

A missing-joint fraction of roughly 20-40% is normal for MediaPipe Hands on
hand-washing video, since the hands occlude each other constantly. That is the
occlusion problem the paper sets out to solve.
