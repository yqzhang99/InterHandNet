# InterHandNet

Official implementation of **InterHandNet: Capturing Two-hand Interaction for
Robust Hand-washing Activity Recognition**, accepted to
[IEEE PerCom 2025](https://doi.org/10.1109/PerCom64205.2025.00021)
(full paper acceptance rate 15/152).

📄 [Paper PDF](paper/Percom2025Hand.pdf) · [DOI](https://doi.org/10.1109/PerCom64205.2025.00021)

InterHandNet recognises the six hand-washing steps defined by the World Health
Organization from a sequence of two-hand 3D skeletons extracted from an RGB-D
camera. It targets two properties that set hand-washing apart from ordinary
activity recognition: the steps are defined by the *interaction* between the two
hands, and the hands *occlude each other* almost constantly.

## The three proposed modules

| Module | Paper | What it does |
| --- | --- | --- |
| **Interaction Graph** | Eq. (2) | Adds cross-hand edges, weighted by the 3D distance between corresponding keypoints, to the spatial graph convolution. |
| **InterHand Temporal Fusion** | Eq. (4) | Reconstructs an occluded hand by querying its own time window with the other hand's feature. |
| **Interaction Attention** | Eq. (6), (7) | Lets each hand attend over the other hand's spatio-temporal features through query-key-value. |

All three are plain `nn.Module`s that keep the interface of a standard STGCN
block, which is the "strong compatibility" of Section III-F: they drop into
ST-GCN, 2s-AGCN, MS-AAGCN, STA-GCN, CTR-GCN and FR Head without touching the
backbone's own logic.

## Results from the paper

Weighted average over the six camera settings of the dataset by Lulla et al.,
5-fold cross-validation per setting (Table II). `*` marks RGB-based methods.

| Method | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| ST-GCN | 0.5339 | 0.5051 | 0.4049 | 0.3849 |
| MobileNetV2* | 0.6403 | – | – | – |
| Xception* | 0.6683 | – | – | – |
| 2s-AGCN | 0.6711 | 0.6834 | 0.6212 | 0.6197 |
| MS-AAGCN | 0.7393 | 0.7616 | 0.6933 | 0.7009 |
| CTR-GCN | 0.7605 | 0.7547 | 0.7140 | 0.7038 |
| FR Head | 0.7660 | 0.7572 | 0.7084 | 0.7133 |
| STA-GCN | 0.7803 | 0.7659 | 0.7334 | 0.7353 |
| MS-G3D | 0.7922 | 0.7931 | 0.7604 | 0.7569 |
| **InterHandNet (FR Head)** | 0.8106 | **0.8195** | 0.7841 | 0.7824 |
| **InterHandNet (STA-GCN)** | **0.8170** | 0.8121 | **0.7964** | **0.7951** |

Adding the modules to a backbone improves it by up to 40% in F1 (Table III), and
one-second inference on an NVIDIA Jetson Orin Nano costs 7.4 ms with the ST-GCN
backbone against more than 6.5 s for a 3D CNN on RGB (Table VI).

## Installation

```bash
git clone https://github.com/yqzhang99/InterHandNet.git
cd InterHandNet
pip install -e .
```

Optional extras: `".[extraction]"` for MediaPipe skeleton extraction, `".[onnx]"`
for ONNX export, `".[dev]"` for the test suite.

## Quick start

```python
import torch
from interhandnet import InterHandNet

model = InterHandNet(num_classes=6)

# One second of two-hand skeleton data at 30 fps:
# 3 coordinates, 30 frames, 42 joints (left hand 0..20, right hand 21..41).
skeleton = torch.randn(1, 3, 30, 42)
logits = model(skeleton)  # (1, 6)
```

Every module can be switched off individually, which reproduces the ablation
rows of Table III and Table IV:

```python
baseline = InterHandNet(
    num_classes=6,
    use_interaction_graph=False,
    use_interaction_attention=False,
    use_interhand_temporal_fusion=False,
)  # plain ST-GCN backbone
```

## Training

Prepare a skeleton archive first — see [docs/dataset.md](docs/dataset.md):

```bash
python tools/extract_skeletons.py \
    --videos data/raw/lulla \
    --output data/handwashing_mediapipe.npz
```

Train one fold:

```bash
python tools/train.py --config configs/interhandnet.yaml --camera 100 --fold 0
```

Run the full evaluation protocol of Section IV-B, five temporally ordered folds
per camera setting combined into a weighted average:

```bash
python tools/cross_validate.py --config configs/interhandnet.yaml
```

Evaluate a checkpoint and print the per-step confusion matrix of Fig. 9:

```bash
python tools/evaluate.py \
    --config configs/interhandnet.yaml \
    --checkpoint work_dirs/interhandnet/best.pt \
    --camera 100 --fold 0
```

Any config value can be overridden from the command line:

```bash
python tools/train.py --config configs/interhandnet.yaml \
    --set training.epochs=20 data.batch_size=32 model.num_heads=8
```

## Configurations

| Config | Modules enabled | Paper row |
| --- | --- | --- |
| `configs/stgcn_baseline.yaml` | none | ST-GCN baseline |
| `configs/ablation_ig.yaml` | IG | `+IG` |
| `configs/ablation_ia.yaml` | IA | `+IA` |
| `configs/ablation_ig_ia.yaml` | IG + IA | `+IG/IA` |
| `configs/interhandnet.yaml` | IG + IA + ITF | `+IG/IA/ITF` |

`configs/base.yaml` holds the shared settings, including the hyperparameters of
Section IV-C: 50 epochs, learning rate 0.01, SGD with momentum 0.9 and weight
decay 0.0005, cross-entropy loss.

## Edge deployment

The paper runs the full pipeline on an NVIDIA Jetson Orin Nano with a Luxonis
OAK-D camera. See [deploy/jetson/README.md](deploy/jetson/README.md):

```bash
python tools/export_onnx.py --config configs/interhandnet.yaml \
    --checkpoint work_dirs/interhandnet/best.pt --output interhandnet.onnx

# On the Jetson:
python deploy/jetson/build_engine.py --onnx interhandnet.onnx \
    --engine interhandnet.engine --precision fp16
python deploy/jetson/realtime_demo.py --engine interhandnet.engine --camera 0
```

Reproduce the latency protocol of Table VI (1,800 samples, first five excluded
as warm-up):

```bash
python tools/benchmark_latency.py --config configs/interhandnet.yaml --device cuda:0
```

## Using the modules with another backbone

The modules do not depend on the ST-GCN backbone. To add them to an existing
STGCN-based network, replace its spatial graph convolution with
`SpatialInteractionGraphConv` and wrap its temporal convolution:

```python
from interhandnet.graph import HandGraph, InteractionGraph
from interhandnet.modules import (
    FeatureExtractor,
    InterHandTemporalFusion,
    InteractionAttention,
    SpatialInteractionGraphConv,
    pairwise_distance_matrix,
)

# Once per forward pass, from the raw input coordinates:
distance = pairwise_distance_matrix(x[:, :3])          # (N, T, V, V)

# Inside a block:
spatial = spatial_conv(x, adjacency, interaction_adjacency, distance)  # Eq. (2)
temporal = temporal_conv(temporal_fusion(spatial))                     # Eq. (4)
temporal = feature_extractor(temporal)                                 # Fig. 5
out = feature_extractor_2(temporal + interaction_attention(temporal))  # Eq. (6), (7)
```

`interhandnet/models/stgc_block.py` is the worked example. The one constraint is
that the distance matrix must have the same temporal length as the features, so a
block after a strided temporal convolution needs a resampled `D`.

## Repository layout

```
interhandnet/
├── graph/          A (physical hand graph) and A_IG (Interaction Graph), Fig. 3
├── modules/        the three proposed modules and the Feature Extractor
├── models/         STGC block and the InterHandNet model, Fig. 2
├── data/           dataset, transforms, skeleton extraction, CV splits
├── engine/         trainer, evaluator, metrics
└── utils/          config, logging, seeding
tools/              train, cross_validate, evaluate, export_onnx, benchmark, extract
deploy/jetson/      ONNX to TensorRT engine and the real-time demo
configs/            experiment configurations
docs/               implementation notes and dataset preparation
tests/              pytest suite
paper/              the PerCom 2025 paper
```

## Documentation

* [docs/implementation_notes.md](docs/implementation_notes.md) — equation-to-code
  map and every decision the paper leaves open, including the two readings of
  `A_IG D` and the query/key assignment in InterHand Temporal Fusion.
* [docs/dataset.md](docs/dataset.md) — obtaining the datasets and building the
  skeleton archive.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite checks the graph construction against Fig. 3, the multi-head tensor
layouts, the shape and behaviour of each module (including that an occluded hand
is reconstructed from its own time window), every ablation combination, the
shipped configs and the ONNX export.

## Citation

```bibtex
@INPROCEEDINGS{11018699,
  author    = {Zhang, Yiqing and Maekawa, Takuya},
  title     = {InterHandNet: Capturing Two-hand Interaction for Robust Hand-washing Activity Recognition},
  booktitle = {2025 IEEE International Conference on Pervasive Computing and Communications (PerCom)},
  year      = {2025},
  pages     = {13--24},
  keywords  = {Hands; Runtime; Activity recognition; Cameras; Skeleton; Robustness; Real-time systems; Wearable devices; Usability; Software development management; Hand-washing activity recognition; hand skeleton; RGB-D camera},
  doi       = {10.1109/PerCom64205.2025.00021}
}
```

## Acknowledgements

The backbone is based on [ST-GCN](https://github.com/yysijie/st-gcn); thanks to
the original authors for their excellent work. Skeletons are extracted with
[MediaPipe Hands](https://arxiv.org/abs/2006.10214). This study is partially
supported by JSPS KAKENHI Grant Number JP21H05299.

## License

Released under the [MIT License](LICENSE).
