# Implementation notes

This document maps the paper onto the code and records the decisions that the
paper leaves open. It is the reference to consult when a number in the code does
not obviously follow from the text.

## Notation and tensor layout

The paper writes node features as `C x T x N`. The code uses the PyTorch
convention `(N, C, T, V)`:

| Paper | Code | Meaning |
| --- | --- | --- |
| `N` (nodes) | `V = 42` | joints, left hand `0..20`, right hand `21..41` |
| `T` | `T = 30` | temporal window, one second at 30 fps |
| `C` | `C` | channels; the input has 3 (x, y, z) |
| `f_l` | block input | features entering the `l`-th STGC block |
| `f^S_l` | `spatial` | output of the spatial graph convolution |
| `f^T_l` | `temporal` | output of the temporal convolution |
| `f^A_l` | block output | output of Interaction Attention |

Joint order per hand follows footnote 1 of Section III-B: palm first, then the
base of the thumb to its tip, ending with the pinky finger.

## Equation to code map

| Paper | Code |
| --- | --- |
| Eq. (1) spatial graph convolution | `modules/interaction_graph.py::SpatialGraphConv` |
| Eq. (2) Interaction Graph | `modules/interaction_graph.py::SpatialInteractionGraphConv` |
| Eq. (3) temporal convolution | `models/stgc_block.py::TemporalConv` |
| Eq. (4) InterHand Temporal Fusion | `modules/interhand_temporal_fusion.py` + `TemporalConv` |
| Eq. (5) scaled dot-product attention | `_attend` in both attention modules |
| Eq. (6), Eq. (7) Interaction Attention | `modules/interaction_attention.py` |
| Fig. 3 graphs `A` and `A_IG` | `graph/hand_graph.py` |
| Fig. 5 Feature Extractor | `modules/feature_extractor.py` |
| Fig. 2 STGC block and network | `models/stgc_block.py`, `models/interhandnet.py` |
| Section III-F backbone compatibility | `models/sta_gcn.py`, `models/base.py` |
| Section IV-B evaluation protocol | `data/splits.py`, `tools/cross_validate.py` |
| Section IV-C hyperparameters | `configs/base.yaml`, `engine/trainer.py` |
| Table VI latency protocol | `tools/benchmark_latency.py` |

## Decisions the paper leaves open

### `M` is a multiplicative mask

Eq. (1) writes `W f (A + M)` and says `M` "can be interpreted flexibly depending
on the specific method used". ST-GCN implements edge importance as an
element-wise mask, so the code uses `A * M` with `M` initialised to ones. Each
temporal branch owns its own mask while sharing the spatial convolution weights,
which is what the original research code did.

### `A_IG D` is a matrix product by default

Section III-C defines `D` as holding "the Euclidean distance in the 3D space
between two corresponding keypoints in the different hands, i.e. the length of
the red edge in Fig. 3", and Eq. (2) adds `A_IG D` to the adjacency. Two
readings are consistent with that text, and both are implemented:

* `distance_fusion="matmul"` (default) computes the matrix product `A_IG @ D`
  over the full pairwise distance matrix. This is what the original research
  code did, so it is the configuration that corresponds to the published
  numbers. Because `A_IG` is the hand-swapping permutation, row `i` of the
  product carries the distances seen from the corresponding joint of the other
  hand.
* `distance_fusion="hadamard"` computes `A_IG * D`, keeping exactly the lengths
  of the 21 red cross-hand edges and zeroing everything else. This is the
  narrower reading of the sentence above.

`D` is computed once per forward pass from the raw input coordinates, before
input normalisation, so that it carries metric distances. The squared distances
are floored with a small epsilon before the square root: the diagonal of `D` is
zero, and `sqrt` has an infinite derivative there, which produced NaN gradients
without the floor.

### Which blocks use the Interaction Graph

`D` is indexed by time, so a block can only use it while its features are at the
same temporal resolution. On the ST-GCN backbone the default applies the
Interaction Graph to the first three blocks, which run at stride 1, following the
published snapshot; the experiment notebooks used blocks 1 and 2 instead. On the
STA-GCN backbone it is applied inside the feature extractor only, because both
branches start with a stride-2 block. When a strided block does use it, the model
resamples `D` with the block's stride so the two stay aligned.

### Query and key assignment in InterHand Temporal Fusion

Eq. (4) reads

```
f^T_{l,t} = sum_k F_k [ (f^{S,R}_{l,t+k} (*) f^{S,L}_l) || (f^{S,L}_{l,t+k} (*) f^{S,R}_l) ]
```

and `(*)` is defined by Eq. (6) with the left operand as the query and the right
operand as key and value. The worked example in Section III-D fixes the intent:

> assume that `f^{S,L}_{0,tau} = 0`, which means the left hand is occluded at
> time `tau`. By employing the attention mechanism, right hand feature
> `f^{S,R}_{0,tau}` automatically selects the most relevant features from left
> hand features `{f^{S,L}_{0,tau-1}, f^{S,L}_{0,tau+1}}` to infer the occluded
> feature `f^{T,L}_{0,tau}`.

So the reconstructed left hand takes its **query from the right hand** and its
**keys and values from the left hand's own time window**. The code implements
this. Note that the original research code had the operands the other way round
(query from the same hand, values from the other hand); with that assignment an
occluded hand contributes a zero query, the softmax degenerates to a uniform
average over the other hand, and nothing is reconstructed from the neighbouring
time steps. The paper's assignment is the one that makes the module do what
Section III-D describes.

Attention runs along the temporal axis independently per joint, because Eq. (4)
mixes time steps within the window and leaves the joint dimension to the graph
convolution.

### Scope of Interaction Attention

Section III-E says Interaction Attention "fuses the spatial-temporal features
from both hands", which reads as if every (time step, joint) pair were a token.
Eq. (6) and Eq. (7) support that reading too: unlike Eq. (1) and Eq. (2), which
carry the time index `f_{l,t}`, they are written over `f_L` and `f_R` with no
time index at all.

The original research code instead attends within each frame, giving a 21 x 21
score matrix per frame and head rather than a 630 x 630 one for `T = 30`. That
is roughly `T` times less attention compute, and it is the version that produced
the numbers in the paper and fits the latency budget of Table VI.

`model.attention_scope` selects between them:

| Value | Score matrix per head | Notes |
| --- | --- | --- |
| `spatial` (default) | `21 x 21` per frame | Original research code, cheap on the Jetson |
| `spatiotemporal` | `(T * 21) x (T * 21)` | Literal reading of Eq. (6)/(7), dominates memory |

Both are exact implementations of Eq. (5) with one hand as query and the other as
key and value; they differ only in what counts as a token. InterHand Temporal
Fusion is unaffected, as Eq. (4) fixes its scope to the temporal window.

### Placement of InterHand Temporal Fusion

Eq. (4) puts the fusion inside the temporal convolution's sum, i.e. attention
first and convolution second, and Fig. 5 labels the stage "the temporal
convolution with InterHand Temporal Fusion". The code follows the equation:
`spatial -> fusion -> temporal convolution`.

### Feature Extractor residual

Fig. 5 describes "a combination of fully connected (FC) layers, ReLU, and
dropout layers [...] which is then element-wise added to the original hand
feature". The code implements `x + Dropout(FC2(ReLU(FC1(x))))` with two
independent FC layers. The two layers are `1 x 1` convolutions so that the same
weights are shared across time steps and joints.

Interaction Attention is wrapped in its own residual as well, following the
"+ Hand Feature" path drawn in Fig. 8. Without it the block would forward only
the other hand's information.

### Temporal kernel sizes

The paper does not state the temporal kernel size. The default is two parallel
branches with kernels 5 and 15, whose outputs are summed, taken from the
original research code (which also had an unused third branch with kernel 29).
Padding is `(k - 1) / 2` in every branch, so all branches produce the same output
length regardless of kernel size. The two branches share the spatial convolution
weights and own separate edge-importance masks, again following that code.

### Number of classes

Section I says the task is to classify a window "into one of the six steps", but
the dataset of Lulla et al. labels each frame with a movement code where `1..6`
are the WHO steps, `0` is "other movement", and `7` (closing the faucet with a
paper towel) is folded into `0` by the dataset authors for classification. The
original research code therefore trains a 7-way classifier, and the confusion
matrices in the notebook are 7 x 7.

`model.num_classes` defaults to 7 for that reason, with class 0 meaning "other".
Keeping the "other" class is also what makes a deployed system able to stay
silent when nobody is washing their hands. Set it to 6 to classify the WHO steps
alone.

### Macro averaging of the metrics

Table II reports accuracy, precision, recall and F1. Precision, recall and F1
are macro-averaged over the classes, computed per class and then averaged
without frequency weighting. This matches the pattern in the table, where the
weaker baselines have a precision well below their accuracy. The original
research code reported accuracy and a row-normalised confusion matrix only, so
the macro-averaging convention is an inference from the table, not from code.

### Temporal-order split

Section IV-B splits each camera setting into five folds by temporal order and
notes that "the validation set is not immediately after the training set in the
temporal sequence". Adjacent windows are nearly identical, so
`cross_validation.gap` drops that many windows on each side of the validation
block. The default is 1; raise it if windows overlap.

## Relation to the original research code

Two earlier code bases exist, and they do not agree with each other:

* the **experiment notebooks** used while writing the paper (`ProjectZ.ipynb` and
  friends), which contain the model in several evolving versions plus the
  training, ONNX export and TensorRT benchmarking cells that produced the
  reported numbers;
* the **published snapshot** that used to live in this repository
  (`net/interhandnet_st_gcn.py`, `modules/*.py`), a cleaned-up extract of the
  above.

Where they disagree, the published snapshot is the later and more deliberate of
the two. Two examples:

* In the notebook the distance matrix is disabled — `D = 0` is passed into the
  blocks and the `torch.bmm(x1, D)` line is commented out, so `A_IG` acts as a
  plain cross-hand graph with no distance weighting. In the published snapshot
  `D` is computed and used. This code follows the published snapshot, since a
  disabled `D` contradicts Eq. (2).
* The notebook applies the Interaction Graph to blocks 1 and 2; the snapshot
  applies it to blocks 0, 1 and 2. The default follows the snapshot.

Conversely, some details exist **only** in the notebooks, because the snapshot
never contained the surrounding pipeline: the seven-class label set, the
`(5, 15)` temporal kernels with `max_hop = 2`, the latency protocol, and the
STA-GCN and FR Head backbones. Those were taken from the notebooks.

One module has no ancestor in either: **InterHand Temporal Fusion does not exist
in the experiment notebooks at all.** It only ever appeared as a standalone file
in the published repository. That is consistent with Table III, where `+IG/IA`
and `+IG/IA/ITF` are separate rows and ITF is described as the later addition.
Anything about ITF in this repository therefore follows Eq. (4) and Fig. 7
directly, with no reference implementation to fall back on, and is the part most
worth validating experimentally.

## Differences from the original research code

The refactor kept the architecture but changed a few things that were either
incorrect or diverged from the paper. They are listed here so the behaviour
change is traceable.

### Fixed: multi-head reshapes scrambled the data

The original code reshaped a `(N, C, T, J)` feature map straight into the
multi-head layout, for example

```python
Q_left.view(N, num_heads, channels_per_head, 21, T)   # from (N, C, T, 21)
Q_right.view(N, num_heads, T * 21, -1)                # from (N, H, d, T, 21)
```

A `view` only relabels the flat buffer, so swapping the last two axes or folding
`(T, J)` into one axis without a transpose mixes channels into the token
dimension. `modules/reshape.py` performs the permutation explicitly, and
`tests/test_reshape.py` asserts the index mapping in both directions.

### Fixed: query and value assignment in InterHand Temporal Fusion

See "Query and key assignment in InterHand Temporal Fusion" above. The original
code queried with the same hand it took values from, which cannot reconstruct an
occluded hand.

### Fixed: the Feature Extractor shared one FC layer

The original code called the same `nn.Conv2d` twice:

```python
out = self.fc(out); out = self.relu(out); out = self.fc(out)
```

Fig. 5 shows two FC layers, so `FeatureExtractor` owns `fc1` and `fc2`. It also
drops the extra `1 x 1` convolution the original applied to the residual path,
keeping the plain element-wise addition the figure describes.

### Fixed: NaN gradients from the distance matrix

`sqrt` at zero has an infinite derivative and the diagonal of `D` is zero. The
squared distances are now floored with an epsilon before the square root.

### Changed: `A_IG` is a single matrix

The original code built `A_IG` with the same hop-partitioning helper as the
physical graph, producing a `(K, V, V)` tensor whose hop-0 subset is the identity
and whose hop-1 subset holds the cross-hand edges — each with its own slice of
the convolution weights. Eq. (2) defines `A_IG` as one `N x N` matrix, so the
code uses a single `(V, V)` matrix and sums the adjacency subsets before the
graph product. Set `interaction_graph_self_loops=True` to put the identity term
back into `A_IG`.

### Changed: InterHand Temporal Fusion sits before the temporal convolution

In the original code the fusion ran after the temporal convolutions, inside the
same module as Interaction Attention. Eq. (4) places it inside the convolution's
sum, so the code now runs `spatial -> fusion -> temporal convolution`.

### Added: block-level residual

The original STGC block of the ST-GCN backbone has no residual around the block;
only Interaction Attention and the Feature Extractors carry one. Every ST-GCN
derivative, including the STA-GCN variant in the same notebook, does add one, and
its absence is a plausible contributor to the unusually low ST-GCN baseline in
Table II (0.5339 accuracy). It is enabled by default and can be switched off with
`model.residual: false`; `configs/interhandnet_stgcn_legacy.yaml` does that
together with the other legacy choices.

### Summary table

| Detail | Experiment notebooks | Published snapshot | This code |
| --- | --- | --- | --- |
| Bone edges, cross-hand edges | 21-joint chain, `i <-> i+21` | same | same |
| `A` partitioning | by hop, `max_hop = 2` | same | same |
| `M` (edge importance) | `A * M`, per temporal branch | same | same |
| `A_IG` | `(K, V, V)`, includes self-loops | same | `(V, V)`, no self-loops |
| Distance matrix `D` | computed then **disabled** | computed, `A_IG @ D` | `A_IG @ D`, `hadamard` optional |
| Interaction Graph blocks | 1, 2 | 0, 1, 2 | 0, 1, 2 |
| Temporal branches | kernels 5 and 15, shared spatial conv | same | same |
| Interaction Attention scope | per frame | per frame | per frame, `spatiotemporal` optional |
| Feature Extractor (Fig. 5) | absent | one FC reused twice | two FC layers, twice per block |
| InterHand Temporal Fusion | absent | present, query/value swapped | Eq. (4) assignment |
| Block residual | ST-GCN: no, STA-GCN: yes | no | yes, optional |
| Classes | 7 | n/a | 7, configurable |
| Window `T` | 28 in the export cells | n/a | 30, per Section IV-A |
| Train/validation split | random 90/10, seed 46 | n/a | temporal 5-fold per camera |
| Backbones | ST-GCN, STA-GCN, FR Head | ST-GCN | ST-GCN, STA-GCN |
| Metrics | accuracy + confusion matrix | n/a | accuracy, macro P/R/F1, confusion matrix |

## Known gaps relative to the paper

* **FR Head backbone.** The paper reports InterHandNet on top of both STA-GCN and
  FR Head (Table II) and integrates the modules into seven more backbones
  (Table III). The ST-GCN and STA-GCN backbones ship here; FR Head, which pulls
  in CTR-GCN topology refinement and the feature-refinement head with its
  contrastive auxiliary losses, does not. The three modules are standalone
  `nn.Module`s so they can be dropped into an external implementation; see
  "Using the modules with another backbone" in the README.
* **Datasets.** The datasets of Lulla et al. and Xie et al. are not
  redistributable. `tools/extract_skeletons.py` builds the skeleton archive from
  local video files.
* **InterWild skeletons.** Table IV uses InterWild instead of MediaPipe Hands to
  remove the effect of missing keypoints. Only the MediaPipe extractor is
  included; an InterWild archive with the same layout can be dropped in.
