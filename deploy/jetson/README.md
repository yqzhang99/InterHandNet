# Edge deployment on NVIDIA Jetson

The paper deploys InterHandNet on an NVIDIA Jetson Orin Nano with a Luxonis
OAK-D Depth AI Camera (Section IV-C). The pipeline has two stages: MediaPipe
Hands extracts 3D skeletons from the RGB-D stream, then a TensorRT engine
classifies one second of skeleton data into one of the six WHO steps.

## Prerequisites

TensorRT and pycuda are part of the JetPack SDK and cannot be installed from
PyPI. Install JetPack on the Jetson first, then add the Python dependencies:

```bash
pip install -r requirements.txt
pip install mediapipe opencv-python
```

## 1. Export the trained model to ONNX

Run this on the training machine:

```bash
python tools/export_onnx.py \
    --config configs/interhandnet.yaml \
    --checkpoint work_dirs/interhandnet/best.pt \
    --output interhandnet.onnx
```

The default input shape is `(1, 3, 30, 42)`: one second of two-hand skeleton
data at 30 fps.

## 2. Build the TensorRT engine

Run this on the Jetson. An engine is tied to the GPU architecture and the
TensorRT version that built it, so it cannot be copied from a workstation.

```bash
python deploy/jetson/build_engine.py \
    --onnx interhandnet.onnx \
    --engine interhandnet.engine \
    --precision fp16
```

## 3. Run real-time recognition

```bash
python deploy/jetson/realtime_demo.py --engine interhandnet.engine --camera 0 --display
```

## Expected latency

Table VI reports the one-second latency measured on the Jetson. The skeleton
extractor dominates the budget, and InterHandNet needs roughly a tenth of it:

| Stage                    | Latency (ms) |
| ------------------------ | -----------: |
| MediaPipe Hands          |      594.000 |
| InterHandNet (ST-GCN)    |        7.433 |
| InterHandNet (STA-GCN)   |       23.952 |
| InterHandNet (FR Head)   |       56.906 |
| 3D CNN on RGB            |    > 6498.79 |

To reproduce the measurement protocol (1,800 samples, first five excluded as
warm-up) with PyTorch instead of TensorRT:

```bash
python tools/benchmark_latency.py --config configs/interhandnet.yaml --device cuda:0
```
