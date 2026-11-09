# StreamRC / SQRCF-UAV: streaming camera-radar 3D UAV detection

Training and evaluation code for streaming camera + mmWave-radar 3D detection of
UAVs (single class `uav.drone`), built on MMDetection3D 1.0.0rc6 and a
SparseBEV-style query decoder with radar-initialised queries and a temporal
QueryStack.

**No weights or data are included.** Checkpoints (`*.pth`), the dataset and the
ImageNet/nuImages pretrained backbones must be obtained separately.

## Layout

| Path | Content |
| --- | --- |
| `configs/r50_fusion_streaming_full.py` | Main R50 camera-radar config (paper setting: 2 decoder layers, 8 sampling points, 320x832 input, 36 epochs) |
| `configs/r101_fusion_streaming_full.py` | R101 variant (inherits the R50 config) |
| `models/` | Detector, head, transformer, radar backbone/sampling, QueryStack; `models/csrc` holds the CUDA sampling op |
| `loaders/` | Dataset, pipelines, streaming samplers and fast multi-GPU test |
| `nuscenes-devkit/` | nuScenes devkit 1.1.x with UAV evaluation additions (`detection_cvpr_uav`), used through `PYTHONPATH` |
| `gen_sweep_info_ming.py`, `readpcd.py` | Build the `*_mingfusion_sweep.pkl` annotation files from nuScenes-format data with radar `.pcd` sweeps |
| `train.py`, `val.py`, `train.sh`, `val.sh`, `tools/*.sh` | Entry points |

## Environment

Tested with: Python 3.8, PyTorch 2.0.0 + CUDA 11.8, torchvision 0.15.0,
mmcv-full 1.6.0, mmdet 2.28.2, mmsegmentation 0.30.0, mmdet3d 1.0.0rc6,
numpy 1.23.5, numba 0.58.1, scipy 1.10.1, Shapely 1.8.5, opencv-python, tensorboard.
See `requirements.txt`.

```bash
conda create -n streamrc python=3.8 -y && conda activate streamrc
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim && mim install mmcv-full==1.6.0
pip install -r requirements.txt
# Optional but recommended: build the CUDA multi-view sampling op
# (falls back to a slower PyTorch grid_sample implementation otherwise).
cd models/csrc && TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" python setup.py build_ext --inplace && cd ../..
```

`tools/env.sh` adds the repo root and `nuscenes-devkit/python-sdk` to `PYTHONPATH`.
Optional variables: `STREAMRC_ENV` (conda env prefix), `STREAMRC_CUDA_HOME`,
`STREAMRC_DATA_ROOT`, `STREAMRC_OUTPUT_ROOT`, `STREAMRC_GPUS`,
`STREAMRC_LOAD_FROM` (pretrained checkpoint; `none` to disable).

## Data

The annotation files store media paths relative to the repository root
(`data/nuscenes/samples/...`), so the dataset must be visible as `data/nuscenes`:

```
data/nuscenes/
  samples/  sweeps/
  full/                          # nuScenes-format tables (version "full")
    streamrc/nuscenes_infos_train_mingfusion_sweep.pkl
    streamrc/nuscenes_infos_val_mingfusion_sweep.pkl
```

```bash
ln -s /path/to/your/dataset data/nuscenes
```

## Pretrained weights

`load_from` defaults to
`pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`
(MMDetection3D nuImages Cascade Mask R-CNN R50). The R101 config uses
`pretrain/resnet101-63fe2227.pth` (torchvision ImageNet). Download them into
`pretrain/` or set `STREAMRC_LOAD_FROM=none`.

## Train

```bash
STREAMRC_GPUS=2 bash train.sh                               # outputs/<config>/<timestamp>
STREAMRC_GPUS=2 bash train.sh --batch-size-per-gpu 8 --work-dir outputs/my_run
bash tools/dist_train.sh configs/r101_fusion_streaming_full.py
```

`train.py` accepts `--override key=value ...` for any config field and
`--stop-after-epochs N`. Validation runs every 2 epochs; the best mAP checkpoint
is kept.

## Evaluate

```bash
STREAMRC_GPUS=1 bash val.sh path/to/checkpoint.pth
bash tools/dist_val.sh configs/r50_fusion_streaming_full.py path/to/checkpoint.pth --override model.pts_bbox_head.transformer.num_layers=4
```

The metric is nuScenes-style mAP/NDS with the `detection_cvpr_uav` evaluation
config from the bundled devkit.

## Notes

- If MMCV lacks the CUDA focal-loss kernel, an equivalent PyTorch formula is used
  automatically (`hpc3_runtime.py`).
- `hpc3_speed.py` contains optional execution speed-ups (TF32, fused AdamW,
  activation checkpointing); they are disabled unless the config defines
  `hpc3_speed=dict(enabled=True, ...)`.
- `train.py --benchmark-iters N` runs a bounded throughput measurement instead of
  full training.

## Third-party code

`nuscenes-devkit/` is from Motional's nuScenes devkit (Apache-2.0, see
`nuscenes-devkit/LICENSE.txt`). The model code builds on SparseBEV (MIT) and
MMDetection3D (Apache-2.0); see their original repositories.
