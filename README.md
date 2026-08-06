# SideMamba

Official research code for **SideMamba**, an efficient side-network framework for video-text retrieval. SideMamba transfers the spatial and semantic priors of a pretrained CLIP model to a lightweight state-space branch, enabling spatiotemporal video modeling without fully fine-tuning a large video backbone.

> This repository is currently being cleaned for public release. Configuration paths and dependency versions may still need to be adapted to your environment.

## Overview

Video-text retrieval requires both strong image-language knowledge and effective temporal modeling. SideMamba keeps a pretrained CLIP encoder as the main branch and inserts a compact Mamba-based side branch at selected visual Transformer layers. The two branches exchange features progressively, and the resulting video representation is matched with text using a fine-grained bidirectional retrieval objective.

The method contains three main components:

- **Efficient Omnidirectional Selective Scan (EOSS).** Video features are scanned along spatial, temporal, and joint spatiotemporal routes. Tri-Dimensional Perception (TDP) exposes complementary 3D token orders, while Quad-directional Strided Pruning (QSP) reduces the number of scanned tokens.
- **Prior-Guided Dynamic Incremental Learning (PGDIL).** Top-down priors from intermediate CLIP layers are injected into the SideMamba branch with learnable fusion gates. Bottom-up dynamic aggregation updates the global token from local features at every side layer.
- **Progressive Spatiotemporal Alignment (PSA).** Progressively fused video features are aligned with text through fine-grained token-frame interaction and a bidirectional contrastive retrieval loss.

The implementation supports **MSVD**, **MSR-VTT**, and **DiDeMo**, and reports standard text-to-video and video-to-text retrieval metrics such as R@1, R@5, R@10, and their sum.

Manuscript: [Overleaf read-only project](https://cn.overleaf.com/read/jpjwpcyvfqzj#7875ca)

## Code structure

```text
SideMamba/
├── src/
│   ├── main_retrieval.py              # Retrieval training/testing entry point
│   ├── config/
│   │   └── retrieval/                 # Sacred dataset, encoder, side-network,
│   │                                  # optimizer, and runtime configurations
│   ├── datamodules/
│   │   └── retrieval/                 # PyTorch Lightning data modules
│   ├── datasets/
│   │   └── retrieval/                 # MSVD, MSR-VTT, and DiDeMo datasets
│   ├── lightning_modules/
│   │   ├── module_retrieval_clip.py   # Training, loss, evaluation, and caching
│   │   ├── module_base.py             # Shared Lightning utilities
│   │   └── retrieval_utils.py         # Retrieval metrics
│   └── models/
│       ├── module_clip.py             # CLIP backbone and side-branch fusion
│       ├── module_vmamba.py           # SideMamba blocks and 3D SSM operators
│       ├── aggregator.py              # Dynamic global-token aggregation
│       ├── module_pos.py              # 2D/3D positional embeddings
│       ├── optimization.py            # Optimizer and learning-rate schedule
│       └── csm/                       # Scan/merge routes and selective scan
└── CODE_USAGE_AUDIT.md                # Static file-dependency audit
```

## Environment

A Linux environment with an NVIDIA GPU is recommended. The main dependencies are:

- Python and PyTorch with a CUDA build compatible with the local driver;
- PyTorch Lightning;
- Sacred, Transformers, timm, and einops;
- NumPy, pandas, OpenCV, Pillow, tqdm, ftfy, and regex;
- torchmetrics, TensorBoard, thop, and torchinfo;
- complexPyTorch;
- Triton and/or a compatible selective-scan CUDA extension for accelerated training.

An example installation is:

```bash
conda create -n sidemamba python=3.10 -y
conda activate sidemamba

# Install a PyTorch build appropriate for your CUDA environment first.
pip install torch torchvision

pip install \
  pytorch-lightning sacred transformers timm einops \
  numpy pandas opencv-python pillow tqdm ftfy regex \
  torchmetrics tensorboard thop torchinfo complexPyTorch triton
```

The selective-scan implementation automatically falls back to a PyTorch version when the CUDA operators are unavailable. The fallback is useful for debugging but is considerably slower.

## Pretrained model

Download the OpenAI CLIP checkpoint used by the selected encoder and place it under the pretrained-model root. For the ViT-L/14 configuration used in the example below:

Other encoder configurations expect the corresponding filenames defined in `src/config/retrieval/encoder_ingredient.py`, for example `ViT-B-16.pt` or `ViT-B-32.pt`.

## Data preparation

### Data sources

The processed data used by this project follows [Cap4Video](https://github.com/whwu95/Cap4Video):

- **MSR-VTT:** use the pre-extracted video frames provided by Cap4Video.
- **MSVD:** use the pre-extracted video frames provided by Cap4Video.
- **DiDeMo:** prepare the frame data from the original videos with the `video2image.py` script provided by Cap4Video. The generated frames should be organized by video ID as shown below.

Please follow the download and preprocessing instructions in the Cap4Video repository and comply with the licenses and terms of the original datasets.

Set `data_root` to a directory containing one or more prepared datasets. The expected MSVD layout is:

```text
<DATA_ROOT>/MSVD-Frames/
├── train_list.txt
├── val_list.txt
├── test_list.txt
├── raw-captions.pkl
└── MSVD_frames/
    ├── <video_id>/
    │   ├── ... extracted frames ...
    │   └── ...
    └── ...

<DATA_ROOT>/MSRVTT/
├── msrvtt_data/
│   ├── MSRVTT_train.9k.csv
│   ├── MSRVTT_JSFUSION_test.csv
│   └── MSRVTT_data.json
└── frames_30fps/<video_id>/...

<DATA_ROOT>/DiDeMo/
├── text/
│   ├── train_list.txt
│   ├── val_list.txt
│   ├── test_list.txt
│   ├── train_data.json
│   ├── val_data.json
│   └── test_data.json
└── DiDeMo_Frames/<video_id>/...
```

The dataloaders consume extracted frame directories. Dataset annotations and video data must be obtained in accordance with the licenses of the original datasets.

## Training

Commands should be run from the `src` directory because the current import paths are relative to it:

```bash
cd src
```

The project uses [Sacred](https://sacred.readthedocs.io/) named configurations. The following command trains the ViT-L/14 SideMamba model on MSVD:

```bash
python main_retrieval.py with \
  dataset.msvd \
  encoder.clip_vit_L_14 \
  side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos \
  data_root=/path/to/data \
  pretrained_model_dir=/path/to/pretrained_models \
  output_dir=../output_retrieval \
  log_dir=../log_retrieval \
  devices=1 \
  batch_size=128 \
  per_gpu_batch_size=16
```

For multi-GPU training, set `devices` to the number of GPUs. `batch_size` is the desired global batch size; gradient accumulation is calculated automatically from `batch_size`, `per_gpu_batch_size`, the number of devices, and the number of nodes.

Example:

```bash
python main_retrieval.py with \
  dataset.msvd \
  encoder.clip_vit_L_14 \
  side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos \
  data_root=/path/to/data \
  pretrained_model_dir=/path/to/pretrained_models \
  devices=8 batch_size=320 per_gpu_batch_size=40 max_epoch=5
```

Checkpoints are written below:

```text
<output_dir>/<dataset>/<experiment_name>/version_<n>/
```

TensorBoard logs are written below `log_dir`.

## Evaluation

Use the `test_only` named configuration and provide a trained Lightning checkpoint:

```bash
cd src

python main_retrieval.py with \
  dataset.msvd \
  encoder.clip_vit_L_14 \
  side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos \
  test_only \
  checkpoint=/path/to/model.ckpt \
  data_root=/path/to/data \
  pretrained_model_dir=/path/to/pretrained_models \
  devices=1
```

The encoder and side-network settings stored in the checkpoint are restored during testing. Keep the dataset configuration consistent with the intended evaluation set.

To resume interrupted training, pass the same experiment configurations together with:

```text
checkpoint=/path/to/last_checkpoint.ckpt
```

## Configuration reference

Frequently used options include:

| Option | Meaning |
|---|---|
| `dataset.msvd`, `dataset.msrvtt`, `dataset.didemo` | Dataset named configuration |
| `encoder.clip_vit_L_14` | CLIP ViT-L/14 backbone and tokenizer |
| `side.vmamba_4_i_patch14_...` | Four-layer interval SideMamba for ViT-L/14 |
| `data_root` | Root directory containing the datasets |
| `pretrained_model_dir` | Root directory containing CLIP checkpoints |
| `devices` | Number of GPUs or a Lightning-compatible device list |
| `batch_size` | Desired global batch size |
| `per_gpu_batch_size` | Batch size processed by each GPU per step |
| `batch_size_val` | Per-loader validation/test batch size |
| `max_epoch` | Maximum number of training epochs |
| `optimizer.init_lr` | Initial learning rate of the side network |
| `checkpoint` | Checkpoint used for testing or resuming training |

Additional named configurations and ablation variants are defined under `src/config/retrieval/`.

## Citation

The paper is currently under preparation. Citation information will be added after publication.

## License

A repository license has not yet been added. Please contact the authors before redistributing the code or trained models.
