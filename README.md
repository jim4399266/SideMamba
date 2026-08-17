# SideMamba
## Overview

Video-text retrieval requires both strong image-language knowledge and effective temporal modeling. SideMamba keeps a pretrained CLIP encoder as the main branch and inserts a compact Mamba-based side branch at selected visual Transformer layers. The two branches exchange features progressively, and the resulting video representation is matched with text using a fine-grained bidirectional retrieval objective.

The method contains three main components:

- **Efficient Omnidirectional Selective Scan (EOSS).** Video features are scanned along spatial, temporal, and joint spatiotemporal routes. Tri-Dimensional Perception (TDP) exposes complementary 3D token orders, while Quad-directional Strided Pruning (QSP) reduces the number of scanned tokens.
- **Prior-Guided Dynamic Incremental Learning (PGDIL).** Top-down priors from intermediate CLIP layers are injected into the SideMamba branch with learnable fusion gates. Bottom-up dynamic aggregation updates the global token from local features at every side layer.
- **Progressive Spatiotemporal Alignment (PSA).** Progressively fused video features are aligned with text through fine-grained token-frame interaction and a bidirectional contrastive retrieval loss.

[//]: # (Manuscript: [Overleaf read-only project]&#40;https://cn.overleaf.com/read/jpjwpcyvfqzj#7875ca&#41;)

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


The selective-scan implementation automatically falls back to a PyTorch version when the CUDA operators are unavailable. The fallback is useful for debugging but is considerably slower.

The installation of mamba_ssm is following 
[VMamba](https://github.com/MzeroMiko/VMamba)
## Pretrained model

Download the OpenAI CLIP checkpoint: [ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)
and
[ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)

The weights of SiM will be released soon.

## Data preparation

### Data sources

The processed data used by this project follows [Cap4Video](https://github.com/whwu95/Cap4Video):

- **MSR-VTT:** use the pre-extracted video frames provided by Cap4Video.
- **MSVD:** use the pre-extracted video frames provided by Cap4Video.
- **DiDeMo:** prepare the frame data from the original videos with the `video2image.py` script provided by Cap4Video. The generated frames should be organized by video ID as shown below.

You can also directly download the datasets from our links:
[MSR-VTT](),
[MSVD](),
[DiDeMo]()

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
