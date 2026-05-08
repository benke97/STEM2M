# STEM2M — Morphology Prediction

3D point-cloud prediction of nanoparticle **morphology** from a single HAADF (high-angle annular dark-field) STEM image. A diffusion model (PVCNN++ backbone) generates the 3D morphology, conditioned on features extracted from the HAADF image by a U-Net-based projection model.

## Overview

The pipeline has three components:

1. **Projection model** (`ModernUNet`) — predicts a per-pixel thickness map and intermediate encoder/decoder features from a HAADF image.
2. **Conditioning** — projects 3D point candidates onto the image plane and samples the U-Net's feature maps at those projected locations to build per-point conditioning vectors.
3. **Diffusion model** (`PVCNN2`) — a point-voxel CNN trained as a DDPM to denoise 3D point clouds conditioned on the per-point features from step 2.

## Repository layout

```
.
├── main.py                      # Hydra entry point (train | test)
├── config/config.yaml           # Hydra configuration
├── regenerate_sim_data.py       # Script that produced the contiguous HDF5 layout
├── requirements.txt
├── mar/
│   ├── data/                    # MorphologyDataset + dataloaders
│   │   └── exp/                 # 5 sample .npz experimental point clouds
│   ├── models/
│   │   ├── modern_unet.py            # U-Net projection model
│   │   ├── resnet_size_estimator.py  # ResNet18 size estimator
│   │   ├── train.py                  # training loops
│   │   ├── test.py                   # inference / visualization
│   │   └── pvcnn/                    # PVCNN++ diffusion backbone
│   └── utils/                   # diffusion utils, helpers, conditioning, viz, RDF loss
└── .gitignore
```

## Dataset and weights

The dataset and the best-performing checkpoint for each model component are hosted on HuggingFace:

> **Dataset & weights:** [Stemson-AI/Morphology_Prediction](https://huggingface.co/datasets/Stemson-AI/Morphology_Prediction)

Download the HDF5 file (e.g. into `./data/`) and update `data.dataset_path` in `config/config.yaml` to point at it. Download the model checkpoints and update `model.pc_model.pretrained_path` and `model.projection_model.pretrained_path` to enable the released weights.

### HDF5 layout

The dataset uses a contiguous-tensor layout for fast random access. Per split (`real`, `synthetic`):

| Path | Shape | Dtype | Notes |
|------|-------|-------|-------|
| `/<split>/haadf_norm` | `(N, 128, 128)` | float32 | Normalized HAADF images |
| `/<split>/haadf_gan` | `(N, 128, 128)` | float32 | CycleGAN-translated HAADF |
| `/<split>/thickness_pt` | `(N, 128, 128)` | float32 | Pt-thickness ground truth |
| `/<split>/com_clean` | `(N, 2)` | float32 | Center-of-mass (clean image) |
| `/<split>/com_gan` | `(N, 2)` | float32 | Center-of-mass (GAN image) |
| `/<split>/pixel_size` | `(N,)` | float32 | nm per pixel |
| `/<split>/sample_ids` | `(N,)` | UTF-8 string | Source sample IDs |
| `/<split>/pc_x` `pc_y` `pc_z` `pc_label` | `(M_total,)` | float32 | Flattened point clouds |
| `/<split>/pc_ptr` | `(N+1,)` | uint64 | CSR-style offsets into `pc_*` |

Point cloud `i` in a split is recovered as `pc_x[pc_ptr[i]:pc_ptr[i+1]]` (and similarly for `y`, `z`, `label`). Labels are stored as `Z / 118` (atomic number normalized); `78 / 118` is Pt.

`regenerate_sim_data.py` is the script that converts the original group-per-sample HDF5 into this layout.

## Installation

```bash
# 1. Create an environment (tested with Python 3.10 + CUDA 11.8)
conda create -n stem2m python=3.10 -y
conda activate stem2m

# 2. Install PyTorch matching your CUDA build
#    See https://pytorch.org/get-started/locally/

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install PyTorch3D (must match torch + CUDA versions)
#    See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# 5. Build the PVCNN custom CUDA extensions (compiled on first import)
#    The C++/CUDA sources live under mar/models/pvcnn/modules/functional/src/.
```

## Usage

The entry point is `main.py`, driven by Hydra. Override any config field on the command line.

### Train

```bash
python main.py main.task=train
```

### Test (inference + visualization on a few samples)

```bash
python main.py main.task=test \
  model.pc_model.pretrained=True \
  model.pc_model.pretrained_path=/path/to/pc_model.pt \
  model.projection_model.pretrained=True \
  model.projection_model.pretrained_path=/path/to/projection_model.pt
```

### Multi-GPU

The training loop is wrapped with HuggingFace Accelerate. Configure once and launch:

```bash
accelerate config
accelerate launch main.py main.task=train
```

## Citation

> _TODO: add citation block once a paper / preprint is available._

## License

> _TODO: choose a license (MIT / Apache-2.0 are common for ML research code)._

## Known TODOs

- [ ] `mar/models/test.py` still has a hardcoded Windows path to the size-estimator weights — wire it through `cfg`.
- [ ] Add a quickstart script that runs inference end-to-end on the bundled `mar/data/exp/*.npz` samples without needing the full dataset.
