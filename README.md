# Monocular Atomic Reconstruction

3D point-cloud prediction of nanoparticle atomic structure conditioned on a single HAADF (high-angle annular dark-field) STEM image. A diffusion model (PVCNN++ backbone) generates the 3D structure, conditioned on features extracted from the HAADF image by a U-Net-based projection model.

## Overview

The pipeline has three components:

1. **Projection model** (`ModernUNet`) — predicts a per-pixel thickness map and intermediate encoder/decoder features from a HAADF image.
2. **Conditioning** — projects 3D point candidates onto the image plane and samples the U-Net's feature maps at those projected locations to build per-point conditioning vectors.
3. **Diffusion model** (`PVCNN2`) — a point-voxel CNN trained as a DDPM to denoise 3D point clouds conditioned on the per-point features from step 2.

## Repository layout

```
.
├── main.py                  # Hydra entry point (train | test)
├── config/config.yaml       # Hydra configuration
├── requirements.txt
├── mar/
│   ├── data/                # HDF5 dataset + dataloaders
│   │   └── exp/             # 5 sample .npz experimental point clouds
│   ├── models/
│   │   ├── modern_unet.py            # U-Net projection model
│   │   ├── resnet_size_estimator.py  # ResNet18 size estimator
│   │   ├── train.py                  # training loops
│   │   ├── test.py                   # inference / visualization
│   │   └── pvcnn/                    # PVCNN++ diffusion backbone
│   └── utils/               # diffusion utils, helpers, conditioning, viz, RDF loss
└── .gitignore
```

## Installation

```bash
# 1. Create an environment (tested with Python 3.10 + CUDA 11.8)
conda create -n mar python=3.10 -y
conda activate mar

# 2. Install PyTorch matching your CUDA build
#    See https://pytorch.org/get-started/locally/

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install PyTorch3D (must match torch + CUDA versions)
#    See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

# 5. Build the PVCNN custom CUDA extensions (compiled on first import)
#    The C++/CUDA sources live under mar/models/pvcnn/modules/functional/src/.
```

## Pretrained weights

Pretrained checkpoints are hosted on HuggingFace Hub. Download the bundle and update the paths in `config/config.yaml`:

- `model.pc_model.pretrained_path` — diffusion model
- `model.projection_model.pretrained_path` — thickness predictor / U-Net

By default `pretrained: False` so a fresh run trains from scratch. Set both to `True` and provide local paths to use the released weights.

> _TODO: insert HuggingFace link once published._

## Dataset

Training uses an HDF5 dataset combining synthetic and real HAADF/structure pairs. Update `data.dataset_path` in `config/config.yaml` to point at your local copy.

A small set of 5 experimental `.npz` point clouds is included under `mar/data/exp/` so the inference path can be exercised without the full dataset.

> _TODO: link to dataset release._

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
- [ ] Move dataset paths in `config/config.yaml` to environment variables or `.env`.
- [ ] Publish HuggingFace weights bundle and update README links.
- [ ] Add a quickstart script that runs inference end-to-end on the bundled `.npz` samples.
