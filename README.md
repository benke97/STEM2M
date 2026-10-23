# STEM2M — Morphology Prediction

3D point-cloud prediction of nanoparticle **morphology** from a single HAADF (high-angle annular dark-field) STEM image. A diffusion model (PVCNN++ backbone) generates the 3D morphology, conditioned on features extracted from the HAADF image by a U-Net-based projection model.

> **Note:** This repository was cleaned up and packaged from the original research code with the help of [Claude Code](https://claude.com/claude-code). If something does not work or the results do not match the paper, please reach out to **haoel@dtu.dk**.

## Overview

The pipeline has five components. Steps 4 and 5 are required to reproduce the paper results:

1. **Projection model** (`ModernUNet`) — predicts a per-pixel thickness map and intermediate encoder/decoder features from a HAADF image.
2. **Conditioning** — projects 3D point candidates onto the image plane and samples the U-Net's feature maps at those projected locations to build per-point conditioning vectors.
3. **Diffusion model** (`PVCNN2`) — a point-voxel CNN trained as a DDPM to denoise 3D point clouds conditioned on the per-point features from step 2. Produces the *coarse* structure.
4. **Refinement model** (`PVCNN2`, same architecture) — a single forward pass at `t = 0` on the conditioned coarse structure that predicts the residual to subtract. It is initialized from the diffusion model and trained on synthetically perturbed ground-truth structures.
5. **Test-time optimization** — moves only z (x/y are fixed by the image) with SGD to minimize the variance of k-nearest-neighbour distances (k = 2, 4, 8, 16), plus a fidelity term that keeps the structure close to the refined one. Uses no ground truth.

Steps 3–5 live in [`mar/models/refinement.py`](mar/models/refinement.py).

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
│   │   ├── train.py                  # diffusion model training loops
│   │   ├── train_refinement.py       # refinement model training
│   │   ├── refinement.py             # coarse sampling, refinement, test-time optimization
│   │   ├── evaluate.py               # full-pipeline evaluation (test_refinement)
│   │   ├── test.py                   # inference / visualization
│   │   └── pvcnn/                    # PVCNN++ diffusion backbone
│   └── utils/                   # diffusion utils, helpers, conditioning, viz, RDF loss
└── .gitignore
```

## Dataset and weights

The dataset and the best-performing checkpoint for each model component are hosted on HuggingFace:

> **Dataset & weights:** [Stemson-AI/Morphology_Prediction](https://huggingface.co/datasets/Stemson-AI/Morphology_Prediction)

### Recommended layout

```
.
├── data/
│   └── morphology_prediction.h5         # contiguous HDF5 dataset
└── weights/
    ├── best_pc_model.pt                              # diffusion model (PVCNN2)
    ├── refinement_model.pt                           # refinement model (PVCNN2)
    ├── particle_thickness_predictor_250k_CycleGAN.pt # projection model (ModernUNet) — for haadf_gan
    ├── particle_thickness_predictor_250k.pt          # projection model — for haadf_norm
    └── ResNet18_Size_estimator_weights.pth           # size estimator backbone
```

`config/config.yaml` ships with paths preset to this layout (relative to the repository root). If your files live elsewhere, override the paths on the command line, e.g.:

```bash
python main.py data.dataset_path=<path/to/morphology_prediction.h5> \
  model.pc_model.pretrained_path=<path/to/best_pc_model.pt> \
  model.refinement_model.pretrained_path=<path/to/refinement_model.pt> \
  model.projection_model.pretrained_path=<path/to/particle_thickness_predictor_250k_CycleGAN.pt>
```

 Both `data/` and `weights/` are gitignored. Paths are resolved through `hydra.utils.to_absolute_path` so they keep working under Hydra's `chdir: true`.

**Switching projection model for `haadf_norm` images** — override on the CLI:

```bash
python main.py data.image_type=haadf_norm \
  model.projection_model.pretrained_path=weights/particle_thickness_predictor_250k.pt
```

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

With the layout above (weights downloaded into `./weights/`), `main.task=test` is enough:

```bash
python main.py main.task=test model.pc_model.pretrained=True
```

To point at a non-default checkpoint:

```bash
python main.py main.task=test \
  model.pc_model.pretrained=True \
  model.pc_model.pretrained_path=weights/my_other_pc_model.pt
```

### Reproduce the paper results (coarse → refinement → optimization)

```bash
python main.py main.task=test_refinement main.seed=42 \
  data.data_type=both data.in_memory=False data.augment=False \
  model.pc_model.pretrained=True \
  evaluation.num_inference_steps=100      # paper: 10, 100 and 1000
```

`main.seed=42 data.data_type=both` regenerates the exact 1000-sample real test split used in the paper. Other seeds or data types give a different test set. The run writes `refinement_metrics.txt` / `.json` in the Hydra output directory. They contain the 3D and 2D (xy-projected) Chamfer distances and the mean / std of first-nearest-neighbour distances. Each metric is reported for the coarse, refined and refined + optimized structures, in nm (normalized coordinates × `cell_size / 2`). Use `evaluation.max_batches=2` for a quick check and `evaluation.save_plots=True` for per-sample projections.

The test-time optimization runs 500 SGD steps per sample, so a full 1000-sample evaluation takes roughly an hour on a single GPU.

### Train the refinement model

```bash
python main.py main.task=train_refinement main.seed=42 \
  data.data_type=both data.batch_size=16 \
  model.pc_model.pretrained_path=weights/best_pc_model.pt
```

This initializes the refinement model from the diffusion model and trains it with a Sinkhorn loss, matching the released `refinement_model.pt` (20 epochs, AdamW, lr 2e-4). The perturbation and optimization settings are under `refinement:` in `config/config.yaml`. Training the Sinkhorn loss requires `geomloss`.

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

- [ ] Add a quickstart script that runs inference end-to-end on the bundled `mar/data/exp/*.npz` samples without needing the full dataset.
