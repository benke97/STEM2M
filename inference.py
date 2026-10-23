"""
inference.py — generate point clouds from the trained diffusion model and dump
GT vs prediction artefacts for visual + numerical inspection.

Supports three "smart inference" levers on top of the trained model:

  --scheduler ddpm | ddim | dpmpp        algorithmic sampler choice
  --num-seeds K                          ensemble K trajectories, per-slot median
  --guidance-strength X                  test-time thickness-map projection (DPS-style)

Run from the repository root with the environment from README.md activated.
Examples:

    python inference.py
    python inference.py --num-seeds 4
    python inference.py --scheduler dpmpp --num-inference-steps 30
    python inference.py --num-seeds 4 --scheduler dpmpp --num-inference-steps 30
    python inference.py --guidance-strength 1.0

GT and prediction are both CoM-centered (mask-aware) before computing chamfer
and before plotting, so apparent xy offsets between them are gone.
"""
import argparse
import datetime
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from omegaconf import OmegaConf

from mar.data.data_module import setup_dataloaders
from mar.models.modern_unet import ModernUNet
from mar.models.pvcnn.pvcnn import PVCNN2
from mar.models.resnet_size_estimator import ResNet18AdaptivePoolFeatureExtractor
from mar.utils.conditioning import condition_point_cloud
from mar.utils.helpers import (
    create_custom_beta_schedule,
    create_custom_beta_schedule_exp,
)
from mar.utils.visualization import center_com

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("inference")


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def load_cfg(config_path: Path):
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    return cfg


def build_scheduler(cfg, name: str):
    """name in {'ddpm','ddim','dpmpp'} — all share the same trained betas."""
    diff = cfg.training.diffusion
    if diff.beta_schedule == "custom":
        betas = create_custom_beta_schedule(
            num_train_timesteps=diff.num_timesteps,
            alpha_bar_power=diff.alpha_bar_power,
            target_alpha_bar_T=diff.target_alpha_bar_T,
            beta_clip_max=diff.beta_clip_max,
        )
    elif diff.beta_schedule == "exp_growth":
        betas = create_custom_beta_schedule_exp()
    else:
        betas = None

    common = dict(num_train_timesteps=diff.num_timesteps, prediction_type="epsilon")
    if betas is not None:
        common["trained_betas"] = betas
    else:
        common["beta_start"] = diff.beta_start
        common["beta_end"] = diff.beta_end
        common["beta_schedule"] = diff.beta_schedule

    if name == "ddpm":
        return DDPMScheduler(**common, clip_sample=False)
    if name == "ddim":
        return DDIMScheduler(**common, clip_sample=False)
    if name == "dpmpp":
        # DPMSolverMultistepScheduler doesn't accept clip_sample
        return DPMSolverMultistepScheduler(**common, algorithm_type="dpmsolver++")
    raise ValueError(f"unknown scheduler {name!r}; choose ddpm | ddim | dpmpp")


def load_pc_model(cfg, ckpt_path: Path, device):
    m = PVCNN2(
        num_classes=cfg.model.pc_model.num_classes,
        embed_dim=cfg.model.pc_model.embed_dim,
        dropout=cfg.model.pc_model.dropout,
        extra_feature_channels=cfg.model.pc_model.extra_feature_channels,
        width_multiplier=cfg.model.pc_model.width_multiplier,
        voxel_resolution_multiplier=cfg.model.pc_model.voxel_resolution_multiplier,
        use_pixel_size_embedding=cfg.model.pc_model.get("use_pixel_size_embedding", True),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(state, PVCNN2):
        state = state.state_dict()
    m.load_state_dict(state)
    log.info(f"Loaded pc_model from {ckpt_path}")
    return m.to(device).eval()


def load_projection_model(cfg, device):
    p = cfg.model.projection_model
    m = ModernUNet(
        n_channels=p.params.n_channels,
        n_classes=p.params.n_classes,
        init_features=p.params.init_features,
        num_groups=p.params.num_groups,
    )
    if p.pretrained:
        path = Path(p.pretrained_path)
        if not path.is_absolute():
            path = (Path(__file__).parent / path).resolve()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded projection_model from {path}")
    return m.to(device).eval()


def load_size_estimator(cfg, device):
    path = Path(cfg.model.size_estimator_weights_path)
    if not path.is_absolute():
        path = (Path(__file__).parent / path).resolve()
    m = ResNet18AdaptivePoolFeatureExtractor(
        input_nc=1,
        output_feature_dim=cfg.model.pc_model.get("size_embed_dim", 64),
        pretrained_estimator_path=str(path),
    )
    log.info(f"Loaded size_estimator backbone from {path}")
    return m.to(device).eval()


# ---------------------------------------------------------------------------
# Differentiable Gaussian splat — used by guidance only
# ---------------------------------------------------------------------------

def gaussian_splat(xy_pixels: torch.Tensor, weights: torch.Tensor, image_size: int, sigma: float):
    """
    xy_pixels: (B, N, 2) pixel coords in [0, image_size).
    weights:   (B, N)    per-point weight (existence).
    Returns:   (B, image_size, image_size) density map. Differentiable in xy_pixels.
    """
    B, N, _ = xy_pixels.shape
    grid = torch.arange(image_size, device=xy_pixels.device, dtype=xy_pixels.dtype)
    gy = grid.view(1, 1, image_size, 1)
    gx = grid.view(1, 1, 1, image_size)
    px = xy_pixels[..., 0].view(B, N, 1, 1)
    py = xy_pixels[..., 1].view(B, N, 1, 1)
    sq = (gx - px) ** 2 + (gy - py) ** 2
    contrib = torch.exp(-sq / (2.0 * sigma * sigma))                   # (B, N, H, W)
    contrib = contrib * weights.view(B, N, 1, 1)
    return contrib.sum(dim=1)                                          # (B, H, W)


def thickness_guidance_grad(
    xyz_centered: torch.Tensor,            # (B, 3, N), CoM-centered, requires_grad
    existence: torch.Tensor,               # (B, 1, N)
    im_com_offset: torch.Tensor,           # (B, 3, 1)
    pixel_sizes: torch.Tensor,             # (B,)
    target_thickness: torch.Tensor,        # (B, H, W) — U-Net prediction
    cell_size: float,
    image_size: int,
    sigma: float,
):
    """Compute -d(MSE(splat(xyz), target)) / d(xyz). Returns (B, 3, N) gradient."""
    B, _, N = xyz_centered.shape
    xyz_world = xyz_centered + im_com_offset                           # (B, 3, N)
    norm_to_img = (cell_size / (2.0 * pixel_sizes)).view(B, 1)         # (B, 1)
    xy = xyz_world[:, :2, :].permute(0, 2, 1)                          # (B, N, 2)
    xy_pixels = xy * norm_to_img.unsqueeze(-1) + image_size / 2.0       # (B, N, 2)
    weights = existence.squeeze(1)                                      # (B, N)

    splat = gaussian_splat(xy_pixels, weights, image_size, sigma)
    # Normalize both to compare shapes; thickness target is in arbitrary units
    splat_n = splat / (splat.sum(dim=(1, 2), keepdim=True) + 1e-6)
    targ_n = target_thickness / (target_thickness.sum(dim=(1, 2), keepdim=True) + 1e-6)
    loss = ((splat_n - targ_n) ** 2).mean()
    grad = torch.autograd.grad(loss, xyz_centered)[0]
    return grad


# ---------------------------------------------------------------------------
# Reverse diffusion
# ---------------------------------------------------------------------------

def sample_one(
    pc_model,
    projection_model,
    size_estimator,
    scheduler,
    batch,
    cfg,
    device,
    num_inference_steps,
    init_noise=None,
    guidance_strength=0.0,
):
    """Single trajectory. Returns (B, 4, N)."""
    images = batch["image"].to(device)
    pixel_sizes = batch["pixel_size"].to(device)
    com_2d = batch["CoM"].to(device)
    gt_pc = batch["point_cloud"].to(device)
    B, _, N = gt_pc.shape

    existence = gt_pc[:, 3:, :]
    if init_noise is None:
        init_noise = torch.randn(B, 3, N, device=device)
    sample = torch.cat([init_noise, existence], dim=1)
    im_com_offset = torch.cat([com_2d, torch.zeros(B, 1, device=device)], dim=1).unsqueeze(-1)

    size_embedding = None
    if size_estimator is not None:
        with torch.no_grad():
            size_embedding = size_estimator(images)

    # If guidance is on, we need the U-Net's predicted thickness map up front.
    target_thickness = None
    if guidance_strength > 0.0:
        with torch.no_grad():
            proj_out = projection_model(images)
        # Try common key names; fall back to taking the first tensor.
        if isinstance(proj_out, dict):
            for k in ("logits", "thickness", "out", "prediction"):
                if k in proj_out:
                    target_thickness = proj_out[k]
                    break
            if target_thickness is None:
                target_thickness = next(iter(proj_out.values()))
        else:
            target_thickness = proj_out
        target_thickness = target_thickness.squeeze(1).detach()        # (B, H, W)

    use_occlusion = cfg.model.conditioning.get("use_occlusion", False)
    image_size = images.shape[-1]

    scheduler.set_timesteps(num_inference_steps, device=device)
    for t in scheduler.timesteps:
        t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
        with torch.no_grad():
            xyz = sample[:, :3, :]
            exist = sample[:, 3:, :]
            xyz_centered = center_com(xyz, mask=exist)

            cond_input = torch.cat([xyz_centered + im_com_offset, exist], dim=1)
            cond_features = condition_point_cloud(cond_input.clone(), pixel_sizes, images, projection_model, device, cfg)
            model_input = cond_features.clone()
            if not use_occlusion:
                model_input[:, :3, :] -= im_com_offset

            noise_pred = pc_model(model_input, t_batch, pixel_sizes, size_embedding)
            noise_pred_centered = center_com(noise_pred, mask=exist)

        prev = scheduler.step(noise_pred_centered, t, xyz_centered).prev_sample
        prev_centered = center_com(prev, mask=exist)

        # Test-time guidance: nudge prev toward thickness consistency.
        if guidance_strength > 0.0 and target_thickness is not None:
            with torch.enable_grad():
                xyz_g = prev_centered.detach().requires_grad_(True)
                grad = thickness_guidance_grad(
                    xyz_centered=xyz_g,
                    existence=exist,
                    im_com_offset=im_com_offset,
                    pixel_sizes=pixel_sizes,
                    target_thickness=target_thickness,
                    cell_size=cfg.data.cell_size,
                    image_size=image_size,
                    sigma=1.0,
                )
            prev_centered = prev_centered.detach() - guidance_strength * grad
            prev_centered = center_com(prev_centered, mask=exist)

        sample = torch.cat([prev_centered, exist], dim=1)

    return sample, gt_pc


def sample_ensemble(
    pc_model, projection_model, size_estimator, scheduler, batch, cfg, device,
    num_inference_steps, num_seeds, guidance_strength, base_seed,
):
    """Run K trajectories with different starting noise. Returns (K, B, 4, N), gt (B, 4, N)."""
    images = batch["image"]
    B = images.shape[0]
    N = batch["point_cloud"].shape[-1]
    preds = []
    gt_pc = None
    for k in range(num_seeds):
        g = torch.Generator(device=device).manual_seed(base_seed + 1000 * k + 1)
        init_noise = torch.randn(B, 3, N, generator=g, device=device)
        pred, gt_pc = sample_one(
            pc_model, projection_model, size_estimator, scheduler, batch, cfg, device,
            num_inference_steps, init_noise=init_noise, guidance_strength=guidance_strength,
        )
        preds.append(pred)
    return torch.stack(preds, dim=0), gt_pc


def aggregate_seeds_median(preds_K: torch.Tensor, existence: torch.Tensor):
    """preds_K: (K, B, 4, N). Returns (B, 4, N) with per-slot median xyz."""
    xyz_K = preds_K[:, :, :3, :]
    median_xyz = xyz_K.median(dim=0).values
    return torch.cat([median_xyz, existence], dim=1)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_xyz(path: Path, xyz: np.ndarray, element="Pt"):
    path.write_text(
        f"{xyz.shape[0]}\n\n" + "\n".join(f"{element} {x:.5f} {y:.5f} {z:.5f}" for x, y, z in xyz)
    )


def render_views(path: Path, gt_xyz: np.ndarray, pred_xyz: np.ndarray, image: np.ndarray, title: str):
    """gt_xyz and pred_xyz are both expected mask-aware-CoM-centered already."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image, cmap="gray", origin="lower")
    axes[0].set_title("HAADF (input)")
    axes[0].axis("off")

    for ax, (i, j, label) in zip(axes[1:], [(0, 1, "xy (top-down)"), (0, 2, "xz (side)"), (1, 2, "yz (side)")]):
        ax.scatter(gt_xyz[:, i], gt_xyz[:, j], s=2, c="black", alpha=0.5, label=f"GT (n={len(gt_xyz)})")
        ax.scatter(pred_xyz[:, i], pred_xyz[:, j], s=2, c="red", alpha=0.5, label=f"pred (n={len(pred_xyz)})")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8)
        lim = max(np.abs(gt_xyz).max(), np.abs(pred_xyz).max(), 0.1) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    da = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1).mean()
    db = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(axis=1).mean()
    return float((da + db) / 2.0)


def mask_aware_center(xyz: np.ndarray, active: np.ndarray) -> np.ndarray:
    """xyz: (3, N), active: (N,) bool. Subtract mean over active points; returns (3, N)."""
    if not active.any():
        return xyz
    centroid = xyz[:, active].mean(axis=1, keepdims=True)
    return xyz - centroid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("weights/best_pc_model.pt"))
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-inference-steps", type=int, default=100)
    ap.add_argument("--scheduler", choices=["ddpm", "ddim", "dpmpp"], default="ddpm")
    ap.add_argument("--num-seeds", type=int, default=1, help=">1 enables median ensembling")
    ap.add_argument("--guidance-strength", type=float, default=0.0, help=">0 enables thickness-map test-time guidance")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    here = Path(__file__).parent.resolve()
    cfg_path = args.config if args.config.is_absolute() else (here / args.config).resolve()
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else (here / args.checkpoint).resolve()
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    cfg = load_cfg(cfg_path)
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = 0
    cfg.data.augment = False
    if cfg.data.dataset_path and not Path(cfg.data.dataset_path).is_absolute():
        cfg.data.dataset_path = str((here / cfg.data.dataset_path).resolve())

    out_root = args.output_dir or (here / "inference_out" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {out_root}")
    OmegaConf.save(cfg, out_root / "config_used.yaml")

    seed = args.seed if args.seed is not None else cfg.main.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    log.info(f"Seed: {seed}  scheduler: {args.scheduler}  steps: {args.num_inference_steps}  K: {args.num_seeds}  guidance: {args.guidance_strength}")

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    train_loader, val_loader, test_loader, _ = setup_dataloaders(cfg.data, seed, task="test")
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    if loader is None:
        log.error(f"No {args.split} loader available.")
        sys.exit(1)
    log.info(f"Using {args.split} loader; {len(loader.dataset)} samples available.")

    pc_model = load_pc_model(cfg, ckpt_path, device)
    projection_model = load_projection_model(cfg, device)
    # The released best_pc_model.pt was trained without the size embedding.
    size_estimator = load_size_estimator(cfg, device) if cfg.training.get("use_size_model", False) else None
    scheduler = build_scheduler(cfg, args.scheduler)
    log.info(f"Scheduler: {type(scheduler).__name__}, {args.num_inference_steps} inference steps")

    metrics_lines = ["sample_idx,n_active,chamfer_per_seed,chamfer_ensemble"]
    written = 0
    for batch in loader:
        if written >= args.num_samples:
            break
        preds_K, gt = sample_ensemble(
            pc_model, projection_model, size_estimator, scheduler, batch, cfg, device,
            num_inference_steps=args.num_inference_steps,
            num_seeds=args.num_seeds,
            guidance_strength=args.guidance_strength,
            base_seed=seed,
        )
        existence = gt[:, 3:, :]
        ensemble = aggregate_seeds_median(preds_K, existence)            # (B, 4, N)

        for b in range(gt.shape[0]):
            if written >= args.num_samples:
                break
            gt_pc = gt[b].cpu().numpy()
            ens_pc = ensemble[b].cpu().numpy()
            active = gt_pc[3] > 0.5
            n_active = int(active.sum())

            gt_xyz_centered = mask_aware_center(gt_pc[:3], active)
            ens_xyz_centered = mask_aware_center(ens_pc[:3], active)

            gt_xyz = gt_xyz_centered[:, active].T
            ens_xyz = ens_xyz_centered[:, active].T
            cd_ensemble = chamfer(gt_xyz, ens_xyz)

            cds_per_seed = []
            for k in range(args.num_seeds):
                pk = preds_K[k, b].cpu().numpy()
                pk_centered = mask_aware_center(pk[:3], active)
                cds_per_seed.append(chamfer(gt_xyz, pk_centered[:, active].T))
            cd_per_seed_str = "|".join(f"{c:.4f}" for c in cds_per_seed)

            metrics_lines.append(f"{written},{n_active},{cd_per_seed_str},{cd_ensemble:.6f}")

            stem = f"sample_{written:03d}"
            write_xyz(out_root / f"{stem}_gt.xyz", gt_xyz)
            write_xyz(out_root / f"{stem}_pred.xyz", ens_xyz)
            np.savez(out_root / f"{stem}.npz",
                     gt=gt_pc, pred_ensemble=ens_pc,
                     pred_seeds=preds_K[:, b].cpu().numpy(),
                     image=batch["image"][b].cpu().numpy())
            render_views(
                out_root / f"{stem}_views.png",
                gt_xyz=gt_xyz,
                pred_xyz=ens_xyz,
                image=batch["image"][b, 0].cpu().numpy(),
                title=f"{stem}  chamfer_ens={cd_ensemble:.4f}  per_seed=[{cd_per_seed_str}]  n={n_active}",
            )
            log.info(f"  {stem}: chamfer_ens={cd_ensemble:.4f}  per_seed=[{cd_per_seed_str}]  n={n_active}")
            written += 1

    (out_root / "metrics.csv").write_text("\n".join(metrics_lines) + "\n")

    cd_ens = []
    cd_seed_avg = []
    for line in metrics_lines[1:]:
        parts = line.split(",")
        if parts[3] != "nan":
            cd_ens.append(float(parts[3]))
        seed_vals = [float(x) for x in parts[2].split("|") if x != "nan"]
        if seed_vals:
            cd_seed_avg.append(sum(seed_vals) / len(seed_vals))
    if cd_ens:
        log.info(f"Done. Mean chamfer (ensemble) over {len(cd_ens)} samples: {sum(cd_ens)/len(cd_ens):.4f}")
    if cd_seed_avg and args.num_seeds > 1:
        log.info(f"Mean chamfer (avg over seeds, no aggregation): {sum(cd_seed_avg)/len(cd_seed_avg):.4f}")
    log.info(f"Outputs in {out_root}")


if __name__ == "__main__":
    main()
