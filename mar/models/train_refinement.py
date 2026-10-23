# mar/models/train_refinement.py
"""
`main.task=train_refinement`: trains the refinement model.

The refinement model has the same PVCNN2 architecture as the PC model and is initialized
from the PC model weights. It is trained at t = 0 to predict the residual between a
synthetically perturbed ground-truth point cloud and the clean one. The perturbation
mimics the errors of the coarse diffusion output: Gaussian noise that, for most points,
only moves z; and padding points scattered around randomly chosen real points.
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from mar.utils.conditioning import condition_point_cloud
from mar.utils.helpers import get_optimizer, get_scheduler, save_checkpoint, save_metrics
from mar.utils.visualization import center_com

log = logging.getLogger(__name__)


def perturb_point_cloud(xyz_centered, existence, noise_level, zero_xy_fraction, padding_noise_factor):
    """Returns (perturbed xyz, noise added to real points). All tensors (B, 3|1, N)."""
    noise = torch.randn_like(xyz_centered) * noise_level
    B, _, N = noise.shape
    zero_mask = torch.rand(B, 1, N, device=noise.device) < zero_xy_fraction
    noise[:, :2, :][zero_mask.expand(-1, 2, -1)] = 0.0
    perturbed = xyz_centered + noise * existence[:, :1, :]

    # Padding (non-existent) points: place them near random real points.
    exist_mask = existence[:, 0, :] > 0.5
    for b in range(B):
        pad_idx = torch.where(~exist_mask[b])[0]
        real_idx = torch.where(exist_mask[b])[0]
        if len(pad_idx) == 0 or len(real_idx) == 0:
            continue
        sources = real_idx[torch.randint(0, len(real_idx), (len(pad_idx),), device=noise.device)]
        offsets = (torch.rand(3, len(pad_idx), device=noise.device) - 0.5) * 2 * noise_level * padding_noise_factor
        offsets[:2, torch.rand(len(pad_idx), device=noise.device) < 0.5] = 0.0
        perturbed[b, :, pad_idx] = xyz_centered[b, :, sources] + offsets

    return perturbed, noise


def _refinement_loss(batch, model, projection_model, cfg, device, noise_level, padding_noise_factor,
                     sinkhorn_loss_fn):
    images = batch['image'].to(device)
    pixel_sizes = batch['pixel_size'].to(device)
    com_2d = batch['CoM'].to(device)
    clean = batch['point_cloud'].to(device)
    xyz_loss_mask = batch['point_cloud_mask'].unsqueeze(1).to(device)
    xyz, existence = clean[:, :3, :], clean[:, 3:, :]
    B = clean.shape[0]

    gt_centered = center_com(xyz, xyz_loss_mask)
    perturbed, noise = perturb_point_cloud(
        gt_centered, existence, noise_level,
        cfg.refinement.training.zero_xy_fraction, padding_noise_factor,
    )

    perturbed_centered = center_com(perturbed, existence)
    im_com = torch.cat([com_2d, torch.zeros(B, 1, device=device, dtype=com_2d.dtype)], dim=1).unsqueeze(-1)
    model_input = condition_point_cloud(
        torch.cat((perturbed_centered + im_com, existence), dim=1),
        pixel_sizes, images, projection_model, device, cfg, mask_feature_map=False,
    )
    model_input[:, :3, :] -= im_com

    timesteps = torch.zeros(B, device=device, dtype=torch.long)
    predicted = model(model_input, timesteps, pixel_sizes, None)

    if sinkhorn_loss_fn is not None:
        pred_x0 = center_com(perturbed_centered - predicted, existence)
        gt_w = xyz_loss_mask.squeeze(1).float()
        pred_w = existence.squeeze(1).float()
        gt_w = gt_w / (gt_w.sum(dim=1, keepdim=True) + 1e-8)
        pred_w = pred_w / (pred_w.sum(dim=1, keepdim=True) + 1e-8)
        losses = sinkhorn_loss_fn(gt_w, gt_centered.permute(0, 2, 1), pred_w, pred_x0.permute(0, 2, 1))
        return (losses * 10.0).mean()

    target = center_com(noise, xyz_loss_mask)
    pred = center_com(predicted, existence)
    per_elem = F.mse_loss(pred, target, reduction='none') * xyz_loss_mask
    count = xyz_loss_mask.sum(dim=[1, 2]) * 3
    per_sample = torch.where(count > 0, per_elem.sum(dim=[1, 2]) / count, torch.zeros_like(count))
    return per_sample.mean()


def train_refinement_model(accelerator, cfg, model, projection_model, train_loader, val_loader):
    projection_model.eval()
    train_cfg = cfg.refinement.training

    sinkhorn_loss_fn = None
    if train_cfg.loss == "Sinkhorn":
        from geomloss import SamplesLoss
        sinkhorn_loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.04, scaling=0.5, diameter=1,
                                       backend='tensorized')
    elif train_cfg.loss != "MSELoss":
        raise ValueError(f"Unknown refinement loss: {train_cfg.loss}")

    optimizer = get_optimizer(cfg, model)
    total_training_steps = cfg.training.epochs * len(train_loader)
    lr_scheduler = get_scheduler(cfg, optimizer, total_training_steps)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history = {"epochs": [], "learning_rate": [], "train_loss_total": [], "val_loss_total": []}
    best_val_loss = float('inf')

    for epoch in range(cfg.training.epochs):
        if accelerator.is_main_process:
            log.info(f"--- Epoch {epoch + 1}/{cfg.training.epochs} ---")

        model.train()
        train_loss_sum = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train {epoch + 1}", disable=not accelerator.is_main_process)
        for batch in progress_bar:
            loss = _refinement_loss(batch, model, projection_model, cfg, accelerator.device,
                                    train_cfg.noise_level, train_cfg.train_padding_noise_factor,
                                    sinkhorn_loss_fn)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            train_loss_sum += loss.item() * batch['image'].shape[0]
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}",
                                      "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch + 1}", disable=not accelerator.is_main_process):
                loss = _refinement_loss(batch, model, projection_model, cfg, accelerator.device,
                                        train_cfg.val_noise_level, train_cfg.val_padding_noise_factor,
                                        sinkhorn_loss_fn)
                val_loss_sum += loss.item() * batch['image'].shape[0]

        # The original training run also stepped the scheduler once per epoch.
        lr_scheduler.step()

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_loss = val_loss_sum / len(val_loader.dataset)
        if not accelerator.is_main_process:
            continue

        log.info(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        history["epochs"].append(epoch)
        history["learning_rate"].append(optimizer.param_groups[0]['lr'])
        history["train_loss_total"].append(train_loss)
        history["val_loss_total"].append(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            log.info(f"New best validation loss: {best_val_loss:.4f}")
        if (epoch + 1) % cfg.training.checkpoint_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_loss,
                'cfg': OmegaConf.to_container(cfg, resolve=True),
            }, is_best, filename=f"checkpoint_epoch_{epoch + 1}.pt", best_filename="best_model.pt")
        save_metrics(history, filename=str(metrics_dir / "training_metrics.json"))

    if accelerator.is_main_process:
        log.info("Refinement training finished.")
