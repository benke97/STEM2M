# mar/models/refinement.py
"""
Full inference pipeline used for the paper results:

  1. sample_coarse()        reverse diffusion with the PC model (coarse structure)
  2. refine()               single forward pass of the refinement model (t = 0),
                            which predicts the residual to subtract from the coarse structure
  3. optimize_uniformity()  test-time optimization of z only, trading off nearest-neighbour
                            uniformity against fidelity to the refined structure

All point clouds are (B, 4, N): xyz + existence mask.
"""
import logging

import torch
import torch.nn as nn

from mar.utils.conditioning import condition_point_cloud
from mar.utils.visualization import center_com

log = logging.getLogger(__name__)


def _im_com_offset(com_2d: torch.Tensor) -> torch.Tensor:
    """(B, 2) image-plane CoM -> (B, 3, 1) offset with z = 0."""
    zeros = torch.zeros(com_2d.shape[0], 1, device=com_2d.device, dtype=com_2d.dtype)
    return torch.cat([com_2d, zeros], dim=1).unsqueeze(-1)


@torch.no_grad()
def sample_coarse(pc_model, projection_model, noise_scheduler, images, pixel_sizes, com_2d,
                  existence, cfg, num_inference_steps):
    """Reverse diffusion with the PC model. Returns a CoM-centered (B, 4, N) sample."""
    device = images.device
    B, _, N = existence.shape
    im_com = _im_com_offset(com_2d)

    sample = torch.cat((torch.randn((B, 3, N), device=device), existence), dim=1)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in noise_scheduler.timesteps:
        xyz_centered = center_com(sample[:, :3, :], mask=existence)
        pc_for_conditioning = torch.cat((xyz_centered + im_com, sample[:, 3:, :]), dim=1)
        model_input = condition_point_cloud(
            point_clouds=pc_for_conditioning,
            pixel_sizes=pixel_sizes,
            images=images,
            projection_model=projection_model,
            device=device,
            cfg=cfg,
            mask_feature_map=False,
        )
        model_input[:, :3, :] -= im_com

        timesteps = torch.full((B,), int(t), device=device, dtype=torch.long)
        predicted_noise = pc_model(model_input, timesteps, pixel_sizes, None)
        predicted_noise = center_com(predicted_noise, mask=existence)

        new_xyz = noise_scheduler.step(predicted_noise, t, sample[:, :3, :]).prev_sample
        sample = torch.cat((new_xyz, sample[:, 3:, :]), dim=1)

    return torch.cat((center_com(sample[:, :3, :], mask=existence), sample[:, 3:, :]), dim=1)


@torch.no_grad()
def refine(refinement_model, projection_model, coarse, images, pixel_sizes, com_2d, cfg):
    """One refinement pass at t = 0. Returns the refined (B, 4, N) point cloud."""
    device = images.device
    B = coarse.shape[0]
    existence = coarse[:, 3:, :]
    im_com = _im_com_offset(com_2d)

    xyz_centered = center_com(coarse[:, :3, :], existence)
    conditioned = condition_point_cloud(
        torch.cat((xyz_centered + im_com, existence), dim=1),
        pixel_sizes,
        images,
        projection_model,
        device,
        cfg,
        mask_feature_map=False,
    )
    conditioned[:, :3, :] -= im_com

    timesteps = torch.zeros(B, device=device, dtype=torch.long)
    residual = refinement_model(conditioned, timesteps, pixel_sizes, None)
    return torch.cat((xyz_centered - residual, existence), dim=1)


class UniformityLoss(nn.Module):
    """
    Penalizes the variance of k-nearest-neighbour distances (several k at once), which
    pushes the point cloud toward a uniform, lattice-like spacing.

    Input shape: (B, 4, N), where the 4th channel is the existence mask (> 0.5 = real point).
    """

    def __init__(self, ks=(2, 4, 8, 16), eps: float = 1e-9):
        super().__init__()
        self.ks = sorted(ks)
        self.ksum = sum(ks)
        self.eps = eps

    @staticmethod
    def _pairwise_distances_squared(x: torch.Tensor) -> torch.Tensor:
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        dists_sq = x_sq - 2 * torch.bmm(x, x.transpose(-1, -2)) + x_sq.transpose(-1, -2)
        return torch.clamp(dists_sq, min=0.0)

    def forward(self, pc_4d: torch.Tensor) -> torch.Tensor:
        B = pc_4d.shape[0]
        xyz = pc_4d[:, :3, :].transpose(1, 2)  # (B, N, 3)
        mask = pc_4d[:, 3, :] > 0.5            # (B, N)
        num_real_points = mask.sum(dim=1).float()

        dists_sq = self._pairwise_distances_squared(xyz)
        valid_pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        dists_sq.masked_fill_(~valid_pair_mask, float('inf'))
        dists_sq.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

        batch_loss_terms = torch.zeros(B, device=pc_4d.device)
        for k in self.ks:
            valid_indices = torch.where(num_real_points > k)[0]
            if len(valid_indices) == 0:
                continue
            knn_dists_sq, _ = torch.topk(dists_sq[valid_indices], k, dim=2, largest=False)
            valid_knn_dists_sq = knn_dists_sq[mask[valid_indices]]
            valid_knn_dists_sq = valid_knn_dists_sq[valid_knn_dists_sq < float('inf')]
            if valid_knn_dists_sq.numel() > 1:
                variance = torch.var(torch.sqrt(valid_knn_dists_sq + self.eps))
                batch_loss_terms[valid_indices] += (1 - k / self.ksum) * variance

        valid_items_mask = num_real_points > 0
        if valid_items_mask.sum() > 0:
            return batch_loss_terms[valid_items_mask].mean()
        return torch.tensor(0.0, device=pc_4d.device)


def optimize_uniformity(samples, steps=500, lr=0.1, momentum=0.9,
                        uniformity_weight=1.0, fidelity_weight=0.8, ks=(2, 4, 8, 16)):
    """
    Test-time optimization applied after refinement. Only z is updated (x/y gradients of
    real points are zeroed, since x/y are constrained by the image). The objective is
        uniformity_weight * UniformityLoss + fidelity_weight * MSE(z-moved, refined)
    Uses no ground truth. Returns the optimized, CoM-centered (B, 4, N) batch.
    """
    loss_fn = UniformityLoss(ks=ks)
    optimized = []

    for i in range(samples.shape[0]):
        existence = samples[i:i + 1, 3:, :]
        mask = existence.squeeze() > 0.5
        xyz = nn.Parameter(samples[i:i + 1, :3, :].detach().clone())
        initial_valid = center_com(xyz.detach().clone(), mask=existence)[0, :, mask]
        optimizer = torch.optim.SGD([xyz], lr=lr, momentum=momentum)

        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()
                xyz_valid = center_com(xyz, mask=mask.unsqueeze(0))[0, :, mask]
                loss_uniformity = loss_fn(torch.cat([xyz, existence.to(xyz.dtype)], dim=1))
                fidelity = torch.mean((xyz_valid - initial_valid) ** 2)
                loss = uniformity_weight * loss_uniformity + fidelity_weight * fidelity
                loss.backward()
                xyz.grad[0, 0, mask] = 0.0
                xyz.grad[0, 1, mask] = 0.0
                optimizer.step()

        xyz_centered = center_com(xyz.detach(), mask=mask.unsqueeze(0))
        optimized.append(torch.cat([xyz_centered, existence], dim=1))

    return torch.cat(optimized, dim=0)
