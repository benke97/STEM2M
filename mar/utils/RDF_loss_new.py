import torch
import torch.nn as nn
import math
from typing import Literal

class RDFLoss(nn.Module):
    """
    Computes a pseudo-RDF loss between two point clouds using Kernel Density Estimation (KDE).

    This loss operates on single point clouds (not batches) and assumes all points
    are valid. It calculates a pseudo-Radial Distribution Function (RDF) for a
    predicted and a target point cloud, and then computes the Mean Squared Error
    between their RDFs, providing a differentiable measure of structural similarity.

    Args:
        cutoff (float): The maximum distance to consider for the RDF.
        n_bins (int): The number of bins to evaluate the RDF over the distance range.
        bandwidth (float): The bandwidth (h) for the Gaussian KDE kernel.
        reduction (Literal['mean', 'sum', 'none']): Specifies the reduction for the
            final MSE loss between the two RDFs. Defaults to 'mean'.
        eps (float): A small epsilon for numerical stability, e.g., in normalization.
    """
    def __init__(
        self,
        cutoff: float,
        n_bins: int = 100,
        bandwidth: float = 0.005,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        eps: float = 1e-7,
    ):
        super().__init__()
        if not (cutoff > 0 and n_bins > 0 and bandwidth > 0):
            raise ValueError("cutoff, n_bins, and bandwidth must be positive.")

        self.cutoff = cutoff
        self.n_bins = n_bins
        self.bandwidth = bandwidth
        self.reduction = reduction
        self.eps = eps

        # Define the points at which the RDF will be evaluated.
        # Register as a buffer so it moves to the correct device with the model.
        r_eval_points = torch.linspace(0, self.cutoff, self.n_bins)
        self.register_buffer('r_eval', r_eval_points)

    def _compute_rdf(self, pc: torch.Tensor) -> torch.Tensor:
        """
        Computes the pseudo-RDF for a single point cloud.

        Args:
            pc (torch.Tensor): A single point cloud of shape [N, 3].

        Returns:
            torch.Tensor: The computed pseudo-RDF of shape [n_bins].
        """
        # 1. Find all valid pairwise distances
        # dist_matrix is [N, N], containing distances between all pairs of points.
        dist_matrix = torch.cdist(pc, pc, p=2.0)

        # We only care about distances > 0 and <= cutoff.
        # This mask implicitly handles self-distances (d=0).
        mask = (dist_matrix > self.eps) & (dist_matrix <= self.cutoff)
        valid_dists = dist_matrix[mask]
        
        # If no pairs are within the cutoff, the RDF is zero.
        if valid_dists.numel() == 0:
            return torch.zeros_like(self.r_eval)

        # 2. Apply Gaussian kernel (KDE)
        # We estimate the density at each `r_eval` point.
        # Shapes: r_eval=[1, n_bins], valid_dists=[M, 1] -> u=[M, n_bins]
        u = (self.r_eval.unsqueeze(0) - valid_dists.unsqueeze(1)) / self.bandwidth
        kernel_vals = torch.exp(-0.5 * u**2)
        
        # Sum the kernel contributions for each bin to get the unnormalized RDF.
        rdf = kernel_vals.sum(dim=0)

        return rdf

    def forward(self, pc_pred: torch.Tensor, pc_gt: torch.Tensor) -> torch.Tensor:
        """
        Calculates the RDF loss between a predicted and a ground truth point cloud.

        Args:
            pc_pred (torch.Tensor): Predicted point cloud, shape [N_pred, 3] or [1, N_pred, 3].
            pc_gt (torch.Tensor): Ground truth point cloud, shape [N_gt, 3] or [1, N_gt, 3].

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Standardize input shape to [N, 3]
        if pc_pred.dim() == 3:
            pc_pred = pc_pred.squeeze(0)
        if pc_gt.dim() == 3:
            pc_gt = pc_gt.squeeze(0)

        # Calculate RDF for the prediction (with gradients)
        rdf_pred = self._compute_rdf(pc_pred)

        # Calculate RDF for the ground truth (without gradients)
        with torch.no_grad():
            rdf_gt = self._compute_rdf(pc_gt)

        #normalize by maximum value in rdf_gt #i think this is better than dividing each by own max
        rdf_gt_max = torch.max(rdf_gt)
        rdf_pred = rdf_pred / (rdf_gt_max + self.eps)
        rdf_gt = rdf_gt / (rdf_gt_max + self.eps)

        # Compute the loss between the two RDFs
        loss = torch.nn.functional.mse_loss(rdf_pred, rdf_gt, reduction=self.reduction)

        return loss

def visualize_rdf(
    loss_module: RDFLoss,
    pc_pred: torch.Tensor,
    pc_gt: torch.Tensor,
    output_path: str = None,
):
    """
    Computes and plots the RDFs for a prediction and ground truth point cloud.

    Args:
        loss_module (RDFLoss): An instance of the RDFLoss class.
        pc_pred (torch.Tensor): The predicted point cloud [N_pred, 3].
        pc_gt (torch.Tensor): The ground truth point cloud [N_gt, 3].
        output_path (str, optional): If provided, saves the plot to this file path.
                                     Otherwise, displays the plot.
    """
    import matplotlib.pyplot as plt

    loss_module.eval()

    # Ensure inputs are on the same device as the model's buffers
    device = loss_module.r_eval.device
    pc_pred = pc_pred.to(device)
    pc_gt = pc_gt.to(device)
    
    # Standardize input shape to [N, 3]
    if pc_pred.dim() == 3:
        pc_pred = pc_pred.squeeze(0)
    if pc_gt.dim() == 3:
        pc_gt = pc_gt.squeeze(0)

    # Compute RDFs
    with torch.no_grad():
        rdf_pred = loss_module._compute_rdf(pc_pred)
        rdf_gt = loss_module._compute_rdf(pc_gt)
    # Normalize RDFs
    rdf_gt_max = torch.max(rdf_gt)
    rdf_pred = rdf_pred / (rdf_gt_max + loss_module.eps)
    rdf_gt = rdf_gt / (rdf_gt_max + loss_module.eps)
    # Prepare data for plotting
    r_np = loss_module.r_eval.cpu().numpy()
    rdf_pred_np = rdf_pred.cpu().numpy()
    rdf_gt_np = rdf_gt.cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(r_np, rdf_gt_np, label=f'Ground Truth RDF (N={pc_gt.shape[0]})', color='blue', lw=2)
    plt.plot(r_np, rdf_pred_np, label=f'Predicted RDF (N={pc_pred.shape[0]})', color='red', ls='--')
    
    plt.title("RDF Comparison")
    plt.xlabel("Distance (r)")
    plt.ylabel("Normalized Pseudo-RDF")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        #print(f"Saved RDF plot to {output_path}")
    else:
        plt.show()
    plt.close()
