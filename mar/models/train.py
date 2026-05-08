# src/mar/models/train.py
import torch
import torch.nn as nn # Import nn
# import torch.cuda.amp as amp # Removed, Accelerate handles AMP
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import traceback # Import traceback for detailed error logging
import hydra # Import hydra
from omegaconf import DictConfig, OmegaConf # For type hinting cfg and OmegaConf utilities
from typing import Tuple, Optional, Dict, Any # For type hints
import matplotlib
matplotlib.use('Agg')        # <- must come before pyplot
import matplotlib.pyplot as plt
from accelerate import Accelerator # Add Accelerator import
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput # For type hinting scheduler output
from mar.models.resnet_size_estimator import ResNet18AdaptivePoolFeatureExtractor
from pytorch3d.loss import chamfer_distance
from mar.utils.conditioning import condition_point_cloud, GrayscalePointCloudProjectionModel
from mar.utils.visualization import visualize_unconditioned, visualize_conditioned, visualize_projection_and_sampling, debug_visualize_scheduler, visualize_refinement
from mar.utils.RDF_loss_new import RDFLoss, visualize_rdf
# Import utilities
# from mar.utils.timestep_sampling import TimestepSampler # REMOVED
from ..utils.helpers import (
    save_checkpoint,
    visualize_predictions, # Assuming this exists elsewhere, might be redundant now
    save_metrics,
    get_optimizer,
    get_scheduler,
    get_loss_function,
    loss_needs_3channels,
)
from collections import defaultdict # Make sure this is imported


log = logging.getLogger(__name__)

# --- Debugging: Per-Batch Loss Contribution Plotting ---
def plot_batch_loss_contributions(timesteps_tensor: torch.Tensor,
                                  mse_losses_per_sample: torch.Tensor,
                                  rdf_losses_per_sample_no_lambda: torch.Tensor,
                                  lambda_rdf: float, # Added lambda_rdf
                                  epoch: int,
                                  batch_idx: int,
                                  cfg: DictConfig,
                                  noise_scheduler_config
                                  ):
    """
    Plots the percentage contribution of MSE and scaled RDF losses for unique timesteps in the current batch.
    Displays the plot using plt.show().
    Args:
        timesteps_tensor (torch.Tensor): Tensor of shape [B] with timesteps for the current batch.
        mse_losses_per_sample (torch.Tensor): Tensor of shape [B] with per-sample MSE losses.
        rdf_losses_per_sample_no_lambda (torch.Tensor): Tensor of shape [B] with per-sample RDF losses (before lambda, after noise norm).
        lambda_rdf (float): The scaling factor applied to the RDF loss.
        epoch (int): Current epoch number.
        batch_idx (int): Current batch index within the epoch.
        cfg (DictConfig): Hydra configuration.
        noise_scheduler_config: The config attribute of the noise_scheduler.
    """
    if timesteps_tensor.numel() == 0:
        log.warning(f"Epoch {epoch}, Batch {batch_idx}: No timesteps to plot.")
        return

    # Aggregate losses per unique timestep
    mse_sum_at_timestep = defaultdict(float)
    rdf_sum_at_timestep = defaultdict(float)
    count_at_timestep = defaultdict(int)

    for i in range(timesteps_tensor.shape[0]):
        t = timesteps_tensor[i].item()
        mse_sum_at_timestep[t] += mse_losses_per_sample[i].item()
        rdf_sum_at_timestep[t] += rdf_losses_per_sample_no_lambda[i].item()
        count_at_timestep[t] += 1

    unique_timesteps_plot = sorted(count_at_timestep.keys())
    if not unique_timesteps_plot:
        log.warning(f"Epoch {epoch}, Batch {batch_idx}: No data to plot after aggregation.")
        return

    mse_percentages_plot = []
    rdf_scaled_percentages_plot = []
    
    avg_mse_values_text = []
    avg_rdf_no_lambda_values_text = []
    avg_scaled_rdf_values_text = []

    for t in unique_timesteps_plot:
        avg_mse_at_t = mse_sum_at_timestep[t] / count_at_timestep[t]
        avg_rdf_no_lambda_at_t = rdf_sum_at_timestep[t] / count_at_timestep[t]
        
        scaled_avg_rdf_at_t = avg_rdf_no_lambda_at_t * lambda_rdf
        total_scaled_loss_at_t = avg_mse_at_t + scaled_avg_rdf_at_t
        epsilon = 1e-8

        perc_mse_t = 0.0
        perc_rdf_scaled_t = 0.0

        if total_scaled_loss_at_t < epsilon:
            if avg_mse_at_t > epsilon: perc_mse_t = 100.0
            if scaled_avg_rdf_at_t > epsilon: perc_rdf_scaled_t = 100.0
        else:
            perc_mse_t = (avg_mse_at_t / total_scaled_loss_at_t) * 100
            perc_rdf_scaled_t = (scaled_avg_rdf_at_t / total_scaled_loss_at_t) * 100
        
        mse_percentages_plot.append(perc_mse_t)
        rdf_scaled_percentages_plot.append(perc_rdf_scaled_t)
        
        avg_mse_values_text.append(f"{avg_mse_at_t:.4f}")
        avg_rdf_no_lambda_values_text.append(f"{avg_rdf_no_lambda_at_t:.4f}")
        avg_scaled_rdf_values_text.append(f"{scaled_avg_rdf_at_t:.4f}")

    plt.figure(figsize=(14, 7)) # Increased figure size for potentially more info
    plt.scatter(unique_timesteps_plot, mse_percentages_plot, label=f'MSE % Contribution (per Timestep Avg)', marker='o', color='blue')
    plt.scatter(unique_timesteps_plot, rdf_scaled_percentages_plot, label=f'Scaled RDF % Contribution (per Timestep Avg, λ={lambda_rdf})', marker='x', color='red')

    # Optionally, annotate points with their actual loss values if plot is not too crowded
    # for i, t_val in enumerate(unique_timesteps_plot):
    #     plt.annotate(f"MSE: {avg_mse_values_text[i]}\nRDF_s: {avg_scaled_rdf_values_text[i]}", (t_val, mse_percentages_plot[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7)

    plt.xlabel("Timestep (present in current batch)")
    plt.ylabel("Percentage of Total Loss Contribution (%)")
    # Create a more concise title, detailed values can be logged or inspected
    title_str = f"Per-Timestep Loss Contribution - Epoch {epoch}, Batch {batch_idx}\n"
    title_str += f"Avg MSEs: {', '.join(avg_mse_values_text)}\n"
    title_str += f"Avg RDFs (no λ): {', '.join(avg_rdf_no_lambda_values_text)}\n"
    title_str += f"Avg Scaled RDFs (λ={lambda_rdf}): {', '.join(avg_scaled_rdf_values_text)}"
    plt.title(title_str, fontsize=10)

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 105)
    if len(unique_timesteps_plot) > 0:
        plt.xlim(min(unique_timesteps_plot) - 5, max(unique_timesteps_plot) + 5)
    elif noise_scheduler_config is not None and hasattr(noise_scheduler_config, 'num_train_timesteps'):
        plt.xlim(0, noise_scheduler_config.num_train_timesteps)
    else:
        plt.xlim(0, 1000)

    log.info(f"Displaying per-timestep loss contribution plot for Epoch {epoch}, Batch {batch_idx}. Close plot window to continue training.")
    plt.show()
    plt.close()
# --- End Debugging ---
def predict_x0_from_xt_eps(xt, eps, t, noise_scheduler):
    """
    Predicts the clean data x0 given noisy data xt, predicted noise eps, and timestep t.
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(xt.device)
    t = t.to(alphas_cumprod.device)
    sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1)

    # Denoise formula: x0 = (xt - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
    pred_x0 = (xt - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
    return pred_x0


def center_com(tensor_xyz, mask=None):
    """
    Centers the Center of Mass (CoM) of a point cloud tensor, making it translationally invariant.
    """
    if tensor_xyz.shape[-2] != 3:
        raise ValueError(f"Expected 3 channels for XYZ, got {tensor_xyz.shape[-2]}")

    if mask is not None:
        masked_tensor = tensor_xyz * mask
        sum_of_coords = masked_tensor.sum(dim=-1)  # Shape: [B, 3]
        num_valid_points = mask.sum(dim=-1)  # Shape: [B, 1]
        com = sum_of_coords / (num_valid_points + 1e-8)  # Shape: [B, 3]
        com = com.unsqueeze(-1)  # Shape: [B, 3, 1] for broadcasting
    else:
        com = tensor_xyz.mean(dim=-1, keepdim=True)

    return tensor_xyz - com


def train_conditioned(accelerator: Accelerator, cfg: DictConfig, model: nn.Module, proj_model: nn.Module,
                      train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                      lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                      noise_scheduler: Any, epoch: int,
                      size_estimator: Optional[nn.Module] = None):
    """
    Cleaned-up training function focusing on CoM-invariant processing with MSE and Sinkhorn losses.
    """
    epoch_loss_total_sum = 0.0
    epoch_uniform_loss_sum = 0.0
    model.train()
    proj_model.eval()
    if size_estimator:
        size_estimator.eval()

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=not accelerator.is_main_process, leave=True)

    # --- Loss Function Initialization ---
    primary_loss_name = cfg.training.loss_fn.primary.name
    sinkhorn_loss_fn = None
    uniform_loss_fn = UniformityLoss()
    if primary_loss_name == "Sinkhorn":
        try:
            from geomloss import SamplesLoss
            # Using recommended parameters from original code
            sinkhorn_loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.04, scaling=0.5, diameter=1, backend='tensorized')
        except ImportError:
            log.error("Geomloss is not installed. Please install it with 'pip install geomloss' to use Sinkhorn loss.")
            raise

    for batch_idx, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            if cfg.main.task == "refinement":
                coarse_point_clouds = batch['coarse_point_cloud'].to(accelerator.device)
            
            clean_point_clouds = batch['point_cloud'].to(accelerator.device)
            images = batch['image'].to(accelerator.device)
            pixel_sizes = batch['pixel_size'].to(accelerator.device)
            CoM = batch['CoM'].to(accelerator.device)
            xyz_loss_mask = batch['point_cloud_mask'].unsqueeze(1).to(accelerator.device)
            current_batch_size = clean_point_clouds.shape[0]

            xyz = clean_point_clouds[:, :3, :]
            existence = clean_point_clouds[:, 3:, :]

            noise_xyz_initial_random = torch.randn_like(xyz)
            if cfg.main.task == "refinement":
                #timestep all 0
                timesteps = torch.zeros(current_batch_size, device=accelerator.device, dtype=torch.long)
                noisy_xyz = coarse_point_clouds[:, :3, :]
                existence_noisy = coarse_point_clouds[:, 3:, :]

            else:
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=accelerator.device).long()
                noisy_xyz = noise_scheduler.add_noise(xyz, noise_xyz_initial_random, timesteps)

            # Center noisy point cloud on its CoM for model input
            if cfg.main.task == "refinement":
                noisy_xyz_centered_for_input = center_com(noisy_xyz, existence_noisy)
            else:
                noisy_xyz_centered_for_input = center_com(noisy_xyz, xyz_loss_mask)

            # Create the 4D point cloud input for conditioning, re-introducing the image CoM
            # to align the point cloud with the image projection space.
            zeros_for_z = torch.zeros(current_batch_size, 1, device=CoM.device, dtype=CoM.dtype)
            im_CoM = torch.cat([CoM, zeros_for_z], dim=1).unsqueeze(-1)  # (B, 3, 1)
            noisy_point_cloud_input_for_conditioning = noisy_xyz_centered_for_input + im_CoM
            if cfg.main.task == "refinement":
                noisy_point_clouds_input = torch.cat((noisy_point_cloud_input_for_conditioning, existence_noisy), dim=1)
            else:
                noisy_point_clouds_input = torch.cat((noisy_point_cloud_input_for_conditioning, existence), dim=1)

            conditioned_point_clouds = condition_point_cloud(
                noisy_point_clouds_input,
                pixel_sizes,
                images,
                proj_model,
                accelerator.device,
                cfg,
                mask_feature_map=False
            )

            conditioned_point_clouds[:, :3, :] -= im_CoM

            size_embeddings = None
            if size_estimator is not None:
                with torch.no_grad():
                    size_embeddings = size_estimator(images)

            predicted_noise_xyz = model(conditioned_point_clouds, timesteps, pixel_sizes, size_embeddings)

            # --- Loss Calculation (CoM Invariant) ---
            # Center the true noise and predicted noise based on their own CoMs to achieve translational invariance.
            noise_xyz_centered_for_loss_target = center_com(noise_xyz_initial_random, xyz_loss_mask)
            if cfg.main.task == "refinement":
                predicted_noise_xyz_centered = center_com(predicted_noise_xyz, existence_noisy)
            else:
                predicted_noise_xyz_centered = center_com(predicted_noise_xyz, xyz_loss_mask)

            # --- Primary Loss Calculation (MSE or Sinkhorn) ---
            primary_loss_term = torch.tensor(0.0, device=accelerator.device)
            pred_x0 = None  

            if primary_loss_name == "MSELoss":
                loss_xyz_per_element = F.mse_loss(predicted_noise_xyz_centered, noise_xyz_centered_for_loss_target, reduction='none')
                masked_loss_per_element = loss_xyz_per_element * xyz_loss_mask
                per_sample_active_elements_count = xyz_loss_mask.sum(dim=[1, 2]) * 3
                per_sample_sum_squared_error_xyz = masked_loss_per_element.sum(dim=[1, 2])

                per_sample_primary_loss = torch.where(
                    per_sample_active_elements_count > 0,
                    per_sample_sum_squared_error_xyz / per_sample_active_elements_count,
                    torch.zeros_like(per_sample_active_elements_count, dtype=torch.float32)
                )
                # Start with the base MSE loss for the whole batch
                primary_loss_term = per_sample_primary_loss.mean()
                if False:
                    # Identify samples with low timesteps to apply uniformity loss
                    low_timestep_mask = timesteps < 50
                    
                    # Check if there are any samples that meet the condition
                    if torch.any(low_timestep_mask):
                        # Filter the necessary tensors for the low-timestep samples
                        pred_x0 = predict_x0_from_xt_eps(
                            noisy_xyz_centered_for_input[low_timestep_mask], 
                            predicted_noise_xyz_centered[low_timestep_mask], 
                            timesteps[low_timestep_mask], 
                            noise_scheduler
                        )
                        
                        # Create the 4D input for the uniformity loss function
                        pred_x0_4d = torch.cat((pred_x0, xyz_loss_mask[low_timestep_mask]), dim=1)
                        
                        # Calculate uniformity loss only on the filtered subset
                        uniform_loss = uniform_loss_fn(pred_x0_4d)
                        
                        # Add the scaled uniformity loss to the total primary loss term
                        alpha = 100 # Weight for the uniformity loss
                        primary_loss_term = primary_loss_term + alpha * uniform_loss
                        epoch_uniform_loss_sum += (alpha * uniform_loss.item()) * current_batch_size
                        
                        #if accelerator.is_main_process: # Log periodically
                        #    log.info(f"Batch {batch_idx}: Applied uniformity loss ({uniform_loss.item():.4f}) to {low_timestep_mask.sum()} samples.")
            elif primary_loss_name == "Sinkhorn":
                if sinkhorn_loss_fn is None:
                    raise RuntimeError("Sinkhorn loss function was not initialized.")

                # Calculate MSE for all samples (will be used for high timesteps or as a fallback)
                if cfg.main.task == "refinement":
                    per_sample_final_loss = torch.zeros(current_batch_size, device=accelerator.device, dtype=torch.float32)
                else:
                    loss_xyz_per_element = F.mse_loss(predicted_noise_xyz_centered, noise_xyz_centered_for_loss_target, reduction='none')
                    masked_loss_per_element = loss_xyz_per_element * xyz_loss_mask
                    per_sample_active_elements_count = xyz_loss_mask.sum(dim=[1, 2]) * 3
                    per_sample_sum_squared_error_xyz = masked_loss_per_element.sum(dim=[1, 2])
                    per_sample_mse_loss = torch.where(
                        per_sample_active_elements_count > 0,
                        per_sample_sum_squared_error_xyz / per_sample_active_elements_count,
                        torch.zeros_like(per_sample_active_elements_count)
                    )

                    # Start with all losses as MSE. Overwrite the ones that need Sinkhorn.
                    per_sample_final_loss = per_sample_mse_loss.clone()

                # Find samples needing Sinkhorn loss (low timesteps)
                if cfg.main.task=="refinement":
                    sinkhorn_mask = torch.ones(current_batch_size, device=accelerator.device, dtype=torch.bool)
                else:
                    sinkhorn_mask = timesteps < 10000
                if torch.any(sinkhorn_mask):
                    if cfg.main.task == "refinement":
                        pred_x0 = noisy_point_clouds_input[:,:3,:]-predicted_noise_xyz # Use noisy input directly for refinement
                    else:
                        pred_x0 = predict_x0_from_xt_eps(noisy_xyz_centered_for_input, predicted_noise_xyz_centered, timesteps, noise_scheduler)
                    sinkhorn_indices = torch.where(sinkhorn_mask)[0]

                    for i in sinkhorn_indices:
                        sample_active_mask = (xyz_loss_mask[i, 0, :] == 1)
                        if not torch.any(sample_active_mask):
                            continue  # Skip if no active points

                        # Center GT and predicted x0 for this single sample
                        gt_xyz_centered_sample = center_com(xyz[i:i+1], xyz_loss_mask[i:i+1])
                        if cfg.main.task == "refinement":
                            pred_x0_centered_sample = center_com(pred_x0[i:i+1], existence_noisy[i:i+1])
                            
                        else:
                            pred_x0_centered_sample = center_com(pred_x0[i:i+1], xyz_loss_mask[i:i+1])

                        gt_points_active = gt_xyz_centered_sample[0, :, sample_active_mask].permute(1, 0)
                        if cfg.main.task == "refinement":
                            sample_active_mask_noisy = (existence_noisy[i, 0, :] == 1)
                            pred_points_active = pred_x0_centered_sample[0, :, sample_active_mask_noisy].permute(1, 0)
                        else:
                            pred_points_active = pred_x0_centered_sample[0, :, sample_active_mask].permute(1, 0)

                        if gt_points_active.numel() > 0 and pred_points_active.numel() > 0:
                            try:
                                sample_loss = sinkhorn_loss_fn(gt_points_active, pred_points_active)
                                if cfg.main.task == "refinement":
                                    per_sample_final_loss[i] = sample_loss
                                else:
                                    sinkhorn_scale_factor = 10.0 # From original code
                                    per_sample_final_loss[i] = sample_loss * sinkhorn_scale_factor
                            except Exception as e:
                                log.warning(f"Sinkhorn loss failed for sample {i}, batch {batch_idx}. Falling back to MSE. Error: {e}")
                                continue

                primary_loss_term = per_sample_final_loss.mean()

            else:
                raise ValueError(f"Unknown primary loss function: {primary_loss_name}")

            # --- Backward Pass ---
            loss = primary_loss_term
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- Logging and Metric Update ---
            epoch_loss_total_sum += loss.item() * current_batch_size
            if accelerator.is_main_process:
                log_dict = {
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    f"Primary Loss ({primary_loss_name})": f"{primary_loss_term.item():.4f}"
                }
                progress_bar.set_postfix(log_dict)

    total_samples_processed = len(train_loader.dataset)
    avg_epoch_loss_total = epoch_loss_total_sum / total_samples_processed if total_samples_processed > 0 else 0.0
    avg_epoch_uniform_loss = epoch_uniform_loss_sum / total_samples_processed if total_samples_processed > 0 else 0.0
    return avg_epoch_loss_total, avg_epoch_uniform_loss


def validate_conditioned(accelerator: Accelerator, cfg: DictConfig, model: nn.Module, proj_model: nn.Module,
                         val_loader: torch.utils.data.DataLoader, noise_scheduler: Any, epoch: int,
                         size_estimator: Optional[nn.Module] = None):
    """
    Cleaned-up validation function, mirroring the logic of train_conditioned.
    """
    epoch_loss_total_sum = 0.0
    epoch_primary_loss_sum = 0.0
    epoch_uniform_loss_sum = 0.0
    model.eval()
    proj_model.eval()
    if size_estimator:
        size_estimator.eval()

    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", disable=not accelerator.is_main_process, leave=True)

    # --- Loss Function Initialization ---
    primary_loss_name = cfg.training.loss_fn.primary.name
    uniform_loss_fn = UniformityLoss()
    sinkhorn_loss_fn = None
    if primary_loss_name == "Sinkhorn":
        try:
            from geomloss import SamplesLoss
            sinkhorn_loss_fn = SamplesLoss(loss='sinkhorn', p=1, blur=0.04, scaling=0.5, diameter=1, backend='tensorized')
        except ImportError:
            log.error("Geomloss is not installed. Please install it with 'pip install geomloss' to use Sinkhorn loss.")
            raise

    for batch_idx, batch in enumerate(progress_bar):
        if cfg.main.task == "refinement":
            coarse_point_clouds = batch['coarse_point_cloud'].to(accelerator.device)

        clean_point_clouds = batch['point_cloud'].to(accelerator.device)
        images = batch['image'].to(accelerator.device)
        pixel_sizes = batch['pixel_size'].to(accelerator.device)
        CoM = batch['CoM'].to(accelerator.device)
        xyz_loss_mask = batch['point_cloud_mask'].unsqueeze(1).to(accelerator.device)
        current_batch_size = clean_point_clouds.shape[0]

        xyz = clean_point_clouds[:, :3, :]
        existence = clean_point_clouds[:, 3:, :]

        noise_xyz_initial_random = torch.randn_like(xyz)
        if cfg.main.task == "refinement":
            #timestep all 0
            timesteps = torch.zeros(current_batch_size, device=accelerator.device, dtype=torch.long)
            noisy_xyz = coarse_point_clouds[:, :3, :]
            existence_noisy = coarse_point_clouds[:, 3:, :]
        else:
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=accelerator.device).long()
            noisy_xyz = noise_scheduler.add_noise(xyz, noise_xyz_initial_random, timesteps)

        # Center noisy point cloud on its CoM for model input
        if cfg.main.task == "refinement":
            noisy_xyz_centered_for_input = center_com(noisy_xyz, existence_noisy)
        else:
            noisy_xyz_centered_for_input = center_com(noisy_xyz, xyz_loss_mask)

        zeros_for_z = torch.zeros(current_batch_size, 1, device=CoM.device, dtype=CoM.dtype)
        im_CoM = torch.cat([CoM, zeros_for_z], dim=1).unsqueeze(-1)
        noisy_point_cloud_input_for_conditioning = noisy_xyz_centered_for_input + im_CoM
        if cfg.main.task == "refinement":
            noisy_point_clouds_input = torch.cat((noisy_point_cloud_input_for_conditioning, existence_noisy), dim=1)
        else:
            noisy_point_clouds_input = torch.cat((noisy_point_cloud_input_for_conditioning, existence), dim=1)

        conditioned_point_clouds = condition_point_cloud(
            noisy_point_clouds_input, pixel_sizes, images, proj_model, accelerator.device, cfg, mask_feature_map=False
        )
        conditioned_point_clouds[:, :3, :] -= im_CoM

        size_embeddings = None
        if size_estimator is not None:
            size_embeddings = size_estimator(images)
        
        with torch.no_grad():
            predicted_noise_xyz = model(conditioned_point_clouds, timesteps, pixel_sizes, size_embeddings)

        noise_xyz_centered_for_loss_target = center_com(noise_xyz_initial_random, xyz_loss_mask)
        if cfg.main.task == "refinement":
            predicted_noise_xyz_centered = center_com(predicted_noise_xyz, existence_noisy)
        else:
            predicted_noise_xyz_centered = center_com(predicted_noise_xyz, xyz_loss_mask)

        # --- Primary Loss Calculation (Mirrors Training Logic) ---
        primary_loss_term = torch.tensor(0.0, device=accelerator.device)
        pred_x0 = None

        if primary_loss_name == "MSELoss":
            loss_xyz_per_element = F.mse_loss(predicted_noise_xyz_centered, noise_xyz_centered_for_loss_target, reduction='none')
            masked_loss_per_element = loss_xyz_per_element * xyz_loss_mask
            per_sample_active_elements_count = xyz_loss_mask.sum(dim=[1, 2]) * 3
            per_sample_sum_squared_error_xyz = masked_loss_per_element.sum(dim=[1, 2])

            per_sample_primary_loss = torch.where(
                per_sample_active_elements_count > 0,
                per_sample_sum_squared_error_xyz / per_sample_active_elements_count,
                torch.zeros_like(per_sample_active_elements_count, dtype=torch.float32)
            )

            

            # Start with the base MSE loss for the whole batch
            primary_loss_term = per_sample_primary_loss.mean()
            if False:  # Uniformity loss application is optional, controlled by a flag
                # Identify samples with low timesteps to apply uniformity loss
                low_timestep_mask = timesteps < 50
                
                # Check if there are any samples that meet the condition
                if torch.any(low_timestep_mask):
                    # Filter the necessary tensors for the low-timestep samples
                    pred_x0 = predict_x0_from_xt_eps(
                        noisy_xyz_centered_for_input[low_timestep_mask], 
                        predicted_noise_xyz_centered[low_timestep_mask], 
                        timesteps[low_timestep_mask], 
                        noise_scheduler
                    )
                    
                    # Create the 4D input for the uniformity loss function
                    pred_x0_4d = torch.cat((pred_x0, xyz_loss_mask[low_timestep_mask]), dim=1)
                    
                    # Calculate uniformity loss only on the filtered subset
                    uniform_loss = uniform_loss_fn(pred_x0_4d)
                    
                    # Add the scaled uniformity loss to the total primary loss term
                    alpha = 100 # Weight for the uniformity loss
                    primary_loss_term = primary_loss_term + alpha * uniform_loss
                    epoch_uniform_loss_sum += (alpha * uniform_loss.item()) * current_batch_size
                    
                    #if accelerator.is_main_process: # Log periodically
                    #    log.info(f"Batch {batch_idx}: Applied uniformity loss ({uniform_loss.item():.4f}) to {low_timestep_mask.sum()} samples.")

        elif primary_loss_name == "Sinkhorn":
            if cfg.main.task == "refinement":
                per_sample_final_loss = torch.zeros(current_batch_size, device=accelerator.device, dtype=torch.float32)
            per_sample_mse_loss = F.mse_loss(predicted_noise_xyz_centered, noise_xyz_centered_for_loss_target, reduction='none').mean(dim=[1,2])
            per_sample_final_loss = per_sample_mse_loss.clone()
            if cfg.main.task == "refinement":
                sinkhorn_mask = torch.ones(current_batch_size, device=accelerator.device, dtype=torch.bool)
            else:
                sinkhorn_mask = timesteps < 10000
            if torch.any(sinkhorn_mask):
                if cfg.main.task == "refinement":
                    pred_x0 = noisy_point_clouds_input[:,:3,:]-predicted_noise_xyz
                else:
                    pred_x0 = predict_x0_from_xt_eps(noisy_xyz_centered_for_input, predicted_noise_xyz_centered, timesteps, noise_scheduler)
                sinkhorn_indices = torch.where(sinkhorn_mask)[0]
                for i in sinkhorn_indices:
                    sample_active_mask = (xyz_loss_mask[i, 0, :] == 1)
                    if not torch.any(sample_active_mask): continue
                    gt_xyz_centered_sample = center_com(xyz[i:i+1], xyz_loss_mask[i:i+1])
                    if cfg.main.task == "refinement":
                        pred_x0_centered_sample = center_com(pred_x0[i:i+1], existence_noisy[i:i+1])
                    else:
                        pred_x0_centered_sample = center_com(pred_x0[i:i+1], xyz_loss_mask[i:i+1])
                    gt_points_active = gt_xyz_centered_sample[0, :, sample_active_mask].permute(1, 0)
                    pred_points_active = pred_x0_centered_sample[0, :, sample_active_mask].permute(1, 0)
                    if gt_points_active.numel() > 0 and pred_points_active.numel() > 0:
                        
                        sample_loss = sinkhorn_loss_fn(gt_points_active, pred_points_active)
                        
                        if cfg.main.task == "refinement":
                            per_sample_final_loss[i] = sample_loss
                        else:
                            sinkhorn_scale_factor = 10.0
                            per_sample_final_loss[i] = sample_loss * sinkhorn_scale_factor
            primary_loss_term = per_sample_final_loss.mean()

        else:
            raise ValueError(f"Unknown primary loss function for validation: {primary_loss_name}")

        loss = primary_loss_term
        epoch_primary_loss_sum += primary_loss_term.item() * current_batch_size
        epoch_loss_total_sum += loss.item() * current_batch_size # In this case, total is the same as primary
        if accelerator.is_main_process:
            progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}"})

    total_samples_processed = len(val_loader.dataset)
    avg_epoch_loss_total = epoch_loss_total_sum / total_samples_processed if total_samples_processed > 0 else 0.0
    avg_epoch_primary_loss = epoch_primary_loss_sum / total_samples_processed if total_samples_processed > 0 else 0.0
    avg_epoch_uniform_loss = epoch_uniform_loss_sum / total_samples_processed if total_samples_processed > 0 else 0.0
    
    return avg_epoch_loss_total, avg_epoch_primary_loss, avg_epoch_uniform_loss


# --- Modified Main Training Loop Function (train_conditioned_structure_predictor) ---
# This function needs to be updated to handle the new return signature of validate_conditioned
# and to store the new metrics in the history.

def train_conditioned_structure_predictor(accelerator: Accelerator, cfg, model, projection_model, train_loader, val_loader, noise_scheduler):
    projection_model.eval()
    optimizer = get_optimizer(cfg, model)

    steps_per_epoch = len(train_loader)
    total_training_steps = cfg.training.epochs * steps_per_epoch
    if accelerator.is_main_process:
        log.info(f"Total training steps calculated for scheduler: {total_training_steps}")

    # --- LR Scheduler Setup (Unchanged) ---
    if cfg.training.scheduler.name == "ReduceLROnPlateau":
        if accelerator.is_main_process:
            log.info("Using ReduceLROnPlateau learning rate scheduler.")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.scheduler.get('rop_mode', 'min'),
            factor=cfg.training.scheduler.get('rop_factor', 0.1),
            patience=cfg.training.scheduler.get('rop_patience', 10),
            min_lr=cfg.training.scheduler.get('rop_min_lr', 0),
            verbose=cfg.training.scheduler.get('rop_verbose', True)
        )
    else:
        lr_scheduler = get_scheduler(cfg, optimizer, total_training_steps)

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # --- Size Estimator Setup (Unchanged) ---
    size_estimator = None
    if cfg.training.get("use_size_model", False):
        size_estimator_weights_path = cfg.model.get("size_estimator_weights_path")
        if accelerator.is_main_process:
            log.info(f"Initializing size estimator from: {size_estimator_weights_path}")
        size_estimator = ResNet18AdaptivePoolFeatureExtractor(
            input_nc=1,
            output_feature_dim=cfg.model.pc_model.get("size_embed_dim", 64),
            pretrained_estimator_path=size_estimator_weights_path,
        )
        size_estimator.eval()
        size_estimator = accelerator.prepare(size_estimator)

    # --- REMOVED: grayscale_proj_model initialization ---
    # This model is no longer used in the cleaned-up train/validate functions.

    # --- SIMPLIFIED: History Dictionary ---
    # Only track the metrics that are now returned.
    best_val_loss = float('inf')
    history = {
        "train_loss_total": [],
        "train_uniform_loss": [],
        "val_loss_total": [],
        "val_primary_loss": [],
        "val_uniform_loss": [],
        "learning_rate": [],
        "epochs": [],
    }

    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "training_metrics.json"

    if cfg.debug.get("visualize_scheduler_behavior", True) and accelerator.is_main_process:
        try:
            example_batch_for_debug = next(iter(train_loader))
            gt_pc_for_debug = example_batch_for_debug['point_cloud'][0]
            debug_visualize_scheduler(gt_pc_for_debug, noise_scheduler, accelerator.device, cfg)
        except Exception as e:
            log.error(f"Error during scheduler debug visualization: {e}", exc_info=True)

    for epoch in range(cfg.training.epochs):
        if accelerator.is_main_process:
            log.info(f"\n--- Epoch {epoch+1}/{cfg.training.epochs} ---")
            history["epochs"].append(epoch)
            history["learning_rate"].append(optimizer.param_groups[0]['lr'])

        # --- UPDATED: Training Call ---
        # The function no longer takes grayscale_proj_model and returns only one value.
        train_loss, train_uniform_loss = train_conditioned(
            accelerator, cfg, model, projection_model,
            train_loader, optimizer,
            lr_scheduler if cfg.training.scheduler.name != "ReduceLROnPlateau" else None,
            noise_scheduler, epoch, size_estimator
        )
        if accelerator.is_main_process:
            history["train_loss_total"].append(train_loss)
            history["train_uniform_loss"].append(train_uniform_loss)

        # --- UPDATED: Validation ---
        if (epoch + 1) % cfg.training.validation_freq == 0:
            # The function no longer takes grayscale_proj_model and returns two values.
            val_loss_total, val_primary_loss, val_uniform_loss = validate_conditioned(
                accelerator, cfg, model, projection_model,
                val_loader, noise_scheduler, epoch, size_estimator
            )
            if accelerator.is_main_process:
                # Simplified logging for the new return values.
                log.info(f"Validation Epoch {epoch+1} - Total Loss: {val_loss_total:.4f}, "
                         f"Primary Loss ({cfg.training.loss_fn.primary.name}): {val_primary_loss:.4f}, "
                         f"Uniform Loss: {val_uniform_loss:.4f}")

                # Simplified history update.
                history["val_loss_total"].append(val_loss_total)
                history["val_primary_loss"].append(val_primary_loss)
                history["val_uniform_loss"].append(val_uniform_loss)

                is_best = val_loss_total < best_val_loss
                if is_best:
                    best_val_loss = val_loss_total
                    log.info(f"New best validation total loss: {best_val_loss:.4f}")

                # Checkpointing logic remains the same.
                if (epoch + 1) % cfg.training.checkpoint_freq == 0 or is_best:
                    checkpoint_name = f"checkpoint_epoch_{epoch+1}.pt" if not is_best else "best_model.pt"
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'val_loss': val_loss_total,
                        'cfg': OmegaConf.to_container(cfg, resolve=True)
                    }, is_best, filename=checkpoint_name, best_filename="best_model.pt")

                save_metrics(history, filename=str(metrics_path))

                # Visualization call remains the same, as it didn't use the removed model.
                if cfg.visualization.enabled and (epoch + 1) % cfg.visualization.freq == 0:
                    log.info(f"Generating visualization for epoch {epoch+1}...")
                    try:
                        vis_batch = next(iter(val_loader))
                        if cfg.main.task == "refinement":
                            visualize_refinement(accelerator.unwrap_model(model),
                                                 accelerator.unwrap_model(projection_model),
                                                 vis_batch,
                                                 epoch + 1,
                                                 accelerator, 
                                                 cfg
                            )
                        else:
                            visualize_conditioned(
                                accelerator.unwrap_model(model),
                                accelerator.unwrap_model(projection_model),
                                noise_scheduler,
                                vis_batch,
                                epoch + 1,
                                accelerator.device,
                                cfg,
                                size_estimator=accelerator.unwrap_model(size_estimator) if size_estimator else None,
                            )
                    except Exception as e:
                        log.error(f"Error during visualization: {e}", exc_info=True)

            # --- LR Scheduler Step (Unchanged) ---
            if lr_scheduler is not None and cfg.training.scheduler.name == "ReduceLROnPlateau":
                lr_scheduler.step(val_loss_total)

        # Early stopping and other scheduler steps remain the same.
        if lr_scheduler is not None and cfg.training.scheduler.name not in ["ReduceLROnPlateau"]:
             lr_scheduler.step()

        # [ ... The rest of the function (early stopping, etc.) is unchanged ... ]
        # The early stopping logic should work as long as the monitored metric (e.g., 'val_loss_total')
        # is still being recorded in the history dictionary, which it is.

    if accelerator.is_main_process:
        log.info("Training finished.")

def train_unconditioned(accelerator: Accelerator, cfg, model, train_loader, optimizer, lr_scheduler, noise_scheduler, epoch):
    epoch_loss = 0.0
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=not accelerator.is_main_process, leave=True)
    for batch in progress_bar:
        clean_point_clouds = batch['point_cloud'] # Already on correct device
        current_batch_size = clean_point_clouds.shape[0]
        noise = torch.randn_like(clean_point_clouds)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=accelerator.device).long()
        noisy_point_clouds = noise_scheduler.add_noise(clean_point_clouds, noise, timesteps)

        predicted_noise = model(noisy_point_clouds, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        # loss.backward() # Replace with accelerator.backward
        accelerator.backward(loss)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        avg_loss = accelerator.gather(loss.repeat(cfg.data.batch_size)).mean()
        epoch_loss += avg_loss.item()
        if accelerator.is_main_process:
            progress_bar.set_postfix(Loss=f"{avg_loss.item():.4f}")

    return epoch_loss / len(train_loader)

def validate_unconditioned(accelerator: Accelerator, cfg, model, val_loader, noise_scheduler, epoch):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", disable=not accelerator.is_main_process, leave=True)
        for batch in progress_bar:
            clean_point_clouds = batch['point_cloud'] # Already on correct device
            current_batch_size = clean_point_clouds.shape[0]
            noise = torch.randn_like(clean_point_clouds)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (current_batch_size,), device=accelerator.device).long()
            noisy_point_clouds = noise_scheduler.add_noise(clean_point_clouds, noise, timesteps)

            predicted_noise = model(noisy_point_clouds, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            avg_loss = accelerator.gather(loss.repeat(cfg.data.batch_size)).mean()
            epoch_loss += avg_loss.item()
            if accelerator.is_main_process:
                progress_bar.set_postfix(Loss=f"{avg_loss.item():.4f}")

    return epoch_loss / len(val_loader)

def train_unconditioned_structure_predictor(accelerator: Accelerator, cfg, model, train_loader, val_loader, noise_scheduler):
    optimizer = get_optimizer(cfg, model)
    steps_per_epoch = len(train_loader)
    total_training_steps = cfg.training.epochs * steps_per_epoch
    if accelerator.is_main_process:
        print(f"Total training steps calculated for scheduler: {total_training_steps}")
    lr_scheduler = get_scheduler(cfg, optimizer, total_training_steps)

    # Prepare optimizer and scheduler
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    best_val_loss = float('inf')
    history = {
        "train_loss": [], "val_loss": [], "learning_rate": [], "epochs": []
    }
    metrics_path = Path(cfg.paths.metrics_dir) / "training_metrics.json"

    for epoch in range(cfg.training.epochs):
        train_loss = train_unconditioned(accelerator, cfg, model, train_loader, optimizer, lr_scheduler, noise_scheduler, epoch)
        val_loss = validate_unconditioned(accelerator, cfg, model, val_loader, noise_scheduler, epoch)

        if accelerator.is_main_process:
            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']
            log.info(f"Epoch {epoch}: Train Loss: {train_loss}, Validation Loss: {val_loss}, Learning Rate: {current_lr}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)
            history["epochs"].append(epoch)
            save_metrics(history, filename=metrics_path)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                unwrapped_model = accelerator.unwrap_model(model)
                # visualize_unconditioned(unwrapped_model, noise_scheduler, epoch, accelerator.device, cfg) # Ensure visualization works correctly
                save_checkpoint(unwrapped_model, is_best=True)
                log.info(f"Saved new best model with validation loss: {best_val_loss}")

            if epoch % cfg.training.checkpoint_freq == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                # visualize_unconditioned(unwrapped_model, noise_scheduler, epoch, accelerator.device, cfg) # Ensure visualization works correctly
                save_checkpoint(unwrapped_model, is_best=False, filename=f"checkpoint_epoch_{epoch}.pt")
                log.info(f"Saved checkpoint for epoch {epoch}")


def visualize_projection_and_sampling0(point_clouds,
                                      pixel_sizes,
                                      images,
                                      SUPER_CELL_SIZE=5.0,
                                      IMAGE_SIZE=128,
                                      batch_idx=0,
                                      device='cpu'):
    """
    1) projects 3D points → pixel coords
    2) builds a [-1,1] grid for grid_sample
    3) samples the image via grid_sample
    4) overlays both raw points and sampled intensities on the image
    """
    # Move to device / CPU
    pc    = point_clouds[batch_idx].to(device)      # [3,N]
    px_sz = pixel_sizes[batch_idx].to(device)       # scalar
    img   = images[batch_idx:batch_idx+1].to(device)  # [1,1,H,W]

    # 1) normalized coords → nm → pixel
    pc_xy = pc[:2] * (SUPER_CELL_SIZE / 2.0)           # [2,N] in nm
    pc_pix = pc_xy / px_sz + (IMAGE_SIZE / 2.0)        # [2,N] in px
    xs, ys = pc_pix[0], pc_pix[1]                      # each [N]

    # 2) build grid in [-1,1] for grid_sample
    # note: grid_sample expects grid in shape [B, H_out, W_out, 2]
    #        or for “pointwise” sampling [B, 1, N, 2]
    # here we do [1,1,N,2]:
    grid_x = xs / (IMAGE_SIZE - 1) * 2.0 - 1.0         # [N]
    grid_y = ys / (IMAGE_SIZE - 1) * 2.0 - 1.0         # [N]
    grid   = torch.stack([grid_x, grid_y], dim=-1)    # [N,2]
    grid   = grid.unsqueeze(0).unsqueeze(1)           # [1,1,N,2]

    # 3) sample the grayscale image
    sampled = F.grid_sample(img, grid,
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=True)     # [1,1,1,N]
    sampled = sampled.view(-1).cpu().numpy()          # [N]

    # 4) plot
    base_img = img[0,0].cpu().numpy()                 # [H,W]
    depth    = pc[2].cpu().numpy()                    # [N] for coloring

    plt.figure(figsize=(5,5))
    plt.imshow(base_img, cmap='gray', origin='upper')
    # raw projected points (colored by Z)
    plt.scatter(xs.cpu(), ys.cpu(), s=5, c=depth, cmap='viridis', alpha=0.5, label='projected points')
    # overlay the actual sampled intensity as an outline
    plt.title(f'Batch {batch_idx} projection & sampling check')
    plt.xlabel('pixel column (x)')
    plt.ylabel('pixel row (y)')
    plt.legend(loc='upper right')
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.colorbar(label='depth / sampled intensity')
    plt.show()

# Add this helper function at the top of the file after imports
def center_z_axis(tensor_xyz, mask=None):
    """
    Centers the Z-axis (channel 2) of a point cloud tensor.
    
    Args:
        tensor_xyz: Tensor of shape [B, 3, N] or [3, N]
        mask: Optional mask of shape [B, 1, N] or [1, N] indicating valid points
    
    Returns:
        Tensor with Z-axis centered
    """
    if tensor_xyz.shape[-2] != 3:
        raise ValueError(f"Expected 3 channels for XYZ, got {tensor_xyz.shape[-2]}")
    
    z_channel = tensor_xyz[..., 2:3, :] # [..., 1, N] - Z channel
    
    if mask is not None:
        # Only consider valid points for centering
        if tensor_xyz.ndim == 3:  # Batched [B, 3, N]
            valid_z = z_channel * mask  # [B, 1, N]
            valid_count = mask.sum(dim=-1, keepdim=True)  # [B, 1, 1]
            z_mean = valid_z.sum(dim=-1, keepdim=True) / (valid_count + 1e-8)  # [B, 1, 1]
        else:  # Single sample [3, N]
            valid_z = z_channel * mask  # [1, N]
            valid_count = mask.sum(dim=-1, keepdim=True)  # [1, 1]
            z_mean = valid_z.sum(dim=-1, keepdim=True) / (valid_count + 1e-8)  # [1, 1]
    else:
        # Use all points for centering
        z_mean = z_channel.mean(dim=-1, keepdim=True)  # [..., 1, 1]
    
    # Center the Z-axis
    tensor_xyz_centered = tensor_xyz.clone()
    tensor_xyz_centered[..., 2:3, :] = z_channel - z_mean
    
    return tensor_xyz_centered

class UniformityLoss(nn.Module):
    """
    Calculates a uniformity loss for batches of point clouds.
    The loss encourages a uniform distribution by penalizing high variance
    in the distances to k-nearest neighbors.

    Input shape: (B, 4, N), where the 4th dim is an existence mask.
    """
    def __init__(self, ks: list[int] = [2, 4, 8, 16], eps: float = 1e-9):
        """
        Args:
            ks (list[int]): A list of k-values for k-nearest neighbors.
            eps (float): A small epsilon for numerical stability.
        """
        super().__init__()
        self.ks = sorted(ks)
        self.ksum = sum(ks)
        self.eps = eps

    def _batched_pairwise_distances_squared(self, x: torch.Tensor) -> torch.Tensor:
        """Computes batched pairwise squared Euclidean distances."""
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        dot_product = torch.bmm(x, x.transpose(-1, -2))
        dists_sq = x_sq - 2 * dot_product + x_sq.transpose(-1, -2)
        return torch.clamp(dists_sq, min=0.0)

    def forward(self, pc_4d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc_4d (torch.Tensor): Batch of point clouds, shape (B, 4, N).
                                  The 4th dim is a mask (> 0.5 for real points).
        Returns:
            torch.Tensor: A scalar tensor for the mean uniformity loss.
        """
        B, _, N = pc_4d.shape
        device = pc_4d.device

        xyz = pc_4d[:, :3, :].transpose(1, 2)  # (B, N, 3)
        mask = pc_4d[:, 3, :] > 0.5            # (B, N)
        num_real_points = mask.sum(dim=1).float()

        dists_sq = self._batched_pairwise_distances_squared(xyz)
        
        valid_pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        dists_sq.masked_fill_(~valid_pair_mask, float('inf'))
        dists_sq.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

        batch_loss_terms = torch.zeros(B, device=device)

        for k in self.ks:
            valid_indices = torch.where(num_real_points > k)[0]
            if len(valid_indices) == 0:
                continue
            
            # Filter batch items that have enough points for this k
            dists_sq_filt = dists_sq[valid_indices]
            mask_filt = mask[valid_indices]
            
            knn_dists_sq, _ = torch.topk(dists_sq_filt, k, dim=2, largest=False)
            
            # Select distances corresponding to real points and flatten
            valid_knn_dists_sq = knn_dists_sq[mask_filt]
            valid_knn_dists_sq = valid_knn_dists_sq[valid_knn_dists_sq < float('inf')]

            if valid_knn_dists_sq.numel() > 1:
                D_k = torch.sqrt(valid_knn_dists_sq + self.eps)
                variance = torch.var(D_k)
                weight = (1 - k / self.ksum)
                batch_loss_terms[valid_indices] += weight * variance
        
        # Average the loss only over items that had real points
        valid_items_mask = num_real_points > 0
        if valid_items_mask.sum() > 0:
            return batch_loss_terms[valid_items_mask].mean()
        else:
            return torch.tensor(0.0, device=device)
        