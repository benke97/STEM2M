import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
import datetime
import numpy as np
import torch.cuda.amp as amp
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

# Removed: HydraConfig import is no longer needed here if relying solely on CWD

log = logging.getLogger(__name__)

# Removed: get_hydra_output_dir() function

def save_checkpoint(state, is_best, filename='checkpoint.pt', best_filename='best_model.pt'):
    """Saves checkpoint relative to the CURRENT working directory (managed by Hydra)."""
    try:
        # --- Construct paths relative to CWD ---
        # Make sure checkpoint path has checkpoints/ directory prefix
        if not filename.startswith('checkpoints/'):
            filename = f"checkpoints/{filename}"
        if not best_filename.startswith('checkpoints/'):
            best_filename = f"checkpoints/{best_filename}"
            
        # Create checkpoints directory if it doesn't exist
        Path('checkpoints').mkdir(exist_ok=True)

        save_path = Path(filename)
        best_path = Path(best_filename)
        # ---
        
        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            # If it's the best model, save directly to best_filename only
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, best_path)
            log.info(f"Saved best model checkpoint to {best_path}")
        else:
            # Only save regular checkpoint if it's not the best
            torch.save(state, save_path)
            log.debug(f"Saved checkpoint to {save_path}")

    except Exception as e:
        log.error(f"Failed to save checkpoint(s): {e}", exc_info=True)


def visualize_predictions(model, dataloader, device, cfg, epoch, use_amp):
    """Generates and saves a plot relative to the CURRENT working directory (managed by Hydra)."""
    cfg_viz = cfg.visualization
    # Using CWD name as a fallback identifier if needed
    run_identifier = cfg.main.get("run_name") or Path(os.getcwd()).name or "unnamed_run"

    if not cfg_viz.get("enabled", False): return

    #log.info(f"Generating visualization for epoch {epoch} (Run ID: {run_identifier})...")

    try:
        # --- Construct path relative to CWD ---
        relative_viz_dir_name = cfg_viz.output_dir # e.g., "visualization"
        viz_dir = Path(relative_viz_dir_name) # Relative to CWD
        viz_dir.mkdir(parents=True, exist_ok=True)
        #log.info(f"Attempting to save visualization in: {viz_dir.resolve()}") # Log absolute path
        # ---
    except Exception as e:
        log.error(f"Could not create target visualization directory: {e}. Skipping visualization.", exc_info=True)
        return

    model.eval()
    try:
        sample_batch = next(iter(dataloader))
        images = sample_batch['image'].to(device)
        thickness_maps = sample_batch['thickness_map'] # Keep on CPU for plotting initially

        with torch.no_grad():
            with amp.autocast(enabled=use_amp):
                pred_thickness, _ = model(images)

        images = images.cpu().numpy()
        thickness_maps = thickness_maps.cpu().numpy() # Ensure it's numpy
        pred_thickness = pred_thickness.float().cpu().numpy()

        num_samples = min(cfg_viz.num_samples, images.shape[0])
        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
        if num_samples == 1:
           axes = axes.reshape(3, 1)

        for i in range(num_samples):
            # Input image
            ax_img = axes[0, i]
            ax_img.imshow(images[i, 0, :, :], cmap='gray')
            ax_img.set_title(f"Input {i}")
            ax_img.axis('off')

            # Ground truth thickness map
            ax_gt = axes[1, i]
            im_gt = ax_gt.imshow(thickness_maps[i].squeeze(), cmap='viridis')
            ax_gt.set_title(f"GT {i}")
            ax_gt.axis('off')

            # Predicted thickness map
            ax_pred = axes[2, i]
            pred_viz = torch.sigmoid(torch.from_numpy(pred_thickness[i].squeeze())).numpy()
            im_pred = ax_pred.imshow(pred_viz, cmap='viridis')
            ax_pred.set_title(f"Pred {i}")
            ax_pred.axis('off')

        # --- Construct final save path relative to CWD ---
        save_filename = f"{run_identifier}_epoch_{epoch}_results.png"
        save_path = viz_dir / save_filename # viz_dir is already relative to CWD
        # ---
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        log.info(f"Visualization saved to {save_path} (Absolute: {save_path.resolve()})")

    except Exception as e:
        log.error(f"Failed to generate visualization: {e}", exc_info=True)
    finally:
        model.train()


def save_metrics(metrics_data, filename="training_metrics.json"):
    """Saves training metrics relative to the CURRENT working directory (managed by Hydra)."""
    try:
        # --- Construct path relative to CWD ---
        # filename is expected to be a relative path like
        # "training_metrics/training_metrics.json" as constructed in train.py
        save_path = Path(filename)
        # ---

        # Ensure parent directory exists (e.g., ./training_metrics/)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        #log.info(f"Attempting to save metrics to: {save_path.resolve()}") # Log absolute path

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if "metadata" not in metrics_data: metrics_data["metadata"] = {}
        metrics_data["metadata"]["saved_at"] = timestamp

        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        log.info(f"Training metrics saved to {save_path} (Absolute: {save_path.resolve()})")

    except Exception as e:
        log.error(f"Could not save metrics to {filename}: {e}", exc_info=True)

# --- Keep the rest of helpers.py (prepare_for_lpips, loss_needs_3channels, etc.) ---
def prepare_for_lpips(tensor):
    return torch.cat((tensor, tensor, tensor), 1)

def loss_needs_3channels(criterion):
    if  isinstance(criterion, LearnedPerceptualImagePatchSimilarity):
        return True
    else:
        return False

def get_loss_function(loss_cfg):
    """Gets a loss function instance based on configuration."""
    loss_name = loss_cfg.get("name", None)
    if loss_name is None:
        raise ValueError("Loss function configuration must include a 'name' key.")

    log.info(f"Initializing loss function: {loss_name}")

    if loss_name == "MSELoss":
        return torch.nn.MSELoss()
    elif loss_name == "L1Loss":
        return torch.nn.L1Loss()
    elif loss_name == "BCEWithLogitsLoss":
         return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "LPIPS":
        # Ensure torchmetrics is installed
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except ImportError:
             raise ImportError("TorchMetrics LPIPS is required but not found/installed. `pip install torchmetrics[image]`")

        net_type = loss_cfg.get("net_type", "squeeze") # Changed default as per docs v1.0+
        reduction = loss_cfg.get("reduction", "mean")
        normalize_input = loss_cfg.get("normalize", True) # Default to True

        log.info(f"  TorchMetrics LPIPS parameters: net_type={net_type}, reduction={reduction}, normalize={normalize_input}")

        return LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            reduction=reduction,
            normalize=normalize_input
        )

    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")

def get_optimizer(cfg, model):
    #adam, admaw, or sgd
    optimizer_name = cfg.training.optimizer.name
    if optimizer_name == "Adam" or optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.training.optimizer.weight_decay)
    elif optimizer_name == "AdamW" or optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.training.optimizer.weight_decay)
    elif optimizer_name == "SGD" or optimizer_name == "sgd":
        #not implemented yet
        raise NotImplementedError("SGD optimizer is not implemented yet.")
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")


def get_scheduler(cfg, optimizer, total_training_steps):
    """Gets a learning rate scheduler based on configuration."""
    scheduler_config = cfg.training.scheduler
    scheduler_name = scheduler_config.name
    warmup_steps = scheduler_config.get('warmup_steps', 0) # Get warmup steps safely

    main_scheduler = None
    # Calculate the duration for the main scheduler
    main_scheduler_duration = total_training_steps - warmup_steps
    if main_scheduler_duration <= 0 and warmup_steps > 0:
        print(f"Warning: Warmup steps ({warmup_steps}) >= total steps ({total_training_steps}). "
              f"Main scheduler will not run.")


    # Create the main scheduler first
    if scheduler_name == "CosineAnnealingLR":
        print(f"Configuring CosineAnnealingLR with T_max = {main_scheduler_duration} steps.")
        main_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=main_scheduler_duration, # Use adjusted duration
                                         eta_min=scheduler_config.eta_min) # Ensure eta_min is in config

    elif scheduler_name == "LinearLR":
        print(f"Configuring LinearLR with total_iters = {main_scheduler_duration} steps.")
        main_scheduler = LinearLR(optimizer,
                                  start_factor=1.0,
                                  end_factor=scheduler_config.get('linear_end_factor', 0.0), # Make end factor configurable
                                  total_iters=main_scheduler_duration) # Use adjusted duration

    elif scheduler_name == "constant":
        print("Configuring Constant LR.")
        # Constant doesn't really have a duration dependency like others
        main_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    # Handle warmup
    if warmup_steps > 0:
        # Add checks for warmup_start_factor
        if 'warmup_start_factor' not in scheduler_config:
             raise ValueError("Warmup specified, but 'warmup_start_factor' is missing in scheduler config.")

        print(f"Adding Linear Warmup for {warmup_steps} steps.")
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=scheduler_config.warmup_start_factor,
                                    total_iters=warmup_steps)
        # Chain them
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_scheduler, main_scheduler],
                                 milestones=[warmup_steps]) # Switch *after* warmup_steps
        print(f"Using scheduler: Linear Warmup -> {type(main_scheduler).__name__}")
    else:
        # No warmup, just return the main scheduler
        scheduler = main_scheduler
        print(f"Using scheduler: {type(main_scheduler).__name__} (no warmup)")

    return scheduler

    

def create_custom_beta_schedule(
    num_train_timesteps=1000,
    alpha_bar_power=2.5,         # Higher value -> Slower initial noise increase. Try 2.5, 3.0, 3.5, 4.0
    target_alpha_bar_T=0.0001,   # Target cumulative alpha at T (ensures high noise). 0.001 is less noise, 0.00001 is more.
    beta_clip_max=0.999          # Clip betas to prevent numerical issues near T
    ):
    """
    Generates a custom numpy array of betas for DDPMScheduler.

    This schedule starts adding noise VERY slowly, controlled by alpha_bar_power,
    and aims for a high noise level at the final timestep T, controlled by
    target_alpha_bar_T.

    Args:
        num_train_timesteps (int): T, the total number of diffusion steps.
        alpha_bar_power (float): Controls the initial flatness of the alpha_bar curve.
                                 Higher power means slower initial noise addition.
        target_alpha_bar_T (float): The desired value of alpha_bar at t=T.
                                   Smaller values mean higher noise at the end.
        beta_clip_max (float): Maximum allowed value for individual beta_t.

    Returns:
        np.ndarray: A numpy array of beta values of shape (num_train_timesteps,).
    """
    T = num_train_timesteps

    # Create timesteps array from 1 to T
    timesteps = np.linspace(1, T, T, dtype=np.float64)

    # Calculate alpha_bar based on the power law interpolation
    # Ensure alpha_bar_t doesn't drop below target_alpha_bar_T prematurely if power is very high
    alpha_bar_t = 1.0 - (1.0 - target_alpha_bar_T) * np.power(timesteps / T, alpha_bar_power)
    alpha_bar_t = np.maximum(alpha_bar_t, target_alpha_bar_T) # Prevent going below target

    # Add alpha_bar_0 = 1.0 at the beginning for calculation purposes
    alpha_bar_full = np.concatenate([np.array([1.0]), alpha_bar_t])

    # Calculate betas: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    # Ensure division is safe by clipping alpha_bar slightly away from 0 if necessary
    alpha_bar_prev = alpha_bar_full[:-1]
    alpha_bar_curr = alpha_bar_full[1:]

    # Avoid division by zero or negative values if alpha_bar becomes extremely small
    epsilon = 1e-12
    betas = 1.0 - alpha_bar_curr / (alpha_bar_prev + epsilon)

    # Clip betas to prevent numerical instability (especially near T)
    betas = np.clip(betas, 0.0, beta_clip_max)

    #betas[-1] = beta_clip_max
    print(betas[-5:])
    print(f"\n--- Custom Beta Schedule Created ---")
    print(f"  alpha_bar_power: {alpha_bar_power}")
    print(f"  target_alpha_bar_T: {target_alpha_bar_T}")
    print(f"  Calculated betas range: [{betas.min():.2e}, {betas.max():.4f}]")

    return betas


def sigmoid_beta_schedule(timesteps: int, start_beta: float = 1e-4, end_beta: float = 0.02, start_x: float = -6.0, end_x: float = 6.0) -> torch.Tensor:
    """
    Generates a sigmoid beta schedule.

    Args:
        timesteps: The total number of diffusion steps.
        start_beta: The minimum beta value in the schedule.
        end_beta: The maximum beta value in the schedule.
        start_x: The starting value for the sigmoid function's input (controls steepness).
        end_x: The ending value for the sigmoid function's input (controls steepness).

    Returns:
        A torch.Tensor of beta values for each timestep.
    """
    # Generate `x` values linearly spaced over the sigmoid's active range
    # e.g., from -6 to 6, which covers most of the sigmoid's curve.
    x = np.linspace(start_x, end_x, timesteps)
    
    # Apply the sigmoid function, then scale to the desired beta range [start_beta, end_beta]
    betas = start_beta + (end_beta - start_beta) / (1 + np.exp(-x))
    
    # Convert to a PyTorch tensor
    return torch.from_numpy(betas).float()

def create_custom_beta_schedule_exp(T=1000, beta_min=1e-6, beta_max=1.0, gamma=4.0):
    """
    Generates a smooth exponential beta schedule from beta_min to beta_max.

    Args:
        T (int): Number of diffusion timesteps
        beta_min (float): Minimum beta value (start of schedule)
        beta_max (float): Maximum beta value (end of schedule)
        gamma (float): Controls how backloaded the growth is

    Returns:
        np.ndarray: Array of beta values for each timestep
    """
    t = np.linspace(0, 1, T)
    ramp = (np.exp(gamma * t) - 1) / (np.exp(gamma) - 1)
    betas = beta_min + (beta_max - beta_min) * ramp
    return betas