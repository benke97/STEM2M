\
import torch
from tqdm import tqdm
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from typing import Dict, Any, Optional, Tuple, Union, Callable
from omegaconf import DictConfig, OmegaConf

# Import custom beta schedule functions
from mar.utils.helpers import create_custom_beta_schedule, create_custom_beta_schedule_exp

# Define a mapping from names to scheduler classes
SCHEDULER_MAP = {
    "ddpm": DDPMScheduler,
    "ddim": DDIMScheduler,
    "dpm++": DPMSolverMultistepScheduler,  # Corresponds to DPM-Solver++
}

def sample_point_cloud(
    pc_model: torch.nn.Module,
    scheduler_name: str,
    initial_noise_shape: Tuple[int, int, int],  # (batch_size, num_coord_dims, num_points)
    num_inference_steps: int,
    device: torch.device,
    cfg: DictConfig,  # Global config, used for cfg.training.diffusion scheduler parameters
    model_forward_kwargs: Dict[str, Any],  # Passed to pc_model.forward()
    input_conditioning_features: Optional[torch.Tensor] = None,  # Concatenated to sample
    eta: float = 0.0,  # For DDIM scheduler
    generator: Optional[torch.Generator] = None,
    disable_tqdm: bool = False,
    step_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,  # Called each step
    conditioning_callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None  # Generate conditioning per step
) -> torch.Tensor:
    """
    Generates point clouds using a specified diffusion model and scheduler.

    Args:
        pc_model: The point cloud diffusion model.
        scheduler_name: Name of the scheduler to use (e.g., "ddpm", "ddim", "dpm++").
        initial_noise_shape: Shape of the initial noise (batch_size, num_coord_dims, num_points).
        num_inference_steps: Number of denoising steps.
        device: PyTorch device.
        cfg: OmegaConf DictConfig object. model.diffusion is used for scheduler params.
        model_forward_kwargs: Dictionary of keyword arguments for the pc_model's forward method
                              (e.g., {'cond_features': ..., 'pixel_sizes': ...}).
        input_conditioning_features: Optional tensor of features to concatenate to the noisy sample
                                     at each step. Shape (B, num_input_cond_dims, N).
        eta: Eta parameter for DDIM scheduler.
        generator: Optional torch.Generator for reproducible noise.
        disable_tqdm: If True, disables the tqdm progress bar.
        step_callback: Optional callback function called after each denoising step.
                      Signature: (step_idx: int, current_sample: Tensor, current_timestep: Tensor) -> None
        conditioning_callback: Optional callback to generate conditioning features dynamically per step.
                              Signature: (current_sample: Tensor) -> Tensor

    Returns:
        A tensor representing the sampled point clouds.
    """
    if scheduler_name.lower() not in SCHEDULER_MAP:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. Available: {list(SCHEDULER_MAP.keys())}"
        )

    SchedulerClass = SCHEDULER_MAP[scheduler_name.lower()]

    # Scheduler parameters are sourced from cfg.training.diffusion,
    # reflecting the settings used during model training.
    s_cfg_dict = OmegaConf.to_container(cfg.training.diffusion, resolve=True)
    
    # Filter out keys that are not part of DDPMScheduler or other schedulers' init
    # This is a basic filter; specific schedulers might have different needs.
    valid_scheduler_keys = [
        "num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
        "variance_type", "clip_sample", "prediction_type", "trained_betas",
        "solver_order", "thresholding", "dynamic_thresholding_ratio",
        "sample_max_value", "algorithm_type", "solver_type", "lower_order_final",
        "set_alpha_to_one", "steps_offset", "rescale_betas_zero_snr"
    ]
    scheduler_init_kwargs = {
        k: v for k, v in s_cfg_dict.items() if k in valid_scheduler_keys
    }
    
    # Ensure num_train_timesteps is present, as it's fundamental for most
    if 'num_train_timesteps' not in scheduler_init_kwargs:
        scheduler_init_kwargs['num_train_timesteps'] = 1000 # A common default

    # Ensure prediction_type is set if model expects it (common for DPM, DDIM)
    # Default to 'epsilon' if not specified, as it's a common setting.
    if scheduler_name.lower() in ["ddim", "dpm++"] and 'prediction_type' not in scheduler_init_kwargs:
        scheduler_init_kwargs['prediction_type'] = 'epsilon'


    # Prepare scheduler initialization arguments
    scheduler_init_kwargs = {
        "num_train_timesteps": cfg.training.diffusion.num_timesteps,
        "prediction_type": cfg.training.diffusion.get('prediction_type', 'epsilon'), # Common for many schedulers
        "clip_sample": cfg.training.diffusion.get('clip_sample', False), # Common for many schedulers
    }

    beta_schedule_type = cfg.training.diffusion.beta_schedule.lower()

    if scheduler_name.lower() == "ddpm":
        if beta_schedule_type == "custom":
            custom_betas = create_custom_beta_schedule(
                num_train_timesteps=cfg.training.diffusion.num_timesteps,
                alpha_bar_power=cfg.training.diffusion.alpha_bar_power,
                target_alpha_bar_T=cfg.training.diffusion.target_alpha_bar_T,
                beta_clip_max=cfg.training.diffusion.beta_clip_max
            )
            scheduler_init_kwargs["trained_betas"] = custom_betas
            # When trained_betas is provided, beta_start, beta_end, beta_schedule are ignored by DDPMScheduler
        elif beta_schedule_type == "exp_growth":
            custom_betas = create_custom_beta_schedule_exp(
                T=cfg.training.diffusion.num_timesteps, # Assuming T is num_train_timesteps
                # Add other params for create_custom_beta_schedule_exp if they exist in your config
                # e.g., beta_min, beta_max, gamma, if defined under cfg.training.diffusion
            )
            scheduler_init_kwargs["trained_betas"] = custom_betas
        else:
            # Standard beta schedules
            scheduler_init_kwargs.update({
                "beta_start": cfg.training.diffusion.beta_start,
                "beta_end": cfg.training.diffusion.beta_end,
                "beta_schedule": cfg.training.diffusion.beta_schedule,
            })
    elif scheduler_name.lower() == "ddim":
        # DDIM specific adjustments, if any, can be added here
        pass

    scheduler = SchedulerClass(**scheduler_init_kwargs)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Prepare initial sample (noise)
    sample = torch.randn(initial_noise_shape, generator=generator, device=device)
      # Scale noise by init_sigma if scheduler expects it (common for DDPM)
    # For other schedulers, this might not be needed or handled differently.
    if hasattr(scheduler, 'init_noise_sigma') and isinstance(scheduler, DDPMScheduler):
         sample = sample * scheduler.init_noise_sigma

    pc_model.eval()  # Ensure model is in evaluation mode

    for step_idx, t in enumerate(tqdm(timesteps, disable=disable_tqdm, desc=f"Sampling with {scheduler_name}")):
        # Generate dynamic conditioning if callback provided
        current_conditioning_features = input_conditioning_features
        if conditioning_callback is not None:
            current_conditioning_features = conditioning_callback(sample)
        
        # Prepare model input (the noisy point cloud + optional concatenated conditioning)
        current_sample_input = sample
        if current_conditioning_features is not None:
            if current_conditioning_features.shape[0] != sample.shape[0]:
                # This might require broadcasting or repeating if batch sizes differ.
                # Assuming prepared correctly by the caller for now.
                # E.g. current_conditioning_features = current_conditioning_features.repeat(sample.shape[0], 1, 1)
                pass # Add handling if necessary
            current_sample_input = torch.cat([sample, current_conditioning_features], dim=1)

        # Predict noise (or x0, or v, depending on scheduler.prediction_type)
        with torch.no_grad():
            # Timestep needs to be a tensor for the model
            timestep_input = t.unsqueeze(0).repeat(sample.shape[0]).to(device) # Shape: (B,)
            
            noise_pred = pc_model(current_sample_input, timestep_input, **model_forward_kwargs)

        # Compute previous noisy sample using the scheduler's step method
        step_kwargs = {}
        if scheduler_name.lower() == "ddim":
            step_kwargs["eta"] = eta
            # DDIM's step can also take a generator for SDE-like behavior if eta > 0
            if generator is not None: # Pass generator if DDIM uses it
                 step_kwargs["generator"] = generator
        
        sample = scheduler.step(noise_pred, t, sample, **step_kwargs).prev_sample

        # Call step callback if provided
        if step_callback is not None:
            step_callback(step_idx, sample.detach().clone(), t)

    return sample

