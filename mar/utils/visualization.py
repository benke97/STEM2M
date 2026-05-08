# src/mar/utils/visualization.py
import torch
import torch.nn as nn
# import torch.cuda.amp as amp # Removed, Accelerate handles AMP
from tqdm import tqdm
import logging
import numpy as np
import imageio
import matplotlib
from pathlib import Path
import torch.nn.functional as F
import traceback
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Optional, Dict, Any, List
from mar.utils.RDF_loss_new import RDFLoss, visualize_rdf

# Attempt to import DPMSolverMultistepScheduler and DDIMScheduler for visualization
try:
    from diffusers.schedulers import DPMSolverMultistepScheduler, DDIMScheduler
except ImportError:
    DPMSolverMultistepScheduler = None # Define as None if import fails
    DDIMScheduler = None

matplotlib.use('Agg') # Changed from TkAgg to Agg
import matplotlib.pyplot as plt
from mar.utils.conditioning import condition_point_cloud

log = logging.getLogger(__name__)

# --- PyTorch3D and Visualization Imports ---
# Attempt imports, log warning if they fail, but don't use a global flag.
# Define dummy types if imports fail to potentially avoid NameErrors later,
# although the logic should prevent their use if setup fails.
try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        OrthographicCameras as P3DOrthographicCameras, # Alias to avoid conflict if defined later
        PointsRasterizationSettings,
        PointsRenderer as P3DPointsRenderer, # Alias
        PointsRasterizer, AlphaCompositor
    )
    from pytorch3d.renderer.cameras import look_at_view_transform
    from pytorch3d.renderer.cameras import CamerasBase as P3DCamerasBase # Alias
    from pytorch3d.transforms import Transform3d
    import imageio # For saving images/gifs
    # import matplotlib.pyplot as plt # Already imported
    from PIL import Image, ImageDraw, ImageFont # Optional for adding text to images
    _PYTORCH3D_AVAILABLE = True
except ImportError as e:
    # Log prominently if visualization dependencies are missing
    print(f"\n{'='*20} WARNING {'='*20}")
    print(f"PyTorch3D or essential visualization libraries (imageio, matplotlib, Pillow) not found: {e}.")
    print("Visualization features will be automatically disabled.")
    print(f"{'='*50}\n")
    _PYTORCH3D_AVAILABLE = False
    # Define dummy types using the aliased names
    P3DPointsRenderer = object
    P3DOrthographicCameras = object
    P3DCamerasBase = object
    Pointclouds = object
    Image = object # type: ignore
    ImageDraw = object # type: ignore
    ImageFont = object # type: ignore

# This re-check block seems redundant if the first one is comprehensive.
# I'll keep the _PYTORCH3D_AVAILABLE from the first block and conditionally import/check ImageFont.
_IMAGEFONT_TRUETYPE_AVAILABLE = False # Initialize
if _PYTORCH3D_AVAILABLE:
    try:
        # Test if ImageFont.truetype is available, as it can be missing in some Pillow installs
        ImageFont.truetype("arial.ttf", 15) # Try loading a common font
        _IMAGEFONT_TRUETYPE_AVAILABLE = True
    except AttributeError:
        print("ImageFont.truetype not available. Text rendering on images will be basic or disabled.")
        _IMAGEFONT_TRUETYPE_AVAILABLE = False
    except Exception as e: # Catch other potential errors like font file not found
        print(f"Could not load default font for ImageFont: {e}. Text rendering might be affected.")
        _IMAGEFONT_TRUETYPE_AVAILABLE = False


def _setup_pytorch3d_renderer(
    image_size: int, point_radius: float, points_per_pixel: int, device: torch.device
) -> Tuple[Optional[object], Optional[Dict[str, object]]]:
    """
    Initializes PyTorch3D renderer and orthographic cameras.
    Returns (None, None) if PyTorch3D is unavailable or setup fails.
    """
    if not _PYTORCH3D_AVAILABLE:
        log.warning("PyTorch3D is not available. Cannot set up renderer.")
        return None, None

    try:
        R_xy, T_xy = look_at_view_transform(dist=2.5, elev=0, azim=0)
        cameras_xy = P3DOrthographicCameras(device=device, R=R_xy, T=T_xy)
        R_yz, T_yz = look_at_view_transform(dist=2.5, elev=0, azim=90)
        cameras_yz = P3DOrthographicCameras(device=device, R=R_yz, T=T_yz)
        R_xz, T_xz = look_at_view_transform(dist=2.5, elev=90, azim=0)
        cameras_xz = P3DOrthographicCameras(device=device, R=R_xz, T=T_xz)

        cameras_dict = {"xy": cameras_xy, "yz": cameras_yz, "xz": cameras_xz}

        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=point_radius,
            points_per_pixel=points_per_pixel,
            bin_size=0 # Important for fine-grained control if not using coarse rasterization
        )
        renderer = P3DPointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras_xy, raster_settings=raster_settings),
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )
        return renderer, cameras_dict
    except Exception as e:
        log.error(f"Error setting up PyTorch3D renderer: {e}", exc_info=True)
        return None, None

def _render_point_cloud_views(
    pc_tensor: torch.Tensor, 
    renderer, # Use aliased type
    cameras_dict: Dict[str, Any], # Use aliased type
    image_size: int,
    device: torch.device,
    title_prefix: str = "",
    color_range: Tuple[float, float] = (-1.0, 1.0),
    cfg: Optional[DictConfig] = None, # For font path
    filter_exists: bool = False 
) -> Optional[np.ndarray]:
    """
    Renders XY, YZ, XZ views of a point cloud and combines them.
    Accepts 4D input (x, y, z, exists). Can filter points where exists < 0.5.
    Returns None if rendering is not possible.
    """
    if not _PYTORCH3D_AVAILABLE or renderer is None or cameras_dict is None:
         log.error("Attempted to render views, but PyTorch3D is not available or setup failed.")
         return None

    C_dim = pc_tensor.shape[0] if pc_tensor.ndim == 2 and pc_tensor.shape[0] in [3,4] else (pc_tensor.shape[1] if pc_tensor.ndim == 2 and pc_tensor.shape[1] in [3,4] else None)
    if pc_tensor.ndim != 2 or C_dim is None:
        log.warning(f"Unexpected point cloud shape {pc_tensor.shape} for visualization '{title_prefix}'. Expected (3/4, N) or (N, 3/4). Skipping render.")
        return None
    
    pc_c_n = pc_tensor.to(device)
    if pc_c_n.shape[1] == C_dim: # Input was (N, C)
        pc_c_n = pc_c_n.transpose(0, 1) # Convert to (C, N)
    
    C, N = pc_c_n.shape
    dtype = pc_c_n.dtype

    pc_to_render_c_n = pc_c_n 
    if C == 4 and filter_exists:
        exists_values = pc_c_n[3, :] 
        mask = exists_values >= 0.5 
        pc_to_render_c_n = pc_c_n[:, mask] 
        #log.info(f"Visualization '{title_prefix}': Filtered points. Retained {pc_to_render_c_n.shape[1]}/{N} points.")

    verts = pc_to_render_c_n[:3, :].transpose(0, 1).contiguous() 

    if verts.shape[0] == 0:
        log.warning(f"Cannot visualize empty point cloud for '{title_prefix}' (possibly after filtering). Skipping render.")
        return None
    if torch.isnan(verts).any() or torch.isinf(verts).any():
        log.warning(f"Skipping visualization for '{title_prefix}' due to NaNs or Infs in vertex data. Skipping render.")
        return None

    rendered_views = []
    cmap = plt.get_cmap("coolwarm") 
    fixed_min, fixed_max = color_range

    view_configs = [
        ('xy', cameras_dict['xy'], 2, "XY View (Z color)"), 
        ('yz', cameras_dict['yz'], 0, "YZ View (X color)"), 
        ('xz', cameras_dict['xz'], 1, "XZ View (Y color)"), 
    ]

    for view_key, camera, coord_idx, view_title_text in view_configs:
        try:
            coords = verts[:, coord_idx] 
            norm_coords = torch.clamp((coords - fixed_min) / (fixed_max - fixed_min + 1e-6), 0, 1)
            colors_np = cmap(norm_coords.cpu().numpy())[:, :3] 
            colors = torch.tensor(colors_np, dtype=dtype, device=device)
            point_cloud = Pointclouds(points=[verts], features=[colors])
            img = renderer(point_cloud, cameras=camera)[0, ..., :3] 
            rendered_views.append(img)
        except Exception as e:
            log.error(f"Error rendering view '{view_key}' for '{title_prefix}': {e}", exc_info=True)
            rendered_views.append(torch.zeros((image_size, image_size, 3), dtype=dtype, device=device))

    if not rendered_views:
        log.warning(f"No views were successfully rendered for '{title_prefix}'. Skipping image composition.")
        return None

    combined_image = torch.cat(rendered_views, dim=1) 
    combined_image_np = (combined_image.cpu().numpy() * 255).astype(np.uint8)

    if _PYTORCH3D_AVAILABLE and _IMAGEFONT_TRUETYPE_AVAILABLE: # and cfg for font path
        try:
            pil_image = Image.fromarray(combined_image_np)
            draw = ImageDraw.Draw(pil_image)
            font_path = cfg.visualization.get("font_path", "arial.ttf") if cfg and hasattr(cfg, 'visualization') else "arial.ttf"
            try:
                font_main = ImageFont.truetype(font_path, 16)
                font_sub = ImageFont.truetype(font_path, 14)
            except IOError:
                log.warning(f"Font '{font_path}' not found. Using default PIL font for 3D view text.")
                font_main = ImageFont.load_default()
                font_sub = ImageFont.load_default()

            base_text_y_pos = 10
            if title_prefix:
                try: # PIL >= 8.0.0
                    bbox = draw.textbbox((0, 0), title_prefix, font=font_main)
                    title_w, title_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text(((combined_image_np.shape[1] - title_w) // 2, 5), title_prefix, fill=(255, 255, 255), font=font_main)
                    base_text_y_pos = 5 + title_h + 5
                except AttributeError: # Fallback
                    title_w = font_main.getlength(title_prefix) if hasattr(font_main, 'getlength') else len(title_prefix) * 10
                    draw.text(((combined_image_np.shape[1] - title_w) // 2, 5), title_prefix, fill=(255, 255, 255), font=font_main)
                    base_text_y_pos = 25
            """
            panel_width = image_size
            range_text = f"Range: [{fixed_min:.2f}, {fixed_max:.2f}]"
            if C == 4: range_text += f" ({'Filt.' if filter_exists else 'All'} Pts)"

            for i, (_, _, _, view_title_text) in enumerate(view_configs):
                try: # PIL >= 8.0.0
                    bbox_sub = draw.textbbox((0,0), view_title_text, font=font_sub)
                    sub_w = bbox_sub[2] - bbox_sub[0]
                    text_x = i * panel_width + (panel_width - sub_w) // 2
                    draw.text((max(10 + i * panel_width, text_x), base_text_y_pos), view_title_text, fill=(220, 220, 220), font=font_sub)
                except AttributeError: # Fallback
                    sub_w = font_sub.getlength(view_title_text) if hasattr(font_sub, 'getlength') else len(view_title_text) * 8
                    text_x = max(10 + i * panel_width, i * panel_width + (panel_width // 2) - (sub_w // 2))
                    draw.text((text_x, base_text_y_pos + 2), view_title_text, fill=(220, 220, 220), font=font_sub)

            try: # PIL >= 8.0.0
                bbox_range = draw.textbbox((0,0), range_text, font=font_sub)
                range_w = bbox_range[2] - bbox_range[0]
                draw.text((combined_image_np.shape[1] - range_w - 10, base_text_y_pos), range_text, fill=(255, 255, 0), font=font_sub)
            except AttributeError: # Fallback
                 range_w = font_sub.getlength(range_text) if hasattr(font_sub, 'getlength') else len(range_text) * 8
                 draw.text((combined_image_np.shape[1] - range_w - 10, base_text_y_pos + 2 ), range_text, fill=(255, 255, 0), font=font_sub)
            
            combined_image_np = np.array(pil_image)
            """
        except Exception as e:
            log.error(f"Error adding text overlay to 3D image for '{title_prefix}': {e}", exc_info=True)
    return combined_image_np

def center_z_axis(tensor_xyz, mask=None):
    """
    Centers the Z-axis (channel 2) of a point cloud tensor.
    
    Args:
        tensor_xyz: Tensor of shape [B, 3, N] or [3, N]
        mask: Optional mask of shape [B, 1, N] or [1, N] indicating valid points
    
    Returns:
        Tensor with Z-axis centered
    """
    if tensor_xyz.ndim not in [2, 3] or tensor_xyz.shape[-2] != 3:
        # Assuming the tensor is [..., 3, N], so the channel dim is -2
        raise ValueError(f"Expected 3 channels for XYZ (shape [B,3,N] or [3,N]), got {tensor_xyz.shape}")

    z_channel_index = 2 # Z is the third channel (index 2)
    
    # Determine if batched or single sample and select the z-channel
    if tensor_xyz.ndim == 3: # Batched: [B, 3, N]
        z_channel = tensor_xyz[:, z_channel_index:z_channel_index+1, :] # [B, 1, N]
        if mask is not None:
            if mask.ndim == 2: # If mask is [B,N], unsqueeze to [B,1,N]
                mask = mask.unsqueeze(1)
            if mask.shape[0] != z_channel.shape[0] or mask.shape[2] != z_channel.shape[2] or mask.shape[1] != 1:
                raise ValueError(f"Mask shape {mask.shape} incompatible with z_channel shape {z_channel.shape} for batched input.")
            valid_z = z_channel * mask
            valid_count = mask.sum(dim=-1, keepdim=True)
            z_mean = valid_z.sum(dim=-1, keepdim=True) / (valid_count + 1e-8) # [B, 1, 1]
        else:
            z_mean = z_channel.mean(dim=-1, keepdim=True) # [B, 1, 1]
    
    else: # Single sample: [3, N]
        z_channel = tensor_xyz[z_channel_index:z_channel_index+1, :] # [1, N]
        if mask is not None:
            if mask.ndim == 1: # If mask is [N], unsqueeze to [1,N]
                mask = mask.unsqueeze(0)
            if mask.shape[1] != z_channel.shape[1] or mask.shape[0] != 1:
                 raise ValueError(f"Mask shape {mask.shape} incompatible with z_channel shape {z_channel.shape} for single sample.")
            valid_z = z_channel * mask
            valid_count = mask.sum(dim=-1, keepdim=True)
            z_mean = valid_z.sum(dim=-1, keepdim=True) / (valid_count + 1e-8) # [1, 1]
        else:
            z_mean = z_channel.mean(dim=-1, keepdim=True) # [1, 1]
            
    tensor_xyz_centered = tensor_xyz.clone()
    if tensor_xyz.ndim == 3:
        tensor_xyz_centered[:, z_channel_index:z_channel_index+1, :] = z_channel - z_mean
    else:
        tensor_xyz_centered[z_channel_index:z_channel_index+1, :] = z_channel - z_mean
        
    return tensor_xyz_centered


def center_com(tensor_xyz, mask=None):
    """
    Centers the Center of Mass (CoM) of a point cloud tensor.
    
    Args:
        tensor_xyz: Tensor of shape [B, 3, N] or [3, N].
        mask: Optional mask of shape [B, 1, N] or [1, N] indicating valid points.
    
    Returns:
        Tensor with CoM at (0,0,0).
    """
    if tensor_xyz.shape[-2] != 3: # Assumes tensor_xyz is [..., C, N]
        raise ValueError(f"Expected 3 channels for XYZ, got {tensor_xyz.shape[-2]}")

    if mask is not None:
        # Mask has shape [B, 1, N] or [1,N] for single sample.
        # It should broadcast correctly for multiplication with tensor_xyz [B,3,N] or [3,N].
        masked_tensor = tensor_xyz * mask # Element-wise multiplication, relies on broadcasting mask from [B,1,N] to [B,3,N]
        # Sum over the N dimension to get the sum of coordinates for valid points.
        sum_of_coords = masked_tensor.sum(dim=-1) # Shape: [B, 3] or [3] if single sample
        
        # Count valid points for each sample in the batch.
        # Mask is [B,1,N] or [1,N]. Sum over N: [B,1] or [1].
        # This count applies to each of the X, Y, Z channels for the CoM calculation.
        num_valid_points = mask.sum(dim=-1) # Shape: [B, 1] or [1]
        
        # com = sum_of_coords / (num_valid_points + 1e-8)
        # sum_of_coords is [B,3], num_valid_points is [B,1]. Broadcasting works.
        com = sum_of_coords / (num_valid_points + 1e-8) # Shape: [B, 3] or [3]
        
        com = com.unsqueeze(-1) # Shape: [B, 3, 1] or [3, 1]
    else:
        # If no mask, compute mean over the N dimension.
        com = tensor_xyz.mean(dim=-1, keepdim=True) # Shape: [B, 3, 1] or [3, 1]
    
    return tensor_xyz - com


def visualize_unconditioned(
    model: nn.Module,
    noise_scheduler: Any, 
    epoch: int,
    device: torch.device,
    cfg: DictConfig,
):
    """
    Generates samples using the unconditional diffusion model and visualizes
    the denoising process using PyTorch3D. Saves images to Hydra output dir.
    Assumes the model predicts 3 channels (noise for XYZ).
    Skips if visualization dependencies are missing, setup fails, or cfg disables it.
    """
    if not cfg.visualization.enabled:
        log.info(f"Unconditioned visualization for epoch {epoch} disabled via cfg.visualization.enabled=False.")
        return
    if not _PYTORCH3D_AVAILABLE: 
        log.warning(f"Skipping unconditioned visualization for epoch {epoch} due to missing PyTorch3D or dependencies.")
        return
    
    # Use the original noise_scheduler for visualization
    effective_scheduler = noise_scheduler
    scheduler_name_for_log = type(noise_scheduler).__name__
    log.info(f"Visualization (unconditioned, epoch {epoch}): Using original scheduler: {scheduler_name_for_log}.")

    vis_cfg = cfg.visualization 
    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=vis_cfg.image_size,
        point_radius=vis_cfg.point_radius,
        points_per_pixel=vis_cfg.points_per_pixel,
        device=device    )
    if renderer is None or cameras_dict is None:
        log.warning(f"Skipping unconditioned visualization for epoch {epoch} due to renderer setup failure.")
        return

    vis_output_dir_uncond = Path(vis_cfg.subdir) / f"epoch_{epoch}" / "unconditional"
    try:
        vis_output_dir_uncond.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create unconditional visualization directory {vis_output_dir_uncond}: {e}")
        return # Cannot proceed if base directory cannot be created

    model.eval()
    num_samples_vis = vis_cfg.num_samples_to_vis
    
    # Optimize inference steps based on scheduler type for efficiency
    def get_optimal_steps_for_scheduler(scheduler, default_steps):
        """Get optimal number of inference steps based on scheduler type."""
        scheduler_name = type(scheduler).__name__
        
        if 'DPMSolverMultistepScheduler' == scheduler_name: # DPM++
            return min(default_steps, 20)
        elif 'DDIMScheduler' == scheduler_name: # DDIM
            return min(default_steps, 50)
        elif 'DDPMScheduler' == scheduler_name: # DDPM
            return default_steps 
        else:
            # Fallback for unknown schedulers or those not explicitly listed
            log.warning(
                f"Scheduler '{scheduler_name}' does not have a specific step optimization rule in get_optimal_steps_for_scheduler. "
                f"Applying a general fallback: min({default_steps}, 100) steps."
            )
            return min(default_steps, 100)
    
    base_inference_steps = vis_cfg.num_inference_steps
    num_inference_steps = get_optimal_steps_for_scheduler(effective_scheduler, base_inference_steps)
      # Log optimization information
    if num_inference_steps != base_inference_steps:
        log.info(f"Optimized inference steps for {type(effective_scheduler).__name__}: {base_inference_steps} -> {num_inference_steps} steps")

    points = cfg.data.get("point_cloud_size", 1024)
    if hasattr(model, 'point_cloud_size'): 
        points = model.point_cloud_size
    elif hasattr(cfg.model, 'point_cloud_size'): 
        points = cfg.model.point_cloud_size

    dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.float32
    shape = (num_samples_vis, 3, points) 
    # Initial sample C_T should be CoM-free, torch.randn already produces this (mean 0).
    sample = torch.randn(shape, device=device, dtype=dtype) 
    
    effective_scheduler.set_timesteps(num_inference_steps)
    log.info(f"--- Epoch {epoch}: Visualizing {num_samples_vis} unconditioned sample(s) ({num_inference_steps} steps, 3-channel XYZ, scheduler: {type(effective_scheduler).__name__}) ---")

    timesteps_iterable = tqdm(effective_scheduler.timesteps, desc=f"Uncond Vis E{epoch}", leave=False, disable=num_samples_vis > 1 or not vis_cfg.get("tqdm_steps", True))

    save_step_indices_for_intermediate_vis = set()
    intermediate_3d_raw_dir, intermediate_3d_filtered_dir = None, None # Init

    if num_samples_vis > 0 and vis_cfg.get("save_intermediate_steps", True): # Check if intermediate saving is enabled
        sample_0_intermediate_base_dir = vis_output_dir_uncond / "sample_0_intermediate_steps"
        intermediate_3d_raw_dir = sample_0_intermediate_base_dir / "3d_raw"
        intermediate_3d_filtered_dir = sample_0_intermediate_base_dir / "3d_filtered"

        try:
            intermediate_3d_raw_dir.mkdir(parents=True, exist_ok=True)
            intermediate_3d_filtered_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Could not create intermediate directories for unconditioned vis under {sample_0_intermediate_base_dir}: {e}")
            # Disable saving intermediates if dirs can't be made
            save_step_indices_for_intermediate_vis = set() 
            intermediate_3d_raw_dir, intermediate_3d_filtered_dir = None, None


        num_total_scheduler_steps = len(effective_scheduler.timesteps)
        max_intermediate_saves = vis_cfg.get("max_intermediate_saves", 100)

        if num_total_scheduler_steps <= max_intermediate_saves:
            save_step_indices_for_intermediate_vis = set(range(num_total_scheduler_steps))
        else:
            save_step_indices_for_intermediate_vis.add(0) 
            save_step_indices_for_intermediate_vis.add(num_total_scheduler_steps - 1) 
            num_additional_saves_needed = max_intermediate_saves - 2
            if num_additional_saves_needed > 0:
                intermediate_step_indices_pool = list(range(1, num_total_scheduler_steps - 1))
                if intermediate_step_indices_pool:
                    if len(intermediate_step_indices_pool) <= num_additional_saves_needed:
                        save_step_indices_for_intermediate_vis.update(intermediate_step_indices_pool)
                    else:
                        selected_indices_in_pool = np.linspace(0, len(intermediate_step_indices_pool) - 1, num_additional_saves_needed, dtype=int)
                        for idx_in_pool in selected_indices_in_pool:
                            save_step_indices_for_intermediate_vis.add(intermediate_step_indices_pool[idx_in_pool])
    
    for step_idx, t in enumerate(timesteps_iterable):
        with torch.no_grad():
            timestep_batch = torch.full((num_samples_vis,), t, device=device, dtype=torch.long)
            
            # GeoDiff Algo 1, Step 3: Shift C_s (which is 'sample' here) to zero CoM.
            # This also prepares the input for the model.
            sample_current_step_centered = center_com(sample)

            # Debug: Check for NaN/Inf before model forward pass
            if torch.isnan(sample_current_step_centered).any() or torch.isinf(sample_current_step_centered).any():
                log.warning(f"Uncond Step {step_idx+1}: sample_current_step_centered contains NaN/Inf before model forward pass")
            
            # Model predicts noise from the centered current state.
            noise_pred = model(sample_current_step_centered, timestep_batch) 
            
            # Predicted noise should be centered for the loss (which is what GeoDiff's alignment provides).
            noise_pred_centered = center_com(noise_pred)

            # Debug: Check for NaN/Inf in model output
            if torch.isnan(noise_pred_centered).any() or torch.isinf(noise_pred_centered).any():
                log.warning(f"Uncond Step {step_idx+1}: noise_pred_centered (after centering) contains NaN/Inf")
        
        if noise_pred_centered.shape[1] != 3 or noise_pred_centered.shape[0] != num_samples_vis or noise_pred_centered.shape[2] != points:
            log.error(f"Unconditioned noise_pred_centered shape {noise_pred_centered.shape} is incorrect. Expected ({num_samples_vis}, 3, {points}). Stopping."); break
        
        # Pass the *centered* current sample (x_t) to the scheduler.
        # This ensures that if the scheduler calculates x0_pred based on x_t and epsilon_pred,
        # that x0_pred will be centered, aligning with the expected properties of the reverse process.
        denoised_result = effective_scheduler.step(noise_pred_centered, t, sample_current_step_centered)
        
        # The result (x_{t-1}) becomes the 'sample' for the *next* iteration.
        # Explicitly center it to ensure it's CoM-free before the next loop's iteration.
        sample = center_com(denoised_result.prev_sample) 
        
        # Debug: Check for NaN/Inf after scheduler step
        if torch.isnan(sample).any() or torch.isinf(sample).any():
            log.warning(f"Uncond Step {step_idx+1}: sample contains NaN/Inf after scheduler step")
            # Additional debugging: print some statistics
            log.warning(f"  sample stats: min={sample.min().item():.6f}, max={sample.max().item():.6f}, mean={sample.mean().item():.6f}")
            log.warning(f"  noise_pred stats: min={noise_pred.min().item():.6f}, max={noise_pred.max().item():.6f}, mean={noise_pred.mean().item():.6f}")

        if num_samples_vis > 0 and step_idx in save_step_indices_for_intermediate_vis and intermediate_3d_raw_dir and intermediate_3d_filtered_dir:
            pc_to_render_intermediate = sample[0].detach().clone() 
            step_img_filename_stem = f"step_{step_idx+1:04d}_t_{t.item():04d}"
            title_3d_intermediate = f"Uncond E{epoch} S{step_idx+1}/{num_inference_steps} T{t.item()} Sample 0"

            rendered_3d_raw = _render_point_cloud_views(
                pc_tensor=pc_to_render_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Raw",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
            )
            if rendered_3d_raw is not None:
                imageio.imwrite(intermediate_3d_raw_dir / (step_img_filename_stem + ".png"), rendered_3d_raw)
            
            rendered_3d_filtered = _render_point_cloud_views(
                pc_tensor=pc_to_render_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Filt.",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_3d_filtered is not None:
                imageio.imwrite(intermediate_3d_filtered_dir / (step_img_filename_stem + ".png"), rendered_3d_filtered)

    # Final visualization of the completed sample(s)
    if num_samples_vis > 0:
        for s_idx in range(num_samples_vis):
            final_pc_to_render = sample[s_idx].detach().clone()
            final_img_filename_stem = f"final_sample_{s_idx}"
            final_title_prefix = f"Uncond E{epoch} Final Sample {s_idx}"

            rendered_final_raw = _render_point_cloud_views(
                pc_tensor=final_pc_to_render, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=final_title_prefix + " Raw",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
            )
            if rendered_final_raw is not None:
                imageio.imwrite(vis_output_dir_uncond / (final_img_filename_stem + "_raw.png"), rendered_final_raw)

            rendered_final_filtered = _render_point_cloud_views(
                pc_tensor=final_pc_to_render, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=final_title_prefix + " Filt.",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_final_filtered is not None:
                imageio.imwrite(vis_output_dir_uncond / (final_img_filename_stem + "_filtered.png"), rendered_final_filtered)
        log.info(f"Saved final unconditioned visualizations for {num_samples_vis} sample(s) to {vis_output_dir_uncond}")

    log.info(f"--- Finished unconditioned visualization for epoch {epoch} ---")
    model.train()

@torch.no_grad()
def visualize_conditioned(
    model: nn.Module,
    projection_model: nn.Module,
    noise_scheduler: Any,
    example_batch: Dict[str, torch.Tensor],
    epoch: int,
    device: torch.device,
    cfg: DictConfig,
    size_estimator: Optional[nn.Module] = None,
):
    """
    Visualizes conditioned diffusion model generation.
    The 4th channel of the input to conditioning is the GT binary existence mask.
    Noise is added to and predicted for only the XYZ channels.
    """
    if not cfg.visualization.enabled: 
        log.info(f"Conditioned visualization for epoch {epoch} disabled via cfg.visualization.enabled=False.")
        return
    if not _PYTORCH3D_AVAILABLE: 
        log.warning(f"Skipping conditioned visualization for epoch {epoch} due to missing PyTorch3D or dependencies.")
        return

    # Use the original noise_scheduler for visualization
    effective_scheduler = noise_scheduler
    scheduler_name_for_log = type(noise_scheduler).__name__
    log.info(f"Visualization (conditioned, epoch {epoch}): Using original scheduler: {scheduler_name_for_log}.")

    vis_cfg = cfg.visualization
    max_samples_to_vis_from_batch = vis_cfg.num_samples_to_vis
    
    try:
        actual_batch_size_for_vis = min(max_samples_to_vis_from_batch, example_batch["image"].shape[0])
        if actual_batch_size_for_vis == 0 and max_samples_to_vis_from_batch > 0:
            log.warning("num_samples_to_vis > 0 but example_batch is empty or 'image' key missing. Skipping conditioned visualization.")
            return
        if actual_batch_size_for_vis == 0 :
             return 
    except KeyError:
        log.error("Conditioned visualization: 'image' key missing in example_batch. Skipping.")
        return

    # Get and validate the index for intermediate visualization
    intermediate_vis_idx = vis_cfg.get("intermediate_vis_sample_idx", 0)
    if not (0 <= intermediate_vis_idx < actual_batch_size_for_vis):
        if actual_batch_size_for_vis > 0: # Only warn if there are samples to visualize
            log.warning(
                f"Configured intermediate_vis_sample_idx {intermediate_vis_idx} is out of bounds "
                f"for actual_batch_size_for_vis ({actual_batch_size_for_vis}). Defaulting to 0."
            )
            intermediate_vis_idx = 0
        # If actual_batch_size_for_vis is 0, intermediate_vis_idx is moot as intermediate saving will be skipped.
        # Setting to 0 here to prevent potential downstream issues if logic changes, though current flow exits.
        elif actual_batch_size_for_vis == 0:
            intermediate_vis_idx = 0


    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=vis_cfg.image_size,
        point_radius=vis_cfg.point_radius,
        points_per_pixel=vis_cfg.points_per_pixel,
        device=device
    )
    if renderer is None or cameras_dict is None:
        log.warning(f"Renderer setup failed for conditioned vis epoch {epoch}. Skipping.")
        return

    base_vis_output_dir = Path(vis_cfg.subdir) / f"epoch_{epoch}" / "conditioned"
    try:
        base_vis_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving conditioned visualizations for epoch {epoch} to: {base_vis_output_dir}")
    except OSError as e:
        log.error(f"Could not create base conditioned visualization directory {base_vis_output_dir}: {e}. Skipping.")
        return

    gt_key_found = None
    for key_candidate in ("target", "points", "point_cloud", "pc"): # "point_cloud" is primary
        if key_candidate in example_batch and example_batch[key_candidate].shape[1] == 4: # Ensure it's 4-channel
            gt_key_found = key_candidate
            break
    if gt_key_found:
        for b_idx in range(actual_batch_size_for_vis):
            try:
                gt_pc_original = example_batch[gt_key_found][b_idx].to(device, dtype=torch.float32) # [4, N] or [N, 4]
                
                # Center the GT point cloud before visualization
                gt_xyz_original = gt_pc_original[:3, :] # [3, N]
                gt_existence_original = gt_pc_original[3:, :] # [1, N]
                
                # Ensure mask is correctly shaped for center_com if it expects [1,N] for single sample
                # center_com handles [3,N] for tensor_xyz and [1,N] for mask
                gt_xyz_centered = center_com(gt_xyz_original, mask=gt_existence_original)
                
                gt_pc_for_vis = torch.cat((gt_xyz_centered, gt_existence_original), dim=0) # [4, N]

                rendered_gt = _render_point_cloud_views(
                    pc_tensor=gt_pc_for_vis, renderer=renderer, cameras_dict=cameras_dict,
                    image_size=vis_cfg.image_size, device=device,
                    title_prefix=f"E{epoch} Sample {b_idx} GT (CoM Centered)",
                    color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg,
                    filter_exists=vis_cfg.get("filter_gt_vis", True) # GT has 'exists', allow filtering
                )
                if rendered_gt is not None:
                    gt_img_path = base_vis_output_dir / f"sample_{b_idx}_gt_point_cloud.png"
                    imageio.imwrite(gt_img_path, rendered_gt)
            except Exception as e:
                log.error(f"Failed to save GT PC visualization for sample {b_idx}: {e}", exc_info=True)
    else:
        log.warning(f"No suitable 4-channel GT point cloud key found in batch for visualization: {list(example_batch.keys())}")

    for b_idx in range(actual_batch_size_for_vis):
        try:
            conditioning_image_single_tensor = example_batch["image"][b_idx].to(device)
            cond_img_np = conditioning_image_single_tensor[0].detach().cpu().numpy() 
            cond_img_np = np.flip(cond_img_np, axis=0) 
            cond_img_path = base_vis_output_dir / f"sample_{b_idx}_conditioning_image.png"
            img_min, img_max = cond_img_np.min(), cond_img_np.max()
            norm_img = ((cond_img_np - img_min) / (img_max - img_min + 1e-6)) * 255.0 if img_min != img_max else np.ones_like(cond_img_np) * 127.5
            imageio.imwrite(cond_img_path, norm_img.astype(np.uint8))
        except Exception as e:
            log.error(f"Failed to save conditioning image for sample {b_idx}: {e}", exc_info=True)
            
    # --- Prepare for Sampling ---
    model.eval()
    projection_model.eval()
    if size_estimator: 
        size_estimator.eval()

    points = cfg.data.point_cloud_size
    dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.float32

    # Prepare initial sample_batch: noisy XYZ + clean existence
    # Ensure "point_cloud" key exists and has 4 channels
    if "point_cloud" not in example_batch or example_batch["point_cloud"].shape[1] != 4:
        log.error(f"Conditioned visualization: 'point_cloud' key missing in example_batch or not 4-channel. Shape: {example_batch.get('point_cloud', torch.empty(0)).shape}. Skipping.")
        model.train() # Restore train mode
        projection_model.train() # Restore train mode if it was changed
        if size_estimator: size_estimator.train()
        return
        
    clean_point_clouds_for_vis = example_batch["point_cloud"][:actual_batch_size_for_vis].to(device, dtype=dtype)
    clean_existence_channel_batch = clean_point_clouds_for_vis[:, 3:, :] # [B, 1, N]
    # Initial noisy_xyz_batch (C_T) should be CoM-free. torch.randn naturally produces this.
    initial_noisy_xyz_batch = torch.randn((actual_batch_size_for_vis, 3, points), device=device, dtype=dtype) # [B, 3, N]
    sample_batch = torch.cat((initial_noisy_xyz_batch, clean_existence_channel_batch), dim=1) # [B, 4, N]

    conditioning_image_batch = example_batch["image"][:actual_batch_size_for_vis].to(device)
    pixel_size_batch = example_batch["pixel_size"][:actual_batch_size_for_vis].to(device)
    size_embedding_batch = None
    if size_estimator:
        with torch.no_grad():
            size_embedding_batch = size_estimator(conditioning_image_batch)
    
    # Optimize inference steps based on scheduler type for efficiency
    def get_optimal_steps_for_scheduler(scheduler, default_steps):
        """Get optimal number of inference steps based on scheduler type."""
        scheduler_name = type(scheduler).__name__

        if 'DPMSolverMultistepScheduler' == scheduler_name: # DPM++
            return min(default_steps, 20)
        elif 'DDIMScheduler' == scheduler_name: # DDIM
            return min(default_steps, 50)
        elif 'DDPMScheduler' == scheduler_name: # DDPM
            return default_steps
        else:
            # Fallback for unknown schedulers or those not explicitly listed
            log.warning(
                f"Scheduler '{scheduler_name}' does not have a specific step optimization rule in get_optimal_steps_for_scheduler. "
                f"Applying a general fallback: min({default_steps}, 100) steps."
            )
            return min(default_steps, 100)

    base_inference_steps = vis_cfg.num_inference_steps
    num_inference_steps = get_optimal_steps_for_scheduler(effective_scheduler, base_inference_steps)
      # Log optimization information
    if num_inference_steps != base_inference_steps:
        log.info(f"Optimized inference steps for {type(effective_scheduler).__name__}: {base_inference_steps} -> {num_inference_steps} steps")

    effective_scheduler.set_timesteps(num_inference_steps) # Use effective_scheduler here
    log.info(f"--- Epoch {epoch}: Visualizing {actual_batch_size_for_vis} conditioned sample(s) ({num_inference_steps} steps, scheduler: {type(effective_scheduler).__name__}) ---")

    timesteps_iterable = tqdm(effective_scheduler.timesteps, desc=f"Cond Vis E{epoch}", leave=False, disable=not vis_cfg.get("tqdm_steps", True)) # Use effective_scheduler here
    
    # Create directories for intermediate visualizations
    intermediate_base_dir = None
    intermediate_3d_raw_dir = None
    intermediate_3d_filtered_dir = None
    intermediate_overlay_dir = None

    if actual_batch_size_for_vis > 0: # Only attempt to create dirs if there's a valid sample index
        intermediate_base_dir = base_vis_output_dir / f"sample_{intermediate_vis_idx}_intermediate_steps"
        intermediate_3d_raw_dir = intermediate_base_dir / "3d_raw"
        intermediate_3d_filtered_dir = intermediate_base_dir / "3d_filtered"
        intermediate_overlay_dir = intermediate_base_dir / "overlay"
        
        try:
            intermediate_3d_raw_dir.mkdir(parents=True, exist_ok=True)
            intermediate_3d_filtered_dir.mkdir(parents=True, exist_ok=True)
            intermediate_overlay_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Could not create intermediate directories for sample {intermediate_vis_idx}: {e}")
            # Set to None to disable intermediate saves if directory creation fails
            intermediate_3d_raw_dir = intermediate_3d_filtered_dir = intermediate_overlay_dir = None
    
    save_step_indices_for_intermediate_vis = set()
    num_total_scheduler_steps = len(effective_scheduler.timesteps)
    max_intermediate_saves = vis_cfg.get("max_intermediate_saves", 100)

    if num_total_scheduler_steps <= max_intermediate_saves:
        save_step_indices_for_intermediate_vis = set(range(num_total_scheduler_steps))
    else:
        save_step_indices_for_intermediate_vis.add(0) 
        save_step_indices_for_intermediate_vis.add(num_total_scheduler_steps - 1) 
        num_additional_saves_needed = max_intermediate_saves - 2
        if num_additional_saves_needed > 0:
            intermediate_step_indices_pool = list(range(1, num_total_scheduler_steps - 1))
            if intermediate_step_indices_pool:
                if len(intermediate_step_indices_pool) <= num_additional_saves_needed:
                    save_step_indices_for_intermediate_vis.update(intermediate_step_indices_pool)
                else:
                    selected_indices_in_pool = np.linspace(0, len(intermediate_step_indices_pool) - 1, num_additional_saves_needed, dtype=int)
                    for idx_in_pool in selected_indices_in_pool:
                        save_step_indices_for_intermediate_vis.add(intermediate_step_indices_pool[idx_in_pool])
    
    for step_idx, t in enumerate(timesteps_iterable):
        with torch.no_grad():
            timestep_batch = torch.full((actual_batch_size_for_vis,), t, device=device, dtype=torch.long)
            try:
                # GeoDiff Algo 1, Step 3: Shift C_s (XYZ part of 'sample_batch') to zero CoM.
                current_xyz_part = sample_batch[:, :3, :]
                current_existence_part = sample_batch[:, 3:, :] 
                current_xyz_part_centered = center_com(current_xyz_part, mask=current_existence_part)

                # --- Prepare CoM offset for conditioning, mirroring training logic ---
                if "CoM" not in example_batch:
                    log.error("Conditioned visualization: 'CoM' key missing in example_batch. Cannot apply CoM offset. Stopping.")
                    model.train(); projection_model.train()
                    if size_estimator: size_estimator.train()
                    return 
                
                CoM_from_batch = example_batch["CoM"][:actual_batch_size_for_vis].to(device, dtype=dtype) 
                if CoM_from_batch.ndim == 1 and actual_batch_size_for_vis == 1 : CoM_from_batch = CoM_from_batch.unsqueeze(0) # ensure [1,2] if B=1 and CoM was [2]
                
                if CoM_from_batch.shape[0] != actual_batch_size_for_vis or CoM_from_batch.shape[1] != 2:
                    log.error(f"Expected CoM from batch to be [{actual_batch_size_for_vis}, 2], got {CoM_from_batch.shape}. Cannot apply CoM offset. Stopping.")
                    model.train(); projection_model.train()
                    if size_estimator: size_estimator.train()
                    return

                zeros_for_z_offset = torch.zeros(actual_batch_size_for_vis, 1, device=device, dtype=dtype)
                # im_CoM_offset will be [B, 3, 1], representing the (X, Y, 0) offset for each sample in the batch
                im_CoM_offset = torch.cat([CoM_from_batch, zeros_for_z_offset], dim=1).unsqueeze(-1)

                # Apply CoM offset to the centered XYZ part before conditioning
                xyz_part_for_conditioning_shifted = current_xyz_part_centered + im_CoM_offset
                sample_batch_for_conditioning_final = torch.cat((xyz_part_for_conditioning_shifted, current_existence_part), dim=1)

                # Check input to condition_point_cloud
                if torch.isnan(sample_batch_for_conditioning_final).any() or torch.isinf(sample_batch_for_conditioning_final).any():
                    log.warning(f"Step {step_idx+1}: sample_batch_for_conditioning_final (input to condition_point_cloud) contains NaN/Inf.")

                # Call condition_point_cloud with the CoM-shifted point cloud
                # This function is assumed to return concatenated (original_4D_cloud_channels, conditioning_feature_channels)
                raw_conditioning_output = condition_point_cloud(
                    sample_batch_for_conditioning_final.clone(), 
                    pixel_size_batch, 
                    conditioning_image_batch, 
                    projection_model, 
                    device, cfg
                )
                
                conditioned_sample_input_for_model = raw_conditioning_output.clone()

                # Conditionally un-shift CoM for the main model input, based on use_occlusion
                # If use_occlusion is False, model expects CoM-centered XYZ.
                # If use_occlusion is True, model expects XYZ shifted by im_CoM_offset.
                use_occlusion = cfg.model.conditioning.get("use_occlusion", False)
                if not use_occlusion:
                    conditioned_sample_input_for_model[:, :3, :] -= im_CoM_offset
                
                current_size_embedding = size_embedding_batch 
                
                if torch.isnan(conditioned_sample_input_for_model).any() or torch.isinf(conditioned_sample_input_for_model).any():
                    nan_inf_source_info = "with CoM shift (use_occlusion=True)" if use_occlusion else "after CoM un-shift (use_occlusion=False)"
                    log.warning(f"Step {step_idx+1}: conditioned_sample_input_for_model ({nan_inf_source_info}) contains NaN/Inf before model forward pass")
                
                noise_pred_xyz = model(conditioned_sample_input_for_model, timestep_batch, pixel_size_batch, current_size_embedding)
                
                # Predicted noise should be centered for the loss (which is what GeoDiff's alignment provides).
                noise_pred_xyz_centered = center_com(noise_pred_xyz, mask=current_existence_part)

                if torch.isnan(noise_pred_xyz_centered).any() or torch.isinf(noise_pred_xyz_centered).any():
                    log.warning(f"Step {step_idx+1}: noise_pred_xyz_centered (after centering with existence mask) contains NaN/Inf")
                
            except Exception as e:
                log.error(f"Conditioned model forward pass failed at step {step_idx+1}, t={t}: {e}", exc_info=True); break
        
        if noise_pred_xyz_centered.shape[1] != 3 or noise_pred_xyz_centered.shape[0] != actual_batch_size_for_vis or noise_pred_xyz_centered.shape[2] != points:
            log.error(f"Conditioned noise_pred_xyz_centered shape {noise_pred_xyz_centered.shape} is incorrect. Expected ({actual_batch_size_for_vis}, 3, {points}). Stopping."); break

        try:
            # Pass the *centered* current sample (x_t) to the scheduler.
            # This ensures that if the scheduler calculates x0_pred based on x_t and epsilon_pred,
            # that x0_pred will be centered, aligning with the expected properties of the reverse process.
            current_noisy_xyz_for_scheduler = current_xyz_part_centered # USE THE CENTERED PART
            
            if torch.isnan(current_noisy_xyz_for_scheduler).any() or torch.isinf(current_noisy_xyz_for_scheduler).any():
                log.warning(f"Step {step_idx+1}: current_noisy_xyz_for_scheduler contains NaN/Inf before scheduler step")
            
            denoised_xyz_result = effective_scheduler.step(noise_pred_xyz_centered, t, current_noisy_xyz_for_scheduler) # Use centered noise and centered current_sample
            prev_denoised_xyz = denoised_xyz_result.prev_sample
            
            # GeoDiff Algo 1, Step 3 (implicit in the loop): Ensure the new sample (x_{t-1}) is centered for the NEXT step.
            prev_denoised_xyz_centered = center_com(prev_denoised_xyz, mask=current_existence_part)

            if torch.isnan(prev_denoised_xyz_centered).any() or torch.isinf(prev_denoised_xyz_centered).any():
                log.warning(f"Step {step_idx+1}: prev_denoised_xyz_centered contains NaN/Inf after scheduler step")
            
            # Reconstruct sample_batch with the new, centered XYZ and the clean existence channel.
            sample_batch = torch.cat((prev_denoised_xyz_centered, current_existence_part), dim=1) 

        except Exception as e:
            log.error(f"Conditioned scheduler step failed at step {step_idx+1}, t={t}: {e}", exc_info=True); break

        # --- Intermediate Visualizations (for sample `intermediate_vis_idx` of the batch) ---
        if step_idx in save_step_indices_for_intermediate_vis and intermediate_3d_raw_dir is not None: # Check if dir creation was successful
            # For 3D views, use the CoM-centered point cloud from sample_batch
            pc_for_3d_intermediate = sample_batch[intermediate_vis_idx].detach().clone() # (C, N)
            step_img_filename_stem = f"step_{step_idx+1:04d}_t_{t.item():04d}"
            
            title_3d_intermediate = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx}"
            rendered_3d_raw = _render_point_cloud_views(
                pc_tensor=pc_for_3d_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Raw",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
            )
            if rendered_3d_raw is not None:
                imageio.imwrite(intermediate_3d_raw_dir / (step_img_filename_stem + ".png"), rendered_3d_raw)
            
            rendered_3d_filtered = _render_point_cloud_views(
                pc_tensor=pc_for_3d_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Filt.",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_3d_filtered is not None:
                imageio.imwrite(intermediate_3d_filtered_dir / (step_img_filename_stem + ".png"), rendered_3d_filtered)

            # For overlay projections, shift the CoM-centered point cloud by im_CoM_offset
            if intermediate_overlay_dir is not None:
                xyz_intermediate_centered = sample_batch[intermediate_vis_idx, :3, :].detach().clone() # [3, N]
                existence_intermediate = sample_batch[intermediate_vis_idx, 3:, :].detach().clone() # [1, N]
                # Get the specific CoM offset for this sample: im_CoM_offset is [B,3,1], so slice it.
                current_im_CoM_offset_intermediate = im_CoM_offset[intermediate_vis_idx, :, :] # Shape [3, 1]
                xyz_intermediate_shifted = xyz_intermediate_centered + current_im_CoM_offset_intermediate # Broadcasting [3,N] + [3,1]
                pc_for_overlay_intermediate = torch.cat((xyz_intermediate_shifted, existence_intermediate), dim=0) # [4, N]
                
                try:
                    title_overlay_intermediate_raw = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx} Overlay (Raw)"
                    visualize_projection_and_sampling(
                        point_clouds=pc_for_overlay_intermediate.unsqueeze(0), # Needs [1, 4, N]
                        pixel_sizes=pixel_size_batch[intermediate_vis_idx:intermediate_vis_idx+1], 
                        images=conditioning_image_batch[intermediate_vis_idx:intermediate_vis_idx+1],
                        cell_size=cfg.data.cell_size, device=device, filter_exists=False,
                        save_path=intermediate_overlay_dir / (step_img_filename_stem + "_raw_overlay.png"),
                        title=title_overlay_intermediate_raw
                    )
                    title_overlay_intermediate_filtered = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx} Overlay (Filtered)"
                    visualize_projection_and_sampling(
                        point_clouds=pc_for_overlay_intermediate.unsqueeze(0), # Needs [1, 4, N]
                        pixel_sizes=pixel_size_batch[intermediate_vis_idx:intermediate_vis_idx+1], 
                        images=conditioning_image_batch[intermediate_vis_idx:intermediate_vis_idx+1],
                        cell_size=cfg.data.cell_size, device=device, filter_exists=True,
                        save_path=intermediate_overlay_dir / (step_img_filename_stem + "_filtered_overlay.png"),
                        title=title_overlay_intermediate_filtered
                    )
                except Exception as e:
                    log.error(f"Failed intermediate overlay vis for E{epoch} S{step_idx+1} Sample {intermediate_vis_idx}: {e}", exc_info=True)
    
    # --- Final Predictions (Batch-wise) ---
    log.info(f"--- Epoch {epoch}: Generating final predictions visualizations ---")
    final_preds_base_dir = base_vis_output_dir / "FINAL_PREDICTIONS"
    final_preds_base_dir.mkdir(parents=True, exist_ok=True)

    for b_idx in range(actual_batch_size_for_vis):
        sample_final_dir = final_preds_base_dir / f"sample_{b_idx}"
        sample_final_dir.mkdir(parents=True, exist_ok=True)
        
        # For 3D views, use the CoM-centered point cloud from the final sample_batch
        final_pc_b_for_3d = sample_batch[b_idx].detach().clone() # (C,N), C=4
        
        title_3d_final = f"E{epoch} Final Sample {b_idx}"
        rendered_final_3d_raw = _render_point_cloud_views(
            pc_tensor=final_pc_b_for_3d, renderer=renderer, cameras_dict=cameras_dict,
            image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_final + " Raw",
            color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
        )
        if rendered_final_3d_raw is not None:
            imageio.imwrite(sample_final_dir / "3D_RAW_final.png", rendered_final_3d_raw)

        rendered_final_3d_filtered = _render_point_cloud_views(
            pc_tensor=final_pc_b_for_3d, renderer=renderer, cameras_dict=cameras_dict,
            image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_final + " Filtered",
            color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
        )
        if rendered_final_3d_filtered is not None:
            imageio.imwrite(sample_final_dir / "3D_FILTERED_final.png", rendered_final_3d_filtered)

        # --- RDF Visualization ---
        if False:
            try:
                # 1. Initialize RDFLoss module to use its _compute_rdf method
                rdf_loss_module = RDFLoss(
                    cutoff=cfg.visualization.get("rdf_cutoff", 1.4),
                    n_bins=cfg.visualization.get("rdf_n_bins", 512),
                    bandwidth=cfg.visualization.get("rdf_bandwidth", 0.005)
                ).to(device)

                # 2. Prepare Predicted Point Cloud (filtered)
                pred_exists_mask = final_pc_b_for_3d[3, :] >= 0.5
                if pred_exists_mask.any():
                    pc_pred_for_rdf = final_pc_b_for_3d[:3, pred_exists_mask].transpose(0, 1).contiguous() # [N_pred, 3]

                    # 3. Prepare Ground Truth Point Cloud (filtered)
                    gt_pc_b_4_n = example_batch[gt_key_found][b_idx].to(device, dtype=torch.float32)
                    gt_exists_mask = gt_pc_b_4_n[3, :] >= 0.5
                    if gt_exists_mask.any():
                        # We use the original uncentered GT for RDF as it's an intrinsic property
                        pc_gt_for_rdf = gt_pc_b_4_n[:3, gt_exists_mask].transpose(0, 1).contiguous() # [N_gt, 3]

                        # 4. Generate and save the plot
                        rdf_plot_path = sample_final_dir / "RDF_GT_vs_PRED_final.png"
                        visualize_rdf(
                            loss_module=rdf_loss_module,
                            pc_pred=pc_pred_for_rdf,
                            pc_gt=pc_gt_for_rdf,
                            output_path=str(rdf_plot_path)
                        )
                    else:
                        log.warning(f"Skipping RDF plot for sample {b_idx}: No valid points in ground truth.")
                else:
                    log.warning(f"Skipping RDF plot for sample {b_idx}: No valid points in prediction.")
            except Exception as e_rdf:
                log.error(f"Failed to generate RDF plot for sample {b_idx}: {e_rdf}", exc_info=True)

        # For overlay projections, shift the CoM-centered point cloud by im_CoM_offset
        xyz_final_centered = sample_batch[b_idx, :3, :].detach().clone() # [3, N]
        existence_final = sample_batch[b_idx, 3:, :].detach().clone() # [1, N]
        current_im_CoM_offset_final = im_CoM_offset[b_idx, :, :] # Shape [3, 1]
        xyz_final_shifted = xyz_final_centered + current_im_CoM_offset_final # Broadcasting [3,N] + [3,1]
        final_pc_b_for_overlay = torch.cat((xyz_final_shifted, existence_final), dim=0) # [4, N]

        try:
            title_overlay_final_raw = f"E{epoch} Final Sample {b_idx} Overlay (Raw)"
            visualize_projection_and_sampling(
                point_clouds=final_pc_b_for_overlay.unsqueeze(0), # Make it [1, 4, N]
                pixel_sizes=pixel_size_batch[b_idx:b_idx+1], 
                images=conditioning_image_batch[b_idx:b_idx+1],
                cell_size=cfg.data.cell_size, device=device, filter_exists=False,
                save_path=sample_final_dir / "OVERLAY_RAW_final.png",
                title=title_overlay_final_raw
            )
            title_overlay_final_filtered = f"E{epoch} Final Sample {b_idx} Overlay (Filtered)"
            visualize_projection_and_sampling(
                point_clouds=final_pc_b_for_overlay.unsqueeze(0), # Make it [1, 4, N]
                pixel_sizes=pixel_size_batch[b_idx:b_idx+1], 
                images=conditioning_image_batch[b_idx:b_idx+1],
                cell_size=cfg.data.cell_size, device=device, filter_exists=True,
                save_path=sample_final_dir / "OVERLAY_FILTERED_final.png",
                title=title_overlay_final_filtered
            )
        except Exception as e:
            log.error(f"Failed final overlay vis for E{epoch} sample {b_idx}: {e}", exc_info=True)

        # --- Combined GT and Prediction 3D Plots ---
        if True: # Hardcoded for easy toggling by changing to 'if False:'
            if gt_key_found and _PYTORCH3D_AVAILABLE and renderer and cameras_dict:
                
                # --- Plot 1: GT (Red) vs. Prediction_RAW (Blue) ---
                try:
                    # Predicted points (Blue) - RAW
                    # Use final_pc_b_for_3d which is the CoM-centered prediction
                    pred_xyz_raw_n_3 = final_pc_b_for_3d[:3, :].transpose(0, 1).contiguous() 
                    if pred_xyz_raw_n_3.shape[0] > 0:
                        pred_raw_colors_n_3 = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_raw_n_3.shape[0], 1) # Blue
                    else:
                        pred_raw_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    # Ground Truth points (Red) - Common for both plots
                    gt_pc_b_4_n = example_batch[gt_key_found][b_idx].to(device, dtype=dtype) # [4, N_gt]
                    
                    # Center the GT point cloud
                    gt_xyz_original_3_n = gt_pc_b_4_n[:3, :] # [3, N_gt]
                    gt_existence_original_1_n = gt_pc_b_4_n[3:, :] # [1, N_gt]
                    
                    # Ensure mask is correctly shaped for center_com if it expects [1,N] for single sample
                    # center_com handles [3,N] for tensor_xyz and [1,N] for mask
                    gt_xyz_centered_3_n = center_com(gt_xyz_original_3_n, mask=gt_existence_original_1_n)
                    gt_xyz_n_3 = gt_xyz_centered_3_n.transpose(0, 1).contiguous() # [N_gt, 3]

                    # Optional: Filter GT points if desired (e.g., using gt_existence_original_1_n)
                    # This filtering should happen *after* centering if based on original existence,
                    # or be applied to the mask used for centering if filtering affects CoM.
                    # For simplicity, if filtering is needed, apply to gt_xyz_n_3 and gt_existence_original_1_n before color assignment.
                    # Example:
                    # gt_exists_mask_n = gt_existence_original_1_n.squeeze() >= 0.5
                    # gt_xyz_n_3 = gt_xyz_n_3[gt_exists_mask_n]

                    if gt_xyz_n_3.shape[0] > 0:
                        gt_colors_n_3 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(gt_xyz_n_3.shape[0], 1) # Red
                    else:
                        gt_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    if pred_xyz_raw_n_3.shape[0] == 0 and gt_xyz_n_3.shape[0] == 0:
                        log.info(f"Skipping combined GT/Pred_Raw 3D plot for sample {b_idx} as both point clouds are empty.")
                    else:
                        # Handle cases where one point cloud might be empty
                        if gt_xyz_n_3.shape[0] > 0 and pred_xyz_raw_n_3.shape[0] > 0:
                            combined_verts_raw = torch.cat([gt_xyz_n_3, pred_xyz_raw_n_3], dim=0) # GT (Red) first
                            combined_colors_raw = torch.cat([gt_colors_n_3, pred_raw_colors_n_3], dim=0)
                        elif gt_xyz_n_3.shape[0] > 0:
                            combined_verts_raw = gt_xyz_n_3
                            combined_colors_raw = gt_colors_n_3
                        elif pred_xyz_raw_n_3.shape[0] > 0:
                            combined_verts_raw = pred_xyz_raw_n_3
                            combined_colors_raw = pred_raw_colors_n_3
                        else: # Should be caught by earlier check
                            combined_verts_raw = torch.empty((0,3), device=device, dtype=dtype)
                            combined_colors_raw = torch.empty((0,3), device=device, dtype=dtype)


                        if combined_verts_raw.shape[0] > 0:
                            point_cloud_combined_raw = Pointclouds(points=[combined_verts_raw], features=[combined_colors_raw])
                            rendered_views_combined_raw = []
                            view_configs_combined = [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]

                            for view_key, camera in view_configs_combined:
                                try:
                                    img_combined = renderer(point_cloud_combined_raw, cameras=camera)[0, ..., :3] 
                                    rendered_views_combined_raw.append(img_combined)
                                except Exception as e_render:
                                    log.error(f"Error rendering combined GT/Pred_Raw view '{view_key}' for sample {b_idx}: {e_render}", exc_info=True)
                                    rendered_views_combined_raw.append(torch.zeros((vis_cfg.image_size, vis_cfg.image_size, 3), dtype=dtype, device=device))
                            

                            if rendered_views_combined_raw:
                                combined_image_tensor_raw = torch.cat(rendered_views_combined_raw, dim=1) 
                                combined_image_np_raw = (combined_image_tensor_raw.cpu().numpy() * 255).astype(np.uint8)
                                
                                title_text_raw = f"E{epoch} Final S{b_idx} GT(Red)/Pred_Raw(Blue)"
                                if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                                    try:
                                        pil_img_raw = Image.fromarray(combined_image_np_raw)
                                        draw_raw = ImageDraw.Draw(pil_img_raw)
                                        font_path = cfg.visualization.get("font_path", "arial.ttf")
                                        font_main = ImageFont.truetype(font_path, 16)
                                        bbox_raw = draw_raw.textbbox((0,0), title_text_raw, font=font_main)
                                        title_w_raw = bbox_raw[2] - bbox_raw[0]
                                        draw_raw.text(((combined_image_np_raw.shape[1] - title_w_raw) // 2, 5), title_text_raw, fill=(255,255,255), font=font_main)
                                        combined_image_np_raw = np.array(pil_img_raw)
                                    except Exception as e_text: log.error(f"Error adding text to GT/Pred_Raw image for sample {b_idx}: {e_text}", exc_info=True)
                                imageio.imwrite(sample_final_dir / "3D_GT_PRED_RAW_overlay.png", combined_image_np_raw)
                        else:
                            log.info(f"Skipping combined GT/Pred_Raw 3D plot for sample {b_idx} as combined_verts_raw is empty.")
                except Exception as e_plot_raw:
                    log.error(f"Failed to generate GT/Pred_Raw 3D plot for sample {b_idx}: {e_plot_raw}", exc_info=True)

                # --- Plot 2: GT (Red) vs. Prediction_FILTERED (Blue) ---
                try:
                    # Predicted points (Blue) - FILTERED
                    # Use final_pc_b_for_3d which is the CoM-centered prediction
                    pred_exists_values = final_pc_b_for_3d[3, :] 
                    pred_mask = pred_exists_values >= 0.5  
                    pred_xyz_filtered_n_3 = final_pc_b_for_3d[:3, pred_mask].transpose(0, 1).contiguous() 
                    
                    if pred_xyz_filtered_n_3.shape[0] > 0:
                        pred_filtered_colors_n_3 = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_filtered_n_3.shape[0], 1) # Blue
                    else:
                        pred_filtered_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    # Ground Truth points (Red) - FILTERED
                    # gt_xyz_n_3 is from the RAW plot block, containing all centered GT points.
                    # gt_existence_original_1_n also from RAW plot block, contains existence values.
                    gt_exists_mask_n = gt_existence_original_1_n.squeeze() >= 0.5
                    gt_xyz_filtered_n_3 = gt_xyz_n_3[gt_exists_mask_n]

                    if gt_xyz_filtered_n_3.shape[0] > 0:
                        gt_filtered_colors_n_3 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(gt_xyz_filtered_n_3.shape[0], 1) # Red
                    else:
                        gt_filtered_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)


                    if pred_xyz_filtered_n_3.shape[0] == 0 and gt_xyz_filtered_n_3.shape[0] == 0:
                        log.info(f"Skipping combined GT/Pred_Filt 3D plot for sample {b_idx} as both point clouds are empty.")
                    else:
                        if gt_xyz_filtered_n_3.shape[0] > 0 and pred_xyz_filtered_n_3.shape[0] > 0:
                            combined_verts_filt = torch.cat([gt_xyz_filtered_n_3, pred_xyz_filtered_n_3], dim=0) # GT (Red) first
                            combined_colors_filt = torch.cat([gt_filtered_colors_n_3, pred_filtered_colors_n_3], dim=0)
                        elif gt_xyz_filtered_n_3.shape[0] > 0:
                            combined_verts_filt = gt_xyz_filtered_n_3
                            combined_colors_filt = gt_filtered_colors_n_3
                        elif pred_xyz_filtered_n_3.shape[0] > 0:
                            combined_verts_filt = pred_xyz_filtered_n_3
                            combined_colors_filt = pred_filtered_colors_n_3
                        else: # Should be caught by earlier check
                            combined_verts_filt = torch.empty((0,3), device=device, dtype=dtype)
                            combined_colors_filt = torch.empty((0,3), device=device, dtype=dtype)

                        if combined_verts_filt.shape[0] > 0:
                            point_cloud_combined_filt = Pointclouds(points=[combined_verts_filt], features=[combined_colors_filt])
                            rendered_views_combined_filt = []
                            view_configs_combined = [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]

                            for view_key, camera in view_configs_combined:
                                try:
                                    img_combined = renderer(point_cloud_combined_filt, cameras=camera)[0, ..., :3] 
                                    rendered_views_combined_filt.append(img_combined)
                                except Exception as e_render:
                                    log.error(f"Error rendering combined GT/Pred_Filt view '{view_key}' for sample {b_idx}: {e_render}", exc_info=True)
                                    rendered_views_combined_filt.append(torch.zeros((vis_cfg.image_size, vis_cfg.image_size, 3), dtype=dtype, device=device))
                            

                            if rendered_views_combined_filt:
                                combined_image_tensor_filt = torch.cat(rendered_views_combined_filt, dim=1) 
                                combined_image_np_filt = (combined_image_tensor_filt.cpu().numpy() * 255).astype(np.uint8)
                                
                                title_text_filt = f"E{epoch} Final S{b_idx} GT(Red)/Pred_Filt(Blue)"
                                if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                                    try:
                                        pil_img_filt = Image.fromarray(combined_image_np_filt)
                                        draw_filt = ImageDraw.Draw(pil_img_filt)
                                        font_path = cfg.visualization.get("font_path", "arial.ttf")
                                        font_main = ImageFont.truetype(font_path, 16)
                                        bbox_filt = draw_filt.textbbox((0,0), title_text_filt, font=font_main)
                                        title_w_filt = bbox_filt[2] - bbox_filt[0]
                                        draw_filt.text(((combined_image_np_filt.shape[1] - title_w_filt) // 2, 5), title_text_filt, fill=(255,255,255), font=font_main)
                                        combined_image_np_filt = np.array(pil_img_filt)
                                    except Exception as e_text: log.error(f"Error adding text to GT/Pred_Filt image for sample {b_idx}: {e_text}", exc_info=True)
                                imageio.imwrite(sample_final_dir / "3D_GT_PRED_FILTERED_overlay.png", combined_image_np_filt)
                        else:
                            log.info(f"Skipping combined GT/Pred_Filt 3D plot for sample {b_idx} as combined_verts_filt is empty.")
                except Exception as e_plot_filt:

                    log.error(f"Failed to generate GT/Pred_Filt 3D plot for sample {b_idx}: {e_plot_filt}", exc_info=True)
            
            elif not gt_key_found:
                log.warning(f"Skipping combined GT/Pred 3D plots for sample {b_idx}: GT key not found in example_batch.")
            # Implicitly, if _PYTORCH3D_AVAILABLE is False or renderer/cameras_dict are None, this block won't execute.

        # -- Post processing to enforce crystallinity using Lennard-Jones style potential --
        if False: # Hard-coded to always run
            # --- Hard-coded Parameters for LJ Relaxation ---
            num_steps = 1000
            equilibrium_distance = 0.03  # r_e: The distance where the potential energy is minimal.
            epsilon = 5.0                 # Depth of the potential well, scales the force magnitude.
            learning_rate = 1e-5          # Step size for updating positions based on force.
            vis_freq = 1                # Frequency to save frames for the GIF.
            # --- End of Hard-coded Parameters ---

            log.info(f"--- Starting Lennard-Jones style relaxation for E{epoch} S{sample_idx} ---")
            relaxation_output_dir = sample_final_dir / "lj_relaxation_process"
            relaxation_output_dir.mkdir(parents=True, exist_ok=True)

            # Start with the final, unrelaxed prediction
            relaxed_pc_4d = final_pc_b_for_3d.clone()
            
            exists_mask = relaxed_pc_4d[3, :] >= 0.5
            if not exists_mask.any():
                log.warning(f"Relaxation E{epoch} S{sample_idx}: No points with exists>=0.5. Skipping this sample.")
                continue # Skip to next sample in the batch

            num_points = exists_mask.sum().item()
            log.info(f"Relaxing {num_points} points with r_e={equilibrium_distance}, lr={learning_rate}.")
            
            # Work only with existing points, shape (N_exist, 3)
            xyz_current = relaxed_pc_4d[:3, exists_mask].T.clone()

            # LJ constants derived from equilibrium_distance
            sigma = equilibrium_distance / (2**(1/6))
            sigma_6 = sigma**6
            sigma_12 = sigma**12
            
            frames = []

            # Relaxation Loop
            for step in tqdm(range(num_steps), desc=f"LJ Relax E{epoch} S{sample_idx}", leave=False):
                # Calculate pairwise distances and vectors
                pdist_matrix = torch.cdist(xyz_current, xyz_current, p=2)
                
                # Avoid division by zero for self-distance
                pdist_matrix.fill_diagonal_(float('inf'))
                
                # Calculate terms for LJ force calculation
                r_inv = 1.0 / pdist_matrix
                r_6_inv = r_inv**6
                r_12_inv = r_inv**12

                # Calculate the magnitude of the LJ force for each pair
                # F(r) = 24 * epsilon / r * [2 * (sigma/r)^12 - (sigma/r)^6]
                force_magnitude = 24 * epsilon * r_inv * (2 * sigma_12 * r_12_inv - sigma_6 * r_6_inv)
                force_magnitude.fill_diagonal_(0) # No self-force

                # Calculate the Z-component of the vector between each pair of points
                z_coords = xyz_current[:, 2]
                z_diffs = z_coords.unsqueeze(1) - z_coords.unsqueeze(0) # z_i - z_j

                # Z-component of the force vector F_ij_z = F_magnitude_ij * (z_i - z_j) / r_ij
                force_z_components = force_magnitude * z_diffs * r_inv
                force_z_components.fill_diagonal_(0)

                # Total Z-force on each point is the sum of forces from all other points
                total_force_z = torch.sum(force_z_components, dim=1)

                # Update only the Z coordinates based on the calculated force
                xyz_current[:, 2] += learning_rate * total_force_z

                # Periodic Visualization for GIF
                if renderer and (step % vis_freq == 0 or step == num_steps - 1):
                    viz_pc_4d = relaxed_pc_4d.clone()
                    viz_pc_4d[:3, exists_mask] = xyz_current.T
                    
                    title = f"LJ Relax E{epoch} S{sample_idx} - Step {step+1}"
                    rendered_frame = _render_point_cloud_views(
                        pc_tensor=viz_pc_4d, renderer=renderer, cameras_dict=cameras_dict,
                        image_size=vis_cfg.image_size, device=device, title_prefix=title,
                        color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, 
                        filter_exists=True
                    )
                    if rendered_frame is not None:
                        frames.append(rendered_frame)

            if frames:
                gif_path = relaxation_output_dir / f"lj_relaxation_e{epoch}_s{sample_idx}.gif"
                imageio.mimsave(gif_path, frames, duration=100)

            # Update the main tensor with the final relaxed coordinates
            relaxed_pc_4d[:3, exists_mask] = xyz_current.T

            # --- Visualizations for the RELAXED Point Cloud ---
            log.info(f"--- Visualizing final LJ-relaxed structure for E{epoch} S{sample_idx} ---")
            
            rendered_final_3d_relaxed = _render_point_cloud_views(
                pc_tensor=relaxed_pc_4d, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=f"E{epoch} Final S{sample_idx} LJ-Relaxed",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_final_3d_relaxed is not None:
                imageio.imwrite(sample_final_dir / "3D_RELAXED_final.png", rendered_final_3d_relaxed)

            if gt_key_found:
                gt_colors_n_3 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(gt_xyz_n_3.shape[0], 1)
                
                pred_exists_mask_relaxed = relaxed_pc_4d[3, :] >= 0.5
                pred_xyz_relaxed_filt_n_3 = relaxed_pc_4d[:3, pred_exists_mask_relaxed].T.contiguous()
                pred_relaxed_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_relaxed_filt_n_3.shape[0], 1)

                if gt_xyz_n_3.shape[0] > 0 and pred_xyz_relaxed_filt_n_3.shape[0] > 0:
                    combined_verts = torch.cat([gt_xyz_n_3, pred_xyz_relaxed_filt_n_3], dim=0)
                    combined_colors = torch.cat([gt_colors_n_3, pred_relaxed_colors], dim=0)
                    
                    point_cloud_combined = Pointclouds(points=[combined_verts], features=[combined_colors])
                    rendered_views_list = []
                    for _, camera in [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]:
                        img_v = renderer(point_cloud_combined, cameras=camera)[0, ..., :3]
                        rendered_views_list.append(img_v)
                    
                    combined_image_np = (torch.cat(rendered_views_list, dim=1).cpu().numpy() * 255).astype(np.uint8)
                    # Add title to the combined relaxed image
                    title_text_relaxed = f"E{epoch} Final S{sample_idx} GT(Red)/Pred_Relaxed(Blue)"
                    if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                        try:
                            pil_img = Image.fromarray(combined_image_np)
                            draw = ImageDraw.Draw(pil_img)
                            font_path = cfg.visualization.get("font_path", "arial.ttf")
                            font_main = ImageFont.truetype(font_path, 16)
                            bbox = draw.textbbox((0,0), title_text_relaxed, font=font_main)
                            title_w = bbox[2] - bbox[0]
                            draw.text(((combined_image_np.shape[1] - title_w) // 2, 5), title_text_relaxed, fill=(255,255,255), font=font_main)
                            combined_image_np = np.array(pil_img)
                        except Exception as e_text: log.error(f"Error adding text to GT/Pred_Relaxed image for sample {b_idx}: {e_text}", exc_info=True)
                    imageio.imwrite(sample_final_dir / "3D_GT_PRED_RELAXED_overlay.png", combined_image_np)


    log.info(f"--- Finished conditioned visualization loop for epoch {epoch} ---")
    model.train()
    projection_model.train()
    if size_estimator: size_estimator.train()


def debug_visualize_scheduler(
    ground_truth_point_cloud: torch.Tensor, # Should be a single PC: [C, N]
    noise_scheduler,
    device: torch.device,
    cfg: DictConfig,
):
    """
    Visualizes how the scheduler modifies a single ground truth point cloud for selected timesteps.
    Saves a configured number of figures, including the first and last scheduler timesteps.
    Can optionally render the original GT for timestep t=0.
    """
    # Ensure the output directory exists
    debug_dir = Path(cfg.visualization.subdir) / "scheduler_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving scheduler debug visualizations to: {debug_dir}")

    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=cfg.visualization.image_size,
        point_radius=cfg.visualization.point_radius,
        points_per_pixel=cfg.visualization.points_per_pixel,
        device=device,
    )
    if renderer is None or cameras_dict is None:
        log.warning("Renderer setup failed. Skipping scheduler visualization.")
        return

    gt_pc_single = ground_truth_point_cloud.to(device)
    if gt_pc_single.ndim == 3 and gt_pc_single.shape[0] == 1: # If [1,C,N] passed
        gt_pc_single = gt_pc_single[0]
    elif gt_pc_single.ndim != 2:
        log.error(f"Scheduler debug expects [C,N] or [1,C,N] GT, got {ground_truth_point_cloud.shape}. Skipping.")
        return

    # Determine timesteps to visualize
    num_inference_steps_for_scheduler = cfg.visualization.get("num_inference_steps_scheduler_debug", cfg.visualization.num_inference_steps)
    noise_scheduler.set_timesteps(num_inference_steps_for_scheduler)
    
    all_scheduler_timesteps = noise_scheduler.timesteps # Typically a 1D tensor
    if all_scheduler_timesteps is None or len(all_scheduler_timesteps) == 0:
        log.warning("Scheduler reported 0 timesteps. Skipping scheduler visualization.")
        return
        
    num_total_scheduler_steps = len(all_scheduler_timesteps)
    num_images_to_save = cfg.visualization.get("num_scheduler_debug_images", 100)
    force_gt_at_t0 = cfg.visualization.get("force_gt_at_scheduler_t0", True)

    selected_indices = []
    if num_total_scheduler_steps <= num_images_to_save:
        selected_indices = np.arange(num_total_scheduler_steps)
    else:
        # Ensure first and last steps are included, and sample in between
        selected_indices = np.linspace(0, num_total_scheduler_steps - 1, num_images_to_save, dtype=int)
    
    # Ensure indices are unique (esp. if num_images_to_save is close to num_total_scheduler_steps)
    selected_indices = np.unique(selected_indices) 
    
    # Handle tensor or list for all_scheduler_timesteps before indexing
    if isinstance(all_scheduler_timesteps, torch.Tensor):
        # Ensure selected_indices can correctly index the tensor (e.g., convert to LongTensor or list)
        # If all_scheduler_timesteps is 1D, numpy int array should work directly for indexing.
        try:
            timesteps_to_visualize = all_scheduler_timesteps[selected_indices]
        except IndexError: # Fallback if numpy array indexing fails for some tensor types
            all_scheduler_timesteps_list = all_scheduler_timesteps.cpu().tolist()
            timesteps_to_visualize = [all_scheduler_timesteps_list[i] for i in selected_indices]
            if isinstance(all_scheduler_timesteps, torch.Tensor): # Convert back to tensor if original was tensor
                 timesteps_to_visualize = torch.tensor(timesteps_to_visualize, device=all_scheduler_timesteps.device, dtype=all_scheduler_timesteps.dtype)

    elif isinstance(all_scheduler_timesteps, list):
        timesteps_to_visualize = [all_scheduler_timesteps[i] for i in selected_indices]
    else:
        log.error(f"Unsupported type for noise_scheduler.timesteps: {type(all_scheduler_timesteps)}. Skipping.")
        return

    log.info(f"Selected {len(timesteps_to_visualize)} timesteps for scheduler visualization (target: {num_images_to_save}).")

    timesteps_iterable = tqdm(timesteps_to_visualize, 
                              desc=f"Visualizing {len(timesteps_to_visualize)} Scheduler Timesteps", 
                              disable=not cfg.visualization.get("tqdm_steps", True))
                              
    for t_val in timesteps_iterable: # t_val is usually a tensor scalar or Python number
        # Ensure t_val is a tensor for scheduler's add_noise if it expects it
        current_t_tensor = t_val if isinstance(t_val, torch.Tensor) else torch.tensor(t_val, device=device)
        # Ensure t is at least 1D for add_noise, and on the correct device
        t_for_add_noise = current_t_tensor.unsqueeze(0).to(device) if current_t_tensor.ndim == 0 else current_t_tensor.to(device)
        if t_for_add_noise.ndim == 0: # some schedulers might take scalar tensor, others 1D
             t_for_add_noise = t_for_add_noise.unsqueeze(0)


        pc_to_render_this_step = None
        title_suffix = ""
        t_val_item = current_t_tensor.item() # For naming and conditions

        # Check if this is the t=0 step and if we should force GT
        if force_gt_at_t0 and abs(t_val_item - 0.0) < 1e-6 : # Compare float t_val_item to 0
            pc_to_render_this_step = gt_pc_single
            title_suffix = " (Forced GT at t=0)"
            log.debug(f"Scheduler Vis: Forcing GT for timestep t={t_val_item}")
        else:
            noise = torch.randn_like(gt_pc_single)
            # add_noise might expect batch for samples, so unsqueeze gt_pc_single and then squeeze result
            noisy_point_cloud_batch = noise_scheduler.add_noise(gt_pc_single.unsqueeze(0), noise.unsqueeze(0), t_for_add_noise)
            pc_to_render_this_step = noisy_point_cloud_batch[0]

        rendered_image_np = _render_point_cloud_views(
            pc_tensor=pc_to_render_this_step,
            renderer=renderer, cameras_dict=cameras_dict,
            image_size=cfg.visualization.image_size, device=device,
            title_prefix=f"Scheduler Timestep {t_val_item:.2f}{title_suffix}", # Format t_val_item
            color_range=cfg.visualization.get("color_range", (-1.0, 1.0)), cfg=cfg,
            filter_exists=cfg.visualization.get("filter_scheduler_debug_vis", False)
        )

        if rendered_image_np is not None:
            # Format t_val_item for filename to handle potential floats and ensure ordering
            # Using a fixed number of digits, perhaps based on max timestep value or just integer part
            img_path = debug_dir / f"timestep_{int(round(t_val_item)):04d}.png"
            try:
                imageio.imwrite(img_path, rendered_image_np)
            except Exception as e:
                log.error(f"Failed to save scheduler vis for timestep {t_val_item}: {e}")
    log.info(f"--- Finished scheduler debug visualization ---")

def visualize_projection_and_sampling(
    point_clouds: torch.Tensor,   # Expected: [1, 4, N]
    pixel_sizes: torch.Tensor,    # Expected: [1]
    images: torch.Tensor,         # Expected: [1, C_img, H, W]
    cell_size: float,
    save_path: Path,
    title: str,
    device: torch.device,
    filter_exists: bool = False
):
    """
    Projects 4D points to pixel coords, samples image, overlays points.
    Assumes inputs are for a single sample, batched to size 1.
    Dimensions are (x, y, z, exists) for point_clouds.
    """
    # --- Input Validation and Preparation ---
    if not (point_clouds.ndim == 3 and point_clouds.shape[0] == 1 and point_clouds.shape[1] == 4):
         log.warning(f"Projection vis expects point_clouds [1, 4, N], got {point_clouds.shape}. Attempting to proceed.")
    if not (images.ndim == 4 and images.shape[0] == 1): # C_img can be > 1
         log.warning(f"Projection vis expects images [1, C_img, H, W], got {images.shape}. Attempting to proceed.")
    if not (pixel_sizes.ndim == 1 and pixel_sizes.shape[0] == 1):
         log.warning(f"Projection vis expects pixel_sizes [1], got {pixel_sizes.shape}. Attempting to proceed.")

    save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    IMAGE_H, IMAGE_W = images.shape[2], images.shape[3]

    # Extract single sample data, move to device
    pc_item = point_clouds[0].to(device)      # [4, N_original]
    px_sz_item = pixel_sizes[0].to(device)    # scalar
    img_item_batch = images.to(device)        # [1, C_img, H, W]

    # --- Apply Filtering (if enabled) ---
    pc_to_plot = pc_item # [4, N_current]
    if filter_exists:
        exists_values = pc_item[3, :] # [N_original]
        mask = exists_values >= 0.5   # boolean mask
        pc_to_plot = pc_item[:, mask] # [4, N_filtered]

    if pc_to_plot.shape[1] == 0:
        log.warning(f"No points to plot for '{title}' (shape after filter: {pc_to_plot.shape}). Skipping plot.")
        # Create a placeholder image indicating no points? Or just skip.
        # For now, skip by returning.
        # To create placeholder:
        # fig, ax = plt.subplots()
        # ax.text(0.5, 0.5, "No points to display", ha='center', va='center')
        # ax.set_title(title)
        # plt.savefig(save_path, dpi=100); plt.close(fig)
        return

    # --- Coordinate Transformation and Grid Sampling ---
    # 1) Normalized coords [-1,1] -> nm -> pixel coordinates
    # pc_to_plot[:2] are x,y coordinates in normalized range [-1,1]
    pc_xy_nm = pc_to_plot[:2] * (cell_size / 2.0)     # [2, N_current] in nm
    
    # Convert nm to pixel coordinates. Origin (0,0) in pixel space is top-left.
    # X (width dimension) mapping:
    pc_pix_x = pc_xy_nm[0] / px_sz_item + (IMAGE_W / 2.0) # [N_current] (cols)
    # Y (height dimension) mapping:
    pc_pix_y = pc_xy_nm[1] / px_sz_item + (IMAGE_H / 2.0) # [N_current] (rows)
    
    xs_plot, ys_plot = pc_pix_x, pc_pix_y             # Each [N_current]

    # 2) Build grid in [-1,1] for F.grid_sample
    # F.grid_sample expects (x,y) where x is width dim, y is height dim.
    # Input image to grid_sample is [B,C,H,W]. Grid is [B,H_out,W_out,2] or [B,1,N_points,2].
    # Grid coords: x from -1 (left) to 1 (right), y from -1 (top) to 1 (bottom).
    grid_x_norm = xs_plot / (IMAGE_W - 1) * 2.0 - 1.0    # [N_current] map to [-1,1] for W
    grid_y_norm = ys_plot / (IMAGE_H - 1) * 2.0 - 1.0    # [N_current] map to [-1,1] for H
    
    grid_for_sampling = torch.stack([grid_x_norm, grid_y_norm], dim=-1) # [N_current, 2]
    grid_for_sampling = grid_for_sampling.unsqueeze(0).unsqueeze(1)     # [1, 1, N_current, 2]

    # 3) Sample the image intensity at projected points
    # Use the first channel of the image if multiple (e.g. for RGB, sample R)
    sampled_intensities = F.grid_sample(
        img_item_batch[:, 0:1, :, :],  # Ensure [1, 1, H, W] for single channel sampling
        grid_for_sampling,
        mode='bilinear',
        padding_mode='zeros', # 'border' might also be useful
        align_corners=True    # Often set to True, consistent with pc_to_pix mapping
    ) # Output: [1, 1, 1, N_current]
    sampled_intensities_np = sampled_intensities.squeeze().cpu().numpy() # [N_current]

    # --- Plotting ---
    # Base image for display (first channel, e.g., grayscale)
    base_display_img = img_item_batch[0, 0].cpu().numpy() # [H, W]
    
    # Z-coordinates for coloring (optional, not used in current scatter plot color)
    # depth_values_np = pc_to_plot[2].cpu().numpy() # [N_current]

    fig_h = 10
    fig_w = fig_h * (IMAGE_W / IMAGE_H) if IMAGE_H > 0 else fig_h
    plt.figure(figsize=(fig_w, fig_h))
    
    # Display the image. 'origin=upper' means (0,0) is top-left.
    # 'extent' maps pixel boundaries to data coords: (left, right, bottom, top)
    plt.imshow(base_display_img, cmap='gray', origin='upper', extent=(0, IMAGE_W, IMAGE_H, 0))

    # Scatter plot: xs_plot (cols) vs ys_plot (rows)
    # For imshow with origin='upper', y increases downwards. Scatter default y increases upwards.
    # By setting ylim(IMAGE_H, 0), scatter plot y-axis also increases downwards.
    plt.scatter(
        xs_plot.cpu().numpy(), 
        ys_plot.cpu().numpy(), 
        s=15,  # Increased size slightly
        c="lime", # Changed color for better visibility on gray
        alpha=0.6, 
        edgecolors='black', linewidths=0.5, # Add edge for clarity
        label=f'Projected Points ({pc_to_plot.shape[1]})'
    )
    # Example: color by sampled intensity (if desired)
    # plt.scatter(xs_plot.cpu().numpy(), ys_plot.cpu().numpy(), s=15, c=sampled_intensities_np, cmap='viridis', alpha=0.6, label='Projected (Intensity Color)')


    plt.xlim(0, IMAGE_W)
    plt.ylim(IMAGE_H, 0) # Y-axis: 0 at top, H at bottom
    plt.axis('off')      # Hide axes ticks and labels
    plt.title(title, fontsize=12)
    plt.legend(loc="upper right", fontsize=8, frameon=True, facecolor='white', framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight') # dpi can be adjusted
    plt.close(plt.gcf()) # Close the figure to free memory


def simple_crystallinity_relaxation(
    predicted_x0: torch.Tensor,
    cfg: DictConfig,
    output_dir: Path,
    epoch: int,
    sample_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs a simple, rule-based relaxation to enforce local crystallinity.

    The process involves iterative steps where for each existing point:
    1. Find its nearest neighbor.
    2. If the neighbor is closer than a threshold:
       - If the neighbor has a larger Z value, move the current point down (negative Z).
       - If the neighbor has a smaller Z value, move the current point up (positive Z).
    3. XY coordinates are fixed.
    4. The process is visualized and saved as a GIF.

    Args:
        predicted_x0 (torch.Tensor): The predicted point cloud, shape [1, 4, N].
        cfg (DictConfig): Config, containing relaxation parameters.
        output_dir (Path): Directory to save the output GIF.
        epoch (int): Current epoch number for titling.
        sample_idx (int): Current sample index for titling.
        device (torch.device): The device to run computations on.

    Returns:
        torch.Tensor: The relaxed point cloud, shape [1, 4, N].
    """
    # Use a new config block for this specific relaxation method
    relax_cfg = cfg.visualization.get("simple_relaxation", {})
    if not relax_cfg.get("enabled", False):
        return predicted_x0 # Return original if not enabled

    log.info(f"--- Starting SIMPLE crystallinity relaxation for E{epoch} S{sample_idx} ---")
    
    # --- 1. Parameter and Data Setup ---
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_steps = relax_cfg.get("num_steps", 100)
    min_dist = relax_cfg.get("min_neighbor_distance", 0.08)
    z_step = relax_cfg.get("z_step", 0.01) # The fixed step size
    vis_freq = relax_cfg.get("vis_frequency", 5)

    if not (predicted_x0.ndim == 3 and predicted_x0.shape[0] == 1):
        log.error(f"Relaxation expects a single sample [1, 4, N], got {predicted_x0.shape}. Skipping.")
        return predicted_x0
    
    pc_4d = predicted_x0.squeeze(0).clone() # Shape (4, N)
    exists_mask = pc_4d[3, :] >= 0.5
    
    if not exists_mask.any():
        log.warning(f"Relaxation E{epoch} S{sample_idx}: No points with exists>=0.5. Returning original.")
        return predicted_x0
    
    log.info(f"Relaxing {exists_mask.sum().item()} points with z_step={z_step}, min_dist={min_dist}.")

    # Work only with existing points. Transpose to (N_exist, 3)
    xyz_current = pc_4d[:3, exists_mask].T.clone()

    # --- 2. Visualization Setup ---
    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=cfg.visualization.image_size,
        point_radius=cfg.visualization.point_radius,
        points_per_pixel=cfg.visualization.points_per_pixel,
        device=device
    )
    frames = []
    
    # --- 3. Relaxation Loop ---
    for step in tqdm(range(num_steps), desc=f"Simple Relax E{epoch} S{sample_idx}", leave=False):
        # Calculate pairwise distances between all existing points
        pdist = torch.cdist(xyz_current, xyz_current, p=2)
        
        # To find the nearest neighbor, ignore self-distance by filling diagonal with infinity
        pdist.fill_diagonal_(float('inf'))
        
        # Find the minimum distance and the index of the nearest neighbor for each point
        min_dists, nn_indices = torch.min(pdist, dim=1)
        
        # Get Z coordinates for all points and their nearest neighbors
        z_coords = xyz_current[:, 2]
        z_neighbors = z_coords[nn_indices]

        # --- Determine which points to move ---
        # Create a tensor to hold the z-updates for this step, initialized to zero
        z_updates = torch.zeros_like(z_coords)

        # Condition 1: The nearest neighbor must be closer than the threshold
        too_close_mask = min_dists < min_dist
        
        # Rule 1: If too close AND neighbor is higher, move this point DOWN
        move_down_mask = too_close_mask & (z_neighbors > z_coords)
        z_updates[move_down_mask] = -z_step
        
        # Rule 2: If too close AND neighbor is lower, move this point UP
        move_up_mask = too_close_mask & (z_neighbors < z_coords)
        z_updates[move_up_mask] = z_step

        # Apply the updates to the Z coordinates
        xyz_current[:, 2] += z_updates
        
        # --- 4. Periodic Visualization ---
        if renderer and (step % vis_freq == 0 or step == num_steps - 1):
            viz_pc_4d = torch.zeros_like(pc_4d)
            viz_pc_4d[:, ~exists_mask] = pc_4d[:, ~exists_mask] # Put back non-existing points
            viz_pc_4d[:3, exists_mask] = xyz_current.T          # Put in the current relaxed points
            viz_pc_4d[3, exists_mask] = 1.0                     # Mark them as existing
            
            title = f"Simple Relax E{epoch} S{sample_idx} - Step {step+1}"
            rendered_frame = _render_point_cloud_views(
                pc_tensor=viz_pc_4d, renderer=renderer, cameras_dict=cameras_dict,
                image_size=cfg.visualization.image_size, device=device, title_prefix=title,
                color_range=cfg.visualization.get("color_range", (-1.0, 1.0)), cfg=cfg, 
                filter_exists=True
            )
            if rendered_frame is not None:
                frames.append(rendered_frame)

    # --- 5. Finalization ---
    if frames:
        gif_path = output_dir / f"simple_relaxation_e{epoch}_s{sample_idx}.gif"
        log.info(f"Saving simple relaxation GIF to {gif_path}")
        imageio.mimsave(gif_path, frames, duration=100)

    # Reconstruct the final full point cloud
    final_pc_4d = pc_4d.clone()
    final_pc_4d[:3, exists_mask] = xyz_current.T
    
    log.info(f"--- Finished simple crystallinity relaxation for E{epoch} S{sample_idx} ---")
    return final_pc_4d.unsqueeze(0)


def _render_and_save_views(point_cloud, renderer, cameras_dict, save_path, title, cfg):
    """Helper to render a Pointclouds object from 3 views, add a title, and save."""
    if point_cloud is None or point_cloud.isempty():
        log.warning(f"Skipping render for '{title}' as point cloud is empty.")
        return

    try:
        rendered_views = []
        view_configs = [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]
        for view_key, camera in view_configs:
            img = renderer(point_cloud, cameras=camera)[0, ..., :3]
            rendered_views.append(img)
        
        if not rendered_views:
            return

        combined_image_tensor = torch.cat(rendered_views, dim=1)
        combined_image_np = (combined_image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
            try:
                pil_img = Image.fromarray(combined_image_np)
                draw = ImageDraw.Draw(pil_img)
                font_path = cfg.visualization.get("font_path", "arial.ttf")
                font = ImageFont.truetype(font_path, 16)
                bbox = draw.textbbox((0, 0), title, font=font)
                title_w = bbox[2] - bbox[0]
                draw.text(((combined_image_np.shape[1] - title_w) // 2, 5), title, fill=(255, 255, 255), font=font)
                combined_image_np = np.array(pil_img)
            except Exception as e_text:
                log.error(f"Error adding text to image for '{title}': {e_text}")

        imageio.imwrite(save_path, combined_image_np)
    except Exception as e:
        log.error(f"Failed during _render_and_save_views for '{title}': {e}", exc_info=True)


def visualize_refinement(model, projection_model, vis_batch, epoch, accelerator, cfg):
    """
    Visualizes the output of the refinement model.
    Saves separate images for GT, Prediction, and an overlay of both.
    """
    model.eval()
    projection_model.eval()

    # --- 1. Get Data and Run Model (Corrected Logic) ---
    noisy_structures = vis_batch["coarse_point_cloud"].to(accelerator.device)
    gt_structures = vis_batch["point_cloud"].to(accelerator.device)
    pixel_sizes = vis_batch['pixel_size'].to(accelerator.device)
    images = vis_batch['image'].to(accelerator.device)
    CoM = vis_batch['CoM'].to(accelerator.device)
    current_batch_size = gt_structures.shape[0]

    noisy_xyz = noisy_structures[:, :3, :]
    existence_noisy = noisy_structures[:, 3:, :]
    timesteps = torch.zeros(current_batch_size, device=accelerator.device, dtype=torch.long)

    # Center coarse point cloud on its own CoM for model input
    noisy_xyz_centered_for_input = center_com(noisy_xyz, existence_noisy)

    # Create the 4D point cloud input for conditioning, re-introducing the image CoM
    zeros_for_z = torch.zeros(current_batch_size, 1, device=CoM.device, dtype=CoM.dtype)
    im_CoM = torch.cat([CoM, zeros_for_z], dim=1).unsqueeze(-1)
    noisy_point_cloud_input_for_conditioning = noisy_xyz_centered_for_input + im_CoM
    
    # Use the coarse existence mask for the input
    noisy_point_clouds_input = torch.cat((noisy_point_cloud_input_for_conditioning, existence_noisy), dim=1)

    # Get conditioning features
    conditioned_point_clouds = condition_point_cloud(
        noisy_point_clouds_input, pixel_sizes, images, projection_model, accelerator.device, cfg
    )
    # Re-center before feeding to the diffusion model
    conditioned_point_clouds[:, :3, :] -= im_CoM

    with torch.no_grad():
        # Model predicts the refined XYZ coordinates
        pred_xyz_batch = model(conditioned_point_clouds, timesteps, pixel_sizes, None)

    # --- 2. Setup Visualization ---
    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=cfg.visualization.image_size,
        point_radius=cfg.visualization.point_radius,
        points_per_pixel=cfg.visualization.points_per_pixel,
        device=accelerator.device
    )
    if renderer is None or cameras_dict is None:
        log.warning("Renderer setup failed. Skipping refinement visualization.")
        return

    vis_output_dir = Path(cfg.visualization.subdir) / f"epoch_{epoch}" / "refinement"
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    dtype = pred_xyz_batch.dtype

    # --- 3. Visualization Loop ---
    for b_idx in range(current_batch_size):
        try:
            # --- Save Conditioning Image ---
            try:
                conditioning_image_single_tensor = images[b_idx] # Already on device
                cond_img_np = conditioning_image_single_tensor[0].detach().cpu().numpy()
                cond_img_np = np.flip(cond_img_np, axis=0)
                cond_img_path = vis_output_dir / f"sample_{b_idx}_conditioning_image.png"
                img_min, img_max = cond_img_np.min(), cond_img_np.max()
                norm_img = ((cond_img_np - img_min) / (img_max - img_min + 1e-6)) * 255.0 if img_min != img_max else np.ones_like(cond_img_np) * 127.5
                imageio.imwrite(cond_img_path, norm_img.astype(np.uint8))
            except Exception as e:
                log.error(f"Failed to save conditioning image for refinement sample {b_idx}: {e}", exc_info=True)

            # --- Prepare GT points (Red) ---
            gt_pc_4d = gt_structures[b_idx]
            gt_exists_mask = gt_pc_4d[3, :] >= 0.5
            if not gt_exists_mask.any():
                log.warning(f"Skipping refinement vis for sample {b_idx}: no valid GT points.")
                continue
            
            gt_xyz_centered = center_com(gt_pc_4d[:3, :].unsqueeze(0), mask=gt_exists_mask.unsqueeze(0).unsqueeze(0)).squeeze(0)
            gt_xyz_filtered = gt_xyz_centered[:, gt_exists_mask].transpose(0, 1).contiguous()
            gt_colors = torch.tensor([[1.0, 0.0, 0.0]], device=accelerator.device, dtype=dtype).repeat(gt_xyz_filtered.shape[0], 1)
            point_cloud_gt = Pointclouds(points=[gt_xyz_filtered], features=[gt_colors])

            # --- Prepare Predicted points (Blue) ---
            pred_xyz_3d = pred_xyz_batch[b_idx]
            # Use the existence mask from the *coarse input* for the prediction
            pred_exists_mask = existence_noisy[b_idx, 0, :] >= 0.5
            if not pred_exists_mask.any():
                log.warning(f"Skipping refinement vis for sample {b_idx}: no valid predicted points.")
                continue
            # pred_x0 is the refined point cloud prediction
            pred_x0 = noisy_xyz_centered_for_input[b_idx, :3, :] - pred_xyz_3d
            
            # Filter points based on the coarse existence mask
            pred_xyz_filtered_3_n = pred_x0[:, pred_exists_mask]
            
            # Center the filtered points and prepare for visualization
            pred_xyz_centered_3_n = center_com(pred_xyz_filtered_3_n) # center_com expects [3, N]
            pred_xyz_filtered = pred_xyz_centered_3_n.transpose(0, 1).contiguous() # Transpose to [N, 3] for Pointclouds

            pred_colors = torch.tensor([[0.0, 0.0, 1.0]], device=accelerator.device, dtype=dtype).repeat(pred_xyz_filtered.shape[0], 1)
            point_cloud_pred = Pointclouds(points=[pred_xyz_filtered], features=[pred_colors])


            # --- Render and Save Individual Images ---
            _render_and_save_views(
                point_cloud_gt, renderer, cameras_dict,
                save_path=vis_output_dir / f"sample_{b_idx}_refinement_gt.png",
                title=f"E{epoch} Refinement S{b_idx} GT (Red)", cfg=cfg
            )
            _render_and_save_views(
                point_cloud_pred, renderer, cameras_dict,
                save_path=vis_output_dir / f"sample_{b_idx}_refinement_prediction.png",
                title=f"E{epoch} Refinement S{b_idx} Prediction (Blue)", cfg=cfg
            )

            # --- Render and Save Overlay Image ---
            combined_verts = torch.cat([gt_xyz_filtered, pred_xyz_filtered], dim=0)
            combined_colors = torch.cat([gt_colors, pred_colors], dim=0)
            point_cloud_combined = Pointclouds(points=[combined_verts], features=[combined_colors])
            _render_and_save_views(
                point_cloud_combined, renderer, cameras_dict,
                save_path=vis_output_dir / f"sample_{b_idx}_refinement_overlay.png",
                title=f"E{epoch} Refinement S{b_idx} GT(Red)/Pred(Blue)", cfg=cfg
            )

        except Exception as e:
            log.error(f"Failed to visualize refinement for sample {b_idx}: {e}", exc_info=True)

    log.info(f"Finished refinement visualization for epoch {epoch}. Saved to {vis_output_dir}")
    model.train()
    projection_model.train()


@torch.no_grad()
def visualize_conditioned_4d(
    model: nn.Module,
    projection_model: nn.Module,
    noise_scheduler: Any,
    example_batch: Dict[str, torch.Tensor],
    epoch: int,
    device: torch.device,
    cfg: DictConfig,
    size_estimator: Optional[nn.Module] = None,
):
    """
    Visualizes conditioned diffusion model generation.
    The 4th channel of the input to conditioning is the GT binary existence mask.
    Noise is added to and predicted for only the XYZ channels.
    """
    if not cfg.visualization.enabled: 
        log.info(f"Conditioned visualization for epoch {epoch} disabled via cfg.visualization.enabled=False.")
        return
    if not _PYTORCH3D_AVAILABLE: 
        log.warning(f"Skipping conditioned visualization for epoch {epoch} due to missing PyTorch3D or dependencies.")
        return

    # Use the original noise_scheduler for visualization
    effective_scheduler = noise_scheduler
    scheduler_name_for_log = type(noise_scheduler).__name__
    log.info(f"Visualization (conditioned, epoch {epoch}): Using original scheduler: {scheduler_name_for_log}.")
    vis_cfg = cfg.visualization
    max_samples_to_vis_from_batch = vis_cfg.num_samples_to_vis
    
    try:
        actual_batch_size_for_vis = min(max_samples_to_vis_from_batch, example_batch["image"].shape[0])
        if actual_batch_size_for_vis == 0 and max_samples_to_vis_from_batch > 0:
            log.warning("num_samples_to_vis > 0 but example_batch is empty or 'image' key missing. Skipping conditioned visualization.")
            return
        if actual_batch_size_for_vis == 0 :
             return 
    except KeyError:
        log.error("Conditioned visualization: 'image' key missing in example_batch. Skipping.")
        return

    # Get and validate the index for intermediate visualization
    intermediate_vis_idx = vis_cfg.get("intermediate_vis_sample_idx", 0)
    if not (0 <= intermediate_vis_idx < actual_batch_size_for_vis):
        if actual_batch_size_for_vis > 0: # Only warn if there are samples to visualize
            log.warning(
                f"Configured intermediate_vis_sample_idx {intermediate_vis_idx} is out of bounds "
                f"for actual_batch_size_for_vis ({actual_batch_size_for_vis}). Defaulting to 0."
            )
            intermediate_vis_idx = 0
        # If actual_batch_size_for_vis is 0, intermediate_vis_idx is moot as intermediate saving will be skipped.
        # Setting to 0 here to prevent potential downstream issues if logic changes, though current flow exits.
        elif actual_batch_size_for_vis == 0:
            intermediate_vis_idx = 0


    renderer, cameras_dict = _setup_pytorch3d_renderer(
        image_size=vis_cfg.image_size,
        point_radius=vis_cfg.point_radius,
        points_per_pixel=vis_cfg.points_per_pixel,
        device=device
    )
    if renderer is None or cameras_dict is None:
        log.warning(f"Renderer setup failed for conditioned vis epoch {epoch}. Skipping.")
        return

    base_vis_output_dir = Path(vis_cfg.subdir) / f"epoch_{epoch}" / "conditioned"
    try:
        base_vis_output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving conditioned visualizations for epoch {epoch} to: {base_vis_output_dir}")
    except OSError as e:
        log.error(f"Could not create base conditioned visualization directory {base_vis_output_dir}: {e}. Skipping.")
        return

    gt_key_found = None
    for key_candidate in ("target", "points", "point_cloud", "pc"): # "point_cloud" is primary
        if key_candidate in example_batch and example_batch[key_candidate].shape[1] == 4: # Ensure it's 4-channel
            gt_key_found = key_candidate
            break
    if gt_key_found:
        for b_idx in range(actual_batch_size_for_vis):
            try:
                gt_pc_original = example_batch[gt_key_found][b_idx].to(device, dtype=torch.float32) # [4, N] or [N, 4]

                rendered_gt = _render_point_cloud_views(
                    pc_tensor=gt_pc_original, renderer=renderer, cameras_dict=cameras_dict,
                    image_size=vis_cfg.image_size, device=device,
                    title_prefix=f"E{epoch} Sample {b_idx} GT (CoM Centered)",
                    color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg,
                    filter_exists=vis_cfg.get("filter_gt_vis", True) # GT has 'exists', allow filtering
                )
                if rendered_gt is not None:
                    gt_img_path = base_vis_output_dir / f"sample_{b_idx}_gt_point_cloud.png"
                    imageio.imwrite(gt_img_path, rendered_gt)
            except Exception as e:
                log.error(f"Failed to save GT PC visualization for sample {b_idx}: {e}", exc_info=True)
    else:
        log.warning(f"No suitable 4-channel GT point cloud key found in batch for visualization: {list(example_batch.keys())}")

    for b_idx in range(actual_batch_size_for_vis):
        try:
            conditioning_image_single_tensor = example_batch["image"][b_idx].to(device)
            cond_img_np = conditioning_image_single_tensor[0].detach().cpu().numpy() 
            cond_img_np = np.flip(cond_img_np, axis=0) 
            cond_img_path = base_vis_output_dir / f"sample_{b_idx}_conditioning_image.png"
            img_min, img_max = cond_img_np.min(), cond_img_np.max()
            norm_img = ((cond_img_np - img_min) / (img_max - img_min + 1e-6)) * 255.0 if img_min != img_max else np.ones_like(cond_img_np) * 127.5
            imageio.imwrite(cond_img_path, norm_img.astype(np.uint8))
        except Exception as e:
            log.error(f"Failed to save conditioning image for sample {b_idx}: {e}", exc_info=True)
            
    # --- Prepare for Sampling ---
    model.eval()
    projection_model.eval()
    if size_estimator: 
        size_estimator.eval()

    points = cfg.data.point_cloud_size
    dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.float32

    # Prepare initial sample_batch: noisy XYZ + clean existence
    # Ensure "point_cloud" key exists and has 4 channels
    if "point_cloud" not in example_batch or example_batch["point_cloud"].shape[1] != 4:
        log.error(f"Conditioned visualization: 'point_cloud' key missing in example_batch or not 4-channel. Shape: {example_batch.get('point_cloud', torch.empty(0)).shape}. Skipping.")
        model.train() # Restore train mode
        projection_model.train() # Restore train mode if it was changed
        if size_estimator: size_estimator.train()
        return
        
    sample_batch = torch.randn((actual_batch_size_for_vis, 4, points), device=device, dtype=dtype) # [B, 4, N]
    conditioning_image_batch = example_batch["image"][:actual_batch_size_for_vis].to(device)
    pixel_size_batch = example_batch["pixel_size"][:actual_batch_size_for_vis].to(device)
    size_embedding_batch = None
    if size_estimator:
        with torch.no_grad():
            size_embedding_batch = size_estimator(conditioning_image_batch)
    
    # Optimize inference steps based on scheduler type for efficiency
    def get_optimal_steps_for_scheduler(scheduler, default_steps):
        """Get optimal number of inference steps based on scheduler type."""
        scheduler_name = type(scheduler).__name__

        if 'DPMSolverMultistepScheduler' == scheduler_name: # DPM++
            return min(default_steps, 20)
        elif 'DDIMScheduler' == scheduler_name: # DDIM
            return min(default_steps, 50)
        elif 'DDPMScheduler' == scheduler_name: # DDPM
            return default_steps
        else:
            # Fallback for unknown schedulers or those not explicitly listed
            log.warning(
                f"Scheduler '{scheduler_name}' does not have a specific step optimization rule in get_optimal_steps_for_scheduler. "
                f"Applying a general fallback: min({default_steps}, 100) steps."
            )
            return min(default_steps, 100)

    base_inference_steps = vis_cfg.num_inference_steps
    num_inference_steps = get_optimal_steps_for_scheduler(effective_scheduler, base_inference_steps)
      # Log optimization information
    if num_inference_steps != base_inference_steps:
        log.info(f"Optimized inference steps for {type(effective_scheduler).__name__}: {base_inference_steps} -> {num_inference_steps} steps")

    effective_scheduler.set_timesteps(num_inference_steps) # Use effective_scheduler here
    log.info(f"--- Epoch {epoch}: Visualizing {actual_batch_size_for_vis} conditioned sample(s) ({num_inference_steps} steps, scheduler: {type(effective_scheduler).__name__}) ---")

    timesteps_iterable = tqdm(effective_scheduler.timesteps, desc=f"Cond Vis E{epoch}", leave=False, disable=not vis_cfg.get("tqdm_steps", True)) # Use effective_scheduler here
    
    # Create directories for intermediate visualizations
    intermediate_base_dir = None
    intermediate_3d_raw_dir = None
    intermediate_3d_filtered_dir = None
    intermediate_overlay_dir = None

    if actual_batch_size_for_vis > 0: # Only attempt to create dirs if there's a valid sample index
        intermediate_base_dir = base_vis_output_dir / f"sample_{intermediate_vis_idx}_intermediate_steps"
        intermediate_3d_raw_dir = intermediate_base_dir / "3d_raw"
        intermediate_3d_filtered_dir = intermediate_base_dir / "3d_filtered"
        intermediate_overlay_dir = intermediate_base_dir / "overlay"
        
        try:
            intermediate_3d_raw_dir.mkdir(parents=True, exist_ok=True)
            intermediate_3d_filtered_dir.mkdir(parents=True, exist_ok=True)
            intermediate_overlay_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log.error(f"Could not create intermediate directories for sample {intermediate_vis_idx}: {e}")
            # Set to None to disable intermediate saves if directory creation fails
            intermediate_3d_raw_dir = intermediate_3d_filtered_dir = intermediate_overlay_dir = None
    
    save_step_indices_for_intermediate_vis = set()
    num_total_scheduler_steps = len(effective_scheduler.timesteps)
    max_intermediate_saves = vis_cfg.get("max_intermediate_saves", 100)

    if num_total_scheduler_steps <= max_intermediate_saves:
        save_step_indices_for_intermediate_vis = set(range(num_total_scheduler_steps))
    else:
        save_step_indices_for_intermediate_vis.add(0) 
        save_step_indices_for_intermediate_vis.add(num_total_scheduler_steps - 1) 
        num_additional_saves_needed = max_intermediate_saves - 2
        if num_additional_saves_needed > 0:
            intermediate_step_indices_pool = list(range(1, num_total_scheduler_steps - 1))
            if intermediate_step_indices_pool:
                if len(intermediate_step_indices_pool) <= num_additional_saves_needed:
                    save_step_indices_for_intermediate_vis.update(intermediate_step_indices_pool)
                else:
                    selected_indices_in_pool = np.linspace(0, len(intermediate_step_indices_pool) - 1, num_additional_saves_needed, dtype=int)
                    for idx_in_pool in selected_indices_in_pool:
                        save_step_indices_for_intermediate_vis.add(intermediate_step_indices_pool[idx_in_pool])
    
    for step_idx, t in enumerate(timesteps_iterable):
        with torch.no_grad():
            timestep_batch = torch.full((actual_batch_size_for_vis,), t, device=device, dtype=torch.long)
            try:
                current_sample = sample_batch
                # Check input to condition_point_cloud
                if torch.isnan(current_sample).any() or torch.isinf(current_sample).any():
                    log.warning(f"Step {step_idx+1}: sample_batch_for_conditioning_final (input to condition_point_cloud) contains NaN/Inf.")

                # Call condition_point_cloud with the CoM-shifted point cloud
                # This function is assumed to return concatenated (original_4D_cloud_channels, conditioning_feature_channels)
                raw_conditioning_output = condition_point_cloud(
                    current_sample.clone(), 
                    pixel_size_batch, 
                    conditioning_image_batch, 
                    projection_model, 
                    device, cfg, mask_feature_map=False
                )
                
                conditioned_sample_input_for_model = raw_conditioning_output.clone()
                
                current_size_embedding = size_embedding_batch 
                
                if torch.isnan(conditioned_sample_input_for_model).any() or torch.isinf(conditioned_sample_input_for_model).any():
                    nan_inf_source_info = "with CoM shift (use_occlusion=True)" if use_occlusion else "after CoM un-shift (use_occlusion=False)"
                    log.warning(f"Step {step_idx+1}: conditioned_sample_input_for_model ({nan_inf_source_info}) contains NaN/Inf before model forward pass")
                
                noise_pred = model(conditioned_sample_input_for_model, timestep_batch, pixel_size_batch, current_size_embedding)
                
                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    log.warning(f"Step {step_idx+1}: noise_pred_xyz_centered (after centering with existence mask) contains NaN/Inf")
                
            except Exception as e:
                log.error(f"Conditioned model forward pass failed at step {step_idx+1}, t={t}: {e}", exc_info=True); break
        
        try:
            
            current_noisy_for_scheduler = current_sample # USE THE CENTERED PART
            
            if torch.isnan(current_noisy_for_scheduler).any() or torch.isinf(current_noisy_for_scheduler).any():
                log.warning(f"Step {step_idx+1}: current_noisy_for_scheduler contains NaN/Inf before scheduler step")
            
            denoised_result = effective_scheduler.step(noise_pred, t, current_noisy_for_scheduler) # Use centered noise and centered current_sample
            prev_denoised = denoised_result.prev_sample


            if torch.isnan(prev_denoised).any() or torch.isinf(prev_denoised).any():
                log.warning(f"Step {step_idx+1}: prev_denoised_xyz_centered contains NaN/Inf after scheduler step")
            
            # Reconstruct sample_batch with the new, centered XYZ and the clean existence channel.
            sample_batch = prev_denoised

        except Exception as e:
            log.error(f"Conditioned scheduler step failed at step {step_idx+1}, t={t}: {e}", exc_info=True); break

        # --- Intermediate Visualizations (for sample `intermediate_vis_idx` of the batch) ---
        if step_idx in save_step_indices_for_intermediate_vis and intermediate_3d_raw_dir is not None: # Check if dir creation was successful
            # For 3D views, use the CoM-centered point cloud from sample_batch
            pc_for_3d_intermediate = sample_batch[intermediate_vis_idx].detach().clone() # (C, N)
            step_img_filename_stem = f"step_{step_idx+1:04d}_t_{t.item():04d}"
            
            title_3d_intermediate = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx}"
            rendered_3d_raw = _render_point_cloud_views(
                pc_tensor=pc_for_3d_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Raw",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
            )
            if rendered_3d_raw is not None:
                imageio.imwrite(intermediate_3d_raw_dir / (step_img_filename_stem + ".png"), rendered_3d_raw)
            
            rendered_3d_filtered = _render_point_cloud_views(
                pc_tensor=pc_for_3d_intermediate, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_intermediate + " Filt.",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_3d_filtered is not None:
                imageio.imwrite(intermediate_3d_filtered_dir / (step_img_filename_stem + ".png"), rendered_3d_filtered)

            # For overlay projections, shift the CoM-centered point cloud by im_CoM_offset
            if intermediate_overlay_dir is not None:
                pc_for_overlay_intermediate = pc_for_3d_intermediate.clone()
                
                try:
                    title_overlay_intermediate_raw = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx} Overlay (Raw)"
                    visualize_projection_and_sampling(
                        point_clouds=pc_for_overlay_intermediate.unsqueeze(0), # Needs [1, 4, N]
                        pixel_sizes=pixel_size_batch[intermediate_vis_idx:intermediate_vis_idx+1], 
                        images=conditioning_image_batch[intermediate_vis_idx:intermediate_vis_idx+1],
                        cell_size=cfg.data.cell_size, device=device, filter_exists=False,
                        save_path=intermediate_overlay_dir / (step_img_filename_stem + "_raw_overlay.png"),
                        title=title_overlay_intermediate_raw
                    )
                    title_overlay_intermediate_filtered = f"E{epoch} S{step_idx+1} T{t.item()} Sample {intermediate_vis_idx} Overlay (Filtered)"
                    visualize_projection_and_sampling(
                        point_clouds=pc_for_overlay_intermediate.unsqueeze(0), # Needs [1, 4, N]
                        pixel_sizes=pixel_size_batch[intermediate_vis_idx:intermediate_vis_idx+1], 
                        images=conditioning_image_batch[intermediate_vis_idx:intermediate_vis_idx+1],
                        cell_size=cfg.data.cell_size, device=device, filter_exists=True,
                        save_path=intermediate_overlay_dir / (step_img_filename_stem + "_filtered_overlay.png"),
                        title=title_overlay_intermediate_filtered
                    )
                except Exception as e:
                    log.error(f"Failed intermediate overlay vis for E{epoch} S{step_idx+1} Sample {intermediate_vis_idx}: {e}", exc_info=True)
    
    # --- Final Predictions (Batch-wise) ---
    log.info(f"--- Epoch {epoch}: Generating final predictions visualizations ---")
    final_preds_base_dir = base_vis_output_dir / "FINAL_PREDICTIONS"
    final_preds_base_dir.mkdir(parents=True, exist_ok=True)

    for b_idx in range(actual_batch_size_for_vis):
        sample_final_dir = final_preds_base_dir / f"sample_{b_idx}"
        sample_final_dir.mkdir(parents=True, exist_ok=True)
        
        # For 3D views, use the CoM-centered point cloud from the final sample_batch
        final_pc_b_for_3d = sample_batch[b_idx].detach().clone() # (C,N), C=4
        
        title_3d_final = f"E{epoch} Final Sample {b_idx}"
        rendered_final_3d_raw = _render_point_cloud_views(
            pc_tensor=final_pc_b_for_3d, renderer=renderer, cameras_dict=cameras_dict,
            image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_final + " Raw",
            color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=False
        )
        if rendered_final_3d_raw is not None:
            imageio.imwrite(sample_final_dir / "3D_RAW_final.png", rendered_final_3d_raw)

        rendered_final_3d_filtered = _render_point_cloud_views(
            pc_tensor=final_pc_b_for_3d, renderer=renderer, cameras_dict=cameras_dict,
            image_size=vis_cfg.image_size, device=device, title_prefix=title_3d_final + " Filtered",
            color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
        )
        if rendered_final_3d_filtered is not None:
            imageio.imwrite(sample_final_dir / "3D_FILTERED_final.png", rendered_final_3d_filtered)

        # --- RDF Visualization ---
        if False:
            try:
                # 1. Initialize RDFLoss module to use its _compute_rdf method
                rdf_loss_module = RDFLoss(
                    cutoff=cfg.visualization.get("rdf_cutoff", 1.4),
                    n_bins=cfg.visualization.get("rdf_n_bins", 512),
                    bandwidth=cfg.visualization.get("rdf_bandwidth", 0.005)
                ).to(device)

                # 2. Prepare Predicted Point Cloud (filtered)
                pred_exists_mask = final_pc_b_for_3d[3, :] >= 0.5
                if pred_exists_mask.any():
                    pc_pred_for_rdf = final_pc_b_for_3d[:3, pred_exists_mask].transpose(0, 1).contiguous() # [N_pred, 3]

                    # 3. Prepare Ground Truth Point Cloud (filtered)
                    gt_pc_b_4_n = example_batch[gt_key_found][b_idx].to(device, dtype=torch.float32)
                    gt_exists_mask = gt_pc_b_4_n[3, :] >= 0.5
                    if gt_exists_mask.any():
                        # We use the original uncentered GT for RDF as it's an intrinsic property
                        pc_gt_for_rdf = gt_pc_b_4_n[:3, gt_exists_mask].transpose(0, 1).contiguous() # [N_gt, 3]

                        # 4. Generate and save the plot
                        rdf_plot_path = sample_final_dir / "RDF_GT_vs_PRED_final.png"
                        visualize_rdf(
                            loss_module=rdf_loss_module,
                            pc_pred=pc_pred_for_rdf,
                            pc_gt=pc_gt_for_rdf,
                            output_path=str(rdf_plot_path)
                        )
                    else:
                        log.warning(f"Skipping RDF plot for sample {b_idx}: No valid points in ground truth.")
                else:
                    log.warning(f"Skipping RDF plot for sample {b_idx}: No valid points in prediction.")
            except Exception as e_rdf:
                log.error(f"Failed to generate RDF plot for sample {b_idx}: {e_rdf}", exc_info=True)

        # For overlay projections, shift the CoM-centered point cloud by im_CoM_offset
        final_pc_b_for_overlay = sample_batch[b_idx].detach().clone()

        try:
            title_overlay_final_raw = f"E{epoch} Final Sample {b_idx} Overlay (Raw)"
            visualize_projection_and_sampling(
                point_clouds=final_pc_b_for_overlay.unsqueeze(0), # Make it [1, 4, N]
                pixel_sizes=pixel_size_batch[b_idx:b_idx+1], 
                images=conditioning_image_batch[b_idx:b_idx+1],
                cell_size=cfg.data.cell_size, device=device, filter_exists=False,
                save_path=sample_final_dir / "OVERLAY_RAW_final.png",
                title=title_overlay_final_raw
            )
            title_overlay_final_filtered = f"E{epoch} Final Sample {b_idx} Overlay (Filtered)"
            visualize_projection_and_sampling(
                point_clouds=final_pc_b_for_overlay.unsqueeze(0), # Make it [1, 4, N]
                pixel_sizes=pixel_size_batch[b_idx:b_idx+1], 
                images=conditioning_image_batch[b_idx:b_idx+1],
                cell_size=cfg.data.cell_size, device=device, filter_exists=True,
                save_path=sample_final_dir / "OVERLAY_FILTERED_final.png",
                title=title_overlay_final_filtered
            )
        except Exception as e:
            log.error(f"Failed final overlay vis for E{epoch} sample {b_idx}: {e}", exc_info=True)

        # --- Combined GT and Prediction 3D Plots ---
        if False: # Hardcoded for easy toggling by changing to 'if False:'
            if gt_key_found and _PYTORCH3D_AVAILABLE and renderer and cameras_dict:
                
                # --- Plot 1: GT (Red) vs. Prediction_RAW (Blue) ---
                try:
                    # Predicted points (Blue) - RAW
                    # Use final_pc_b_for_3d which is the CoM-centered prediction
                    pred_xyz_raw_n_3 = final_pc_b_for_3d[:3, :].transpose(0, 1).contiguous() 
                    if pred_xyz_raw_n_3.shape[0] > 0:
                        pred_raw_colors_n_3 = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_raw_n_3.shape[0], 1) # Blue
                    else:
                        pred_raw_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    # Ground Truth points (Red) - Common for both plots
                    gt_pc_b_4_n = example_batch[gt_key_found][b_idx].to(device, dtype=dtype) # [4, N_gt]
                    
                    # Center the GT point cloud
                    gt_xyz_original_3_n = gt_pc_b_4_n[:3, :] # [3, N_gt]
                    gt_existence_original_1_n = gt_pc_b_4_n[3:, :] # [1, N_gt]
                    
                    # Ensure mask is correctly shaped for center_com if it expects [1,N] for single sample
                    # center_com handles [3,N] for tensor_xyz and [1,N] for mask
                    gt_xyz_centered_3_n = center_com(gt_xyz_original_3_n, mask=gt_existence_original_1_n)
                    gt_xyz_n_3 = gt_xyz_centered_3_n.transpose(0, 1).contiguous() # [N_gt, 3]

                    # Optional: Filter GT points if desired (e.g., using gt_existence_original_1_n)
                    # This filtering should happen *after* centering if based on original existence,
                    # or be applied to the mask used for centering if filtering affects CoM.
                    # For simplicity, if filtering is needed, apply to gt_xyz_n_3 and gt_existence_original_1_n before color assignment.
                    # Example:
                    # gt_exists_mask_n = gt_existence_original_1_n.squeeze() >= 0.5
                    # gt_xyz_n_3 = gt_xyz_n_3[gt_exists_mask_n]

                    if gt_xyz_n_3.shape[0] > 0:
                        gt_colors_n_3 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(gt_xyz_n_3.shape[0], 1) # Red
                    else:
                        gt_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    if pred_xyz_raw_n_3.shape[0] == 0 and gt_xyz_n_3.shape[0] == 0:
                        log.info(f"Skipping combined GT/Pred_Raw 3D plot for sample {b_idx} as both point clouds are empty.")
                    else:
                        # Handle cases where one point cloud might be empty
                        if gt_xyz_n_3.shape[0] > 0 and pred_xyz_raw_n_3.shape[0] > 0:
                            combined_verts_raw = torch.cat([gt_xyz_n_3, pred_xyz_raw_n_3], dim=0) # GT (Red) first
                            combined_colors_raw = torch.cat([gt_colors_n_3, pred_raw_colors_n_3], dim=0)
                        elif gt_xyz_n_3.shape[0] > 0:
                            combined_verts_raw = gt_xyz_n_3
                            combined_colors_raw = gt_colors_n_3
                        elif pred_xyz_raw_n_3.shape[0] > 0:
                            combined_verts_raw = pred_xyz_raw_n_3
                            combined_colors_raw = pred_raw_colors_n_3
                        else: # Should be caught by earlier check
                            combined_verts_raw = torch.empty((0,3), device=device, dtype=dtype)
                            combined_colors_raw = torch.empty((0,3), device=device, dtype=dtype)


                        if combined_verts_raw.shape[0] > 0:
                            point_cloud_combined_raw = Pointclouds(points=[combined_verts_raw], features=[combined_colors_raw])
                            rendered_views_combined_raw = []
                            view_configs_combined = [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]

                            for view_key, camera in view_configs_combined:
                                try:
                                    img_combined = renderer(point_cloud_combined_raw, cameras=camera)[0, ..., :3] 
                                    rendered_views_combined_raw.append(img_combined)
                                except Exception as e_render:
                                    log.error(f"Error rendering combined GT/Pred_Raw view '{view_key}' for sample {b_idx}: {e_render}", exc_info=True)
                                    rendered_views_combined_raw.append(torch.zeros((vis_cfg.image_size, vis_cfg.image_size, 3), dtype=dtype, device=device))
                            

                            if rendered_views_combined_raw:
                                combined_image_tensor_raw = torch.cat(rendered_views_combined_raw, dim=1) 
                                combined_image_np_raw = (combined_image_tensor_raw.cpu().numpy() * 255).astype(np.uint8)
                                
                                title_text_raw = f"E{epoch} Final S{b_idx} GT(Red)/Pred_Raw(Blue)"
                                if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                                    try:
                                        pil_img_raw = Image.fromarray(combined_image_np_raw)
                                        draw_raw = ImageDraw.Draw(pil_img_raw)
                                        font_path = cfg.visualization.get("font_path", "arial.ttf")
                                        font_main = ImageFont.truetype(font_path, 16)
                                        bbox_raw = draw_raw.textbbox((0,0), title_text_raw, font=font_main)
                                        title_w_raw = bbox_raw[2] - bbox_raw[0]
                                        draw_raw.text(((combined_image_np_raw.shape[1] - title_w_raw) // 2, 5), title_text_raw, fill=(255,255,255), font=font_main)
                                        combined_image_np_raw = np.array(pil_img_raw)
                                    except Exception as e_text: log.error(f"Error adding text to GT/Pred_Raw image for sample {b_idx}: {e_text}", exc_info=True)
                                imageio.imwrite(sample_final_dir / "3D_GT_PRED_RAW_overlay.png", combined_image_np_raw)
                        else:
                            log.info(f"Skipping combined GT/Pred_Raw 3D plot for sample {b_idx} as combined_verts_raw is empty.")
                except Exception as e_plot_raw:
                    log.error(f"Failed to generate GT/Pred_Raw 3D plot for sample {b_idx}: {e_plot_raw}", exc_info=True)

                # --- Plot 2: GT (Red) vs. Prediction_FILTERED (Blue) ---
                try:
                    # Predicted points (Blue) - FILTERED
                    # Use final_pc_b_for_3d which is the CoM-centered prediction
                    pred_exists_values = final_pc_b_for_3d[3, :] 
                    pred_mask = pred_exists_values >= 0.5  
                    pred_xyz_filtered_n_3 = final_pc_b_for_3d[:3, pred_mask].transpose(0, 1).contiguous() 
                    
                    if pred_xyz_filtered_n_3.shape[0] > 0:
                        pred_filtered_colors_n_3 = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_filtered_n_3.shape[0], 1) # Blue
                    else:
                        pred_filtered_colors_n_3 = torch.empty((0,3), device=device, dtype=dtype)

                    # GT points (Red) are the same as above (gt_xyz_n_3, gt_colors_n_3)

                    if pred_xyz_filtered_n_3.shape[0] == 0 and gt_xyz_n_3.shape[0] == 0:
                        log.info(f"Skipping combined GT/Pred_Filt 3D plot for sample {b_idx} as both point clouds are empty.")
                    else:
                        if gt_xyz_n_3.shape[0] > 0 and pred_xyz_filtered_n_3.shape[0] > 0:
                            combined_verts_filt = torch.cat([gt_xyz_n_3, pred_xyz_filtered_n_3], dim=0) # GT (Red) first
                            combined_colors_filt = torch.cat([gt_colors_n_3, pred_filtered_colors_n_3], dim=0)
                        elif gt_xyz_n_3.shape[0] > 0:
                            combined_verts_filt = gt_xyz_n_3
                            combined_colors_filt = gt_colors_n_3
                        elif pred_xyz_filtered_n_3.shape[0] > 0:
                            combined_verts_filt = pred_xyz_filtered_n_3
                            combined_colors_filt = pred_filtered_colors_n_3
                        else: # Should be caught by earlier check
                            combined_verts_filt = torch.empty((0,3), device=device, dtype=dtype)
                            combined_colors_filt = torch.empty((0,3), device=device, dtype=dtype)

                        if combined_verts_filt.shape[0] > 0:
                            point_cloud_combined_filt = Pointclouds(points=[combined_verts_filt], features=[combined_colors_filt])
                            rendered_views_combined_filt = []
                            view_configs_combined = [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]

                            for view_key, camera in view_configs_combined:
                                try:
                                    img_combined = renderer(point_cloud_combined_filt, cameras=camera)[0, ..., :3] 
                                    rendered_views_combined_filt.append(img_combined)
                                except Exception as e_render:
                                    log.error(f"Error rendering combined GT/Pred_Filt view '{view_key}' for sample {b_idx}: {e_render}", exc_info=True)
                                    rendered_views_combined_filt.append(torch.zeros((vis_cfg.image_size, vis_cfg.image_size, 3), dtype=dtype, device=device))
                            

                            if rendered_views_combined_filt:
                                combined_image_tensor_filt = torch.cat(rendered_views_combined_filt, dim=1) 
                                combined_image_np_filt = (combined_image_tensor_filt.cpu().numpy() * 255).astype(np.uint8)
                                
                                title_text_filt = f"E{epoch} Final S{b_idx} GT(Red)/Pred_Filt(Blue)"
                                if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                                    try:
                                        pil_img_filt = Image.fromarray(combined_image_np_filt)
                                        draw_filt = ImageDraw.Draw(pil_img_filt)
                                        font_path = cfg.visualization.get("font_path", "arial.ttf")
                                        font_main = ImageFont.truetype(font_path, 16)
                                        bbox_filt = draw_filt.textbbox((0,0), title_text_filt, font=font_main)
                                        title_w_filt = bbox_filt[2] - bbox_filt[0]
                                        draw_filt.text(((combined_image_np_filt.shape[1] - title_w_filt) // 2, 5), title_text_filt, fill=(255,255,255), font=font_main)
                                        combined_image_np_filt = np.array(pil_img_filt)
                                    except Exception as e_text: log.error(f"Error adding text to GT/Pred_Filt image for sample {b_idx}: {e_text}", exc_info=True)
                                imageio.imwrite(sample_final_dir / "3D_GT_PRED_FILTERED_overlay.png", combined_image_np_filt)
                        else:
                            log.info(f"Skipping combined GT/Pred_Filt 3D plot for sample {b_idx} as combined_verts_filt is empty.")
                except Exception as e_plot_filt:

                    log.error(f"Failed to generate GT/Pred_Filt 3D plot for sample {b_idx}: {e_plot_filt}", exc_info=True)
            
            elif not gt_key_found:
                log.warning(f"Skipping combined GT/Pred 3D plots for sample {b_idx}: GT key not found in example_batch.")
            # Implicitly, if _PYTORCH3D_AVAILABLE is False or renderer/cameras_dict are None, this block won't execute.

        # -- Post processing to enforce crystallinity using Lennard-Jones style potential --
        if False: # Hard-coded to always run
            # --- Hard-coded Parameters for LJ Relaxation ---
            num_steps = 1000
            equilibrium_distance = 0.03  # r_e: The distance where the potential energy is minimal.
            epsilon = 5.0                 # Depth of the potential well, scales the force magnitude.
            learning_rate = 1e-5          # Step size for updating positions based on force.
            vis_freq = 1                # Frequency to save frames for the GIF.
            # --- End of Hard-coded Parameters ---

            log.info(f"--- Starting Lennard-Jones style relaxation for E{epoch} S{sample_idx} ---")
            relaxation_output_dir = sample_final_dir / "lj_relaxation_process"
            relaxation_output_dir.mkdir(parents=True, exist_ok=True)

            # Start with the final, unrelaxed prediction
            relaxed_pc_4d = final_pc_b_for_3d.clone()
            
            exists_mask = relaxed_pc_4d[3, :] >= 0.5
            if not exists_mask.any():
                log.warning(f"Relaxation E{epoch} S{sample_idx}: No points with exists>=0.5. Skipping this sample.")
                continue # Skip to next sample in the batch

            num_points = exists_mask.sum().item()
            log.info(f"Relaxing {num_points} points with r_e={equilibrium_distance}, lr={learning_rate}.")
            
            # Work only with existing points, shape (N_exist, 3)
            xyz_current = relaxed_pc_4d[:3, exists_mask].T.clone()

            # LJ constants derived from equilibrium_distance
            sigma = equilibrium_distance / (2**(1/6))
            sigma_6 = sigma**6
            sigma_12 = sigma**12
            
            frames = []

            # Relaxation Loop
            for step in tqdm(range(num_steps), desc=f"LJ Relax E{epoch} S{sample_idx}", leave=False):
                # Calculate pairwise distances and vectors
                pdist_matrix = torch.cdist(xyz_current, xyz_current, p=2)
                
                # Avoid division by zero for self-distance
                pdist_matrix.fill_diagonal_(float('inf'))
                
                # Calculate terms for LJ force calculation
                r_inv = 1.0 / pdist_matrix
                r_6_inv = r_inv**6
                r_12_inv = r_inv**12

                # Calculate the magnitude of the LJ force for each pair
                # F(r) = 24 * epsilon / r * [2 * (sigma/r)^12 - (sigma/r)^6]
                force_magnitude = 24 * epsilon * r_inv * (2 * sigma_12 * r_12_inv - sigma_6 * r_6_inv)
                force_magnitude.fill_diagonal_(0) # No self-force

                # Calculate the Z-component of the vector between each pair of points
                z_coords = xyz_current[:, 2]
                z_diffs = z_coords.unsqueeze(1) - z_coords.unsqueeze(0) # z_i - z_j

                # Z-component of the force vector F_ij_z = F_magnitude_ij * (z_i - z_j) / r_ij
                force_z_components = force_magnitude * z_diffs * r_inv
                force_z_components.fill_diagonal_(0)

                # Total Z-force on each point is the sum of forces from all other points
                total_force_z = torch.sum(force_z_components, dim=1)

                # Update only the Z coordinates based on the calculated force
                xyz_current[:, 2] += learning_rate * total_force_z

                # Periodic Visualization for GIF
                if renderer and (step % vis_freq == 0 or step == num_steps - 1):
                    viz_pc_4d = relaxed_pc_4d.clone()
                    viz_pc_4d[:3, exists_mask] = xyz_current.T
                    
                    title = f"LJ Relax E{epoch} S{sample_idx} - Step {step+1}"
                    rendered_frame = _render_point_cloud_views(
                        pc_tensor=viz_pc_4d, renderer=renderer, cameras_dict=cameras_dict,
                        image_size=vis_cfg.image_size, device=device, title_prefix=title,
                        color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, 
                        filter_exists=True
                    )
                    if rendered_frame is not None:
                        frames.append(rendered_frame)

            if frames:
                gif_path = relaxation_output_dir / f"lj_relaxation_e{epoch}_s{sample_idx}.gif"
                imageio.mimsave(gif_path, frames, duration=100)

            # Update the main tensor with the final relaxed coordinates
            relaxed_pc_4d[:3, exists_mask] = xyz_current.T

            # --- Visualizations for the RELAXED Point Cloud ---
            log.info(f"--- Visualizing final LJ-relaxed structure for E{epoch} S{sample_idx} ---")
            
            rendered_final_3d_relaxed = _render_point_cloud_views(
                pc_tensor=relaxed_pc_4d, renderer=renderer, cameras_dict=cameras_dict,
                image_size=vis_cfg.image_size, device=device, title_prefix=f"E{epoch} Final S{sample_idx} LJ-Relaxed",
                color_range=vis_cfg.get("color_range", (-1.0, 1.0)), cfg=cfg, filter_exists=True
            )
            if rendered_final_3d_relaxed is not None:
                imageio.imwrite(sample_final_dir / "3D_RELAXED_final.png", rendered_final_3d_relaxed)

            if gt_key_found:
                gt_colors_n_3 = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(gt_xyz_n_3.shape[0], 1)
                
                pred_exists_mask_relaxed = relaxed_pc_4d[3, :] >= 0.5
                pred_xyz_relaxed_filt_n_3 = relaxed_pc_4d[:3, pred_exists_mask_relaxed].T.contiguous()
                pred_relaxed_colors = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).repeat(pred_xyz_relaxed_filt_n_3.shape[0], 1)

                if gt_xyz_n_3.shape[0] > 0 and pred_xyz_relaxed_filt_n_3.shape[0] > 0:
                    combined_verts = torch.cat([gt_xyz_n_3, pred_xyz_relaxed_filt_n_3], dim=0)
                    combined_colors = torch.cat([gt_colors_n_3, pred_relaxed_colors], dim=0)
                    
                    point_cloud_combined = Pointclouds(points=[combined_verts], features=[combined_colors])
                    rendered_views_list = []
                    for _, camera in [('xy', cameras_dict['xy']), ('yz', cameras_dict['yz']), ('xz', cameras_dict['xz'])]:
                        img_v = renderer(point_cloud_combined, cameras=camera)[0, ..., :3]
                        rendered_views_list.append(img_v)
                    
                    combined_image_np = (torch.cat(rendered_views_list, dim=1).cpu().numpy() * 255).astype(np.uint8)
                    # Add title to the combined relaxed image
                    title_text_relaxed = f"E{epoch} Final S{sample_idx} GT(Red)/Pred_Relaxed(Blue)"
                    if _IMAGEFONT_TRUETYPE_AVAILABLE and cfg and hasattr(cfg, 'visualization'):
                        try:
                            pil_img = Image.fromarray(combined_image_np)
                            draw = ImageDraw.Draw(pil_img)
                            font_path = cfg.visualization.get("font_path", "arial.ttf")
                            font_main = ImageFont.truetype(font_path, 16)
                            bbox = draw.textbbox((0,0), title_text_relaxed, font=font_main)
                            title_w = bbox[2] - bbox[0]
                            draw.text(((combined_image_np.shape[1] - title_w) // 2, 5), title_text_relaxed, fill=(255,255,255), font=font_main)
                            combined_image_np = np.array(pil_img)
                        except Exception as e_text: log.error(f"Error adding text to GT/Pred_Relaxed image for sample {b_idx}: {e_text}", exc_info=True)
                    imageio.imwrite(sample_final_dir / "3D_GT_PRED_RELAXED_overlay.png", combined_image_np)


    log.info(f"--- Finished conditioned visualization loop for epoch {epoch} ---")
    model.train()
    projection_model.train()
    if size_estimator: size_estimator.train()