import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage # Added for distance transform
import matplotlib.pyplot as plt # Added for debugging plot

def plot_thickness_and_binary_mask(raw_thickness_map_batch, binary_mask_batch, binarization_threshold, batch_idx_to_plot=0):
    """
    Plots the raw thickness map and the binarized mask side-by-side for a specific item in the batch.
    Args:
        raw_thickness_map_batch (torch.Tensor): Batch of raw thickness maps [B, 1, H, W].
        binary_mask_batch (torch.Tensor): Batch of binarized masks [B, 1, H, W].
        binarization_threshold (float): The threshold used for binarization.
        batch_idx_to_plot (int): Index of the batch item to plot.
    """
    if batch_idx_to_plot >= raw_thickness_map_batch.shape[0]:
        print(f"Debug plot: batch_idx_to_plot {batch_idx_to_plot} is out of bounds for batch size {raw_thickness_map_batch.shape[0]}. Skipping plot.")
        return

    raw_map = raw_thickness_map_batch[batch_idx_to_plot, 0].cpu().numpy()
    binary_map = binary_mask_batch[batch_idx_to_plot, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axes[0].imshow(raw_map, cmap='viridis')
    axes[0].set_title(f'Raw Thickness Map (Batch Item {batch_idx_to_plot})')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(binary_map, cmap='gray')
    axes[1].set_title(f'Binarized Mask (Thresh: {binarization_threshold:.2f})')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def condition_point_cloud_new(point_clouds, pixel_sizes, images, projection_model, device, cfg):
    """
    Conditions a point cloud with multi-scale features from an image projection model.

    This function takes a batch of 3D point clouds and corresponding 2D images,
    uses a U-Net (projection_model) to extract features from the images, and then
    samples these features at the projected 2D locations of the 3D points. The
    sampled features are then concatenated to the original point cloud features.

    Args:
        point_clouds: Tensor of shape [B, C_in, N], e.g., C_in=4 for XYZ+existence.
        pixel_sizes:  Tensor of shape [B], nm-per-pixel for each image.
        images:       Tensor of shape [B, 1, H, W], the conditioning grayscale images.
        projection_model: An nn.Module (e.g., ModernUNet) that takes an image tensor
                      and returns a dictionary containing 'logits', 'encoder_features',
                      and 'decoder_features'.
        device:       The torch.device to run computations on.
        cfg:          A Hydra/DictConfig object containing model and data configurations.

    Returns:
        conditioned_pc: A tensor of shape [B, C_in + C_features, N], where C_features
                        is the total number of channels from all sampled feature maps.
    """
    IMAGE_SIZE = images.shape[-1]
    SUPER_CELL_SIZE = cfg.data.cell_size

    # Ensure all tensors and models are on the correct device
    projection_model = projection_model.to(device).eval()
    images = images.to(device)
    pixel_sizes = pixel_sizes.to(device)
    point_clouds = point_clouds.to(device)

    B, _, N = point_clouds.shape
    H_img, W_img = images.shape[-2], images.shape[-1]

    # 1. ENCODE IMAGES AND SELECT FEATURE SET
    # ==========================================
    with torch.no_grad():
        # The projection model now returns a dictionary containing multiple outputs
        model_output = projection_model(images)
        thickness_map_raw = model_output["logits"]

        # Choose whether to use 'encoder' or 'decoder' features based on the config
        feature_type_to_use = cfg.model.conditioning.get('feature_type', 'encoder').lower()
        if feature_type_to_use == 'decoder':
            features_to_sample = model_output["decoder_features"]
        else:  # Default to encoder features if not specified or invalid
            features_to_sample = model_output["encoder_features"]

    # 2. PROJECT 3D POINTS TO 2D PIXEL COORDINATES
    # ===============================================
    # pc_xy contains the X and Y coordinates of the point cloud
    pc_xy = point_clouds[:, :2, :]                      # Shape: [B, 2, N]
    # Convert normalized coordinates to nanometers
    pc_nm = pc_xy * (SUPER_CELL_SIZE / 2.0)             # Shape: [B, 2, N] in nm
    # Reshape pixel_sizes for broadcasting
    px_per_batch = pixel_sizes.view(B, 1, 1)            # Shape: [B, 1, 1]
    # Convert nanometers to pixel coordinates (origin at top-left)
    pc_pix = pc_nm / px_per_batch + (IMAGE_SIZE / 2.0)  # Shape: [B, 2, N] (col, row)

    # 3. SAMPLE FEATURES FROM SELECTED MAPS
    # =======================================
    sampled_feats_all = []

    # Get the indices of the feature maps to use from the config
    # e.g., [0, 2, 4] means the 1st, 3rd, and 5th maps from the chosen tuple
    features_to_use_idx = cfg.model.conditioning.get('features_to_use', [0, 2, 4])

    for i in features_to_use_idx:
        # Gracefully handle cases where the index might be out of bounds
        if i >= len(features_to_sample):
            print(f"Warning: feature index {i} is out of bounds for {feature_type_to_use} features (max index {len(features_to_sample)-1}). Skipping.")
            continue

        feat_map = features_to_sample[i].to(device) # Shape: [B, C, H, W]
        _, C, H, W = feat_map.shape

        # Normalize point coordinates to the [-1, 1] range required by grid_sample,
        # specific to the resolution (H, W) of the current feature map.
        grid_x = pc_pix[:, 0, :] / (W - 1) * 2.0 - 1.0
        grid_y = pc_pix[:, 1, :] / (H - 1) * 2.0 - 1.0

        # Stack to create the grid for grid_sample
        grid = torch.stack((grid_x, grid_y), dim=-1)   # Shape: [B, N, 2]
        # Unsqueeze to match grid_sample's expected 4D input: [B, H_out, W_out, 2]
        # Here we sample N points, so H_out=1, W_out=N.
        grid = grid.unsqueeze(1)                      # Shape: [B, 1, N, 2]

        # Sample the feature map at the grid locations
        samp = F.grid_sample(feat_map, grid,
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=True)      # Shape: [B, C, 1, N]

        # Remove the singleton dimension
        samp = samp.squeeze(2)                       # Shape: [B, C, N]
        sampled_feats_all.append(samp)

    # 4. SAMPLE FROM THE FINAL THICKNESS MAP (OPTIONAL)
    # ==================================================
    if cfg.model.conditioning.get('use_thickness_map', True):
        # Normalize to the original image resolution for sampling the thickness map
        grid_x_img_res = pc_pix[:, 0, :] / (W_img - 1) * 2.0 - 1.0
        grid_y_img_res = pc_pix[:, 1, :] / (H_img - 1) * 2.0 - 1.0
        grid_img_res = torch.stack((grid_x_img_res, grid_y_img_res), dim=-1).unsqueeze(1) # Shape: [B, 1, N, 2]

        sampled_thickness = F.grid_sample(thickness_map_raw.to(device), grid_img_res,
                                          mode='bilinear',
                                          padding_mode='zeros',
                                          align_corners=True)  # Shape: [B, 1, 1, N]
        sampled_thickness = sampled_thickness.squeeze(2)       # Shape: [B, 1, N]
        sampled_feats_all.append(sampled_thickness)

    # 5. CONCATENATE ALL SAMPLED FEATURES
    # =====================================
    if not sampled_feats_all:
        # A safeguard in case no features were selected for sampling.
        # Returning the original point cloud is a reasonable default action.
        print("Warning: No features were sampled for conditioning. Returning original point cloud.")
        return point_clouds

    all_sampled_features = torch.cat(sampled_feats_all, dim=1)

    # 6. APPEND TO ORIGINAL POINT CLOUD
    # ===================================
    conditioned_pc = torch.cat((point_clouds, all_sampled_features), dim=1)

    return conditioned_pc

def condition_point_cloud(point_clouds, pixel_sizes, images, projection_model, device, cfg, mask_feature_map=True, return_thickness_map=False):
    """
    Args:
        point_clouds: Tensor of shape [B, C_in, N], e.g. C_in=4 for XYZ+existence.
        pixel_sizes:   Tensor of shape [B], nm-per-pixel for each batch.
        images:        Tensor of shape [B, 1, H, W], grayscale.
        projection_model: nn.Module returning (logits, encoder_features_tuple).
        device:        torch.device
        cfg:           Hydra config.
    Returns:
        conditioned_pc: Tensor [B, C_in + sum(C_encoder) + 1 (mask) + 2 (dist_vec) + 1 (raw_thickness), N].
    """ 
    IMAGE_SIZE = images.shape[-1]
    SUPER_CELL_SIZE = cfg.data.cell_size

    projection_model = projection_model.to(device).eval()
    images = images.to(device)
    pixel_sizes = pixel_sizes.to(device)
    point_clouds = point_clouds.to(device)
    
    B, _, N = point_clouds.shape
    H_img, W_img = images.shape[-2], images.shape[-1]


    # 1) encode images and get segmentation logits
    with torch.no_grad():
        # projection_model now returns (thickness_map_raw, encoder_features_tuple)
        thickness_map_raw, encoder_features = projection_model(images) 
        # thickness_map_raw shape: [B, 1, H_img, W_img] (assuming n_classes=1 for thickness)
        # encoder_features: tuple of feature maps (x5, x4, x3, x2, x1)

    # 2) Process thickness map to get binary mask and distance vectors
    # Binarize the raw thickness map
    # Use a threshold from config, default to 0.0 (any positive thickness is foreground)
    binarization_threshold = cfg.model.projection_model.get('binarization_threshold', 0.01)
    binary_mask_tensor = (thickness_map_raw > binarization_threshold).float() # [B, 1, H_img, W_img]

    # --- Debug Plot ---
    if cfg.get('debug', {}).get('plot_conditioning_masks', False):
        # Plot for the first item in the batch, or make batch_idx_to_plot configurable
        #images min max
        print(f"Images min: {images.min()}, max: {images.max()}")
        print(f"Thickness map raw min: {thickness_map_raw.min()}, max: {thickness_map_raw.max()}")
        plot_thickness_and_binary_mask(thickness_map_raw, binary_mask_tensor, binarization_threshold, batch_idx_to_plot=0)
    # --- End Debug Plot ---
    """
    # Calculate distance vectors (points from pixel to closest mask pixel)
    # This part uses scipy and runs on CPU per batch item, which can be slow.
    # For optimal performance, a GPU-based distance transform would be preferred.
    batch_dist_vec_maps = []
    for b_idx in range(B):
        # Ensure binary_mask_tensor is on CPU for numpy operations
        mask_np = binary_mask_tensor[b_idx, 0].cpu().numpy() # [H_img, W_img]
        inv_mask_np = 1 - mask_np # 0 for foreground (mask=1), 1 for background (mask=0)

        # distances: distance to nearest foreground pixel
        # (indices_y, indices_x): coordinates of the nearest foreground pixel
        _, (indices_y, indices_x) = scipy.ndimage.distance_transform_edt(
            inv_mask_np, return_indices=True
        )

        grid_y_np, grid_x_np = np.mgrid[0:H_img, 0:W_img]

        # Vector from current pixel to nearest foreground pixel
        # For foreground pixels, (indices_y, indices_x) == (grid_y_np, grid_x_np), so vector is (0,0)
        vec_y_np = indices_y - grid_y_np
        vec_x_np = indices_x - grid_x_np
        
        # Stack to get [2, H_img, W_img]
        dist_vec_map_np = np.stack((vec_x_np, vec_y_np), axis=0).astype(np.float32)
        batch_dist_vec_maps.append(torch.from_numpy(dist_vec_map_np))
    
    distance_vectors_map_tensor = torch.stack(batch_dist_vec_maps).to(device) # [B, 2, H_img, W_img]
    """
    # 3) project to pixel coords
    pc_xy = point_clouds[:, :2, :]                      # [B,2,N]
    pc_nm = pc_xy * (SUPER_CELL_SIZE / 2.0)              # [B,2,N] in nm
    px_per_batch = pixel_sizes.view(B, 1, 1)             # [B,1,1]
    pc_pix = pc_nm / px_per_batch + (IMAGE_SIZE / 2.0)   # [B,2,N]: (col, row)

    verification = False
    if verification:
        outside_mask = (
            (pc_pix[:, 0, :] < 0) | (pc_pix[:, 0, :] >= IMAGE_SIZE) |
            (pc_pix[:, 1, :] < 0) | (pc_pix[:, 1, :] >= IMAGE_SIZE)
        )  # [B,N]
        num_outside = outside_mask.sum(dim=1)                # [B]
        for b in range(B):
            #print(f"Batch {b}: {num_outside[b].item()} / {N} points outside image boundary.")
            if num_outside[b] > 0:
                #print coords of points outside image boundary
                outside_coords = pc_pix[b, :, outside_mask[b]].T  # [num_outside,2]
                outside_coords = outside_coords.cpu().numpy()
                #print(f"Batch {b} outside coords: {outside_coords}")
                plot_pc_over_image(pc_pix, images)

    # 4) sample from each feature map (encoder features, mask, distance vectors)
    sampled_feats_all = []

    # Sample from encoder features

    features_to_use_idx = cfg.model.conditioning.get('features_to_use', [0,2, 4]) # [0, 2, 4] for x5 x3 and x1

    for i in features_to_use_idx:
        feat_map = encoder_features[i]                     # [B,C,H,W]
        _, C, H, W = feat_map.shape

        #new thing masking the highest resolution feature map
        if H == 128 and W == 128 and mask_feature_map:
            # Mask out the highest resolution feature map (e.g., x5) if specified
            feat_map = feat_map * binary_mask_tensor.to(device)
            #pass

        #print(f"Feature map shape: {feat_map.shape}")
        feat_map = feat_map.to(device) # Ensure on correct device
        pc_pix_centered_x = pc_pix[:, 0, :] + 0.5
        pc_pix_centered_y = pc_pix[:, 1, :] + 0.5
        # Normalize point coordinates to [-1, 1]
        grid_x = pc_pix_centered_x / (W_img - 1) * 2.0 - 1.0  # [B, N]
        grid_y = pc_pix_centered_y / (H_img - 1) * 2.0 - 1.0  # [B, N]
        
        #assert (grid_x <= 1.0).all() and (grid_x >= -1.0).all(), "Grid x out of bounds"
        #assert (grid_y <= 1.0).all() and (grid_y >= -1.0).all(), "Grid y out of bounds"

        grid = torch.stack((grid_x, grid_y), dim=-1)   # [B, N, 2]
        grid = grid.unsqueeze(2)                       # [B, N, 1, 2] -> 1D spatial structure

        samp = F.grid_sample(feat_map, grid,
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=True)       # [B, C, N, 1]

        samp = samp.squeeze(-1)                        # [B, C, N]
        sampled_feats_all.append(samp)
        if False: # Debugging plot
            import matplotlib.pyplot as plt

            # Project a few normalized points back to pixel coordinates for display
            x_px = ((grid_x[0] + 1) / 2) * (feat_map.shape[-1] - 1)
            y_px = ((grid_y[0] + 1) / 2) * (feat_map.shape[-2] - 1)

            # Plot a single channel from feature map
            plt.imshow(feat_map[0, 0].detach().cpu(), cmap='viridis')  # e.g., channel 0
            plt.scatter(x_px.cpu(), y_px.cpu(), c='r', s=5)  # plot all projected points
            plt.title("Projected Points on Feature Map")
            plt.show()
    # Prepare grid for sampling mask and distance vectors (use H_img, W_img from original image)
    """
    grid_x_img_res = pc_pix[:, 0, :] / (W_img - 1) * 2.0 - 1.0 # [B,N]
    grid_y_img_res = pc_pix[:, 1, :] / (H_img - 1) * 2.0 - 1.0 # [B,N]
    grid_img_res = torch.stack((grid_x_img_res, grid_y_img_res), dim=-1) # [B,N,2]
    grid_img_res = grid_img_res.unsqueeze(1) # [B,1,N,2]
    # Sample binary mask values
    sampled_mask = F.grid_sample(binary_mask_tensor.to(device), grid_img_res,
                                 mode='nearest', # Use nearest for binary mask
                                 padding_mode='zeros',
                                 align_corners=True) # [B,1,1,N]
    sampled_mask = sampled_mask[:, :, 0, :] # [B,1,N]
    sampled_feats_all.append(sampled_mask)

    # Sample distance vectors
    sampled_dist_vectors = F.grid_sample(distance_vectors_map_tensor.to(device), grid_img_res,
                                         mode='bilinear', # Bilinear for vectors
                                         padding_mode='zeros',
                                         align_corners=True) # [B,2,1,N]
    sampled_dist_vectors = sampled_dist_vectors[:, :, 0, :] # [B,2,N]
    sampled_feats_all.append(sampled_dist_vectors)

    # Sample raw thickness values
    """
    if cfg.model.conditioning.get('use_thickness_map', True):
        pc_pix_centered_x = pc_pix[:, 0, :] + 0.5
        pc_pix_centered_y = pc_pix[:, 1, :] + 0.5
        # Normalize point coordinates to [-1, 1]
        grid_x = pc_pix_centered_x / (W_img - 1) * 2.0 - 1.0  # [B, N]
        grid_y = pc_pix_centered_y / (H_img - 1) * 2.0 - 1.0  # [B, N]    

        grid = torch.stack((grid_x, grid_y), dim=-1)   # [B, N, 2]
        grid = grid.unsqueeze(2)                       # [B, N, 1, 2] -> 1D spatial structure    

        thickness_maps = thickness_map_raw.to(device)  # [B, 1, H_img, W_img]
        sampled_thickness = F.grid_sample(thickness_maps, grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=True)  # [B, 1, N, 1]
        sampled_thickness = sampled_thickness.squeeze(-1)  # [B, 1, N]
        sampled_feats_all.append(sampled_thickness)
        

    # 5) concat all sampled features
    all_sampled_features = torch.cat(sampled_feats_all, dim=1) # [B, sum(C_encoder) + 1 (mask) + 2 (dist_vec) + 1 (raw_thickness), N]

    # 6) append to original point cloud features (e.g., XYZ, existence)
    conditioned_pc = torch.cat((point_clouds, all_sampled_features), dim=1)

    if return_thickness_map:
        return conditioned_pc, thickness_map_raw
    else:
        return conditioned_pc


def plot_pc_over_image(point_clouds_xy, images):
    """
    Visualize point clouds overlaid on images.
    Points inside the image boundary are red, points outside are blue.
    
    Args:
        point_clouds_xy: Tensor of shape [B, 2, N], XY coords in pixel space
        images:          Tensor of shape [B, 1, 128, 128], grayscale.
    """
    import matplotlib.pyplot as plt


    for i in range(point_clouds_xy.shape[0]):
        plt.figure(figsize=(8, 8))
        img = images[i, 0].cpu().numpy()
        plt.imshow(img, cmap='gray')
        
        # Convert to numpy for easy handling
        x_coords = point_clouds_xy[i, 0].cpu().numpy()
        y_coords = point_clouds_xy[i, 1].cpu().numpy()
        
        # Create mask for points outside image boundary
        image_size = images.shape[-1]
        outside_mask = (x_coords < 0) | (x_coords >= image_size) | (y_coords < 0) | (y_coords >= image_size)
        
        # Plot points inside boundary in red
        plt.scatter(x_coords[~outside_mask], y_coords[~outside_mask], c='green', s=10)
        
        # Plot points outside boundary in blue
        plt.scatter(x_coords[outside_mask], y_coords[outside_mask], c='red', s=20)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

#pytorch3d imports
from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase, OrthographicCameras, FoVOrthographicCameras
from pytorch3d.structures import Pointclouds
from torch import Tensor
from typing import Tuple

def _debug_and_plot_rasterization(
    fragments: torch.Tensor,
    points_list: list,
    cameras: Optional[FoVOrthographicCameras],
    images: Optional[torch.Tensor],
    pc_pix: torch.Tensor,
    existence_mask: torch.Tensor,
    raster_settings: PointsRasterizationSettings,
    b_idx_debug: int = 0,
    debug_culled_points: bool = True,
):
    """
    Helper function to debug culled/occluded points and plot rasterization results.
    """
    B = images.shape[0] if images is not None else 0 # Get batch size from images if available

    if B > 0 and debug_culled_points and images is not None:
        camera_debug = None
        if cameras is not None:
            if hasattr(cameras, 'R') and cameras.R.shape[0] == B:
                camera_debug = cameras[b_idx_debug]
            elif hasattr(cameras, 'R') and B == 1:
                camera_debug = cameras

        if camera_debug is not None and b_idx_debug < len(points_list) and len(points_list[b_idx_debug]) > 0:
            points_to_check_world_b = points_list[b_idx_debug]

            rasterized_indices_in_list_b_flat = fragments.idx[b_idx_debug, ..., :raster_settings.points_per_pixel].flatten()
            unique_rasterized_indices = torch.unique(rasterized_indices_in_list_b_flat)
            unique_rasterized_indices = unique_rasterized_indices[unique_rasterized_indices != -1]

            print(f"\n--- Debugging Occluded/Culled Points for Batch {b_idx_debug} ---")
            print(f"Total valid points in points_list[{b_idx_debug}]: {len(points_to_check_world_b)}")
            print(f"Number of these points rasterized (visible): {len(unique_rasterized_indices)}")

            num_occluded_to_print = 5
            occluded_printed_count = 0
            total_occluded_in_batch = 0

            visible_points_coords = []
            occluded_points_coords = []

            for pt_idx_in_list_b in range(len(points_to_check_world_b)):
                point_world_coords = points_to_check_world_b[pt_idx_in_list_b]
                if pt_idx_in_list_b in unique_rasterized_indices:
                    visible_points_coords.append(point_world_coords.cpu().numpy())
                else:
                    total_occluded_in_batch += 1
                    occluded_points_coords.append(point_world_coords.cpu().numpy())
                    if occluded_printed_count < num_occluded_to_print:
                        point_ndc_transformed = camera_debug.transform_points_ndc(point_world_coords.unsqueeze(0).unsqueeze(0))
                        point_ndc_coords_squeezed = point_ndc_transformed.squeeze(0).squeeze(0)

                        print(f"\nOccluded/Culled Point #{occluded_printed_count+1} (Index in points_list[{b_idx_debug}]: {pt_idx_in_list_b}):")
                        print(f"  World Coords (X,Y,Z): {point_world_coords.cpu().numpy()}")
                        print(f"  NDC Coords (X,Y,Z):   {point_ndc_coords_squeezed.cpu().numpy()}")

                        xw, yw, zw = point_world_coords
                        xc, yc, zc = -xw, -yw, zw

                        print(f"  Cam Coords (Xc,Yc,Zc): [{xc:.4f}, {yc:.4f}, {zc:.4f}] (Derived from world using R)")
                        cam_min_x = camera_debug.min_x.item()
                        cam_max_x = camera_debug.max_x.item()
                        cam_min_y = camera_debug.min_y.item()
                        cam_max_y = camera_debug.max_y.item()
                        cam_znear = camera_debug.znear.item()
                        cam_zfar = camera_debug.zfar.item()
                        print(f"  Camera Frustum (Batch {b_idx_debug}):")
                        print(f"    X-Range: [{cam_min_x:.4f}, {cam_max_x:.4f}] -> Xc is {'IN' if cam_min_x <= xc <= cam_max_x else 'OUT'}")
                        print(f"    Y-Range: [{cam_min_y:.4f}, {cam_max_y:.4f}] -> Yc is {'IN' if cam_min_y <= yc <= cam_max_y else 'OUT'}")
                        print(f"    Z-Range: [{cam_znear:.4f}, {cam_zfar:.4f}] -> Zc is {'IN' if cam_znear <= zc <= cam_zfar else 'OUT'}")

                        if not torch.all((point_ndc_coords_squeezed >= -1.0 - 1e-5) & (point_ndc_coords_squeezed <= 1.0 + 1e-5)):
                            print(f"  --> This point's NDC coords are OUTSIDE [-1, 1] range (Culled by Frustum).")
                        else:
                            print(f"  --> This point's NDC coords are INSIDE [-1, 1] range (Likely Occluded).")
                        occluded_printed_count += 1

            if total_occluded_in_batch == 0 and len(points_to_check_world_b) > 0 and len(points_to_check_world_b) == len(unique_rasterized_indices):
                print(f"All {len(points_to_check_world_b)} valid points in batch {b_idx_debug} were rasterized (visible).")
            elif len(points_to_check_world_b) == 0:
                print(f"No valid points in points_list[{b_idx_debug}] to debug.")
            else:
                print(f"Printed details for {occluded_printed_count} out of {total_occluded_in_batch} occluded/culled points in batch {b_idx_debug}.")

            if points_to_check_world_b.numel() > 0:
                fig_3d = plt.figure(figsize=(10, 8))
                ax_3d = fig_3d.add_subplot(111, projection='3d')

                if visible_points_coords:
                    visible_points_np = np.array(visible_points_coords)
                    ax_3d.scatter(visible_points_np[:, 0], visible_points_np[:, 1], visible_points_np[:, 2], c='green', label=f'Visible ({len(visible_points_np)})', s=20, depthshade=True)

                if occluded_points_coords:
                    occluded_points_np = np.array(occluded_points_coords)
                    ax_3d.scatter(occluded_points_np[:, 0], occluded_points_np[:, 1], occluded_points_np[:, 2], c='red', label=f'Occluded/Culled ({len(occluded_points_np)})', s=20, depthshade=True)

                ax_3d.set_xlabel('World X')
                ax_3d.set_ylabel('World Y')
                ax_3d.set_zlabel('World Z')
                ax_3d.set_title(f'3D Point Visualization (Batch {b_idx_debug})\nGreen=Visible, Red=Occluded/Culled')

                all_points_np = points_to_check_world_b.cpu().numpy()
                if all_points_np.size > 0:
                    min_coords = all_points_np.min(axis=0)
                    max_coords = all_points_np.max(axis=0)
                    mids = (min_coords + max_coords) / 2
                    ranges = max_coords - min_coords
                    plot_radius = 0.5 * max(ranges[0], ranges[1], ranges[2], 1e-3)

                    ax_3d.set_xlim(mids[0] - plot_radius, mids[0] + plot_radius)
                    ax_3d.set_ylim(mids[1] - plot_radius, mids[1] + plot_radius)
                    ax_3d.set_zlim(mids[2] - plot_radius, mids[2] + plot_radius)
                    ax_3d.set_box_aspect((1,1,1))

                ax_3d.legend()
                plt.show()
            
            print(f"--- End Debugging Occluded/Culled Points for Batch {b_idx_debug} ---")

        elif camera_debug is None:
            print(f"--- Debugging Occluded/Culled Points: Camera for batch {b_idx_debug} not available. ---")
        elif not (b_idx_debug < len(points_list) and len(points_list[b_idx_debug]) > 0):
            print(f"--- Debugging Occluded/Culled Points: No points in points_list[{b_idx_debug}] to check. ---")

    # --- 2D Plotting ---
    if B > 0 and images is not None and b_idx_debug < images.shape[0] and b_idx_debug < pc_pix.shape[0] and b_idx_debug < existence_mask.shape[0]:
        rasterized_image = fragments.idx[b_idx_debug, ..., 0].cpu().squeeze()
        input_image = images[b_idx_debug, 0].cpu().squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        im1 = axes[0].imshow(rasterized_image, cmap='viridis', origin='lower')
        axes[0].set_title(f"Rasterized Point Indices (Batch {b_idx_debug}, Point 0)")
        axes[0].set_xlabel("Pixel X")
        axes[0].set_ylabel("Pixel Y")
        fig.colorbar(im1, ax=axes[0], label="Point Index (-1 for no point)", fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(input_image, cmap='gray', origin='lower')
        axes[1].set_title(f"Input Image with Projected Points (Batch {b_idx_debug})")
        axes[1].set_xlabel("Pixel X")
        axes[1].set_ylabel("Pixel Y")
        fig.colorbar(im2, ax=axes[1], label="Pixel Intensity", fraction=0.046, pad=0.04)

        valid_points_mask_batch_debug = existence_mask[b_idx_debug].cpu().numpy()
        # Ensure pc_pix has enough points for the current batch before indexing
        if valid_points_mask_batch_debug.shape[0] == pc_pix.shape[2]: # Check if mask length matches num points
            pc_pix_batch_debug_x = pc_pix[b_idx_debug, 0, valid_points_mask_batch_debug].cpu().numpy()
            pc_pix_batch_debug_y = pc_pix[b_idx_debug, 1, valid_points_mask_batch_debug].cpu().numpy()
            
            axes[0].scatter(pc_pix_batch_debug_x, pc_pix_batch_debug_y, c='red', s=5, label='Projected Points (pc_pix)')
            axes[1].scatter(pc_pix_batch_debug_x, pc_pix_batch_debug_y, c='red', s=5, label='Projected Points (pc_pix)')
        else:
            print(f"Warning: Mismatch between existence_mask and pc_pix dimensions for batch {b_idx_debug}. Skipping scatter plot of pc_pix.")


        axes[0].legend()
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


def get_most_common_rasterized_index_per_batch_item(fragments_idx: torch.Tensor) -> torch.Tensor:
    """
    For each item in a batch of rasterized fragment indices, finds the point index 
    that appears most frequently, excluding -1.

    The indices found are typically local indices into a list of points that were 
    rasterized for that specific batch item.

    Args:
        fragments_idx (torch.Tensor): Tensor of shape [B, H, W, K] containing
                                      rasterized point indices. -1 indicates no point.
                                      B is batch_size, H, W are spatial dimensions,
                                      K is points_per_pixel.

    Returns:
        torch.Tensor: Tensor of shape [B, 1] containing the most common point index
                      for each batch item. If a batch item has no valid points 
                      (all -1 or no points rasterized), -1 is returned for that item.
    """
    batch_size = fragments_idx.shape[0]
    device = fragments_idx.device
    most_common_indices_list = []

    for b_idx in range(batch_size):
        # Get fragments for the current batch item: [H, W, K]
        current_fragments = fragments_idx[b_idx] 
        
        # Flatten to [H*W*K]
        flat_indices = current_fragments.reshape(-1)
        
        # Filter out -1 values
        valid_indices = flat_indices[flat_indices != -1]
        
        if valid_indices.numel() == 0:
            # No valid points (e.g., all -1 or no points rasterized) for this batch item
            most_common_indices_list.append(torch.tensor(-1, device=device, dtype=torch.long))
        else:
            # Get unique values and their counts
            unique_elements, counts = torch.unique(valid_indices, return_counts=True)
            
            # Find the index of the maximum count
            max_count_idx = torch.argmax(counts)
            
            # Get the most common element (which is a point index)
            most_common_element = unique_elements[max_count_idx]
            most_common_indices_list.append(most_common_element)

    if most_common_indices_list:
        # Stack the results and ensure shape [B, 1]
        output_most_common_indices = torch.stack(most_common_indices_list).unsqueeze(1)
    else:
        # Handle empty input batch case (B=0)
        output_most_common_indices = torch.empty((0, 1), device=device, dtype=torch.long)
            
    return output_most_common_indices

class GrayscalePointCloudProjectionModel(ModelMixin):
    """
    Point cloud projection model adapted for grayscale images and orthographic cameras.
    Key differences from the original PointCloudProjectionModel:
    - Uses grayscale images (1 channel) instead of RGB
    - Uses orthographic camera with larger z meaning closer to camera
    - Handles 4D point clouds where 4th dimension is existence (binary)
    - Filters out padded points during rasterization
    - Samples features from U-Net encoder outputs at different scales
    """
    
    def __init__(
        self,
        image_size: int = 128,
        use_encoder_features: bool = True,
        use_mask: bool = True,
        use_distance_transform: bool = True,
        use_thickness_map: bool = True,
        # Rasterization settings
        raster_radius: float = 0.025,
        raster_points_per_pixel: int = 1,
        # Encoder feature sampling
        features_to_use: list = None,
        scale_factor: float = 1.0,
        # Add binarization_threshold and depth_epsilon to __init__ for consistency
        binarization_threshold: float = 0.5, # Assuming a default threshold
        depth_epsilon: float = 0.5, # Assuming a default epsilon
    ):
        super().__init__()
        self.image_size = image_size
        self.use_encoder_features = use_encoder_features
        self.use_mask = use_mask
        self.use_distance_transform = use_distance_transform
        self.use_thickness_map = use_thickness_map
        self.scale_factor = scale_factor
        self.binarization_threshold = binarization_threshold
        self.depth_epsilon = depth_epsilon

        if features_to_use is None:
            features_to_use = [0, 2, 4] # Default to using features from U-Net layers x5, x3, x1
        self.features_to_use = features_to_use
        
        self.raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=raster_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=0, # bin_size=0 means naive rasterization used by default.
        )
        
    def create_orthographic_camera(self, batch_size: int, pixel_sizes: Tensor,cell_size, device: torch.device) -> FoVOrthographicCameras: # Changed return type hint
        """
        Create orthographic camera.
        - Projection is along the Z-axis.
        - World (x=0, y=0) is the center of projection in the XY plane.
        - Larger world Z values are treated as closer to the camera.
          To achieve this with standard NDC depth (where smaller z_ndc values are closer),
          the camera's Z-axis is inverted relative to the world's Z-axis.
        """
        R_mat = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device))
        R = R_mat.unsqueeze(0).repeat(batch_size, 1, 1)
        
        T = torch.zeros(batch_size, 3, device=device)
        
        # Define base frustum bounds assuming world coordinates are normalized [-1.0, 1.0]
        # and after R_zz=-1.0, camera Z is also in [-1.0, 1.0]

        #We need to set x and y bounds based on image size and pixel size and each item in batch needs it$s own camera
        #x extent is (image_size * pixel_size / 2.0) / cell_size
        #y extent is (image_size * pixel_size / 2.0) / cell_size
        
        min_x_cam = -((self.image_size * pixel_sizes) / 2.0) / (cell_size / 2.0)  # Convert to world coordinates
        max_x_cam = ((self.image_size * pixel_sizes) / 2.0) / (cell_size / 2.0)
        min_y_cam = -((self.image_size * pixel_sizes) / 2.0) / (cell_size / 2.0)
        max_y_cam = ((self.image_size * pixel_sizes) / 2.0) / (cell_size / 2.0)

        # Z-clipping planes in camera space (smaller Z is closer)
        # If world Z: [-1.0 (far), 1.0 (near)]
        # Then camera Z (-world Z): [-1.0 (near), 1.0 (far)]
        znear_cam = -100.0 # Nearest clipping plane in camera Z (most negative value)
        zfar_cam = 100.0   # Farthest clipping plane in camera Z (most positive value)
        
        camera = FoVOrthographicCameras( # CHANGED to FoVOrthographicCameras
            R=R,
            T=T,
            min_x=min_x_cam,
            max_x=max_x_cam,
            min_y=min_y_cam,
            max_y=max_y_cam,
            znear=znear_cam,
            zfar=zfar_cam,
            device=device
            # REMOVED image_size argument as FoVOrthographicCameras does not use it
        )
        return camera
        
    def filter_padded_points(self, point_clouds_4d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        points_3d = point_clouds_4d[:, :3, :]
        existence_mask = point_clouds_4d[:, 3, :] > 0.5
        return points_3d, existence_mask
               
    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection_orthographic(
        self,
        points: torch.Tensor,
        existence_mask: torch.Tensor,
        local_features: torch.Tensor,
        pixel_sizes: torch.Tensor,
        cell_size: float,
        cameras: Optional[FoVOrthographicCameras] = None, # Updated type hint
        images: Optional[torch.Tensor] = None,
        debug_plots: bool = False, # Add a flag to control debugging
    ) -> torch.Tensor:
        B, C_pts_in, N_points = points.shape # C_pts_in is likely 3 for XYZ
        # H_feat, W_feat are from local_features
        _, C_feat, H_feat_map, W_feat_map = local_features.shape 
        device = local_features.device
        R_points_per_pixel = self.raster_settings.points_per_pixel
        
        # print(f"Surface projection with {B} batches, {N_points} points, feature map size {H_feat_map}x{W_feat_map}, points per pixel {R_points_per_pixel}")
        
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=self.raster_settings)

        # IMPORTANT CHANGE: Clone points before any modification for rasterization
        # to avoid affecting the original 'points' tensor (which is noisy_xyz_input).
        points_for_rasterizer_world = points.permute(0, 2, 1).clone() # [B, N, 3]
        
        # The following Z-shift was likely for rasterizer stability.
        # It should operate on the clone, not the original.
        # Consider if this shift is still necessary given the camera's znear/zfar.
        # For now, we keep it but ensure it's on the clone.
        # points_for_rasterizer_world_shifted_z = points_for_rasterizer_world.clone()
        # points_for_rasterizer_world_shifted_z[:, :, 2] -= points_for_rasterizer_world_shifted_z[:, :, 2].min(dim=1, keepdim=True).values
        # For Pytorch3D, Z is typically positive, increasing with distance.
        # Our camera R matrix makes Z_cam = Z_world. If Z_world is [-1, 1] (near, far),
        # then Z_cam is also [-1, 1]. The rasterizer should handle this with znear/zfar.
        # Let's try removing this ad-hoc shift and rely on the camera.
        # If issues persist, this line can be reinstated on points_for_rasterizer_world.clone()

        # print(f"Points shape for rasterizer: {points_for_rasterizer_world.shape}, existence mask shape: {existence_mask.shape}")
        # print(f"Points per batch: {points_for_rasterizer.shape[0]}, Points per batch item: {points_for_rasterizer.shape[1]}, Features per point: {points_for_rasterizer.shape[2]}")
        # min_z = points_for_rasterizer[:, :, 2].min(dim=1).values
        # max_z = points_for_rasterizer[:, :, 2].max(dim=1).values
        # print(f"Min Z: {min_z}, Max Z: {max_z}")
        
        points_list = []
        for b_idx in range(B):
            valid_mask_b = existence_mask[b_idx] # [N_points]
            # Use the original Z values from points_for_rasterizer_world
            points_b = points_for_rasterizer_world[b_idx, valid_mask_b, :] # [num_valid_points_b, 3]
            points_list.append(points_b)
        
        pointclouds = Pointclouds(points=points_list, features=None)
        # fragments.idx shape: [B, H_img, W_img, R_points_per_pixel]
        # H_img, W_img are from self.raster_settings.image_size
        fragments = rasterizer(pointclouds)

        pc_xy = points[:, :2, :]
        pc_nm = pc_xy * (cell_size / 2.0)
        px_per_batch = pixel_sizes.view(B, 1, 1)
        # pc_pix: [B, 2, N_points] (col, row) in image pixel coordinates
        pc_pix = pc_nm / px_per_batch + (self.image_size / 2.0) 

        if debug_plots: # Control debugging via the flag
            _debug_and_plot_rasterization(
                fragments=fragments,
                points_list=points_list, # List of [num_valid_points_b, 3]
                cameras=cameras,
                images=images,
                pc_pix=pc_pix, # Used for plotting original projections
                existence_mask=existence_mask, # [B, N_points]
                raster_settings=self.raster_settings,
                b_idx_debug=0, 
                debug_culled_points=True 
            )


        #now we have 128x128 image with indices of points projected to it, 128x128 input image, a feature map that is either Cx128x128, Cx32x32, or Cx8x8
        """
        for each element in batch, check fragments to find the most common value besides -1. 
        The number of pixels that have this value, is the base size. For each other index in fragments, 
        divide number of pixels with this index by the base size to get the relative size of the point.
        Now make a vector of visibility weights for each element in batch which will act as a scale factor for the features.
        """
        
        # Step 1: Find the most common rasterized index (index into points_list[b_idx]) for each batch item.
        # This uses the function you requested.
        most_common_indices_in_points_list_per_batch = get_most_common_rasterized_index_per_batch_item(fragments.idx)
        # most_common_indices_in_points_list_per_batch has shape [B, 1]

        # Continue with the logic described in the docstring to calculate visibility weights
        visibility_weights_all_batches = []
        for b_idx in range(B):
            # Get all rasterized indices for this batch item (these are indices into points_list[b_idx])
            flat_rasterized_indices_b = fragments.idx[b_idx].reshape(-1)
            valid_rasterized_indices_b = flat_rasterized_indices_b[flat_rasterized_indices_b != -1]

            # Initialize weights for all N_points (original number of points for this batch item) to 0.
            current_visibility_weights = torch.zeros(N_points, device=device, dtype=torch.float32)

            if valid_rasterized_indices_b.numel() > 0:
                unique_rasterized_idxs_in_list, counts_per_rasterized_idx = torch.unique(valid_rasterized_indices_b, return_counts=True)
                
                # Determine base_size: count of the most common point for this batch item
                most_common_idx_val_in_list = most_common_indices_in_points_list_per_batch[b_idx].item()
                base_size = 0.0
                if most_common_idx_val_in_list != -1:
                    # Find its count among the unique_rasterized_idxs_in_list
                    match_mask = (unique_rasterized_idxs_in_list == most_common_idx_val_in_list)
                    if torch.any(match_mask):
                        base_size = counts_per_rasterized_idx[match_mask][0].float()
                
                if base_size == 0: # Fallback if most_common_idx_val_in_list was -1 or not found (e.g. if all points were unique)
                    if counts_per_rasterized_idx.numel() > 0: # Ensure there are counts
                         base_size = torch.max(counts_per_rasterized_idx).float() # Use max count as base
                    else: # Should not happen if valid_rasterized_indices_b.numel() > 0
                        base_size = 1.0 
                if base_size < 1e-6 : base_size = 1.0 # Avoid division by zero or very small numbers

                # `original_indices_for_valid_points_b` maps from index in `points_list[b_idx]` to original index in `0..N_points-1`
                original_indices_for_valid_points_b = torch.where(existence_mask[b_idx])[0] # Shape [num_valid_points_b]

                for i in range(unique_rasterized_idxs_in_list.numel()):
                    rasterized_idx_in_list = unique_rasterized_idxs_in_list[i] # This is an index into points_list[b_idx]
                    count_of_this_idx = counts_per_rasterized_idx[i].float()
                    relative_size = count_of_this_idx / base_size
                    
                    # Map this local rasterized_idx_in_list back to its original point index (0 to N_points-1)
                    if rasterized_idx_in_list.item() < len(original_indices_for_valid_points_b):
                        original_pt_idx = original_indices_for_valid_points_b[rasterized_idx_in_list.item()]
                        current_visibility_weights[original_pt_idx] = relative_size
            
            visibility_weights_all_batches.append(current_visibility_weights)

        if visibility_weights_all_batches:
            visibility_weights_tensor = torch.stack(visibility_weights_all_batches) # Shape [B, N_points]
        else: # Should not happen if B > 0
            visibility_weights_tensor = torch.zeros((B, N_points), device=device, dtype=torch.float32)


        #print(f"Visibility weights tensor shape: {visibility_weights_tensor.shape}, dtype: {visibility_weights_tensor.dtype}")
        # Sample the local_features using pc_pix
        # local_features: [B, C_feat, H_feat_map, W_feat_map]
        # pc_pix: [B, 2, N_points] (col, row)
        
        # Normalize pc_pix for grid_sample. Grid should be in range [-1, 1]
        # grid_x corresponds to W_feat_map, grid_y to H_feat_map
        grid_x = pc_pix[:, 0, :] / (W_feat_map - 1) * 2.0 - 1.0    # [B, N_points]
        grid_y = pc_pix[:, 1, :] / (H_feat_map - 1) * 2.0 - 1.0    # [B, N_points]
        grid = torch.stack((grid_x, grid_y), dim=-1)             # [B, N_points, 2]
        grid = grid.unsqueeze(1).float()                         # [B, 1, N_points, 2] for F.grid_sample

        sampled_lf = F.grid_sample(local_features.float(), grid,
                                   mode='bilinear', padding_mode='zeros', align_corners=True) # [B, C_feat, 1, N_points]
        sampled_lf = sampled_lf.squeeze(2) # [B, C_feat, N_points]

        # Apply visibility weights
        # visibility_weights_tensor: [B, N_points]
        # sampled_lf: [B, C_feat, N_points]
        # Unsqueeze visibility_weights_tensor to [B, 1, N_points] for broadcasting
        scaled_sampled_lf = sampled_lf * visibility_weights_tensor.unsqueeze(1) 
        
        # Permute to [B, N_points, C_feat] as expected by the calling function structure
        output_features = scaled_sampled_lf.permute(0, 2, 1) 

        return output_features

    def get_local_conditioning(
        self,
        point_clouds_4d: torch.Tensor,
        pixel_sizes: torch.Tensor,
        images: torch.Tensor,
        projection_model: torch.nn.Module, # U-Net model
        cell_size: float,
        device: torch.device,
        debug_occlusion: bool = False
    ) -> torch.Tensor:
        B, _, N_points = point_clouds_4d.shape
        H_img, W_img = images.shape[-2:]

        points_3d, existence_mask = self.filter_padded_points(point_clouds_4d)
        # points_3d: [B, 3, N_points], existence_mask: [B, N_points]

        cameras = self.create_orthographic_camera(
            batch_size=B, 
            pixel_sizes=pixel_sizes, 
            cell_size=cell_size, 
            device=device 
        )

        with torch.no_grad():
            # Assuming projection_model (U-Net) returns (thickness_map_raw, encoder_features_tuple)
            raw_thickness_map, unet_encoder_features_tuple = projection_model(images)
        
        all_sampled_features_list = []

        if self.use_encoder_features:
            for i in self.features_to_use:
                if i < len(unet_encoder_features_tuple):
                    current_unet_features = unet_encoder_features_tuple[i].to(device)
                    sampled_encoder_f = self.surface_projection_orthographic(
                        points=points_3d,
                        existence_mask=existence_mask,
                        local_features=current_unet_features,
                        pixel_sizes=pixel_sizes,
                        cell_size=cell_size,
                        cameras=cameras,
                        images=images if debug_occlusion else None,
                        debug_plots=debug_occlusion
                    )
                    # surface_projection_orthographic returns [B, N_points, C_feat]
                    # Permute to [B, C_feat, N_points] for concatenation
                    all_sampled_features_list.append(sampled_encoder_f.permute(0, 2, 1))
                else:
                    print(f"Warning: GrayscalePointCloudProjectionModel: feature index {i} is out of bounds for unet_encoder_features_tuple (length {len(unet_encoder_features_tuple)})")

        pc_xy = point_clouds_4d[:, :2, :]
        pc_nm = pc_xy * (cell_size / 2.0)
        px_per_batch = pixel_sizes.view(B, 1, 1)
        pc_pix = pc_nm / px_per_batch + (self.image_size / 2.0)

        grid_x_img_res = pc_pix[:, 0, :] / (W_img - 1) * 2.0 - 1.0 
        grid_y_img_res = pc_pix[:, 1, :] / (H_img - 1) * 2.0 - 1.0 
        grid_img_res = torch.stack((grid_x_img_res, grid_y_img_res), dim=-1).to(device) 
        grid_img_res = grid_img_res.unsqueeze(1) 

        if self.use_thickness_map:
            sampled_raw_thickness = F.grid_sample(
                raw_thickness_map.to(device), 
                grid_img_res,
                mode='bilinear', padding_mode='zeros', align_corners=True
            ) 
            all_sampled_features_list.append(sampled_raw_thickness.squeeze(2))

        binary_mask_tensor = None
        if self.use_mask or self.use_distance_transform:
            binary_mask_tensor = (raw_thickness_map.to(device) > self.binarization_threshold).float()

            if self.use_mask:
                sampled_mask = F.grid_sample(
                    binary_mask_tensor, 
                    grid_img_res,
                    mode='nearest', padding_mode='zeros', align_corners=True
                )
                all_sampled_features_list.append(sampled_mask.squeeze(2))

        if self.use_distance_transform:
            if binary_mask_tensor is None:
                 binary_mask_tensor = (raw_thickness_map.to(device) > self.binarization_threshold).float()
            
            batch_dist_vec_maps = []
            for b_idx in range(B):
                mask_np = binary_mask_tensor[b_idx, 0].cpu().numpy()
                inv_mask_np = 1 - mask_np
                _, (indices_y, indices_x) = scipy.ndimage.distance_transform_edt(
                    inv_mask_np, return_indices=True, sampling=[1,1] # Added sampling for consistency
                )
                grid_y_np, grid_x_np = np.mgrid[0:H_img, 0:W_img]
                vec_y_np = indices_y - grid_y_np
                vec_x_np = indices_x - grid_x_np
                dist_vec_map_np = np.stack((vec_x_np, vec_y_np), axis=0).astype(np.float32)
                batch_dist_vec_maps.append(torch.from_numpy(dist_vec_map_np))
            
            distance_vectors_map_tensor = torch.stack(batch_dist_vec_maps).to(device)

            sampled_dist_vectors = F.grid_sample(
                distance_vectors_map_tensor, 
                grid_img_res,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )
            all_sampled_features_list.append(sampled_dist_vectors.squeeze(2))

        if not all_sampled_features_list:
            return torch.empty(B, 0, N_points, device=device) 
        
        concatenated_features = torch.cat(all_sampled_features_list, dim=1)
        return concatenated_features

# Your existing function outside the class
def condition_point_cloud_with_projection_model(
    point_clouds, 
    pixel_sizes, 
    images, 
    projection_model, 
    device, 
    cfg
):
    conditioning_cfg = cfg.model.get('conditioning', {})
    debug_occlusion = True
    
    grayscale_proj_model = GrayscalePointCloudProjectionModel(
        image_size=images.shape[-1],
        use_encoder_features=conditioning_cfg.get('use_encoder_features', True),
        use_mask=conditioning_cfg.get('use_mask', True),
        use_distance_transform=conditioning_cfg.get('use_distance_transform', True),
        use_thickness_map=conditioning_cfg.get('use_thickness_map', True),
        raster_radius=conditioning_cfg.get('raster_radius', 0.025),
        raster_points_per_pixel=conditioning_cfg.get('raster_points_per_pixel', 8),
        depth_epsilon=conditioning_cfg.get('depth_epsilon', 0.5),
        features_to_use=conditioning_cfg.get('features_to_use', [0, 2, 4]),
        binarization_threshold=conditioning_cfg.get('binarization_threshold', 0.01),
    )
    
    conditioning_features = grayscale_proj_model.get_local_conditioning(
        point_clouds_4d=point_clouds.to(device),
        pixel_sizes=pixel_sizes.to(device),
        images=images.to(device),
        projection_model=projection_model.to(device).eval(),
        cell_size=cfg.data.cell_size,
        device=device,
        debug_occlusion=debug_occlusion,
    )
    
    conditioning_features_transposed = conditioning_features.transpose(1, 2)
    conditioned_pc = torch.cat((point_clouds.to(device), conditioning_features_transposed), dim=1)
    
    return conditioned_pc