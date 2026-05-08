import torch
from tqdm import tqdm
from pathlib import Path
import logging
from accelerate import Accelerator
from hydra.utils import to_absolute_path
from mar.utils.conditioning import condition_point_cloud
from mar.utils.visualization import visualize_conditioned
from mar.models.resnet_size_estimator import ResNet18AdaptivePoolFeatureExtractor

log = logging.getLogger(__name__)

def test_conditioned(accelerator: Accelerator, pc_model, projection_model, test_loader, noise_scheduler, cfg):
    """
    Test function for conditioned point cloud generation with Accelerator support.
    
    Args:
        accelerator: The Accelerator instance to handle distributed execution
        pc_model: The point cloud diffusion model
        projection_model: The 2D projection model for conditioning
        test_loader: DataLoader for test data
        noise_scheduler: Noise scheduler for diffusion sampling
        cfg: Configuration object
    """
    # Only run visualization on the main process
    if not accelerator.is_main_process:
        return
    
    log.info(f"Testing model with {cfg.test.num_samples_to_vis} samples")
    
    # Ensure models are in eval mode
    pc_model.eval()
    projection_model.eval()
    
    # Create output directory if it doesn't exist
    #output_dir = Path(cfg.paths.test_output_dir)
    #output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Initializing size estimator...")
    size_estimator_weights_path = to_absolute_path(cfg.model.size_estimator_weights_path)

    size_estimator = ResNet18AdaptivePoolFeatureExtractor(
        input_nc=1,
        output_feature_dim=64,
        pretrained_estimator_path=size_estimator_weights_path
    )
    # The ResNet18AdaptivePoolFeatureExtractor's __init__ method now handles loading the weights
    # into its backbone. The load_pretrained_backbone_weights method loads to CPU by default,
    # which is fine as we move the model to the accelerator.device next.
    
    log.info(f"Size estimator initialized. Pretrained weights for backbone loaded from {size_estimator_weights_path}.")
    size_estimator.to(accelerator.device)
    size_estimator.eval()
    # Sample from test_loader
    test_iter = iter(test_loader)
    
    # Use tqdm for progress feedback
    for i in tqdm(range(cfg.test.num_samples_to_vis), desc="Generating test samples"):
        try:
            example_batch = next(test_iter)
        except StopIteration:
            # Reset iterator when exhausted
            test_iter = iter(test_loader)
            example_batch = next(test_iter)
        
        try:
            # visualize_conditioned expects models to be unwrapped
            unwrapped_pc_model = accelerator.unwrap_model(pc_model)
            unwrapped_proj_model = accelerator.unwrap_model(projection_model)
            
            # Visualization function
            visualize_conditioned(
                unwrapped_pc_model,
                unwrapped_proj_model,
                size_estimator,
                noise_scheduler,
                example_batch,
                epoch=i,  # Using i instead of epoch since we're in test mode
                device=accelerator.device,
                cfg=cfg,
            )
            
            log.info(f"Generated test sample {i+1}/{cfg.test.num_samples_to_vis}")
            
        except Exception as e:
            log.error(f"Error generating test sample {i}: {e}", exc_info=True)
    
    #log.info(f"Testing complete. Results saved to {output_dir}")

