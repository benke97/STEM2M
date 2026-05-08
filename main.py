# main.py
import logging
import random

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler, DDPMScheduler
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from mar.data.data_module import setup_dataloaders
from mar.models.modern_unet import ModernUNet
from mar.models.pvcnn.pvcnn import PVCNN2
from mar.models.test import test_conditioned
from mar.models.train import (
    train_conditioned_structure_predictor,
    train_unconditioned_structure_predictor,
)
from mar.utils.helpers import (
    create_custom_beta_schedule,
    create_custom_beta_schedule_exp,
)

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_noise_scheduler(cfg: DictConfig):
    """Construct a DDPM/DDIM scheduler from cfg.training.diffusion."""
    diff = cfg.training.diffusion

    if diff.beta_schedule == "custom":
        custom_betas = create_custom_beta_schedule(
            num_train_timesteps=diff.num_timesteps,
            alpha_bar_power=diff.alpha_bar_power,
            target_alpha_bar_T=diff.target_alpha_bar_T,
            beta_clip_max=diff.beta_clip_max,
        )
    elif diff.beta_schedule == "exp_growth":
        custom_betas = create_custom_beta_schedule_exp()
    else:
        custom_betas = None

    if diff.scheduler == "ddpm":
        if custom_betas is not None:
            return DDPMScheduler(
                num_train_timesteps=diff.num_timesteps,
                trained_betas=custom_betas,
                prediction_type="epsilon",
                clip_sample=False,
            )
        return DDPMScheduler(
            num_train_timesteps=diff.num_timesteps,
            beta_start=diff.beta_start,
            beta_end=diff.beta_end,
            beta_schedule=diff.beta_schedule,
            prediction_type="epsilon",
            clip_sample=False,
        )

    if diff.scheduler == "ddim":
        if custom_betas is None:
            raise ValueError("DDIM currently requires beta_schedule='custom'.")
        return DDIMScheduler(
            num_train_timesteps=diff.num_timesteps,
            trained_betas=custom_betas,
            prediction_type="epsilon",
            clip_sample=False,
        )

    raise ValueError(f"Unknown diffusion scheduler: {diff.scheduler}")


def _instantiate_pc_model(cfg: DictConfig, accelerator: Accelerator) -> PVCNN2:
    if cfg.model.pc_model.name != "pvcnn":
        raise ValueError(f"Unknown pc_model name: {cfg.model.pc_model.name}")

    pc_model = PVCNN2(
        num_classes=cfg.model.pc_model.num_classes,
        embed_dim=cfg.model.pc_model.embed_dim,
        dropout=cfg.model.pc_model.dropout,
        extra_feature_channels=cfg.model.pc_model.extra_feature_channels,
        width_multiplier=cfg.model.pc_model.width_multiplier,
        voxel_resolution_multiplier=cfg.model.pc_model.voxel_resolution_multiplier,
        use_pixel_size_embedding=cfg.model.pc_model.get("use_pixel_size_embedding", True),
    )

    if cfg.model.pc_model.pretrained:
        pc_path = to_absolute_path(cfg.model.pc_model.pretrained_path)
        checkpoint = torch.load(pc_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            pc_model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, PVCNN2):
            pc_model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict):
            pc_model.load_state_dict(checkpoint)
        else:
            raise TypeError(f"Unknown checkpoint type: {type(checkpoint)}")
        if accelerator.is_main_process:
            log.info(f"Loaded pretrained pc_model weights from {pc_path}")

    return pc_model


def _instantiate_projection_model(cfg: DictConfig, accelerator: Accelerator) -> ModernUNet:
    if cfg.model.projection_model.name != "ModernUNet":
        raise ValueError(f"Unknown projection_model name: {cfg.model.projection_model.name}")

    projection_model = ModernUNet(
        n_channels=cfg.model.projection_model.params.n_channels,
        n_classes=cfg.model.projection_model.params.n_classes,
        init_features=cfg.model.projection_model.params.init_features,
        num_groups=cfg.model.projection_model.params.num_groups,
    )

    if cfg.model.projection_model.pretrained:
        proj_path = to_absolute_path(cfg.model.projection_model.pretrained_path)
        checkpoint = torch.load(proj_path, map_location="cpu", weights_only=False)
        projection_model.load_state_dict(checkpoint["model_state_dict"])
        if accelerator.is_main_process:
            log.info(f"Loaded pretrained projection_model weights from {proj_path}")

    return projection_model


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator()

    if accelerator.is_main_process:
        log.info(f"Accelerator state: {accelerator.state}")
        log.info("Starting run with configuration:")
        log.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.main.seed)
    if accelerator.is_main_process:
        log.info(f"Random seed set to: {cfg.main.seed}")

    if accelerator.is_main_process:
        if accelerator.device.type == "cuda":
            log.info(f"Using GPU: {torch.cuda.get_device_name(0)} (managed by Accelerate)")
        else:
            log.info(f"Using {accelerator.device.type.upper()} (managed by Accelerate)")

    cfg.data.augment = cfg.data.get("augment", True)
    cfg.data.dataset_path = to_absolute_path(cfg.data.dataset_path)

    train_loader, val_loader, test_loader, _ = setup_dataloaders(
        cfg.data, cfg.main.seed, cfg.main.task
    )
    if train_loader is None:
        if accelerator.is_main_process:
            log.error("Failed to setup dataloaders. Exiting.")
        return

    if accelerator.is_main_process:
        log.info(f"Instantiating diffusion model: {cfg.model.pc_model.name}")
    pc_model = _instantiate_pc_model(cfg, accelerator)
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in pc_model.parameters() if p.requires_grad)
        log.info(f"pc_model: {type(pc_model).__name__} ({n_params:,} params)")

    if accelerator.is_main_process:
        log.info("Instantiating projection model")
    projection_model = _instantiate_projection_model(cfg, accelerator)
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in projection_model.parameters() if p.requires_grad)
        log.info(f"projection_model: {type(projection_model).__name__} ({n_params:,} params)")

    if accelerator.is_main_process:
        log.info("Setting up diffusion process...")
    noise_scheduler = _build_noise_scheduler(cfg)

    pc_model, projection_model, train_loader, val_loader, test_loader = accelerator.prepare(
        pc_model, projection_model, train_loader, val_loader, test_loader
    )

    if cfg.main.task == "train":
        if cfg.training.diffusion.with_conditioning:
            if accelerator.is_main_process:
                log.info("Training with conditioning...")
            train_conditioned_structure_predictor(
                accelerator=accelerator,
                model=pc_model,
                projection_model=projection_model,
                train_loader=train_loader,
                val_loader=val_loader,
                noise_scheduler=noise_scheduler,
                cfg=cfg,
            )
        else:
            if accelerator.is_main_process:
                log.info("Training without conditioning...")
            train_unconditioned_structure_predictor(
                accelerator=accelerator,
                model=pc_model,
                train_loader=train_loader,
                val_loader=val_loader,
                noise_scheduler=noise_scheduler,
                cfg=cfg,
            )
    elif cfg.main.task == "test":
        if accelerator.is_main_process:
            log.info("Testing the model...")
        test_conditioned(
            accelerator=accelerator,
            pc_model=pc_model,
            projection_model=projection_model,
            test_loader=test_loader,
            noise_scheduler=noise_scheduler,
            cfg=cfg,
        )
    else:
        if accelerator.is_main_process:
            log.error(
                f"Unknown task: {cfg.main.task!r}. Supported tasks in this slim build: 'train', 'test'."
            )


if __name__ == "__main__":
    main()
