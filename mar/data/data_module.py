# mar/data/data_module.py
import datetime
import json
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

from .dataset import MorphologyDataset

log = logging.getLogger(__name__) # Get logger for this module

def save_split_info(split_info, split_file):
    """Saves split information to a JSON file."""
    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    try:
        with open(split_file, "w") as f:
            json.dump(split_info, f, indent=2)
        log.info(f"Split indices saved to {split_file}")
    except IOError as e:
        log.error(f"Could not save split info to {split_file}: {e}")

def load_split_info(split_file):
    """Loads split information from a JSON file."""
    split_file = Path(split_file)
    if not split_file.is_file():
        log.error(f"Split file not found: {split_file}")
        return None
    try:
        with open(split_file, "r") as f:
            split_info = json.load(f)
        log.info(f"Loaded split indices from {split_file}")
        return split_info
    except (IOError, json.JSONDecodeError) as e:
        log.error(f"Could not load or parse split info from {split_file}: {e}")
        return None


def setup_dataloaders(cfg_data, seed, task):
    """
    Loads the dataset, performs splitting, and creates DataLoaders.

    Args:
        cfg_data: The data configuration section (OmegaConf object).
        seed: The random seed for shuffling and splitting.

    Returns:
        tuple: (train_loader, val_loader, test_loader, full_dataset)
               Returns (None, None, None, None) if dataset loading fails.
    """
    log.info("Setting up dataset and dataloaders...")
    try:
        full_dataset = MorphologyDataset(
            hdf5_path=cfg_data.dataset_path,
            data_type=cfg_data.data_type,
            image_type=cfg_data.image_type,
            point_cloud_size=cfg_data.point_cloud_size,
            cell_size=cfg_data.cell_size,
            augment=cfg_data.get('augment', True),
        )
        log.info(f"Full dataset loaded. Length: {len(full_dataset)}")
    except FileNotFoundError:
         log.error(f"Dataset file not found at {cfg_data.dataset_path}. Aborting.")
         return None, None, None, None
    except Exception as e:
         log.error(f"Failed to load dataset: {e}")
         return None, None, None, None


    train_indices, val_indices, test_indices = None, None, None
    split_info = None

    # Option 1: Load existing split
    if cfg_data.split.use_existing_split:
        split_info = load_split_info(cfg_data.split.use_existing_split)
        if split_info:
            train_indices = split_info.get("train", {}).get("indices")
            val_indices = split_info.get("val", {}).get("indices")
            test_indices = split_info.get("test", {}).get("indices")
            if not all([train_indices, val_indices, test_indices]):
                log.warning("Loaded split file is missing indices for train, val, or test. Will generate new split.")
                split_info = None # Force regeneration

    # Option 2: Generate new split if not loaded or forced
    if not split_info:
        log.info("Generating new train/val/test split...")
        np.random.seed(seed) # Use the global seed

        # Map enumerated indices to original HDF5 indices for later type checking
        # full_dataset.indices is a list of (data_type, original_idx)
        # data_type 0 = real, 1 = synthetic
        # We store the enumerated index 'i'
        real_indices_enum_set = {i for i, (data_type, _) in enumerate(full_dataset.indices) if data_type == 0}
        synthetic_indices_enum_set = {i for i, (data_type, _) in enumerate(full_dataset.indices) if data_type == 1}

        real_indices_list = list(real_indices_enum_set)
        synthetic_indices_list = list(synthetic_indices_enum_set)

        log.info(f"Found {len(real_indices_list)} real samples and {len(synthetic_indices_list)} synthetic samples.")

        if len(real_indices_list) < cfg_data.split.test_real_samples:
             log.error(f"Not enough real samples ({len(real_indices_list)}) for the requested test set ({cfg_data.split.test_real_samples}).")
             return None, None, None, full_dataset

        np.random.shuffle(real_indices_list)

        # Allocate real samples for the test set
        test_indices = real_indices_list[:cfg_data.split.test_real_samples]
        remaining_real_indices = real_indices_list[cfg_data.split.test_real_samples:]

        # Pool remaining real samples and all synthetic samples for train/val split
        pool_for_train_val = remaining_real_indices + synthetic_indices_list
        np.random.shuffle(pool_for_train_val)

        # Split the pool into validation and training sets using val_split ratio
        num_val_samples_from_pool = int(len(pool_for_train_val) * cfg_data.split.val_split)
        
        val_indices = pool_for_train_val[:num_val_samples_from_pool]
        train_indices = pool_for_train_val[num_val_samples_from_pool:]

        log.info(f"Split generated: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Calculate real/synthetic counts for each set
        train_real_count = sum(1 for i in train_indices if i in real_indices_enum_set)
        train_synthetic_count = len(train_indices) - train_real_count
        
        val_real_count = sum(1 for i in val_indices if i in real_indices_enum_set)
        val_synthetic_count = len(val_indices) - val_real_count
        
        test_real_count = len(test_indices) # Test set is all real

        # Prepare split info for saving
        split_info = {
            "train": {
                "indices": train_indices,
                "size": len(train_indices),
                "real_count": train_real_count,
                "synthetic_count": train_synthetic_count
            },
            "val": {
                "indices": val_indices,
                "size": len(val_indices),
                "real_count": val_real_count,
                "synthetic_count": val_synthetic_count # Added synthetic count for val
            },
            "test": {
                "indices": test_indices,
                "size": len(test_indices),
                "real_count": test_real_count # Test set is all real
            },
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "seed": seed,
                "dataset_path": str(cfg_data.dataset_path),
                "config_params": { # Log key config params used for split
                     "test_real_samples": cfg_data.split.test_real_samples,
                     "val_split_applied_to_remainder": cfg_data.split.val_split 
                     # cfg_data.split.val_real_samples is not directly used in this new logic for val set size
                }
            }
        }

        # Save the generated split
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        split_filename = f"split_indices_{timestamp}.json"
        split_output_dir = Path(cfg_data.split.output_dir)
        split_output_dir.mkdir(parents=True, exist_ok=True)
        split_file_path = split_output_dir / split_filename
        save_split_info(split_info, split_file_path)
        log.info(f"Saved newly generated split to {split_file_path}")


    # Create datasets using the indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg_data.batch_size,
        shuffle=True,
        num_workers=cfg_data.num_workers,
        pin_memory=True # Good practice if using GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg_data.batch_size,
        shuffle=False,
        num_workers=cfg_data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg_data.batch_size,
        shuffle=False,
        num_workers=cfg_data.num_workers,
        pin_memory=True
    )

    log.info("Dataloaders created successfully.")
    return train_loader, val_loader, test_loader, full_dataset