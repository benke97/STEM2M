# src/mar/data/dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)


# Module-level or Class-level constants for better maintainability
_OLD_CELL_SIZE = 8.0  # nm
_PT_LABEL = 78 / 118.0 # Ensure float division for compatibility
_TYPE_KEY_MAP = {0: 'real', 1: 'synthetic'}

# Augmentation constants, derived from original hardcoded values (64.0, 63.5),
# which imply an image size of 128x128 for the transformation space.
# If self.IMAGE_SIZE is available and should be used, these can be dynamic.
_AUG_IMG_CENTER_XYZ = torch.tensor([64.0, 64.0, 0.0], dtype=torch.float32)
_AUG_ROT_AXIS_XYZ = torch.tensor([63.5, 63.5, 0.0], dtype=torch.float32)


class AtomicReconstructionDataset(Dataset):
    # Keep the __init__ method as it was in the code you provided in the prompt
    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized', point_cloud_size=8192):
        """
        Args:
            hdf5_path: Path to the HDF5 file
            data_type: 'real', 'synthetic', or 'both'
            image_type: 'haadf_normalized' or 'haadf_cycleGAN'
            point_cloud_size: Target number of points in the point cloud.
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
             raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.CELL_SIZE = 8 #nm

        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.indices = [] # Will store (data_type_flag, original_index)

        # --- Use the more robust loading logic from previous versions ---
        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []

                # Load REAL data
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys)), key=int)
                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images
                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes
                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")

                # Load SYNTHETIC data
                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                 pc_group = f["point_clouds"]["synthetic"][key]
                                 synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds
                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
             raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")
    # --- End of __init__ ---


    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = 'real' if data_type_flag == 0 else 'synthetic'

        # Dictionary to hold the data for this item
        batch_data = {}

        try: # Added try-except block around data loading/processing
            # Load point cloud data
            point_cloud_data = self.point_clouds[type_key][key_str]

            # Extract coordinates
            x_data = point_cloud_data['x']
            y_data = point_cloud_data['y']
            z_data = point_cloud_data['z']

            # Stack coordinates
            point_cloud_np = np.stack([x_data, y_data, z_data], axis=1).astype(np.float32)

            # --- Point cloud size handling: Preserve originals when padding ---
            current_size = point_cloud_np.shape[0]
            target_size = self.point_cloud_size

            if current_size == 0:
                # If empty, fill with zeros (or potentially random noise if preferred)
                point_cloud_final_np = np.zeros((target_size, 3), dtype=np.float32)
                # Mask is all zeros for empty clouds
                mask = np.zeros(target_size, dtype=np.float32) # Indicates all points are padding

            elif current_size < target_size:
                # --- Use (0, 0, 0) padding instead of duplicating random points ---
                additional_needed = target_size - current_size

                # Create padding points (0, 0, 0)
                padding_points = np.zeros((additional_needed, 3), dtype=np.float32)

                # Combine original points with padding points
                point_cloud_final_np = np.vstack((point_cloud_np, padding_points))

                # Mask indicates which points are original (1) and which are padding (0)
                mask = np.zeros(target_size, dtype=np.float32)
                mask[:current_size] = 1.0  # Mark the original points as valid/real

            elif current_size > target_size:
                # Sample points randomly (Subsampling)
                # This inherently selects a *subset* of original points.
                # The *values* of the kept points are unaltered.
                indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                point_cloud_final_np = point_cloud_np[indices_to_keep]
                # Mask is all ones (all points are real, just sampled)
                mask = np.ones(target_size, dtype=np.float32)

            else: # current_size == target_size
                point_cloud_final_np = point_cloud_np
                # Mask is all ones (all points are original)
                mask = np.ones(target_size, dtype=np.float32)

            # Data augmentation: rotation
            num_rotations = np.random.randint(0, 4) # Randomly choose a rotation (0, 90, 180, or 270 degrees)
            num_rotations = 0
            #rotate point cloud around z-axis
            # Convert to tensor for rotation
            pt_cloud_tensor = torch.from_numpy(point_cloud_final_np).float()
            # Create rotation matrix using torch
            theta = torch.tensor(np.pi/2 * num_rotations).float()
            rotation_matrix = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
            ]).float()
            # Apply rotation using torch operations
            #point_cloud_final_np = torch.matmul(pt_cloud_tensor, rotation_matrix.t()).numpy()
            point_cloud_final_np = pt_cloud_tensor.numpy()
            # Data augmentation: horizontal reflection (flip along x-axis)
            should_flip = np.random.random() > 0.5  # 50% chance of flipping
            should_flip = False
            if should_flip:
                # Flip point cloud along x-axis by negating x-coordinates
                point_cloud_final_np[:, 0] = -point_cloud_final_np[:, 0]

            # Convert point cloud to tensor C×N format (3 × target_size)
            point_cloud_tensor = torch.from_numpy(point_cloud_final_np).float().permute(1, 0)
            batch_data['point_cloud'] = point_cloud_tensor

            # Convert mask to tensor (target_size,)
            mask_tensor = torch.from_numpy(mask).float()
            batch_data['point_cloud_mask'] = mask_tensor # Include the mask

            # --- Load image and pixel size ---
            image_np = self.images[type_key][original_index]
            
            # Rotate the image to match point cloud rotation
            image_np = np.rot90(image_np, k=(-num_rotations)%4, axes=(0, 1))
            
            # Apply the same horizontal flip to the image if applicable
            if should_flip:
                image_np = np.fliplr(image_np)
                
            # As contiguous array
            image_np = np.ascontiguousarray(image_np) # Ensure the image is contiguous in memory
            image_tensor = torch.from_numpy(image_np).float().unsqueeze(0) # Add channel dim -> (1, H, W)

            batch_data['image'] = image_tensor

            pixel_size_val = self.pixel_sizes[type_key][original_index]
            batch_data['pixel_size'] = torch.tensor(pixel_size_val).float()

            return batch_data

        except KeyError as e:
            log.error(f"KeyError encountered for index {idx} (type: {type_key}, orig_idx: {original_index}): {e}.")
            log.warning(f"Returning dummy data for index {idx} due to KeyError.")
            # Return dummy data (zeros)
            dummy_pc = torch.zeros((3, self.point_cloud_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.point_cloud_size, dtype=torch.float32) # Mask is zeros for dummy
            dummy_img = torch.zeros((1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)
            dummy_px = torch.tensor(0.0).float()
            return {
                'point_cloud': dummy_pc,
                'point_cloud_mask': dummy_mask,
                'image': dummy_img,
                'pixel_size': dummy_px
            }
        except Exception as e:
            log.error(f"Unexpected error processing index {idx} (type: {type_key}, orig_idx: {original_index}): {e}", exc_info=True)
            raise # Re-raise unexpected errors

    def __len__(self):
        return len(self.indices)
    


class AtomicReconstructionDataset2(Dataset):
    # Keep the __init__ method as it was in the code you provided in the prompt
    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized', point_cloud_size=8192, cell_size=8.0):
        """
        Args:
            hdf5_path: Path to the HDF5 file
            data_type: 'real', 'synthetic', or 'both'
            image_type: 'haadf_normalized' or 'haadf_cycleGAN'
            point_cloud_size: Target number of points in the point cloud.
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
             raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.cell_size = cell_size #nm
        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.indices = [] # Will store (data_type_flag, original_index)

        # --- Use the more robust loading logic from previous versions ---
        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []

                # Load REAL data
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys)), key=int)
                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            #find max number of Pt atoms in a single point cloud
                            """
                            max_num_Pt_atoms = 0
                            for key in real_keys:
                                num_Pt_atoms = np.sum(real_point_clouds[key]['label'] == 78/118)
                                if num_Pt_atoms > max_num_Pt_atoms:
                                    max_num_Pt_atoms = num_Pt_atoms
                            print(f"Max number of Pt atoms in a single point cloud: {max_num_Pt_atoms}")
                            """
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images
                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes
                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")

                # Load SYNTHETIC data
                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                 pc_group = f["point_clouds"]["synthetic"][key]
                                 synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds
                            """
                            max_num_Pt_atoms = 0
                            for key in synthetic_keys:
                                num_Pt_atoms = np.sum(synthetic_point_clouds[key]['label'] == 78/118)
                                if num_Pt_atoms > max_num_Pt_atoms:
                                    max_num_Pt_atoms = num_Pt_atoms
                            print(f"Max number of Pt atoms in a single point cloud: {max_num_Pt_atoms}")
                            """
                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
             raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")
    # --- End of __init__ ---


    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = 'real' if data_type_flag == 0 else 'synthetic'
        old_cell_size = 8.0 #nm
        new_cell_size = self.cell_size #nm
        # Dictionary to hold the data for this item
        batch_data = {}

        try: # Added try-except block around data loading/processing
            # Load point cloud data
            point_cloud_data = self.point_clouds[type_key][key_str]

            # Extract coordinates
            x_data = point_cloud_data['x']
            y_data = point_cloud_data['y']
            z_data = point_cloud_data['z']

            # Stack coordinates
            point_cloud_np = np.stack([x_data, y_data, z_data], axis=1).astype(np.float32)

            # -- NEW: Convert point cloud to new cell size, assert all Pt atoms are within 
            point_cloud_np = point_cloud_np * (old_cell_size/2) / (new_cell_size/2) #convert to new cell size
            #shift point cloud hull to positive quadrant
            #x_shift = np.min(point_cloud_np[:,0])
            #y_shift = np.min(point_cloud_np[:,1])
            #z_shift = np.min(point_cloud_np[:,2])
            #point_cloud_np = point_cloud_np - np.array([x_shift, y_shift, z_shift])
            #divide by new cell size
            #point_cloud_np = point_cloud_np / new_cell_size
            #shift back
            #point_cloud_np = point_cloud_np + np.array([x_shift, y_shift, z_shift])/new_cell_size
            
            labels = point_cloud_data['label']
            Pt = 78/118
            #assert all Pt atoms are within -1,1
            #assert there are points with label 78 (Pt) in the point cloud
            #assert np.any(labels == Pt), "No Pt atoms in point cloud"
            #assert np.all(point_cloud_np[:,0] >= -1) and np.all(point_cloud_np[:,0] <= 1), "X coordinates out of bounds"
            #assert np.all(point_cloud_np[:,1] >= -1) and np.all(point_cloud_np[:,1] <= 1), "Y coordinates out of bounds"
            #assert np.all(point_cloud_np[:,2] >= -1) and np.all(point_cloud_np[:,2] <= 1), "Z coordinates out of bounds"

            #remove all points that are not Pt atoms
            point_cloud_np = point_cloud_np[labels == Pt]

            #shift point cloud mean to z = 0
            z_mean = np.mean(point_cloud_np[:,2])
            point_cloud_np[:,2] = point_cloud_np[:,2] - z_mean
            assert np.all(point_cloud_np[:,0] >= -1) and np.all(point_cloud_np[:,0] <= 1), "X coordinates out of bounds"
            assert np.all(point_cloud_np[:,1] >= -1) and np.all(point_cloud_np[:,1] <= 1), "Y coordinates out of bounds"
            assert np.all(point_cloud_np[:,2] >= -1) and np.all(point_cloud_np[:,2] <= 1), "Z coordinates out of bounds"





            # --- Point cloud size handling: Preserve originals when padding ---
            current_size = point_cloud_np.shape[0]
            target_size = self.point_cloud_size

            if current_size == 0:
                # If empty, fill with zeros (or potentially random noise if preferred)
                point_cloud_final_np = np.zeros((target_size, 3), dtype=np.float32)
                # Mask is all zeros for empty clouds
                mask = np.zeros(target_size, dtype=np.float32) # Indicates all points are padding

            elif current_size < target_size:
                # --- Use (0, 0, 0) padding instead of duplicating random points ---
                additional_needed = target_size - current_size

                # Create padding points (0, 0, 0)
                padding_points = np.zeros((additional_needed, 3), dtype=np.float32)

                # Combine original points with padding points
                point_cloud_final_np = np.vstack((point_cloud_np, padding_points))

                # Mask indicates which points are original (1) and which are padding (0)
                mask = np.zeros(target_size, dtype=np.float32)
                mask[:current_size] = 1.0  # Mark the original points as valid/real

            elif current_size > target_size:
                # Sample points randomly (Subsampling)
                # This inherently selects a *subset* of original points.
                # The *values* of the kept points are unaltered.
                indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                point_cloud_final_np = point_cloud_np[indices_to_keep]
                # Mask is all ones (all points are real, just sampled)
                mask = np.ones(target_size, dtype=np.float32)

            else: # current_size == target_size
                point_cloud_final_np = point_cloud_np
                # Mask is all ones (all points are original)
                mask = np.ones(target_size, dtype=np.float32)

            # Data augmentation: rotation
            num_rotations = np.random.randint(0, 4) # Randomly choose a rotation (0, 90, 180, or 270 degrees)
            num_rotations = 0
            #rotate point cloud around z-axis
            # Convert to tensor for rotation
            pt_cloud_tensor = torch.from_numpy(point_cloud_final_np).float()
            # Create rotation matrix using torch
            theta = torch.tensor(np.pi/2 * num_rotations).float()
            rotation_matrix = torch.tensor([
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1]
            ]).float()
            # Apply rotation using torch operations
            #point_cloud_final_np = torch.matmul(pt_cloud_tensor, rotation_matrix.t()).numpy()
            point_cloud_final_np = pt_cloud_tensor.numpy()
            # Data augmentation: horizontal reflection (flip along x-axis)
            should_flip = np.random.random() > 0.5  # 50% chance of flipping
            should_flip = False
            if should_flip:
                # Flip point cloud along x-axis by negating x-coordinates
                point_cloud_final_np[:, 0] = -point_cloud_final_np[:, 0]

            # Convert point cloud to tensor C×N format (3 × target_size)
            point_cloud_tensor = torch.from_numpy(point_cloud_final_np).float().permute(1, 0)
            batch_data['point_cloud'] = point_cloud_tensor

            # Convert mask to tensor (target_size,)
            mask_tensor = torch.from_numpy(mask).float()
            batch_data['point_cloud_mask'] = mask_tensor # Include the mask

            # --- Load image and pixel size ---
            image_np = self.images[type_key][original_index]
            
            # Rotate the image to match point cloud rotation
            image_np = np.rot90(image_np, k=(-num_rotations)%4, axes=(0, 1))
            
            # Apply the same horizontal flip to the image if applicable
            if should_flip:
                image_np = np.fliplr(image_np)
                
            # As contiguous array
            image_np = np.ascontiguousarray(image_np) # Ensure the image is contiguous in memory
            image_tensor = torch.from_numpy(image_np).float().unsqueeze(0) # Add channel dim -> (1, H, W)

            batch_data['image'] = image_tensor

            pixel_size_val = self.pixel_sizes[type_key][original_index]
            batch_data['pixel_size'] = torch.tensor(pixel_size_val).float()

            return batch_data

        except KeyError as e:
            log.error(f"KeyError encountered for index {idx} (type: {type_key}, orig_idx: {original_index}): {e}.")
            log.warning(f"Returning dummy data for index {idx} due to KeyError.")
            # Return dummy data (zeros)
            dummy_pc = torch.zeros((3, self.point_cloud_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.point_cloud_size, dtype=torch.float32) # Mask is zeros for dummy
            dummy_img = torch.zeros((1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)
            dummy_px = torch.tensor(0.0).float()
            return {
                'point_cloud': dummy_pc,
                'point_cloud_mask': dummy_mask,
                'image': dummy_img,
                'pixel_size': dummy_px
            }
        except Exception as e:
            log.error(f"Unexpected error processing index {idx} (type: {type_key}, orig_idx: {original_index}): {e}", exc_info=True)
            raise # Re-raise unexpected errors

    def __len__(self):
        return len(self.indices)
    


class AtomicReconstructionDataset3(Dataset):
    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized', point_cloud_size=1024, cell_size=8.0, augment=True, refinement=False, pad_with_noise=False, noise_std=0.3): # Changed default point_cloud_size to 1024 for consistency with prompt
        """
        Args:
            with padding and existance channel
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
             raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.cell_size = cell_size #nm
        self.augment = augment
        self.refinement = refinement # Whether to load coarse structures for refinement
        self.pad_with_noise = pad_with_noise
        self.noise_std = noise_std
        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size', 'CoM_clean']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.CoMs = {} # Predicted center of mass of particle
        self.indices = [] # Will store (data_type_flag, original_index)
        self.coarse_pcs = {} # Coarse point clouds for refinement
        self.coarse_perms = {} # Permutations for coarse point clouds

        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if refinement:
                        coarse_pc_group_exists = "coarse_structure" in f and "real" in f["coarse_structure"]
                        coarse_pc_perm_exists = "coarse_permutations" in f and "real" in f["coarse_permutations"]
                        print(coarse_pc_perm_exists)
                        print(coarse_pc_perm_exists)
                    #print(1)
                    if image_type == 'haadf_normalized':
                        CoM_group_exists = "CoM_clean" in f and "real" in f["CoM_clean"]
                    else:
                        CoM_group_exists = "CoM_CycleGAN" in f and "real" in f["CoM_CycleGAN"]
                    #print(2)
                    if pc_group_exists and img_group_exists and px_group_exists and CoM_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        
                        if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                            real_coarse_pc_keys = set(f["coarse_structure"]["real"].keys())
                            print(len(real_coarse_pc_keys), "coarse pc keys")

                            real_coarse_perm_keys = set(f["coarse_permutations"]["real"].keys())
                            print(len(real_coarse_perm_keys), "coarse perm keys")

                        if image_type == 'haadf_normalized':
                            real_CoM_keys = set(f["CoM_clean"]["real"].keys())
                        else:
                            real_CoM_keys = set(f["CoM_CycleGAN"]["real"].keys())

                        #print(3)
                        if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                            real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_coarse_pc_keys).intersection(real_px_keys).intersection(real_CoM_keys).intersection(real_coarse_perm_keys)), key=int)
                        else:
                            real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys).intersection(real_CoM_keys)), key=int)

                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            
                            if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                                real_coarse_structures = {}
                                real_coarse_perms = {}
                                for key in tqdm(real_keys, desc="Real coarse structures"):
                                    pc_group = f["coarse_structure"]["real"][key]
                                    perm_group = f["coarse_permutations"]["real"][key]
                                    
                                    # Load the raw (4, N) array
                                    pc_data = pc_group[()]
                                    # Manually create the dictionary from the array slices
                                    real_coarse_structures[key] = {
                                        'x': pc_data[0, :],
                                        'y': pc_data[1, :],
                                        'z': pc_data[2, :],
                                        'label': pc_data[3, :]
                                    }
                                    real_coarse_perms[key] = perm_group[()]
                                self.coarse_perms['real'] = real_coarse_perms
                                self.coarse_pcs['real'] = real_coarse_structures

                            #print(4)
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images

                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes

                            real_CoMs = {}
                            if image_type == 'haadf_normalized':
                                for key in tqdm(real_keys, desc="Real CoMs (Clean)"): real_CoMs[int(key)] = f["CoM_clean"]["real"][key][()]
                            else:
                                for key in tqdm(real_keys, desc="Real CoMs (CycleGAN)"): real_CoMs[int(key)] = f["CoM_CycleGAN"]["real"][key][()]
                            self.CoMs['real'] = real_CoMs

                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")


                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    if refinement:
                        coarse_pc_group_exists = "coarse_structure" in f and "synthetic" in f["coarse_structure"]
                        coarse_pc_perm_exists = "coarse_permutations" in f and "synthetic" in f["coarse_permutations"]
                        #print(coarse_pc_perm_exists)
                        #print(coarse_pc_perm_exists)
                        #print(coarse_pc_perm_exists)
                        #print(coarse_pc_perm_exists)
                    
                    if image_type == 'haadf_normalized':
                        CoM_group_exists = "CoM_clean" in f and "synthetic" in f["CoM_clean"]
                    else:
                        CoM_group_exists = "CoM_CycleGAN" in f and "synthetic" in f["CoM_CycleGAN"]

                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                            synth_coarse_pc_keys = set(f["coarse_structure"]["synthetic"].keys())
                            synth_coarse_perm_keys = set(f["coarse_permutations"]["synthetic"].keys())
                            print(len(synth_coarse_pc_keys), "coarse pc keys")
                            print(len(synth_coarse_perm_keys), "coarse perm keys")
                            #sort keys and print lowest idx
                            print("Lowest idx in coarse pc keys:", sorted(synth_coarse_pc_keys, key=int)[0])
                            print("Largest idx in coarse pc keys:", sorted(synth_coarse_pc_keys, key=int)[-1])

                        if image_type == 'haadf_normalized':
                            synth_CoM_keys = set(f["CoM_clean"]["synthetic"].keys())
                        else:
                            synth_CoM_keys = set(f["CoM_CycleGAN"]["synthetic"].keys())
                        #missing_coarse_keys = synth_pc_keys - synth_coarse_pc_keys
                        #if missing_coarse_keys:
                        #    log.warning(f"Discrepancy found: {len(missing_coarse_keys)} synthetic samples have point clouds/images but are MISSING coarse structures.")  
                        #    log.info(f"Missing keys: {missing_coarse_keys}") 
                        
                                                 
                        if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                            #subtract 6000 from coarse pc keys to match the point cloud keys because im stupid
                            #synth_coarse_pc_keys = {str(int(key) - 6000) for key in synth_coarse_pc_keys}
                            synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_coarse_pc_keys).intersection(synth_px_keys).intersection(synth_CoM_keys).intersection(synth_coarse_perm_keys)), key=int)
                        else:
                            synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys).intersection(synth_CoM_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                 pc_group = f["point_clouds"]["synthetic"][key]
                                 synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds

                            if refinement and coarse_pc_group_exists and coarse_pc_perm_exists:
                                synthetic_coarse_perms = {}
                                synthetic_coarse_structures = {}
                                for key in tqdm(synthetic_keys, desc="Synthetic coarse structures"):
                                    pc_group = f["coarse_structure"]["synthetic"][key]
                                    perm_group = f["coarse_permutations"]["synthetic"][key]
                                    # Load the raw (4, N) array
                                    pc_data = pc_group[()]
                                    
                                    synthetic_coarse_perms[key] = perm_group[()]
                                    # Manually create the dictionary from the array slices
                                    synthetic_coarse_structures[key] = {
                                        'x': pc_data[0, :],
                                        'y': pc_data[1, :],
                                        'z': pc_data[2, :],
                                        'label': pc_data[3, :]
                                    }
                                self.coarse_perms['synthetic'] = synthetic_coarse_perms
                                self.coarse_pcs['synthetic'] = synthetic_coarse_structures

                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            synthetic_CoMs = {}
                            if image_type == 'haadf_normalized':
                                for key in tqdm(synthetic_keys, desc="Synthetic CoMs (Clean)"): synthetic_CoMs[int(key)] = f["CoM_clean"]["synthetic"][key][()]
                            else:
                                for key in tqdm(synthetic_keys, desc="Synthetic CoMs (CycleGAN)"): synthetic_CoMs[int(key)] = f["CoM_CycleGAN"]["synthetic"][key][()]
                            self.CoMs['synthetic'] = synthetic_CoMs
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
             raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")

    def _transform_coords_for_augmentation(self, coords_tensor, s_norm_to_img, s_img_to_norm, operation_fn, device):
        """Helper to apply augmentation transformations in the defined coordinate system."""
        # Ensure augmentation constants are on the correct device
        aug_img_center_xyz_dev = _AUG_IMG_CENTER_XYZ.to(device)
        aug_rot_axis_xyz_dev = _AUG_ROT_AXIS_XYZ.to(device)

        # Stage 1 & 2: Normalized Coords -> Image Space -> Centered for Operation
        temp_coords = coords_tensor * s_norm_to_img  # s_norm_to_img is a scalar
        temp_coords = temp_coords + aug_img_center_xyz_dev
        temp_coords = temp_coords - aug_rot_axis_xyz_dev
        
        # Stage 3: Apply specific geometric operation (e.g., rotation, flip)
        temp_coords = operation_fn(temp_coords)
        
        # Stage 4 & 5: Un-center -> Image Space -> Normalized Coords
        temp_coords = temp_coords + aug_rot_axis_xyz_dev
        temp_coords = temp_coords - aug_img_center_xyz_dev
        temp_coords = temp_coords * s_img_to_norm  # s_img_to_norm is a scalar
        return temp_coords

    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = _TYPE_KEY_MAP[data_type_flag]
        
        current_new_cell_size = self.cell_size # nm
        batch_data = {}

        try:
            point_cloud_data = self.point_clouds[type_key][key_str]
            # Ensure x, y, z data are NumPy arrays. Stack and ensure float32.
            point_cloud_xyz_np = np.stack([
                point_cloud_data['x'], 
                point_cloud_data['y'], 
                point_cloud_data['z']
            ], axis=1).astype(np.float32)

            coarse_pc_data = self.coarse_pcs[type_key][key_str] if self.refinement else None
            if coarse_pc_data is not None:
                coarse_point_cloud = np.stack([
                    coarse_pc_data['x'], 
                    coarse_pc_data['y'], 
                    coarse_pc_data['z'],
                    coarse_pc_data['label']
                ], axis=1).astype(np.float32)
        
            # Scale point cloud based on cell sizes
            # Original: point_cloud_xyz_np * (_OLD_CELL_SIZE / 2) / (current_new_cell_size / 2)
            point_cloud_xyz_np *= (_OLD_CELL_SIZE / current_new_cell_size)
           
            pixel_size_val = self.pixel_sizes[type_key][original_index] # Should be a float
            batch_data['pixel_size'] = torch.tensor(pixel_size_val, dtype=torch.float32)
            
            labels = point_cloud_data['label'] # Assuming NumPy array
            
            # Filter for Pt atoms
            pt_mask = (labels == _PT_LABEL)
            point_cloud_xyz_np = point_cloud_xyz_np[pt_mask]

            # Center z-coordinates
            if point_cloud_xyz_np.shape[0] > 0:
                z_mean = np.mean(point_cloud_xyz_np[:, 2])
                point_cloud_xyz_np[:, 2] -= z_mean

            # Assert coordinates are within [-1, 1] AFTER filtering and centering
            if point_cloud_xyz_np.shape[0] > 0:
                # Using a small epsilon for float comparisons can be robust, but original used direct comparison.
                # To maintain exact functionality, direct comparison is kept.
                assert np.all(point_cloud_xyz_np[:,0] >= -1) and np.all(point_cloud_xyz_np[:,0] <= 1), f"X coordinates out of bounds for {type_key}/{key_str}"
                assert np.all(point_cloud_xyz_np[:,1] >= -1) and np.all(point_cloud_xyz_np[:,1] <= 1), f"Y coordinates out of bounds for {type_key}/{key_str}"
                assert np.all(point_cloud_xyz_np[:,2] >= -1) and np.all(point_cloud_xyz_np[:,2] <= 1), f"Z coordinates out of bounds for {type_key}/{key_str}"

            current_size = point_cloud_xyz_np.shape[0]
            target_size = self.point_cloud_size

            # Initialize final point cloud array [x,y,z,existence] and loss mask
            if self.pad_with_noise:
                # Initialize with Gaussian noise
                point_cloud_final_np = np.random.normal(0, self.noise_std, (target_size, 4)).astype(np.float32)
                point_cloud_final_np[:, 3] = 0.0 # Set existence channel to 0 for all initially
            else:
                # Initialize with zeros
                point_cloud_final_np = np.zeros((target_size, 4), dtype=np.float32)
            
            xyz_loss_mask_np = np.zeros(target_size, dtype=np.float32)

            if current_size > 0:
                if current_size < target_size:
                    # Fill the first part of the array with real data
                    point_cloud_final_np[:current_size, :3] = point_cloud_xyz_np
                    point_cloud_final_np[:current_size, 3] = 1.0 # Existence = 1 for real points
                    xyz_loss_mask_np[:current_size] = 1.0
                elif current_size > target_size:
                    indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                    point_cloud_final_np[:, :3] = point_cloud_xyz_np[indices_to_keep]
                    point_cloud_final_np[:, 3] = 1.0 # All points are real (subsampled)
                    xyz_loss_mask_np[:] = 1.0
                else: # current_size == target_size
                    point_cloud_final_np[:, :3] = point_cloud_xyz_np
                    point_cloud_final_np[:, 3] = 1.0 # All points are real
                    xyz_loss_mask_np[:] = 1.0
            # If current_size is 0, the array remains as initialized (either zeros or noise).

            if self.refinement:
                perm = self.coarse_perms[type_key][key_str]
            else:
                perm = np.random.permutation(target_size)
            point_cloud_final_np = point_cloud_final_np[perm]
            xyz_loss_mask_np = xyz_loss_mask_np[perm]
            batch_data['permutation'] = torch.from_numpy(perm).long()
            #assert that clean_cloud[perm] channel 3 is equal to coarse_point_cloud channel 3
            if self.refinement:
                assert np.array_equal(point_cloud_final_np[:, 3], coarse_point_cloud[:, 3]), f"Existence channel mismatch after permutation for {type_key}/{key_str}"
            # COM
            CoM_np_2d = self.CoMs[type_key][original_index]
            # Create a temporary 3D tensor with a dummy Z=0 coordinate
            temp_com_3d_tensor = torch.tensor(
                [CoM_np_2d[0], CoM_np_2d[1], 0.0], dtype=torch.float32
            )


            

            # --- Data augmentation ---
            # Augment XYZ coordinates; existence channel ([:, 3]) is not augmented.
            xyz_coords_np = point_cloud_final_np[:, :3].copy() # Use .copy() as in original logic
            xyz_tensor = torch.from_numpy(xyz_coords_np).float() # Converts to CPU tensor by default
            _device = xyz_tensor.device # Will be 'cpu' unless default tensor type is changed

            # Scaling factors for norm <-> image space conversion during augmentation
            s_norm_to_img = current_new_cell_size / (2.0 * pixel_size_val)
            s_img_to_norm = (2.0 * pixel_size_val) / current_new_cell_size

            num_rotations = 0
            should_flip = False
            if self.augment:
                num_rotations = np.random.randint(0, 4) # 0, 1, 2, or 3 rotations
            if num_rotations > 0:
                # Rotation matrix for CCW rotation by theta: [[cos, -sin], [sin, cos]]
                # Original code implies CW rotation for points: [x,y] @ [[c,s],[-s,c]]
                # To achieve this with R_ccw.T, R_ccw should be [[c,-s],[s,c]]
                theta_val = np.pi / 2.0 * num_rotations
                cos_t, sin_t = np.cos(theta_val), np.sin(theta_val)
                # This is a standard CCW rotation matrix for angle theta_val
                R_ccw_matrix = torch.tensor([
                    [cos_t, -sin_t, 0.0],
                    [sin_t,  cos_t, 0.0],
                    [0.0,    0.0,   1.0]
                ], dtype=torch.float32, device=_device)
                
                # op is P' = P @ M. For CW rotation, M = R_ccw.T
                rotate_op = lambda c: torch.matmul(c, R_ccw_matrix.t())
                xyz_tensor = self._transform_coords_for_augmentation(
                    xyz_tensor, s_norm_to_img, s_img_to_norm, rotate_op, _device
                )
                temp_com_3d_tensor = self._transform_coords_for_augmentation(
                    temp_com_3d_tensor.unsqueeze(0), s_norm_to_img, s_img_to_norm, rotate_op, _device
                ).squeeze(0)

            if self.augment:
                should_flip = np.random.random() > 0.5
            if should_flip:
                def flip_op(coords_tensor):
                    # Flip Y coordinate in the centered-for-transform space
                    flipped_coords = coords_tensor.clone() # Use clone if coords_tensor might be reused
                    flipped_coords[:, 1] = -flipped_coords[:, 1]
                    return flipped_coords
                
                xyz_tensor = self._transform_coords_for_augmentation(
                    xyz_tensor, s_norm_to_img, s_img_to_norm, flip_op, _device
                )

                temp_com_3d_tensor = self._transform_coords_for_augmentation(
                    temp_com_3d_tensor.unsqueeze(0), s_norm_to_img, s_img_to_norm, flip_op, _device
                ).squeeze(0)
            final_com_2d_tensor = temp_com_3d_tensor[:2]
            batch_data['CoM'] = final_com_2d_tensor            
            # Combine augmented XYZ with the un-augmented existence channel
            existence_channel_tensor = torch.from_numpy(point_cloud_final_np[:, 3]).float().unsqueeze(1).to(_device)
            final_point_cloud_N_4 = torch.cat((xyz_tensor, existence_channel_tensor), dim=1) # Shape (target_size, 4)
            batch_data['point_cloud'] = final_point_cloud_N_4.permute(1, 0) # Shape (4, target_size)

            batch_data['point_cloud_mask'] = torch.from_numpy(xyz_loss_mask_np).float() # Shape (target_size,)

            # Load image (assuming NumPy array from self.images)
            image_np = self.images[type_key][original_index] 
            
            # Apply corresponding augmentations to image
            if num_rotations > 0:
                # Point rotation was CW. np.rot90(k>0) is CCW. So k must be negative for CW.
                image_np = np.rot90(image_np, k=(-num_rotations) % 4, axes=(0, 1))
            if should_flip:
                # Flipping Y for points (y_new = -y_old in transform space) corresponds to
                # flipping the image vertically (top becomes bottom).
                image_np = np.flipud(image_np)
                
            # Ensure contiguous array before tensor conversion for performance
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_np)).float().unsqueeze(0) # Shape (1, H, W)
            batch_data['image'] = image_tensor
            batch_data['dataset_idx'] = idx 
            batch_data['original_index'] = original_index
            batch_data['data_type'] = type_key # 'real' or 'synthetic'

            if self.refinement:
                batch_data['coarse_point_cloud'] = torch.from_numpy(coarse_point_cloud).float().permute(1, 0)


            debugging = False # Set to True to enable plotting
            if debugging:
                import matplotlib.pyplot as plt # Import only when needed

                plt.figure()
                plt.imshow(image_np, cmap='gray') # image_np is already augmented
                
                # Use augmented xyz_tensor for plotting, convert to NumPy
                plot_xyz_np = xyz_tensor.cpu().numpy() 
                
                # Transform points to image plot coordinates for overlay
                # This mirrors the first part of _transform_coords_for_augmentation
                # And the original debugging plot logic.
                plot_coords_xy = plot_xyz_np[:, :2] * s_norm_to_img # Scale
                plot_coords_xy += _AUG_IMG_CENTER_XYZ.cpu().numpy()[:2] # Offset to image center

                # Filter out non-existent (padded) points for cleaner visualization
                real_points_mask_plot = (point_cloud_final_np[:, 3] == 1.0)
                
                plt.scatter(
                    plot_coords_xy[real_points_mask_plot, 0], 
                    plot_coords_xy[real_points_mask_plot, 1], 
                    c='r', s=1
                )
                plot_com_np_2d = final_com_2d_tensor.cpu().numpy()
                # Create a temporary 3D point just for transformation to image space
                plot_com_np_3d_for_transform = np.array([plot_com_np_2d[0], plot_com_np_2d[1], 0.0])
                plot_com_xy = plot_com_np_3d_for_transform[:2] * s_norm_to_img
                plot_com_xy += _AUG_IMG_CENTER_XYZ.cpu().numpy()[:2]
                plt.scatter(plot_com_xy[0], plot_com_xy[1], c='cyan', s=50, marker='x', label='CoM')                
                # Determine image display size for plot limits
                img_display_size = getattr(self, 'IMAGE_SIZE', _AUG_IMG_CENTER_XYZ.cpu().numpy()[0] * 2.0)
                plt.xlim(0, img_display_size)
                plt.ylim(img_display_size, 0) # Match imshow: (0,0) top-left, Y increases downwards
                # Note: original code had plt.ylim(0, self.IMAGE_SIZE). If Y needs to increase downwards
                # to match imshow, then ylim(max, min) is used. scatter Y will be plotted accordingly.

                plt.title(f"Point Cloud over Image (idx: {idx}, type: {type_key}, orig_idx: {original_index})")
                plt.show()

            return batch_data

        except KeyError as e:
            # Provide context for KeyError, common with dictionary/HDF5 access
            print(f"KeyError: {e} for type_key='{type_key}', original_index='{key_str}' (dataset index {idx}).")
            raise # Re-raise the exception to halt or be caught by a DataLoader

    def __len__(self):
        return len(self.indices)
    

class Refinement(Dataset):
    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized', point_cloud_size=1024, cell_size=8.0, augment=True): # Changed default point_cloud_size to 1024 for consistency with prompt
        """
        Args:
            with padding and existance channel
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
             raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.cell_size = cell_size #nm
        self.augment = augment
        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size', 'CoM_clean']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.CoMs = {} # Predicted center of mass of particle
        self.indices = [] # Will store (data_type_flag, original_index)

        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if image_type == 'haadf_normalized':
                        CoM_group_exists = "CoM_clean" in f and "real" in f["CoM_clean"]
                    else:
                        CoM_group_exists = "CoM_cycleGAN" in f and "real" in f["CoM_cycleGAN"]
                    if pc_group_exists and img_group_exists and px_group_exists and CoM_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        if image_type == 'haadf_normalized':
                            real_CoM_keys = set(f["CoM_clean"]["real"].keys())
                        else:
                            real_CoM_keys = set(f["CoM_cycleGAN"]["real"].keys())
                        real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys).intersection(real_CoM_keys)), key=int)
                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images
                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes
                            real_CoMs = {}
                            if image_type == 'haadf_normalized':
                                for key in tqdm(real_keys, desc="Real CoMs (Clean)"): real_CoMs[int(key)] = f["CoM_clean"]["real"][key][()]
                            else:
                                for key in tqdm(real_keys, desc="Real CoMs (CycleGAN)"): real_CoMs[int(key)] = f["CoM_cycleGAN"]["real"][key][()]
                            self.CoMs['real'] = real_CoMs

                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")


                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    if image_type == 'haadf_normalized':
                        CoM_group_exists = "CoM_clean" in f and "synthetic" in f["CoM_clean"]
                    else:
                        CoM_group_exists = "CoM_cycleGAN" in f and "synthetic" in f["CoM_cycleGAN"]

                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        if image_type == 'haadf_normalized':
                            synth_CoM_keys = set(f["CoM_clean"]["synthetic"].keys())
                        else:
                            synth_CoM_keys = set(f["CoM_cycleGAN"]["synthetic"].keys())
                        
                        synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys).intersection(synth_CoM_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                 pc_group = f["point_clouds"]["synthetic"][key]
                                 synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds
                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            synthetic_CoMs = {}
                            if image_type == 'haadf_normalized':
                                for key in tqdm(synthetic_keys, desc="Synthetic CoMs (Clean)"): synthetic_CoMs[int(key)] = f["CoM_clean"]["synthetic"][key][()]
                            else:
                                for key in tqdm(synthetic_keys, desc="Synthetic CoMs (CycleGAN)"): synthetic_CoMs[int(key)] = f["CoM_cycleGAN"]["synthetic"][key][()]
                            self.CoMs['synthetic'] = synthetic_CoMs
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
             raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")

    def _transform_coords_for_augmentation(self, coords_tensor, s_norm_to_img, s_img_to_norm, operation_fn, device):
        """Helper to apply augmentation transformations in the defined coordinate system."""
        # Ensure augmentation constants are on the correct device
        aug_img_center_xyz_dev = _AUG_IMG_CENTER_XYZ.to(device)
        aug_rot_axis_xyz_dev = _AUG_ROT_AXIS_XYZ.to(device)

        # Stage 1 & 2: Normalized Coords -> Image Space -> Centered for Operation
        temp_coords = coords_tensor * s_norm_to_img  # s_norm_to_img is a scalar
        temp_coords = temp_coords + aug_img_center_xyz_dev
        temp_coords = temp_coords - aug_rot_axis_xyz_dev
        
        # Stage 3: Apply specific geometric operation (e.g., rotation, flip)
        temp_coords = operation_fn(temp_coords)
        
        # Stage 4 & 5: Un-center -> Image Space -> Normalized Coords
        temp_coords = temp_coords + aug_rot_axis_xyz_dev
        temp_coords = temp_coords - aug_img_center_xyz_dev
        temp_coords = temp_coords * s_img_to_norm  # s_img_to_norm is a scalar
        return temp_coords

    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = _TYPE_KEY_MAP[data_type_flag]
        
        current_new_cell_size = self.cell_size # nm
        batch_data = {}

        try:
            point_cloud_data = self.point_clouds[type_key][key_str]
            # Ensure x, y, z data are NumPy arrays. Stack and ensure float32.
            point_cloud_xyz_np = np.stack([
                point_cloud_data['x'], 
                point_cloud_data['y'], 
                point_cloud_data['z']
            ], axis=1).astype(np.float32)

            # Scale point cloud based on cell sizes
            # Original: point_cloud_xyz_np * (_OLD_CELL_SIZE / 2) / (current_new_cell_size / 2)
            point_cloud_xyz_np *= (_OLD_CELL_SIZE / current_new_cell_size)
           
            pixel_size_val = self.pixel_sizes[type_key][original_index] # Should be a float
            batch_data['pixel_size'] = torch.tensor(pixel_size_val, dtype=torch.float32)
            
            labels = point_cloud_data['label'] # Assuming NumPy array
            
            # Filter for Pt atoms
            pt_mask = (labels == _PT_LABEL)
            point_cloud_xyz_np = point_cloud_xyz_np[pt_mask]

            # Center z-coordinates
            if point_cloud_xyz_np.shape[0] > 0:
                z_mean = np.mean(point_cloud_xyz_np[:, 2])
                point_cloud_xyz_np[:, 2] -= z_mean

            # Assert coordinates are within [-1, 1] AFTER filtering and centering
            if point_cloud_xyz_np.shape[0] > 0:
                # Using a small epsilon for float comparisons can be robust, but original used direct comparison.
                # To maintain exact functionality, direct comparison is kept.
                assert np.all(point_cloud_xyz_np[:,0] >= -1) and np.all(point_cloud_xyz_np[:,0] <= 1), f"X coordinates out of bounds for {type_key}/{key_str}"
                assert np.all(point_cloud_xyz_np[:,1] >= -1) and np.all(point_cloud_xyz_np[:,1] <= 1), f"Y coordinates out of bounds for {type_key}/{key_str}"
                assert np.all(point_cloud_xyz_np[:,2] >= -1) and np.all(point_cloud_xyz_np[:,2] <= 1), f"Z coordinates out of bounds for {type_key}/{key_str}"

            current_size = point_cloud_xyz_np.shape[0]
            target_size = self.point_cloud_size

            # Initialize final point cloud array [x,y,z,existence] and loss mask
            point_cloud_final_np = np.zeros((target_size, 4), dtype=np.float32)
            xyz_loss_mask_np = np.zeros(target_size, dtype=np.float32)

            if current_size > 0:
                if current_size < target_size:
                    point_cloud_final_np[:current_size, :3] = point_cloud_xyz_np
                    point_cloud_final_np[:current_size, 3] = 1.0 # Existence = 1 for real points
                    xyz_loss_mask_np[:current_size] = 1.0
                elif current_size > target_size:
                    indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                    point_cloud_final_np[:, :3] = point_cloud_xyz_np[indices_to_keep]
                    point_cloud_final_np[:, 3] = 1.0 # All points are real (subsampled)
                    xyz_loss_mask_np[:] = 1.0
                else: # current_size == target_size
                    point_cloud_final_np[:, :3] = point_cloud_xyz_np
                    point_cloud_final_np[:, 3] = 1.0 # All points are real
                    xyz_loss_mask_np[:] = 1.0
            # If current_size is 0, arrays remain all zeros, which is correct.
            perm = np.random.permutation(target_size)
            point_cloud_final_np = point_cloud_final_np[perm]
            xyz_loss_mask_np = xyz_loss_mask_np[perm]

            # COM
            CoM_np_2d = self.CoMs[type_key][original_index]
            # Create a temporary 3D tensor with a dummy Z=0 coordinate
            temp_com_3d_tensor = torch.tensor(
                [CoM_np_2d[0], CoM_np_2d[1], 0.0], dtype=torch.float32
            )


            

            # --- Data augmentation ---
            # Augment XYZ coordinates; existence channel ([:, 3]) is not augmented.
            xyz_coords_np = point_cloud_final_np[:, :3].copy() # Use .copy() as in original logic
            xyz_tensor = torch.from_numpy(xyz_coords_np).float() # Converts to CPU tensor by default
            _device = xyz_tensor.device # Will be 'cpu' unless default tensor type is changed

            # Scaling factors for norm <-> image space conversion during augmentation
            s_norm_to_img = current_new_cell_size / (2.0 * pixel_size_val)
            s_img_to_norm = (2.0 * pixel_size_val) / current_new_cell_size

            num_rotations = 0
            should_flip = False
            if self.augment:
                num_rotations = np.random.randint(0, 4) # 0, 1, 2, or 3 rotations
            if num_rotations > 0:
                # Rotation matrix for CCW rotation by theta: [[cos, -sin], [sin, cos]]
                # Original code implies CW rotation for points: [x,y] @ [[c,s],[-s,c]]
                # To achieve this with R_ccw.T, R_ccw should be [[c,-s],[s,c]]
                theta_val = np.pi / 2.0 * num_rotations
                cos_t, sin_t = np.cos(theta_val), np.sin(theta_val)
                # This is a standard CCW rotation matrix for angle theta_val
                R_ccw_matrix = torch.tensor([
                    [cos_t, -sin_t, 0.0],
                    [sin_t,  cos_t, 0.0],
                    [0.0,    0.0,   1.0]
                ], dtype=torch.float32, device=_device)
                
                # op is P' = P @ M. For CW rotation, M = R_ccw.T
                rotate_op = lambda c: torch.matmul(c, R_ccw_matrix.t())
                xyz_tensor = self._transform_coords_for_augmentation(
                    xyz_tensor, s_norm_to_img, s_img_to_norm, rotate_op, _device
                )
                temp_com_3d_tensor = self._transform_coords_for_augmentation(
                    temp_com_3d_tensor.unsqueeze(0), s_norm_to_img, s_img_to_norm, rotate_op, _device
                ).squeeze(0)

            if self.augment:
                should_flip = np.random.random() > 0.5
            if should_flip:
                def flip_op(coords_tensor):
                    # Flip Y coordinate in the centered-for-transform space
                    flipped_coords = coords_tensor.clone() # Use clone if coords_tensor might be reused
                    flipped_coords[:, 1] = -flipped_coords[:, 1]
                    return flipped_coords
                
                xyz_tensor = self._transform_coords_for_augmentation(
                    xyz_tensor, s_norm_to_img, s_img_to_norm, flip_op, _device
                )

                temp_com_3d_tensor = self._transform_coords_for_augmentation(
                    temp_com_3d_tensor.unsqueeze(0), s_norm_to_img, s_img_to_norm, flip_op, _device
                ).squeeze(0)
            final_com_2d_tensor = temp_com_3d_tensor[:2]
            batch_data['CoM'] = final_com_2d_tensor            
            # Combine augmented XYZ with the un-augmented existence channel
            existence_channel_tensor = torch.from_numpy(point_cloud_final_np[:, 3]).float().unsqueeze(1).to(_device)
            final_point_cloud_N_4 = torch.cat((xyz_tensor, existence_channel_tensor), dim=1) # Shape (target_size, 4)
            batch_data['point_cloud'] = final_point_cloud_N_4.permute(1, 0) # Shape (4, target_size)

            batch_data['point_cloud_mask'] = torch.from_numpy(xyz_loss_mask_np).float() # Shape (target_size,)

            # Load image (assuming NumPy array from self.images)
            image_np = self.images[type_key][original_index] 
            
            # Apply corresponding augmentations to image
            if num_rotations > 0:
                # Point rotation was CW. np.rot90(k>0) is CCW. So k must be negative for CW.
                image_np = np.rot90(image_np, k=(-num_rotations) % 4, axes=(0, 1))
            if should_flip:
                # Flipping Y for points (y_new = -y_old in transform space) corresponds to
                # flipping the image vertically (top becomes bottom).
                image_np = np.flipud(image_np)
                
            # Ensure contiguous array before tensor conversion for performance
            image_tensor = torch.from_numpy(np.ascontiguousarray(image_np)).float().unsqueeze(0) # Shape (1, H, W)
            batch_data['image'] = image_tensor
            batch_data['dataset_idx'] = idx 
            batch_data['original_index'] = original_index
            batch_data['data_type'] = type_key # 'real' or 'synthetic'
            debugging = False # Set to True to enable plotting
            if debugging:
                import matplotlib.pyplot as plt # Import only when needed

                plt.figure()
                plt.imshow(image_np, cmap='gray') # image_np is already augmented
                
                # Use augmented xyz_tensor for plotting, convert to NumPy
                plot_xyz_np = xyz_tensor.cpu().numpy() 
                
                # Transform points to image plot coordinates for overlay
                # This mirrors the first part of _transform_coords_for_augmentation
                # And the original debugging plot logic.
                plot_coords_xy = plot_xyz_np[:, :2] * s_norm_to_img # Scale
                plot_coords_xy += _AUG_IMG_CENTER_XYZ.cpu().numpy()[:2] # Offset to image center

                # Filter out non-existent (padded) points for cleaner visualization
                real_points_mask_plot = (point_cloud_final_np[:, 3] == 1.0)
                
                plt.scatter(
                    plot_coords_xy[real_points_mask_plot, 0], 
                    plot_coords_xy[real_points_mask_plot, 1], 
                    c='r', s=1
                )
                plot_com_np_2d = final_com_2d_tensor.cpu().numpy()
                # Create a temporary 3D point just for transformation to image space
                plot_com_np_3d_for_transform = np.array([plot_com_np_2d[0], plot_com_np_2d[1], 0.0])
                plot_com_xy = plot_com_np_3d_for_transform[:2] * s_norm_to_img
                plot_com_xy += _AUG_IMG_CENTER_XYZ.cpu().numpy()[:2]
                plt.scatter(plot_com_xy[0], plot_com_xy[1], c='cyan', s=50, marker='x', label='CoM')                
                # Determine image display size for plot limits
                img_display_size = getattr(self, 'IMAGE_SIZE', _AUG_IMG_CENTER_XYZ.cpu().numpy()[0] * 2.0)
                plt.xlim(0, img_display_size)
                plt.ylim(img_display_size, 0) # Match imshow: (0,0) top-left, Y increases downwards
                # Note: original code had plt.ylim(0, self.IMAGE_SIZE). If Y needs to increase downwards
                # to match imshow, then ylim(max, min) is used. scatter Y will be plotted accordingly.

                plt.title(f"Point Cloud over Image (idx: {idx}, type: {type_key}, orig_idx: {original_index})")
                plt.show()

            return batch_data

        except KeyError as e:
            # Provide context for KeyError, common with dictionary/HDF5 access
            print(f"KeyError: {e} for type_key='{type_key}', original_index='{key_str}' (dataset index {idx}).")
            raise # Re-raise the exception to halt or be caught by a DataLoader

    def __len__(self):
        return len(self.indices)


class AtomicReconstructionDataset4(Dataset):
    PT_NORMALIZED_LABEL = 78/118 # Define as class constant or in init

    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized',
                point_cloud_size=4096, # Changed to 4096 as per your new request
                cell_size=8.0,
                source_cell_size=8.0, # Assuming input HDF5 data is normalized to this cell size
                augment=False,
                num_duplicates_per_atom=4, # New parameter
                gaussian_noise_sigma=0.001): # New parameter for noise std dev
        """
        Args:
            with padding and existance channel
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.cell_size = cell_size #nm
        self.source_cell_size = source_cell_size
        self.augment = augment
        self.num_duplicates_per_atom = num_duplicates_per_atom
        self.gaussian_noise_sigma = gaussian_noise_sigma

        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.indices = []

        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys)), key=int)
                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images
                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes
                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")


                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                pc_group = f["point_clouds"]["synthetic"][key]
                                synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds
                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
            raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")

    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = 'real' if data_type_flag == 0 else 'synthetic'
        new_cell_size = self.cell_size #nm
        batch_data = {}

        try:
            point_cloud_data = self.point_clouds[type_key][key_str]
            x_data = point_cloud_data['x']
            y_data = point_cloud_data['y']
            z_data = point_cloud_data['z']
            # Initial point cloud from HDF5 (N_total_atoms, 3)
            point_cloud_xyz_np = np.stack([x_data, y_data, z_data], axis=1).astype(np.float32)

            # Scale if source cell size is different from target cell size normalization
            if self.source_cell_size != new_cell_size and new_cell_size > 0:
                point_cloud_xyz_np = point_cloud_xyz_np * (self.source_cell_size / new_cell_size)
            
            labels = point_cloud_data['label']
            
            # Filter for Pt atoms: (N_pt_atoms, 3)
            point_cloud_xyz_np = point_cloud_xyz_np[labels == self.PT_NORMALIZED_LABEL]

            # Center z-coordinates for Pt atoms
            if point_cloud_xyz_np.shape[0] > 0:
                z_mean = np.mean(point_cloud_xyz_np[:,2])
                point_cloud_xyz_np[:,2] = point_cloud_xyz_np[:,2] - z_mean

            # --- Start: New logic for duplicating points and adding noise ---
            if point_cloud_xyz_np.shape[0] > 0 and self.num_duplicates_per_atom > 0:
                num_original_atoms = point_cloud_xyz_np.shape[0]
                # Repeat each atom's coordinates: (N_pt_atoms * num_duplicates, 3)
                expanded_xyz = np.repeat(point_cloud_xyz_np, self.num_duplicates_per_atom, axis=0)
                
                # Add Gaussian noise
                if self.gaussian_noise_sigma > 0:
                    noise = np.random.normal(0, self.gaussian_noise_sigma, expanded_xyz.shape).astype(np.float32)
                    perturbed_xyz = expanded_xyz + noise
                else:
                    perturbed_xyz = expanded_xyz # No noise if sigma is zero

                # Optional: Clip coordinates to [-1, 1] range if they go out of bounds
                # This depends on whether your downstream model expects strict [-1,1]
                # perturbed_xyz = np.clip(perturbed_xyz, -1.0, 1.0)
                
                point_cloud_xyz_processed = perturbed_xyz
            else:
                # If no Pt atoms, or num_duplicates_per_atom is 0, start with an empty array (or original if num_duplicates is 0)
                point_cloud_xyz_processed = point_cloud_xyz_np # which could be (0,3)
            # --- End: New logic ---

            # Assertions (check after noise, may need clipping or wider bounds if noise is large)
            if point_cloud_xyz_processed.shape[0] > 0:
                # Adjust bounds if noise can push them out, or clip before this assertion
                assert np.all(point_cloud_xyz_processed >= -1.1) and np.all(point_cloud_xyz_processed <= 1.1), \
                    f"Coordinates out of typical bounds for {type_key}/{key_str} after noise. Min: {point_cloud_xyz_processed.min()}, Max: {point_cloud_xyz_processed.max()}"


            # current_size is now the number of (potentially duplicated and perturbed) points
            current_size = point_cloud_xyz_processed.shape[0]
            target_size = self.point_cloud_size

            # Create existence channel for real (perturbed) points
            # point_cloud_np will be (current_size, 4) -> (x,y,z,existence)
            if current_size > 0:
                existence_channel_real = np.ones((current_size, 1), dtype=np.float32)
                point_cloud_with_existence = np.hstack((point_cloud_xyz_processed, existence_channel_real))
            else:
                # If there are no points (either no Pt or num_duplicates=0 on empty), start with an empty (0,4) array
                point_cloud_with_existence = np.empty((0, 4), dtype=np.float32)


            # Point cloud size handling (padding/truncation)
            xyz_loss_mask = np.zeros(target_size, dtype=np.float32) # Mask for XYZ loss

            if current_size == 0:
                # All points are padding points (0,0,0 for XYZ, 0 for existence)
                padding_xyz = np.zeros((target_size, 3), dtype=np.float32)
                padding_existence = np.zeros((target_size, 1), dtype=np.float32)
                point_cloud_final_np = np.hstack((padding_xyz, padding_existence))
                # xyz_loss_mask remains all zeros

            elif current_size < target_size:
                additional_needed = target_size - current_size
                
                padding_xyz_coords = np.zeros((additional_needed, 3), dtype=np.float32)
                padding_existence_scores = np.zeros((additional_needed, 1), dtype=np.float32)
                padding_4d = np.hstack((padding_xyz_coords, padding_existence_scores))
                
                point_cloud_final_np = np.vstack((point_cloud_with_existence, padding_4d))
                xyz_loss_mask[:current_size] = 1.0  # Mark original (perturbed) points for XYZ loss

            elif current_size > target_size:
                # Randomly sample from the perturbed points
                indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                point_cloud_final_np = point_cloud_with_existence[indices_to_keep]
                xyz_loss_mask[:] = 1.0 # All points are real (subsampled from perturbed)

            else: # current_size == target_size
                point_cloud_final_np = point_cloud_with_existence
                xyz_loss_mask[:] = 1.0 # All points are real (perturbed)

            # --- Data augmentation ---
            point_cloud_augmented_np = point_cloud_final_np.copy()
            image_np = self.images[type_key][original_index].copy()

            num_rotations_aug = 0
            should_flip_aug = False

            if self.augment:
                num_rotations_aug = np.random.randint(0, 4)
                if num_rotations_aug > 0:
                    theta = torch.tensor(np.pi/2 * num_rotations_aug).float()
                    rotation_matrix_torch = torch.tensor([
                        [torch.cos(theta), -torch.sin(theta), 0],
                        [torch.sin(theta), torch.cos(theta), 0],
                        [0, 0, 1]
                    ]).float()
                    
                    xyz_to_rotate = torch.from_numpy(point_cloud_augmented_np[:, :3]).float()
                    rotated_xyz = torch.matmul(xyz_to_rotate, rotation_matrix_torch.t())
                    point_cloud_augmented_np[:, :3] = rotated_xyz.numpy()
                    image_np = np.rot90(image_np, k=num_rotations_aug, axes=(0, 1))

                should_flip_aug = np.random.random() > 0.5
                if should_flip_aug:
                    point_cloud_augmented_np[:, 0] = -point_cloud_augmented_np[:, 0]
                    image_np = np.fliplr(image_np)
            
            point_cloud_tensor = torch.from_numpy(point_cloud_augmented_np).float().permute(1, 0)
            batch_data['point_cloud'] = point_cloud_tensor

            mask_tensor = torch.from_numpy(xyz_loss_mask).float()
            batch_data['point_cloud_mask'] = mask_tensor
            
            image_np_contiguous = np.ascontiguousarray(image_np)
            image_tensor = torch.from_numpy(image_np_contiguous).float().unsqueeze(0)
            batch_data['image'] = image_tensor

            pixel_size_val = self.pixel_sizes[type_key][original_index]
            batch_data['pixel_size'] = torch.tensor(pixel_size_val).float()

            return batch_data

        except KeyError as e:
            log.error(f"KeyError encountered for index {idx} (type: {type_key}, orig_idx: {original_index}): {e}.")
            log.warning(f"Returning dummy data for index {idx} due to KeyError.")
            dummy_pc = torch.zeros((4, self.point_cloud_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.point_cloud_size, dtype=torch.float32)
            dummy_img = torch.zeros((1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)
            dummy_px = torch.tensor(0.0).float()
            return {
                'point_cloud': dummy_pc,
                'point_cloud_mask': dummy_mask,
                'image': dummy_img,
                'pixel_size': dummy_px
            }
        except Exception as e:
            log.error(f"Unexpected error processing index {idx} (type: {type_key}, orig_idx: {original_index}): {e}", exc_info=True)
            # Depending on strictness, either raise or return dummy data
            # raise
            log.warning(f"Returning dummy data for index {idx} due to unexpected error: {e}")
            dummy_pc = torch.zeros((4, self.point_cloud_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.point_cloud_size, dtype=torch.float32)
            dummy_img = torch.zeros((1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)
            dummy_px = torch.tensor(0.0).float()
            return {
                'point_cloud': dummy_pc,
                'point_cloud_mask': dummy_mask,
                'image': dummy_img,
                'pixel_size': dummy_px
            }

    def __len__(self):
        return len(self.indices)
    

class AtomicReconstructionDataset5(Dataset):
    def __init__(self, hdf5_path, data_type='both', image_type='haadf_normalized', point_cloud_size=1024, cell_size=8.0): # Changed default point_cloud_size to 1024 for consistency with prompt
        """
        Args:
            with padding and existance channel
        """
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
             raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        self.data_type = data_type
        self.image_type = image_type
        self.point_cloud_size = point_cloud_size
        self.IMAGE_SIZE = 128
        self.cell_size = cell_size #nm
        # Fixed outputs
        self.outputs_to_load = ['image', 'point_cloud', 'pixel_size']

        # Data storage dictionaries
        self.point_clouds = {}
        self.images = {}
        self.pixel_sizes = {}
        self.indices = [] # Will store (data_type_flag, original_index)

        try:
            with h5py.File(hdf5_path, 'r') as f:
                real_keys, synthetic_keys = [], []
                if data_type in ['real', 'both']:
                    print("Loading real data...")
                    # ... (rest of your HDF5 loading logic for real data - assuming it's correct)
                    real_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "real" in f["point_clouds"]
                    img_group_exists = image_type in f and "real" in f[image_type]
                    px_group_exists = "pixel_size" in f and "real" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        real_pc_keys = set(f["point_clouds"]["real"].keys())
                        real_img_keys = set(f[image_type]["real"].keys())
                        real_px_keys = set(f["pixel_size"]["real"].keys())
                        real_keys = sorted(list(real_pc_keys.intersection(real_img_keys).intersection(real_px_keys)), key=int)
                        print(f"Found {len(real_keys)} consistent real samples.")
                        if real_keys:
                            real_point_clouds = {}
                            for key in tqdm(real_keys, desc="Real point clouds"):
                                pc_group = f["point_clouds"]["real"][key]
                                real_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['real'] = real_point_clouds
                            real_images = {}
                            for key in tqdm(real_keys, desc="Real images"): real_images[int(key)] = f[image_type]["real"][key][()]
                            self.images['real'] = real_images
                            real_pixel_sizes = {}
                            for key in tqdm(real_keys, desc="Real pixel sizes"): real_pixel_sizes[int(key)] = f["pixel_size"]["real"][key][()]
                            self.pixel_sizes['real'] = real_pixel_sizes
                            self.indices.extend([(0, int(i)) for i in real_keys])
                            real_data_loaded = True
                    if data_type == 'real' and not real_data_loaded: log.warning(f"Requested 'real' data, but no consistent samples found.")


                if data_type in ['synthetic', 'both']:
                    print("Loading synthetic data...")
                    # ... (rest of your HDF5 loading logic for synthetic data - assuming it's correct)
                    synthetic_data_loaded = False
                    pc_group_exists = "point_clouds" in f and "synthetic" in f["point_clouds"]
                    img_group_exists = image_type in f and "synthetic" in f[image_type]
                    px_group_exists = "pixel_size" in f and "synthetic" in f["pixel_size"]
                    if pc_group_exists and img_group_exists and px_group_exists:
                        synth_pc_keys = set(f["point_clouds"]["synthetic"].keys())
                        synth_img_keys = set(f[image_type]["synthetic"].keys())
                        synth_px_keys = set(f["pixel_size"]["synthetic"].keys())
                        synthetic_keys = sorted(list(synth_pc_keys.intersection(synth_img_keys).intersection(synth_px_keys)), key=int)
                        print(f"Found {len(synthetic_keys)} consistent synthetic samples.")
                        if synthetic_keys:
                            synthetic_point_clouds = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic point clouds"):
                                 pc_group = f["point_clouds"]["synthetic"][key]
                                 synthetic_point_clouds[key] = {'x': pc_group['x'][()], 'y': pc_group['y'][()], 'z': pc_group['z'][()], 'label': pc_group['label'][()]}
                            self.point_clouds['synthetic'] = synthetic_point_clouds
                            synthetic_images = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic images"): synthetic_images[int(key)] = f[image_type]["synthetic"][key][()]
                            self.images['synthetic'] = synthetic_images
                            synthetic_pixel_sizes = {}
                            for key in tqdm(synthetic_keys, desc="Synthetic pixel sizes"): synthetic_pixel_sizes[int(key)] = f["pixel_size"]["synthetic"][key][()]
                            self.pixel_sizes['synthetic'] = synthetic_pixel_sizes
                            self.indices.extend([(1, int(i)) for i in synthetic_keys])
                            synthetic_data_loaded = True
                    if data_type == 'synthetic' and not synthetic_data_loaded: log.warning(f"Requested 'synthetic' data, but no consistent samples found.")

        except Exception as e:
            log.error(f"Error loading HDF5 file {self.hdf5_path}: {e}", exc_info=True)
            raise

        if not self.indices:
             raise RuntimeError(f"Dataset initialization failed: No loadable samples found for data_type='{self.data_type}' and image_type='{self.image_type}' in {self.hdf5_path}")
        else:
            print(f"Dataset initialized successfully with {len(self.indices)} samples.")
    # --- End of __init__ ---


    def __getitem__(self, idx):
        data_type_flag, original_index = self.indices[idx]
        key_str = str(original_index)
        type_key = 'real' if data_type_flag == 0 else 'synthetic'
        old_cell_size = 8.0 #nm
        new_cell_size = self.cell_size #nm
        batch_data = {}

        try:
            point_cloud_data = self.point_clouds[type_key][key_str]
            x_data = point_cloud_data['x']
            y_data = point_cloud_data['y']
            z_data = point_cloud_data['z']
            point_cloud_xyz_np = np.stack([x_data, y_data, z_data], axis=1).astype(np.float32)

            point_cloud_xyz_np = point_cloud_xyz_np * (old_cell_size / 2) / (new_cell_size / 2)
            
            labels = point_cloud_data['label']
            Pt = 78/118 # Assuming this is the normalized label for Pt
            
            # Filter for Pt atoms
            point_cloud_xyz_np = point_cloud_xyz_np[labels == Pt]

            # Center z-coordinates
            if point_cloud_xyz_np.shape[0] > 0:
                z_mean = np.mean(point_cloud_xyz_np[:,2])
                point_cloud_xyz_np[:,2] = point_cloud_xyz_np[:,2] - z_mean

            if point_cloud_xyz_np.shape[0] > 0: 
                assert np.all(point_cloud_xyz_np[:,0] >= -1) and np.all(point_cloud_xyz_np[:,0] <= 1), f"X coordinates out of bounds for {type_key}/{key_str} before padding"
                assert np.all(point_cloud_xyz_np[:,1] >= -1) and np.all(point_cloud_xyz_np[:,1] <= 1), f"Y coordinates out of bounds for {type_key}/{key_str} before padding"
                assert np.all(point_cloud_xyz_np[:,2] >= -1) and np.all(point_cloud_xyz_np[:,2] <= 1), f"Z coordinates out of bounds for {type_key}/{key_str} before padding"

            current_size = point_cloud_xyz_np.shape[0]
            target_size = self.point_cloud_size

            if current_size > 0:
                existence_channel_real = np.ones((current_size, 1), dtype=np.float32)
                point_cloud_np = np.hstack((point_cloud_xyz_np, existence_channel_real))
            else:
                point_cloud_np = np.empty((0, 4), dtype=np.float32)


            xyz_loss_mask = np.zeros(target_size, dtype=np.float32) 

            if current_size == 0:
                padding_xyz = np.zeros((target_size, 3), dtype=np.float32)
                padding_existence = np.zeros((target_size, 1), dtype=np.float32)
                point_cloud_final_np = np.hstack((padding_xyz, padding_existence))

            elif current_size < target_size:
                additional_needed = target_size - current_size
                padding_xyz_coords = np.zeros((additional_needed, 3), dtype=np.float32)
                padding_existence_scores = np.zeros((additional_needed, 1), dtype=np.float32)
                padding_4d = np.hstack((padding_xyz_coords, padding_existence_scores))
                point_cloud_final_np = np.vstack((point_cloud_np, padding_4d))
                xyz_loss_mask[:current_size] = 1.0

            elif current_size > target_size:
                indices_to_keep = np.random.choice(current_size, size=target_size, replace=False)
                point_cloud_final_np = point_cloud_np[indices_to_keep]
                xyz_loss_mask[:] = 1.0 

            else: 
                point_cloud_final_np = point_cloud_np
                xyz_loss_mask[:] = 1.0

            # --- Data augmentation ---
            point_cloud_augmented_np = point_cloud_final_np.copy()
            # Load original image for augmentation
            image_to_augment_np = self.images[type_key][original_index].copy()


            # Determine augmentations
            # num_rotations: 0, 1, 2, or 3 steps of 90-degree counter-clockwise rotation
            num_rotations = np.random.randint(0, 4) 
            num_rotations = 1
            should_flip = np.random.random() > 0.5
            shouldflip = True

            # Apply rotation to point cloud
            if num_rotations > 0:
                theta = torch.tensor(np.pi/2 * num_rotations).float()
                # Standard CCW rotation matrix for column vectors v' = R * v
                # For row vectors p' = p * R.T
                rotation_matrix_3d_z_ccw = torch.tensor([
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta),  torch.cos(theta), 0],
                    [0,                 0,                1]
                ]).float()
                
                xyz_to_rotate = torch.from_numpy(point_cloud_augmented_np[:, :3]).float()
                rotated_xyz = torch.matmul(xyz_to_rotate, rotation_matrix_3d_z_ccw.T) # p' = p * R.T
                point_cloud_augmented_np[:, :3] = rotated_xyz.numpy()

                # Apply corresponding rotation to image (np.rot90 rotates CCW)
                image_to_augment_np = np.rot90(image_to_augment_np, k=num_rotations, axes=(0, 1))

            # Apply horizontal reflection to point cloud
            if should_flip:
                point_cloud_augmented_np[:, 0] = -point_cloud_augmented_np[:, 0] # Flip X-coordinate

                # Apply corresponding flip to image
                image_to_augment_np = np.fliplr(image_to_augment_np)

            point_cloud_tensor = torch.from_numpy(point_cloud_augmented_np).float().permute(1, 0)
            batch_data['point_cloud'] = point_cloud_tensor

            mask_tensor = torch.from_numpy(xyz_loss_mask).float()
            batch_data['point_cloud_mask'] = mask_tensor

            # Prepare augmented image tensor
            image_augmented_np_contiguous = np.ascontiguousarray(image_to_augment_np)
            image_tensor = torch.from_numpy(image_augmented_np_contiguous).float().unsqueeze(0)
            batch_data['image'] = image_tensor

            pixel_size_val = self.pixel_sizes[type_key][original_index]
            batch_data['pixel_size'] = torch.tensor(pixel_size_val).float()

            return batch_data

        except KeyError as e:
            log.error(f"KeyError encountered for index {idx} (type: {type_key}, orig_idx: {original_index}): {e}.")
            log.warning(f"Returning dummy data for index {idx} due to KeyError.")
            # Return dummy data (zeros)
            dummy_pc = torch.zeros((3, self.point_cloud_size), dtype=torch.float32)
            dummy_mask = torch.zeros(self.point_cloud_size, dtype=torch.float32) # Mask is zeros for dummy
            dummy_img = torch.zeros((1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=torch.float32)
            dummy_px = torch.tensor(0.0).float()
            return {
                'point_cloud': dummy_pc,
                'point_cloud_mask': dummy_mask,
                'image': dummy_img,
                'pixel_size': dummy_px
            }
        except Exception as e:
            log.error(f"Unexpected error processing index {idx} (type: {type_key}, orig_idx: {original_index}): {e}", exc_info=True)
            raise # Re-raise unexpected errors

    def __len__(self):
        return len(self.indices)