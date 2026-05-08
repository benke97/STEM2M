# mar/data/dataset.py
"""
Morphology dataset reader.

Reads the contiguous HDF5 layout produced by `regenerate_sim_data.py`:

    /<split>/haadf_gan       (N, 128, 128) float32
    /<split>/haadf_norm      (N, 128, 128) float32
    /<split>/thickness_pt    (N, 128, 128) float32
    /<split>/com_gan         (N, 2)        float32
    /<split>/com_clean       (N, 2)        float32
    /<split>/pixel_size      (N,)          float32
    /<split>/sample_ids      (N,)          variable-length string
    /<split>/pc_x            (M_total,)    float32
    /<split>/pc_y            (M_total,)    float32
    /<split>/pc_z            (M_total,)    float32
    /<split>/pc_label        (M_total,)    float32
    /<split>/pc_ptr          (N+1,)        uint64

Splits are 'real' and 'synthetic'. Sample ids are not used by training; they
are kept for traceability.
"""
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# Coordinate cell size used when the source point clouds were stored.
# A morphology produced by the dataset is rescaled from this cell size to the
# `cell_size` requested by the user via cfg.data.cell_size.
_OLD_CELL_SIZE = 8.0  # nm

# Atomic-number labels in the point cloud are stored as Z / 118. Pt = 78.
_PT_LABEL = 78 / 118.0

# (data_type_flag, split_name) — kept for back-compat with the split logic in
# data_module.py which expects integer flags.
_TYPE_KEY_MAP = {0: "real", 1: "synthetic"}

# Augmentation constants (image size 128, rotation pivot at the geometric center).
_AUG_IMG_CENTER_XYZ = torch.tensor([64.0, 64.0, 0.0], dtype=torch.float32)
_AUG_ROT_AXIS_XYZ = torch.tensor([63.5, 63.5, 0.0], dtype=torch.float32)


class MorphologyDataset(Dataset):
    """Dataset of (HAADF image, 3D morphology) pairs in the contiguous HDF5 layout."""

    IMAGE_SIZE = 128

    def __init__(
        self,
        hdf5_path,
        data_type: str = "both",          # 'real', 'synthetic', or 'both'
        image_type: str = "haadf_norm",   # 'haadf_norm' or 'haadf_gan'
        point_cloud_size: int = 1024,
        cell_size: float = 8.0,
        augment: bool = True,
        pad_with_noise: bool = False,
        noise_std: float = 0.3,
        in_memory: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at: {self.hdf5_path}")

        if data_type not in ("real", "synthetic", "both"):
            raise ValueError(f"data_type must be 'real', 'synthetic', or 'both'; got {data_type!r}")
        if image_type not in ("haadf_norm", "haadf_gan"):
            raise ValueError(f"image_type must be 'haadf_norm' or 'haadf_gan'; got {image_type!r}")

        self.data_type = data_type
        self.image_type = image_type
        self.com_field = "com_clean" if image_type == "haadf_norm" else "com_gan"
        self.point_cloud_size = point_cloud_size
        self.cell_size = cell_size
        self.augment = augment
        self.pad_with_noise = pad_with_noise
        self.noise_std = noise_std
        self.in_memory = in_memory

        # When in_memory=False, each worker process lazily opens its own HDF5
        # handle (see _h5()). When in_memory=True, the per-split arrays are
        # eagerly loaded below into self._mem and the HDF5 file is closed.
        self._h5_handle = None
        self._h5_pid = None
        self._mem = {} if in_memory else None

        # Discover sample counts and build (data_type_flag, local_idx) index list.
        self.indices = []
        with h5py.File(self.hdf5_path, "r") as f:
            for flag, split in _TYPE_KEY_MAP.items():
                if data_type not in (split, "both"):
                    continue
                if split not in f:
                    log.warning(f"Split '{split}' missing from {self.hdf5_path}; skipping")
                    continue
                grp = f[split]
                if image_type not in grp:
                    log.warning(f"Field '{image_type}' missing from /{split}; skipping")
                    continue
                n = grp[image_type].shape[0]
                self.indices.extend([(flag, i) for i in range(n)])

                if in_memory:
                    log.info(f"Loading /{split} into RAM ...")
                    self._mem[split] = {
                        image_type: grp[image_type][:],
                        self.com_field: grp[self.com_field][:],
                        "pixel_size": grp["pixel_size"][:],
                        "pc_x": grp["pc_x"][:],
                        "pc_y": grp["pc_y"][:],
                        "pc_z": grp["pc_z"][:],
                        "pc_label": grp["pc_label"][:],
                        "pc_ptr": grp["pc_ptr"][:],
                    }
                    bytes_loaded = sum(a.nbytes for a in self._mem[split].values())
                    log.info(f"  /{split}: {n} samples, {bytes_loaded / (1024**3):.2f} GB in RAM")

        if not self.indices:
            raise RuntimeError(
                f"No samples loadable from {self.hdf5_path} for "
                f"data_type={data_type!r}, image_type={image_type!r}"
            )

        log.info(f"MorphologyDataset: {len(self.indices)} samples from {self.hdf5_path} (in_memory={in_memory})")

    def __len__(self):
        return len(self.indices)

    def _h5(self):
        """Lazily open the HDF5 file once per worker process."""
        if self._h5_handle is None or self._h5_pid != os.getpid():
            self._h5_handle = h5py.File(self.hdf5_path, "r")
            self._h5_pid = os.getpid()
        return self._h5_handle

    def _split_data(self, split):
        """Return a key->array view for `split`, either in-memory dict or h5py group."""
        if self.in_memory:
            return self._mem[split]
        return self._h5()[split]

    def _read_point_cloud(self, split_data, local_idx):
        """Read one variable-length point cloud via the CSR-style pc_ptr index."""
        ptr = split_data["pc_ptr"]
        start = int(ptr[local_idx])
        end = int(ptr[local_idx + 1])
        x = split_data["pc_x"][start:end]
        y = split_data["pc_y"][start:end]
        z = split_data["pc_z"][start:end]
        label = split_data["pc_label"][start:end]
        return x, y, z, label

    def _transform_coords_for_augmentation(
        self, coords_tensor, s_norm_to_img, s_img_to_norm, operation_fn
    ):
        device = coords_tensor.device
        center = _AUG_IMG_CENTER_XYZ.to(device)
        pivot = _AUG_ROT_AXIS_XYZ.to(device)
        out = coords_tensor * s_norm_to_img + center - pivot
        out = operation_fn(out)
        out = (out + pivot - center) * s_img_to_norm
        return out

    def __getitem__(self, idx):
        data_type_flag, local_idx = self.indices[idx]
        split = _TYPE_KEY_MAP[data_type_flag]
        grp = self._split_data(split)

        # --- Point cloud: filter to Pt, rescale, pad/sample to fixed length ---
        x, y, z, label = self._read_point_cloud(grp, local_idx)
        pt_mask = (label == _PT_LABEL)
        xyz = np.stack([x[pt_mask], y[pt_mask], z[pt_mask]], axis=1).astype(np.float32)
        xyz *= (_OLD_CELL_SIZE / self.cell_size)

        # Center z so the morphology straddles z=0.
        if xyz.shape[0] > 0:
            xyz[:, 2] -= xyz[:, 2].mean()
            assert np.all(np.abs(xyz) <= 1.0 + 1e-5), (
                f"Coordinates out of [-1, 1] for /{split}/{local_idx} "
                f"(min={xyz.min():.4f}, max={xyz.max():.4f})"
            )

        target = self.point_cloud_size
        if self.pad_with_noise:
            pc_full = np.random.normal(0, self.noise_std, (target, 4)).astype(np.float32)
            pc_full[:, 3] = 0.0
        else:
            pc_full = np.zeros((target, 4), dtype=np.float32)
        loss_mask = np.zeros(target, dtype=np.float32)

        cur = xyz.shape[0]
        if cur > 0:
            if cur < target:
                pc_full[:cur, :3] = xyz
                pc_full[:cur, 3] = 1.0
                loss_mask[:cur] = 1.0
            elif cur > target:
                keep = np.random.choice(cur, size=target, replace=False)
                pc_full[:, :3] = xyz[keep]
                pc_full[:, 3] = 1.0
                loss_mask[:] = 1.0
            else:
                pc_full[:, :3] = xyz
                pc_full[:, 3] = 1.0
                loss_mask[:] = 1.0

        perm = np.random.permutation(target)
        pc_full = pc_full[perm]
        loss_mask = loss_mask[perm]

        # --- CoM (2D) and image ---
        com_2d = np.asarray(grp[self.com_field][local_idx], dtype=np.float32).reshape(-1)
        com_3d = torch.tensor([com_2d[0], com_2d[1], 0.0], dtype=torch.float32)

        image_np = grp[self.image_type][local_idx]
        pixel_size_val = float(grp["pixel_size"][local_idx])

        # --- Augmentation (rotation by 0/90/180/270 + optional flip) ---
        xyz_t = torch.from_numpy(pc_full[:, :3].copy()).float()
        s_norm_to_img = self.cell_size / (2.0 * pixel_size_val)
        s_img_to_norm = (2.0 * pixel_size_val) / self.cell_size

        num_rot = np.random.randint(0, 4) if self.augment else 0
        do_flip = (self.augment and np.random.random() > 0.5)

        if num_rot > 0:
            theta = np.pi / 2.0 * num_rot
            c, s = np.cos(theta), np.sin(theta)
            R_ccw = torch.tensor(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
            )
            rotate = lambda v: torch.matmul(v, R_ccw.t())
            xyz_t = self._transform_coords_for_augmentation(xyz_t, s_norm_to_img, s_img_to_norm, rotate)
            com_3d = self._transform_coords_for_augmentation(
                com_3d.unsqueeze(0), s_norm_to_img, s_img_to_norm, rotate
            ).squeeze(0)
            image_np = np.rot90(image_np, k=(-num_rot) % 4, axes=(0, 1))

        if do_flip:
            def flip(v):
                out = v.clone()
                out[:, 1] = -out[:, 1]
                return out
            xyz_t = self._transform_coords_for_augmentation(xyz_t, s_norm_to_img, s_img_to_norm, flip)
            com_3d = self._transform_coords_for_augmentation(
                com_3d.unsqueeze(0), s_norm_to_img, s_img_to_norm, flip
            ).squeeze(0)
            image_np = np.flipud(image_np)

        existence = torch.from_numpy(pc_full[:, 3]).float().unsqueeze(1)
        point_cloud = torch.cat((xyz_t, existence), dim=1).permute(1, 0)  # (4, target)

        return {
            "point_cloud": point_cloud,                              # (4, N)
            "point_cloud_mask": torch.from_numpy(loss_mask).float(), # (N,)
            "image": torch.from_numpy(np.ascontiguousarray(image_np)).float().unsqueeze(0),  # (1, H, W)
            "pixel_size": torch.tensor(pixel_size_val, dtype=torch.float32),
            "CoM": com_3d[:2],                                       # (2,)
            "permutation": torch.from_numpy(perm).long(),
            "dataset_idx": idx,
            "original_index": local_idx,
            "data_type": split,                                      # 'real' or 'synthetic'
        }
