"""
Rewrite E:\\data\\combined_dataset.h5 into a contiguous, fast layout.

Source layout (slow):
  /<field>/<split>/<sample_id>                       -> dataset
  /particle_thickness_maps/78/<split>/<sample_id>    -> dataset (extra level)
  /point_clouds/<split>/<sample_id>                  -> dataset or group{x,y,z,label}

Sample IDs are integer strings ('0'..'N-1'). HDF5 returns them in lexicographic
order ('0','1','10','100',...) so we sort numerically before writing.

Target layout:
  /<split>/haadf_gan       (N, 128, 128) float32
  /<split>/haadf_norm      (N, 128, 128) float32
  /<split>/thickness_pt    (N, 128, 128) float32
  /<split>/com_gan         (N, 2)        float32
  /<split>/com_clean       (N, 2)        float32
  /<split>/pixel_size      (N,)          float32
  /<split>/sample_ids      (N,)          variable-length string
  /<split>/pc_x/y/z/label  (M_total,)    float32
  /<split>/pc_ptr          (N+1,)        uint64
"""
import sys
import time
import h5py
import numpy as np
from pathlib import Path

SRC = r"E:\data\combined_dataset.h5"
DST = "combined_dataset_contiguous.h5"

SPLITS = ["real", "synthetic"]

# (src_path_parts, dst_field, kind)
# src_path_parts: list of group names below the root, then split is appended,
# then the sample id. Lets us handle the extra '78' level for thickness.
FIELD_SPECS = [
    (["haadf_cycleGAN"],              "haadf_gan",    "image"),
    (["haadf_normalized"],            "haadf_norm",   "image"),
    (["particle_thickness_maps", "78"], "thickness_pt", "image"),
    (["CoM_CycleGAN"],                "com_gan",      "vec2"),
    (["CoM_clean"],                   "com_clean",    "vec2"),
    (["pixel_size"],                  "pixel_size",   "scalar"),
]
PC_PARTS = ["point_clouds"]   # handled separately

IMAGE_SHAPE = (128, 128)
IMAGE_DTYPE = np.float32
IMAGE_CHUNK = (1, 128, 128)
COMPRESSION = "lzf"

LOG_EVERY = 2000
PC_FLUSH_EVERY = 1024


def resolve(src, parts, split):
    """Walk down parts then split. Return None if missing."""
    node = src
    for p in parts:
        if p not in node:
            return None
        node = node[p]
    if split not in node:
        return None
    return node[split]


def get_sorted_sample_ids(src_split_grp):
    """Get sample IDs sorted numerically (not lexicographically)."""
    print(f"    listing keys (slow for 250k)...", flush=True)
    t0 = time.time()
    raw = list(src_split_grp.keys())
    print(f"    got {len(raw)} keys in {time.time()-t0:.1f}s, sorting numerically...", flush=True)
    # All keys should be int strings; if not, fall back to natural-ish sort
    try:
        ids = sorted(raw, key=int)
    except ValueError as e:
        print(f"    [WARN] non-integer key found: {e}. Using string sort.")
        ids = sorted(raw)
    return ids


def read_pc(src_pc_split, sid):
    """Read one point cloud sample."""
    obj = src_pc_split[sid]
    if isinstance(obj, h5py.Dataset):
        arr = obj[...]
        if arr.ndim == 2 and arr.shape[1] >= 4:
            return arr[:, 0].astype(np.float32), arr[:, 1].astype(np.float32), \
                   arr[:, 2].astype(np.float32), arr[:, 3].astype(np.float32)
        elif arr.ndim == 2 and arr.shape[1] == 3:
            return arr[:, 0].astype(np.float32), arr[:, 1].astype(np.float32), \
                   arr[:, 2].astype(np.float32), np.zeros(arr.shape[0], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected point_cloud shape for {sid}: {arr.shape}")
    elif isinstance(obj, h5py.Group):
        keymap = {k.lower(): k for k in obj.keys()}
        def get(*names):
            for n in names:
                if n in keymap:
                    return obj[keymap[n]][...]
            return None
        x = get("x"); y = get("y"); z = get("z")
        lbl = get("label", "labels")
        if x is None or y is None or z is None:
            raise ValueError(f"point_cloud group {sid} keys: {list(obj.keys())}")
        if lbl is None:
            lbl = np.zeros_like(x, dtype=np.float32)
        return (np.asarray(x, dtype=np.float32),
                np.asarray(y, dtype=np.float32),
                np.asarray(z, dtype=np.float32),
                np.asarray(lbl, dtype=np.float32))
    else:
        raise ValueError(f"Unexpected type for point_cloud/{sid}: {type(obj)}")


def probe_first_sample(src, split, sample_ids):
    print(f"\n  probing first sample (id='{sample_ids[0]}') of /{split}...")
    sid = sample_ids[0]
    for parts, dst_field, kind in FIELD_SPECS:
        grp = resolve(src, parts, split)
        if grp is None:
            print(f"    [WARN] {'/'.join(parts)}/{split} missing")
            continue
        if sid not in grp:
            print(f"    [WARN] {'/'.join(parts)}/{split}/{sid} missing")
            continue
        obj = grp[sid]
        if isinstance(obj, h5py.Dataset):
            print(f"    {'/'.join(parts)} -> {dst_field} ({kind}): "
                  f"shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"    {'/'.join(parts)} -> {dst_field} ({kind}): GROUP")
    pc_grp = resolve(src, PC_PARTS, split)
    if pc_grp is not None and sid in pc_grp:
        x, y, z, lbl = read_pc(pc_grp, sid)
        print(f"    point_clouds -> pc_*: M={len(x)}")


def convert_split(src, dst, split):
    print(f"\n=== Converting split: {split} ===")

    # Use haadf_normalized as master id source (could use any field)
    master = resolve(src, ["haadf_normalized"], split)
    if master is None:
        print(f"  no haadf_normalized for /{split}, skipping")
        return
    sample_ids = get_sorted_sample_ids(master)
    N = len(sample_ids)
    if N == 0:
        return

    probe_first_sample(src, split, sample_ids)

    grp = dst.create_group(split)

    # Resolve & cache source group handles for each field
    src_field_grps = {}     # dst_field -> (src_group, kind)
    for parts, dst_field, kind in FIELD_SPECS:
        sg = resolve(src, parts, split)
        if sg is not None:
            src_field_grps[dst_field] = (sg, kind)
        else:
            print(f"  [WARN] skipping field {dst_field} (source missing)")

    # Pre-create destination datasets
    dsets = {}
    for dst_field, (_, kind) in src_field_grps.items():
        if kind == "image":
            dsets[dst_field] = grp.create_dataset(
                dst_field, shape=(N, *IMAGE_SHAPE), dtype=IMAGE_DTYPE,
                chunks=IMAGE_CHUNK, compression=COMPRESSION,
            )
        elif kind == "vec2":
            dsets[dst_field] = grp.create_dataset(
                dst_field, shape=(N, 2), dtype=np.float32,
            )
        elif kind == "scalar":
            dsets[dst_field] = grp.create_dataset(
                dst_field, shape=(N,), dtype=np.float32,
            )

    str_dt = h5py.string_dtype(encoding="utf-8")
    sid_dset = grp.create_dataset("sample_ids", shape=(N,), dtype=str_dt)

    # Point clouds
    pc_grp_src = resolve(src, PC_PARTS, split)
    have_pc = pc_grp_src is not None
    if have_pc:
        pc_x = grp.create_dataset("pc_x", shape=(0,), maxshape=(None,),
                                  dtype=np.float32, chunks=(65536,), compression=COMPRESSION)
        pc_y = grp.create_dataset("pc_y", shape=(0,), maxshape=(None,),
                                  dtype=np.float32, chunks=(65536,), compression=COMPRESSION)
        pc_z = grp.create_dataset("pc_z", shape=(0,), maxshape=(None,),
                                  dtype=np.float32, chunks=(65536,), compression=COMPRESSION)
        pc_label = grp.create_dataset("pc_label", shape=(0,), maxshape=(None,),
                                      dtype=np.float32, chunks=(65536,), compression=COMPRESSION)
        pc_ptr_arr = np.zeros(N + 1, dtype=np.uint64)
        buf_x, buf_y, buf_z, buf_l = [], [], [], []
        pc_total = 0

        def flush_pc():
            nonlocal pc_total
            if not buf_x:
                return
            bx = np.concatenate(buf_x); by = np.concatenate(buf_y)
            bz = np.concatenate(buf_z); bl = np.concatenate(buf_l)
            new_total = pc_total + len(bx)
            pc_x.resize((new_total,)); pc_x[pc_total:new_total] = bx
            pc_y.resize((new_total,)); pc_y[pc_total:new_total] = by
            pc_z.resize((new_total,)); pc_z[pc_total:new_total] = bz
            pc_label.resize((new_total,)); pc_label[pc_total:new_total] = bl
            pc_total = new_total
            buf_x.clear(); buf_y.clear(); buf_z.clear(); buf_l.clear()

    t0 = time.time()
    missing_log = {df: 0 for df in src_field_grps}
    missing_pc = 0

    for i, sid in enumerate(sample_ids):
        sid_dset[i] = sid

        for dst_field, (sg, kind) in src_field_grps.items():
            if sid not in sg:
                missing_log[dst_field] += 1
                continue
            arr = sg[sid][...]
            if kind == "image":
                dsets[dst_field][i] = arr.astype(IMAGE_DTYPE, copy=False)
            elif kind == "vec2":
                v = np.asarray(arr, dtype=np.float32).reshape(-1)
                if v.size != 2:
                    raise ValueError(f"{dst_field}/{sid} expected size 2, got {v.size}")
                dsets[dst_field][i] = v
            elif kind == "scalar":
                dsets[dst_field][i] = float(np.asarray(arr).reshape(-1)[0])

        if have_pc:
            if sid in pc_grp_src:
                x, y, z, lbl = read_pc(pc_grp_src, sid)
                buf_x.append(x); buf_y.append(y); buf_z.append(z); buf_l.append(lbl)
                pc_ptr_arr[i + 1] = pc_ptr_arr[i] + len(x)
            else:
                pc_ptr_arr[i + 1] = pc_ptr_arr[i]
                missing_pc += 1
            if (i + 1) % PC_FLUSH_EVERY == 0:
                flush_pc()

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"  [{split}] {i+1}/{N}  ({rate:.1f} samples/s, eta {eta/60:.1f} min)",
                  flush=True)

    if have_pc:
        flush_pc()
        grp.create_dataset("pc_ptr", data=pc_ptr_arr)
        print(f"  /{split}/pc_ptr written; total pc points = {pc_total:,}; "
              f"missing pc samples = {missing_pc}")

    for df, n_missing in missing_log.items():
        if n_missing:
            print(f"  [WARN] /{split}/{df}: {n_missing} samples were missing in source")

    print(f"  /{split} done in {(time.time()-t0)/60:.1f} min")


def verify(dst_path):
    print(f"\n=== Verifying {dst_path} ===")
    with h5py.File(dst_path, "r") as f:
        for split in SPLITS:
            if split not in f:
                continue
            g = f[split]
            print(f"\n  /{split}:")
            for k in g.keys():
                d = g[k]
                if isinstance(d, h5py.Dataset):
                    print(f"    {k}: shape={d.shape} dtype={d.dtype} "
                          f"chunks={d.chunks} compression={d.compression}")
            if "pc_ptr" in g:
                ptr = g["pc_ptr"][...]
                ok_mono = bool((np.diff(ptr) >= 0).all())
                ok_match = int(ptr[-1]) == g["pc_x"].shape[0]
                print(f"    pc_ptr: monotonic={ok_mono}, "
                      f"final={int(ptr[-1])} matches pc_x: {ok_match}")
            # Quick speed test on rewritten file
            if "haadf_norm" in g:
                t0 = time.time()
                d = g["haadf_norm"]
                idxs = np.linspace(0, d.shape[0] - 1, 100).astype(int)
                for i in idxs:
                    _ = d[i]
                dt = time.time() - t0
                print(f"    speed test: 100 random reads of haadf_norm in "
                      f"{dt*1000:.1f} ms ({dt*10:.2f} ms/sample)")


def main():
    dst_path = Path(DST).resolve()
    if dst_path.exists():
        print(f"[ERROR] {dst_path} already exists. Delete it or change DST.")
        sys.exit(1)

    print(f"Reading from: {SRC}")
    print(f"Writing to:   {dst_path}")

    t_all = time.time()
    with h5py.File(SRC, "r") as src, h5py.File(dst_path, "w") as dst:
        for split in SPLITS:
            convert_split(src, dst, split)

    print(f"\nTotal time: {(time.time()-t_all)/60:.1f} min")
    verify(dst_path)


if __name__ == "__main__":
    main()