# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import argparse, os, os.path as osp, time
import numpy as np
import trimesh
import mesh2sdf
import h5py
from skimage.measure import marching_cubes


def normalize_mesh_to_unit_box(mesh):
    # V: (N, 3) vertices
    V = mesh.vertices.copy()

    # Axis-aligned bounding box
    vmin = V.min(axis=0)
    vmax = V.max(axis=0)

    # Center of the box
    center = (vmax + vmin) * 0.5

    # Scale so the largest side of the box becomes 2.0 (for [-1, 1])
    bbox_size = (vmax - vmin).max()
    if bbox_size == 0:
        raise ValueError("Degenerate mesh with zero-size bounding box")

    scale = 1.9 / bbox_size

    # Apply normalization
    V_norm = (V - center) * scale

    mesh.vertices = V_norm
    return mesh

# ========== CLI ==========
ap = argparse.ArgumentParser()
ap.add_argument("--name", type=str, required=True)
ap.add_argument("--input_dir", default="")
ap.add_argument("--output_dir", default="")
ap.add_argument("--size", type=int, default=256, help="SDF grid size (N)")
args = ap.parse_args()

full_name = args.name
name = full_name.split(".")[0]
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

h5_out  = osp.join(output_dir, f"{name}_mc.h5")

if osp.exists(h5_out):
    raise SystemExit(0)

t0 = time.time()

# ========== Load mesh (no transformations) ==========
mesh_path = osp.join(input_dir, f"{full_name}")
mesh = trimesh.load(mesh_path, force='mesh')
mesh = normalize_mesh_to_unit_box(mesh)
if mesh is None or mesh.is_empty:
    raise ValueError(f"Failed to load mesh: {mesh_path}")

vertices = np.asarray(mesh.vertices, dtype=np.float32)
faces    = np.asarray(mesh.faces, dtype=np.int32)



# ========== Compute SDF ==========
size  = int(args.size)
level = 2.0 / size  # voxel size in normalized [-1,1] space (used by mesh2sdf)
sdf = mesh2sdf.compute(vertices, faces, size, fix=True, level=level)  # return_mesh=False



# ========== Extract iso-surface and save ==========
# NOTE: marching_cubes returns vertices in voxel index coords [0, N-1]^3.
# Map to [-1, 1]^3 to match the SDF normalization convention:
# # ========== Save SDF ==========
with h5py.File(h5_out, "w") as f:
    f.create_dataset("sdf", data=sdf, compression="gzip", compression_opts=4)
