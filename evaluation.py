# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import argparse, os, time, numpy as np, trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from model.decompose import (
sample_interior_points,
sample_surface_intersection,
_estimate_boundary_areas_by_retention,
_alloc_counts_by_weight,
hausdorff_two_way_points,
concavity_volbound,
)



def concavity2(submesh, full_mesh,  interior_points, intersectors, full_mesh_volumn, n_samples=1000, seed=42, k=0.3,num=0):
    total_time = time.time()
    a = time.time()
    H = trimesh.convex.convex_hull(submesh)
    H_intersector = RayMeshIntersector(H)
    print("convex_hull time: ", time.time() - a)

    a = time.time()
    A_SinH, A_HinS = _estimate_boundary_areas_by_retention(full_mesh, intersectors, H, H_intersector, 10000,seed=seed+11)
    counts = _alloc_counts_by_weight([A_SinH, A_HinS], n_samples)
    nS, nH = int(counts[0]), int(counts[1])
    print("alloc_counts_by_weight time: ", time.time() - a, A_SinH, A_HinS, nS, nH)

    ####sample on surface points
    a = time.time()
    # SA1,_ = trimesh.sample.sample_surface(submesh, nS, seed=seed+13)
    # SA1 = sample_surface_intersection_convex(full_mesh, H, nS, seed=seed+13)
    SA1 = sample_surface_intersection(full_mesh, H, H_intersector, nS, seed=seed+13)
    SA1 = np.array(SA1)
    # trimesh.Trimesh(vertices = SA1).export(f"SA1_{num}.ply")
    print("SA1 time: ", time.time() - a)
    
    a = time.time()
    SA2 = sample_surface_intersection(H, full_mesh, intersectors, nH, seed=seed+17)
    # trimesh.Trimesh(vertices = SA2).export(f"SA2_{num}.ply")
    print("SA2 time: ", time.time() - a)


    a = time.time()
    SA  = np.vstack([x for x in (SA1, SA2) if x.size > 0]) if (SA1.size+SA2.size)>0 else np.zeros((0,3))
    # trimesh.Trimesh(vertices = SA).export(f"SA_{num}.ply")
    # H.export(f"H_{num}.ply")
    # full_mesh.export(f"full_mesh.ply")
    # Ps_hull_ref,_ = trimesh.sample.sample_surface(H, n_samples, seed=seed+19)
    # H = trimesh.convex.convex_hull(trimesh.Trimesh(vertices = SA), repair=True)
    # H_intersector = RayMeshIntersector(H)

    Ps_hull_ref,_ = trimesh.sample.sample_surface(H, n_samples, seed=seed+19)
    # trimesh.Trimesh(vertices = Ps_hull_ref).export(f"Ps_hull_ref_{num}.ply")
    print("preprare_time: ", time.time() - a)
    
    a = time.time()
    if SA.shape[0] == 0 or Ps_hull_ref.shape[0] == 0:
        Hb = 0.0
        print("hausdorff_two_way_points skipped (empty samples)")
    else:
        Hb = hausdorff_two_way_points(SA, Ps_hull_ref)
        Hb = np.array(Hb)
    print("hausdorff_two_way_points time: ", time.time() - a)
    
    a = time.time()
    kRv = concavity_volbound(H, H_intersector, interior_points, full_mesh_volumn, k=k)
    print("concavity_volbound time: ", time.time() - a)
    
    print("total time: ", time.time() - total_time, "Hb:", Hb, "kRv:", kRv)
    print(nS, nH)
    # exit(0)
    return max(Hb, kRv)


def _safe_stem(s):
    base = os.path.basename(s.rstrip(os.sep))
    if base == "":
        base = "mesh"
    return base.replace(" ", "_")

def _fmt4_strip(v):
    # match examples like: 0.0501, 0.029, 0.0696
    x = v.item() if hasattr(v, "item") else v
    s = f"{float(x):.4f}"
    s = s.rstrip("0").rstrip(".")
    return s if s != "" else "0"

def _ours_stem_from_mesh_path(mesh_path):
    # e.g. /.../homer_mc.ply -> homer_mc_OURS_NEW_metrics
    base = os.path.basename(mesh_path)
    if "." in base:
        base = os.path.splitext(base)[0]
    return base + "_OURS_NEW_metrics"

def save_metrics_txt_vhco_ca(out_txt_path, concavity_vals):
    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)

    max_ca = float(np.max(concavity_vals)) if len(concavity_vals) else 0.0

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("# Per-hull concavity metrics (VH/CO/CA):\n")
        for i, c in enumerate(concavity_vals):
            # VH/CO are 0 in your OURS file; CA stores concavity
            f.write(f"hull[{i:03d}]  VH/CO/CA = 0/0/{_fmt4_strip(c)}\n")

        f.write("\n# Max concavity across hulls:\n")
        f.write("max_VHACD             = 0\n")
        f.write("max_COACD             = 0\n")
        f.write(f"max_COACD_APPROXIMATE = {_fmt4_strip(max_ca)}\n")

        f.write("\n# Reconstruction metrics (union vs original):\n")
        f.write("RECON_SURF = 0\n")
        f.write("RECON_VOL  = 0\n")
        f.write("RECON_BOTH = 0\n")

    print(f"[saved] {out_txt_path}")

def load_hulls(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    if not isinstance(m, trimesh.Trimesh):
        return []
    parts = m.split(only_watertight=False)
    hulls = [p for p in parts if p is not None and p.faces is not None and p.faces.shape[0] > 0]
    hulls.sort(key=lambda h: float(np.mean(h.vertices[:, 0])) if h.vertices.shape[0] else 0.0)
    return hulls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to the full original mesh (ply/obj)")
    ap.add_argument("--hulls", required=True, help="Path to the hulls file (multi-part ply/obj)")
    ap.add_argument("--out_dir", default="./metric_new", help="Where to write the metrics txt")
    ap.add_argument("--out_name", default="", help="Optional custom output stem name (without .txt)")
    ap.add_argument("--samples", type=int, default=20000, help="n_samples passed into concavity()")
    ap.add_argument("--interior_samples", type=int, default=10000, help="Interior points sampled once on full mesh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=float, default=0.3, help="k passed into concavity()")
    ap.add_argument("--simplify_full_faces", type=int, default=0,
                    help="(unused here) kept for CLI compatibility")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full = trimesh.load(args.mesh, process=False)
    if isinstance(full, trimesh.Scene):
        full = full.dump(concatenate=True)
    if (not isinstance(full, trimesh.Trimesh)) or full.is_empty or full.faces.shape[0] == 0:
        raise RuntimeError(f"Failed to load a valid mesh from: {args.mesh}")

    hulls = load_hulls(args.hulls)
    if len(hulls) == 0:
        raise RuntimeError(f"No hull components found in: {args.hulls}")

    full_mesh_volumn = float(full.volume)
    intersector = RayMeshIntersector(full)
    interior_points = sample_interior_points(full, intersector, int(args.interior_samples), int(args.seed))

    concavity_vals = []
    t0 = time.time()
    for i, h in enumerate(hulls):
        if h.is_empty or h.faces is None or h.faces.shape[0] == 0 or h.vertices.shape[0] < 4:
            concavity_vals.append(0.0)
            continue

        c = concavity2(
            submesh=h,
            full_mesh=full,
            interior_points=interior_points,
            intersectors=intersector,
            full_mesh_volumn=full_mesh_volumn,
            n_samples=int(args.samples),
            seed=int(args.seed),
            k=float(args.k),
            num=i,
        )
        concavity_vals.append(float(c))
        print(f"[{i:03d}/{len(hulls):03d}] concavity={float(c):.6g}")

    print(f"[done] hulls={len(hulls)} total_time={time.time() - t0:.3f}s")

    stem = args.out_name.strip() if args.out_name.strip() else _ours_stem_from_mesh_path(args.mesh)
    out_txt = os.path.join(args.out_dir, f"{stem}.txt")
    save_metrics_txt_vhco_ca(out_txt, concavity_vals)

if __name__ == "__main__":
    main()
    

###python evaluation.py --mesh {mesh_path}/{name}.ply --hulls {base_path}/{dataset}/{name}/{name}_hull.ply --out_dir {base_path}/{dataset}/metric_new
