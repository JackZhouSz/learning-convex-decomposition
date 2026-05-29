# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import argparse, os, time, numpy as np, trimesh, glob
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import coo_matrix, csr_matrix
import h5py
import heapq
import time
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.spatial import cKDTree
from collections import deque
import torch
from model.dataloader import build_global_patches_numba
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx = self.find(x); ry = self.find(y)
        if rx != ry:
            if self.rank[rx] > self.rank[ry]:
                self.parent[ry] = rx
            elif self.rank[rx] < self.rank[ry]:
                self.parent[rx] = ry
            else:
                self.parent[ry] = rx
                self.rank[rx] += 1

def remap_hierarchical(hierarchical_labels):
    mapping = {}; next_id = 0
    for arr in reversed(hierarchical_labels):
        uniq = np.unique(arr)
        for u in uniq:
            if u not in mapping:
                mapping[u] = next_id; next_id += 1
    remapped = []
    for arr in hierarchical_labels:
        arr = np.asarray(arr)
        remapped.append(np.vectorize(mapping.get)(arr))
    return remapped


def hierarchical_clustering_labels(children, n_samples, max_cluster=20):
    uf = UnionFind(2 * n_samples - 1)
    current_cluster_count = n_samples
    hierarchical_labels = []
    for i, (a, b) in enumerate(children):
        uf.union(a, i + n_samples); uf.union(b, i + n_samples)
        current_cluster_count -= 1
        if current_cluster_count <= max_cluster:
            labels = [uf.find(i) for i in range(n_samples)]
            hierarchical_labels.append(labels)
    hierarchical_labels = remap_hierarchical(hierarchical_labels)
    return hierarchical_labels



def classify_inside7(intersector,  points, eps=1e-3, jitter=1e-5, seed=0):
    n = points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)

    P = np.asarray(points, dtype=np.float64)

    # 3 directions: +x, +y, +z
    D = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    Pj = P + jitter * rng.standard_normal(P.shape)

    # expand to (3N, 3)
    O3 = np.repeat(Pj, 3, axis=0)
    D3 = np.tile(D, (n, 1))

    # shift origin backward along each ray direction
    O3 = O3 - eps * D3

    locations, ray_ids, _ = intersector.intersects_location(
        ray_origins=O3,
        ray_directions=D3,
        multiple_hits=True
    )

    if ray_ids.size == 0:
        return np.zeros((n,), dtype=bool)

    t = np.einsum("ij,ij->i", locations - O3[ray_ids], D3[ray_ids])
    ray_v = ray_ids[t > eps]

    hit_count = np.bincount(ray_v, minlength=3 * n)
    inside_ray = (hit_count % 2 == 1).reshape(n, 3)

    # majority vote across 3 rays (at least 2/3)
    return inside_ray.sum(axis=1) >= 2


classify_inside = classify_inside7

def sample_surface_intersection(mesh_src, mesh_mask, intersector,n, seed=0, batch_mul=5):
    if n == 0:
        return np.zeros((0,3))
    rng = np.random.default_rng(seed)

    out = []
    remaining = n
    tries = 0
    while remaining > 0 and tries < 5:
        tries += 1
        batch = batch_mul * remaining

        pts, _ = trimesh.sample.sample_surface(mesh_src, batch)
        if pts.shape[0] == 0:
            break

        inside = classify_inside(intersector, pts)
        kept = pts[inside]

        if kept.shape[0] == 0:
            continue

        take = min(remaining, kept.shape[0])
        out.append(kept[:take])
        remaining -= take
    if len(out) == 0:
        return np.zeros((0, 3))
    return np.vstack(out)

def concavity_volbound(H, H_intersector,interior_points, full_mesh_volumn, k=0.3):
    N     = interior_points.shape[0]
    V_H   = H.volume
    inside_hull = classify_inside(H_intersector, interior_points)
    V_C   = full_mesh_volumn * (np.count_nonzero(inside_hull) / N)
    dV    = max(V_H - V_C, 0.0)
    return k * ((3.0 * dV) / (4.0 * np.pi)) ** (1.0 / 3.0)

def _estimate_boundary_areas_by_retention(full_mesh, full_intersector, hull, hull_intersector, n_probe,seed=42):
    Ps_full,_ = trimesh.sample.sample_surface(full_mesh, n_probe,seed=seed)
    Ps_hull,_ = trimesh.sample.sample_surface(hull,      n_probe,seed=seed)
    Ps_full = np.array(Ps_full)
    Ps_hull = np.array(Ps_hull)
    inside_hull = classify_inside(hull_intersector, Ps_full)
    inside_full = classify_inside(full_intersector, Ps_hull)
    cS = int(np.count_nonzero(inside_hull))
    cH = int(np.count_nonzero(inside_full))
    A_S_in_H = (cS / n_probe) * full_mesh.area
    A_H_in_S = (cH / n_probe) * hull.area
    return A_S_in_H, A_H_in_S

def concavity(submesh, full_mesh,  interior_points, intersectors, full_mesh_volumn, n_samples=1000, seed=42, k=0.3,num=0):
    H = trimesh.convex.convex_hull(submesh)
    H_intersector = RayMeshIntersector(H)

    A_SinH, A_HinS = _estimate_boundary_areas_by_retention(full_mesh, intersectors, H, H_intersector, 1000,seed=seed+11)
    counts = _alloc_counts_by_weight([A_SinH, A_HinS], n_samples)
    nS, nH = int(counts[0]), int(counts[1])

    # SA1,_ = trimesh.sample.sample_surface(submesh, nS, seed=seed+13)
    SA1 = sample_surface_intersection(full_mesh, H, H_intersector, nS, seed=seed+13)
    # SA1 = np.array(SA1)
    SA2 = sample_surface_intersection(H, full_mesh, intersectors, nH, seed=seed+17)
    SA  = np.vstack([x for x in (SA1, SA2) if x.size > 0]) if (SA1.size+SA2.size)>0 else np.zeros((0,3))

    Ps_hull_ref,_ = trimesh.sample.sample_surface(H, n_samples, seed=seed+19)

    Hb = hausdorff_two_way_points(SA, Ps_hull_ref)
    Hb = np.array(Hb)

    kRv = concavity_volbound(H, H_intersector, interior_points, full_mesh_volumn, k=k)

    return max(Hb, kRv)


# ----------------- face connectivity (on trimmed submesh) -----------------

def split_k2_features(face_indices, face_to_patch, PP_global_components, Xp_global):
    """
    Split faces into two clusters using agglomerative clustering.
    
    Args:
        face_indices: Face indices to split
        face_to_patch: Face to patch mapping
        PP_global_components: Dict with keys 'row', 'col', 'data', 'shape' for sparse matrix
        Xp_global: Patch features
    """
    I = np.asarray(face_indices, dtype=np.int64)

    pid_full = face_to_patch[I]
    ok = pid_full >= 0
    pid = pid_full[ok]

    # unique global patch ids involved in this node
    patch_ids = np.unique(pid)

    # local remap patch ids -> [0..Psub-1]
    max_pid = int(patch_ids.max())
    map_pid = -np.ones(max_pid + 1, dtype=np.int64)
    map_pid[patch_ids] = np.arange(patch_ids.size, dtype=np.int64)

    pid_local = map_pid[pid]

    Xp_sub = Xp_global[patch_ids]
    
    # Assemble sparse matrix from components
    shape_raw = PP_global_components["shape"]
    n = int(shape_raw.item()) if hasattr(shape_raw, "item") else int(shape_raw)
    PP_global = coo_matrix(
        (PP_global_components['data'], (PP_global_components['row'], PP_global_components['col'])),
        shape=(n, n)
    ).tocsr()
    PP_sub = PP_global[patch_ids][:, patch_ids]
    

    patch_labels = AgglomerativeClustering(
        n_clusters=2, linkage="ward", connectivity=PP_sub
    ).fit(Xp_sub).labels_.astype(np.int32)

    face_lr = -np.ones((I.size,), dtype=np.int32)
    face_lr[ok] = patch_labels[pid_local]

    IA = I[face_lr == 0]
    IB = I[face_lr == 1]
    return IA, IB

def submesh_from_faces_trimmed(verts_all, faces_all, face_idx):
    if face_idx.size == 0:
        return np.zeros((0, 3), dtype=verts_all.dtype), np.zeros((0, 3), dtype=np.int64)
    f_subset = faces_all[face_idx]
    flat = f_subset.reshape(-1)
    uniq, inv = np.unique(flat, return_inverse=True)
    subV = verts_all[uniq]
    subF = inv.reshape(-1, 3).astype(np.int64)
    return subV, subF


# ---------------- termination conditions ----------------


def sample_interior_points(mesh, intersector, n_samples, seed=0, batch_mul=5):

    rng = np.random.default_rng(seed)


    bmin, bmax = mesh.bounds
    extent = bmax - bmin

    out = []
    remaining = n_samples

    # fixed ray direction (+x)
    ray_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    while remaining > 0:
        batch = batch_mul * remaining

        # uniform proposal in AABB
        pts = rng.random((batch, 3)) * extent + bmin

        # ray origins and directions
        origins = pts
        directions = np.repeat(ray_dir[None, :], batch, axis=0)

        # ray casting (multiple hits!)
        locations, ray_ids, _ = intersector.intersects_location(
            origins, directions, multiple_hits=True
        )

        # parity test
        hit_count = np.bincount(ray_ids, minlength=batch)
        inside = (hit_count % 2) == 1

        inside_pts = pts[inside]
        if inside_pts.shape[0] == 0:
            continue

        take = min(remaining, inside_pts.shape[0])
        out.append(inside_pts[:take])
        remaining -= take

    return np.concatenate(out, axis=0)

def _alloc_counts_by_weight(weights, n_total):
    w = np.asarray(weights, dtype=np.float64)
    n_total = int(n_total)

    if n_total <= 0:
        return np.zeros_like(w, dtype=np.int64)

    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return np.zeros_like(w, dtype=np.int64)

    ideal = w / s * n_total
    out = np.floor(ideal).astype(np.int64)

    rem = n_total - out.sum()
    if rem > 0:
        frac = ideal - out
        idx = np.argpartition(-frac, rem - 1)[:rem]
        out[idx] += 1

    return out

def hausdorff_two_way_points(A, B, debug=False):
    treeA = cKDTree(A); treeB = cKDTree(B)
    dA, idxB = treeB.query(A, k=1, workers=-1)
    dB, idxA = treeA.query(B, k=1, workers=-1)
    max_dA = dA.max() if dA.size else 0.0
    max_dB = dB.max() if dB.size else 0.0
    if max_dA >= max_dB:
        i = int(dA.argmax()); pts = np.vstack([A[i], B[idxB[i]]])
        return (float(max_dA), pts) if debug else float(max_dA)
    else:
        j = int(dB.argmax()); pts = np.vstack([A[idxA[j]], B[j]])
        return (float(max_dB), pts) if debug else float(max_dB)



def build_patch_mean_features(face_to_patch, face_feats):
    pid = np.asarray(face_to_patch, dtype=np.int64)
    X = np.asarray(face_feats, dtype=np.float32)

    ok = pid >= 0
    pid = pid[ok]
    X = X[ok]
    if pid.size == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    P = int(pid.max() + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X).to(device=device)
    pidt = torch.from_numpy(pid).to(device=device)

    Xsum = torch.zeros((P, Xt.shape[1]), device=Xt.device, dtype=Xt.dtype)
    cnt = torch.zeros((P, 1), device=Xt.device, dtype=Xt.dtype)

    Xsum.index_add_(0, pidt, Xt)
    cnt.index_add_(0, pidt, torch.ones((pidt.numel(), 1), device=Xt.device, dtype=Xt.dtype))

    return (Xsum / torch.clamp(cnt, min=1)).cpu().numpy().astype(np.float32)


def convex_decomposition(
    mesh,
    face_feats,
    eps,
    max_parts,
    save_dir,
    interior_samples=10000,
    seed=42,
    n_samples=20000,
    face_to_patch=None,
    PP_global_components=None,
):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces, dtype=np.int64)
    F = int(faces.shape[0])

    os.makedirs(save_dir, exist_ok=True)

    full_mesh_volumn = mesh.volume
    intersector = RayMeshIntersector(mesh)
    interior_points = sample_interior_points(mesh, intersector, int(interior_samples), int(seed))

    Xp_global = build_patch_mean_features(face_to_patch, face_feats)

    nodes = []
    next_id = 0

    def new_node(parent, depth, idx_subset, concavity_val, is_leaf):
        nonlocal next_id, nodes
        nid = next_id
        next_id += 1
        nodes.append({
            "id": int(nid),
            "parent": parent,
            "depth": int(depth),
            "concavity": float(concavity_val),
            "is_leaf": bool(is_leaf),
            "faces": np.asarray(idx_subset, dtype=np.int64).tolist(),
        })
        return nid

    idx_all = np.arange(F, dtype=np.int64)
    subV0, subF0 = submesh_from_faces_trimmed(verts, faces, idx_all)
    submesh0 = trimesh.Trimesh(vertices=subV0, faces=subF0, process=False)
    c0 = concavity(
        submesh0,
        mesh,
        interior_points,
        intersector,
        full_mesh_volumn,
        n_samples=int(n_samples),
        seed=int(seed),
    )

    root_id = new_node(parent=None, depth=0, idx_subset=idx_all, concavity_val=c0, is_leaf=False)

    pq = []
    heapq.heappush(pq, (-float(c0), 0, int(root_id), idx_all, submesh0))
    parts = []

    def make_leaf(nid, I, cX, node_mesh):
        nodes[nid]["is_leaf"] = True
        face_idx = np.asarray(I, dtype=np.int64)
        subV, subF = submesh_from_faces_trimmed(verts, faces, face_idx)
        part_mesh = trimesh.Trimesh(vertices=subV, faces=subF, process=False)

        hull = None
        if part_mesh.vertices.shape[0] >= 4 and part_mesh.faces.shape[0] >= 1:
            h = part_mesh.convex_hull
            if (not h.is_empty) and (h.faces is not None) and (h.faces.shape[0] > 0):
                hull = h
        parts.append(
            {
                "id": int(nid),
                "concavity": float(cX),
                "faces": face_idx,
                "mesh": part_mesh,
                "hull": hull,
            }
        )

    while pq:
        neg_c, depth, nid, I, node_mesh = heapq.heappop(pq)
        cX = -float(neg_c)
        n_items = int(I.size)

        K_before = int(len(parts) + len(pq) + 1)
        stop_now = (n_items < 2) or (cX <= float(eps)) or (max_parts is not None and K_before >= int(max_parts))

        if stop_now:
            make_leaf(nid, I, cX, node_mesh)
            continue

        IA, IB = split_k2_features(I, face_to_patch, PP_global_components, Xp_global)
        if IA.size == 0 or IB.size == 0:
            make_leaf(nid, I, cX, node_mesh)
            continue

        subVA, subFA = submesh_from_faces_trimmed(verts, faces, IA)
        subVB, subFB = submesh_from_faces_trimmed(verts, faces, IB)
        subA = trimesh.Trimesh(vertices=subVA, faces=subFA, process=False)
        subB = trimesh.Trimesh(vertices=subVB, faces=subFB, process=False)
        cA = concavity(
            subA,
            mesh,
            interior_points,
            intersector,
            full_mesh_volumn,
            n_samples=int(n_samples),
            seed=int(seed),
        )
        cB = concavity(
            subB,
            mesh,
            interior_points,
            intersector,
            full_mesh_volumn,
            n_samples=int(n_samples),
            seed=int(seed),
        )

        idA = new_node(parent=int(nid), depth=int(depth) + 1, idx_subset=IA, concavity_val=cA, is_leaf=False)
        idB = new_node(parent=int(nid), depth=int(depth) + 1, idx_subset=IB, concavity_val=cB, is_leaf=False)

        heapq.heappush(pq, (-float(cA), int(depth) + 1, int(idA), IA, subA))
        heapq.heappush(pq, (-float(cB), int(depth) + 1, int(idB), IB, subB))

    parts = sorted(parts, key=lambda d: int(d["id"]))

    return parts
