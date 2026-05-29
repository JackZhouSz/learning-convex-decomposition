# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import trimesh
import numpy as np
import torch
from skimage import measure
from scipy.spatial import cKDTree
import h5py
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh.proximity import ProximityQuery
from scipy.sparse import coo_matrix, csr_matrix
import numba as nb
import random


def refine_large_faces(mesh, ratio):
    import numpy as np
    import trimesh

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    def face_areas(V, F):
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    max_area = face_areas(V, F).sum() / ratio

    while True:
        A = face_areas(V, F)
        bad = A > max_area
        if not np.any(bad):
            break

        Fb = F[bad]
        e01 = np.linalg.norm(V[Fb[:, 0]] - V[Fb[:, 1]], axis=1)
        e12 = np.linalg.norm(V[Fb[:, 1]] - V[Fb[:, 2]], axis=1)
        e20 = np.linalg.norm(V[Fb[:, 2]] - V[Fb[:, 0]], axis=1)
        choice = np.argmax(np.stack([e01, e12, e20], axis=1), axis=1)

        # edges to split (undirected for caching)
        E = np.empty((Fb.shape[0], 2), dtype=np.int64)
        i0, i1, i2 = Fb[:, 0], Fb[:, 1], Fb[:, 2]
        E[choice == 0] = np.stack([i0[choice == 0], i1[choice == 0]], axis=1)
        E[choice == 1] = np.stack([i1[choice == 1], i2[choice == 1]], axis=1)
        E[choice == 2] = np.stack([i2[choice == 2], i0[choice == 2]], axis=1)
        E = np.sort(E, axis=1)
        E = np.unique(E, axis=0)

        base = len(V)
        V = np.vstack([V, 0.5 * (V[E[:, 0]] + V[E[:, 1]])])
        mid_index = {(int(a), int(b)): base + t for t, (a, b) in enumerate(E)}

        def mid(a, b):
            if a < b:
                return mid_index.get((int(a), int(b)), -1)
            return mid_index.get((int(b), int(a)), -1)

        newF = []
        for (i, j, k) in F:
            m01 = mid(i, j)  # midpoint on oriented edge i->j
            m12 = mid(j, k)  # j->k
            m20 = mid(k, i)  # k->i

            s01 = m01 != -1
            s12 = m12 != -1
            s20 = m20 != -1
            s = int(s01) + int(s12) + int(s20)

            if s == 0:
                newF.append([i, j, k])

            elif s == 1:
                if s01:
                    newF += [[i, m01, k], [m01, j, k]]
                elif s12:
                    newF += [[j, m12, i], [m12, k, i]]
                else:  # s20
                    newF += [[k, m20, j], [m20, i, j]]

            elif s == 2:
                # Always preserve (i,j,k) orientation by using the vertex where both split edges meet
                if s01 and s12:        # split edges (i,j) and (j,k) meet at j
                    newF += [[j, m12, m01], [i, m01, k], [m01, m12, k]]
                elif s12 and s20:      # meet at k
                    newF += [[k, m20, m12], [j, m12, i], [m12, m20, i]]
                else:                  # meet at i (s20 and s01)
                    newF += [[i, m01, m20], [k, m20, j], [m20, m01, j]]

            else:  # s == 3
                newF += [[i, m01, m20], [m01, j, m12], [m20, m12, k], [m01, m12, m20]]

        F = np.asarray(newF, dtype=np.int64)

    return trimesh.Trimesh(V, F, process=False)



# ---------------------------------------------------------------------
# Patching functions for grouping faces into patches
# ---------------------------------------------------------------------
def _face_adjacency_naive_fast_local(faces_local, n_vertices_local):
    F = faces_local.shape[0]
    if F == 0:
        return csr_matrix((0, 0))
    face_idx = np.repeat(np.arange(F), 3)
    vert_idx = faces_local.reshape(-1)
    data = np.ones_like(face_idx, dtype=np.int8)
    FV = coo_matrix((data, (face_idx, vert_idx)), shape=(F, n_vertices_local)).tocsr()
    FF = (FV @ FV.T).tocsr()
    FF.setdiag(0); FF.eliminate_zeros()
    mask = FF.data >= 2
    FF.data[:] = 0
    FF.data[mask] = 1
    return FF.tocsr()


@nb.njit(cache=False)
def _grow_area_cap_numba(indptr, indices,
                         labels, face_area, region_area,
                         cap_area,
                         q_face, q_rid,
                         head, tail):
    nF = labels.shape[0]
    assigned = 0
    for i in range(nF):
        if labels[i] != -1:
            assigned += 1

    while head < tail and assigned < nF:
        u = q_face[head]
        rid = q_rid[head]
        head += 1

        if region_area[rid] >= cap_area:
            continue

        a = indptr[u]
        b = indptr[u + 1]
        for t in range(a, b):
            v = indices[t]
            if labels[v] != -1:
                continue
            av = face_area[v]
            if region_area[rid] + av > cap_area:
                continue

            labels[v] = rid
            region_area[rid] += av
            assigned += 1

            q_face[tail] = v
            q_rid[tail] = rid
            tail += 1

            if region_area[rid] >= cap_area:
                break

    return assigned


@nb.njit(cache=False)
def _fill_area_cap_numba(indptr, indices,
                         labels, face_area, region_area,
                         cap_area):
    nF = labels.shape[0]

    for u in range(nF):
        if labels[u] != -1:
            continue

        au = face_area[u]
        best_rid = -1
        best_rem = -1.0

        a = indptr[u]
        b = indptr[u + 1]

        # prefer patch with remaining area
        for t in range(a, b):
            v = indices[t]
            rid = labels[v]
            if rid < 0:
                continue
            rem = cap_area - region_area[rid]
            if rem >= au and rem > best_rem:
                best_rem = rem
                best_rid = rid

        # fallback: attach to any neighbor patch
        if best_rid < 0:
            for t in range(a, b):
                v = indices[t]
                rid = labels[v]
                if rid >= 0:
                    best_rid = rid
                    break

        if best_rid >= 0:
            labels[u] = best_rid
            region_area[best_rid] += au


def build_global_patches_numba(verts_all, faces_all, K=20000, seed=42):


    V = np.asarray(verts_all)
    F = np.asarray(faces_all, dtype=np.int64)
    nF = int(F.shape[0])

    if nF == 0:
        return [], -np.ones((0,), dtype=np.int32), csr_matrix((0, 0))

    FF = _face_adjacency_naive_fast_local(F, n_vertices_local=V.shape[0])

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_area = (
        0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    ).astype(np.float64)

    K = int(min(int(K), nF))
    if K < 2:
        face_to_patch = np.zeros((nF,), dtype=np.int32)
        patches = [np.arange(nF, dtype=np.int64)]
        PP_global = csr_matrix((1, 1))
        return patches, face_to_patch, PP_global

    rng = np.random.default_rng(int(seed))
    seeds = rng.choice(nF, size=K, replace=False).astype(np.int64)

    labels = -np.ones(nF, dtype=np.int32)
    labels[seeds] = np.arange(K, dtype=np.int32)

    total_area = float(face_area.sum())
    cap_area = total_area / float(K)

    region_area = np.zeros(K, dtype=np.float64)
    region_area[:] = face_area[seeds]



    # ---- NUMBA GROW ----


    indptr = np.ascontiguousarray(FF.indptr.astype(np.int64))
    indices = np.ascontiguousarray(FF.indices.astype(np.int64))

    q_face = np.empty(nF, dtype=np.int64)
    q_rid = np.empty(nF, dtype=np.int32)
    q_face[:K] = seeds
    q_rid[:K] = np.arange(K, dtype=np.int32)

    _grow_area_cap_numba(
        indptr, indices,
        labels, face_area, region_area,
        cap_area,
        q_face, q_rid,
        0, K
    )



    # ---- NUMBA FILL ----
    _fill_area_cap_numba(
        indptr, indices,
        labels, face_area, region_area,
        cap_area
    )


    # ---- PATCH BUILD ----


    order = np.argsort(labels, kind="mergesort")
    labs_sorted = labels[order]
    ok = labs_sorted >= 0
    order = order[ok]
    labs_sorted = labs_sorted[ok]

    patches = []
    if labs_sorted.size > 0:
        cuts = np.nonzero(labs_sorted[1:] != labs_sorted[:-1])[0] + 1
        for s in np.split(order, cuts):
            if s.size > 0:
                patches.append(s.astype(np.int64))

    P = len(patches)
    face_to_patch = -np.ones(nF, dtype=np.int32)
    for p in range(P):
        face_to_patch[patches[p]] = p

    if P <= 0:
        PP_global = csr_matrix((0, 0))
    else:
        rows, cols = FF.nonzero()
        pa = face_to_patch[rows]
        pb = face_to_patch[cols]
        m = (pa >= 0) & (pb >= 0) & (pa != pb)
        rr = np.concatenate([pa[m], pb[m]])
        cc = np.concatenate([pb[m], pa[m]])
        PP_global = coo_matrix(
            (np.ones(rr.size, dtype=np.int8), (rr, cc)),
            shape=(P, P)
        ).tocsr()


    return patches, face_to_patch, PP_global


def quad_to_triangle_mesh(F):
    """
    Converts a quad-dominant mesh into a pure triangle mesh by splitting quads into two triangles.

    Parameters:
        quad_mesh (trimesh.Trimesh): Input mesh with quad faces.

    Returns:
        trimesh.Trimesh: A new mesh with only triangle faces.
    """
    faces = F

    ### If already a triangle mesh -- skip
    if len(faces[0]) == 3:
        return F

    new_faces = []

    for face in faces:
        if len(face) == 4:  # Quad face
            # Split into two triangles
            new_faces.append([face[0], face[1], face[2]])  # Triangle 1
            new_faces.append([face[0], face[2], face[3]])  # Triangle 2
        else:
            print(f"Warning: Skipping non-triangle/non-quad face {face}")

    new_faces = np.array(new_faces)

    return new_faces

def sample_interior_points_grid(sdf_grid: torch.Tensor, n: int) -> torch.Tensor:
    """
    Sample `n` points uniformly from the interior of the shape defined by `sdf_grid` (<0),
    with voxel-centred jitter, normalized to [-1, 1]^3.

    Parameters
    ----------
    sdf_grid : torch.Tensor, shape (Nx, Ny, Nz)
        Signed-distance field where <0 indicates inside.
    n : int
        Number of points to sample.

    Returns
    -------
    pts : torch.Tensor, shape (n, 3), dtype torch.float32
        Sampled points in [-1, 1]^3.
    """
    # find all interior voxel linear indices
    mask = sdf_grid < 0
    interior_idx = mask.view(-1).nonzero(as_tuple=False).squeeze(1)
    if interior_idx.numel() == 0:
        raise ValueError("No interior voxels (sdf_grid<0) to sample from.")

    # sample exactly n of them (with replacement)
    idx = torch.randint(0, interior_idx.numel(), (n,), device=sdf_grid.device)
    choose = interior_idx[idx]

    # convert flat indices → (ix, iy, iz)
    Nx, Ny, Nz = sdf_grid.shape
    ix = choose // (Ny * Nz)
    iy = (choose // Nz) % Ny
    iz = choose % Nz

    # add uniform jitter in [-0.5, +0.5] around voxel centres
    jitter = torch.rand(n, 3, device=sdf_grid.device, dtype=torch.float32) - 0.5
    pts = torch.stack([ix, iy, iz], dim=1).to(torch.float32) + 0.5 + jitter

    # normalize from voxel coords [0..Nx-1] → [-1..1]
    scale = torch.tensor([Nx - 1, Ny - 1, Nz - 1], device=sdf_grid.device, dtype=torch.float32)
    pts = pts / scale * 2.0 - 1.0

    return pts




# 30 pre-picked distinct-ish RGBs (uint8). Use whatever you like here.
_PALETTE = np.array([
    [242,  72,  72],  # 0
    [ 57,  96, 191],  # 1
    [171, 242,  72],  # 2
    [191,  57, 174],  # 3
    [ 72, 242, 213],  # 4
    [191, 129,  57],  # 5
    [114,  72, 242],  # 6
    [ 63, 191,  57],  # 7
    [242,  72, 129],  # 8
    [ 57, 141, 191],  # 9
    [228, 242,  72],  # 10
    [163,  57, 191],  # 11
    [ 72, 242, 157],  # 12
    [191,  85,  57],  # 13
    [ 72,  87, 242],  # 14
    [107, 191,  57],  # 15
    [242,  72, 186],  # 16
    [ 57, 185, 191],  # 17
    [242, 199,  72],  # 18
    [118,  57, 191],  # 19
    [ 72, 242, 100],  # 20
    [191,  57,  74],  # 21
    [ 72, 143, 242],  # 22
    [152, 191,  57],  # 23
    [241,  72, 242],  # 24
    [ 57, 191,  30],  # 25
    [242, 119,  72],  # 26
    [ 57,  57, 191],  # 27
    [200, 242,  72],  # 28
    [191,  57, 157],  # 29
    [ 72, 242, 199],  # 30
    [191, 101,  57],  # 31
    [128,  72, 242],  # 32
    [ 89, 191,  57],  # 33
    [242,  72, 113],  # 34
    [ 57, 128, 191],  # 35
    [242, 242,  72],  # 36
    [145,  57, 191],  # 37
    [ 72, 242, 142],  # 38
    [191,  72,  57],  # 39
    [ 72, 103, 242],  # 40
    [119, 191,  57],  # 41
    [242,  72, 170],  # 42
    [ 57, 170, 191],  # 43
    [242, 213,  72],  # 44
    [130,  57, 191],  # 45
    [ 72, 242,  85],  # 46
    [191,  57,  88],  # 47
    [ 72, 157, 242],  # 48
    [136, 191,  57],  # 49
    [229,  72, 242],  # 50
    [ 57, 191,  44],  # 51
    [242,  86,  72],  # 52
    [ 57, 111, 191],  # 53
    [214, 242,  72],  # 54
    [176,  57, 191],  # 55
    [ 72, 242, 228],  # 56
    [191, 115,  57],  # 57
    [101,  72, 242],  # 58
    [ 76, 191,  57],  # 59
    [242,  72, 142],  # 60
    [ 57, 155, 191],  # 61
    [242, 228,  72],  # 62
    [160,  57, 191],  # 63
    [ 72, 242, 171],  # 64
    [191,  72,  57],  # 65
    [ 72,  90, 242],  # 66
    [113, 191,  57],  # 67
    [242,  72, 199],  # 68
    [ 57, 198, 191],  # 69
    [242, 186,  72],  # 70
    [119,  57, 191],  # 71
    [ 72, 242, 114],  # 72
    [191,  57,  72],  # 73
    [ 72, 145, 242],  # 74
    [149, 191,  57],  # 75
    [242,  72, 228],  # 76
    [ 57, 191,  20],  # 77
    [242, 100,  72],  # 78
    [ 57,  83, 191],  # 79
    [200, 242,  72],  # 80
    [174,  57, 191],  # 81
    [ 72, 242, 142],  # 82
    [191,  88,  57],  # 83
    [ 72, 103, 242],  # 84
    [125, 191,  57],  # 85
    [242,  72, 170],  # 86
    [ 57, 168, 191],  # 87
    [242, 206,  72],  # 88
    [135,  57, 191],  # 89
    [ 72, 242,  98],  # 90
    [191,  57,  90],  # 91
    [ 72, 157, 242],  # 92
    [141, 191,  57],  # 93
    [229,  72, 242],  # 94
    [ 57, 191,  30],  # 95
    [242,  86,  72],  # 96
    [ 57, 111, 191],  # 97
    [214, 242,  72],  # 98
    [163,  57, 191],  # 99
    [ 72, 242, 228],  # 100
    [191, 101,  57],  # 101
    [103,  72, 242],  # 102
    [ 90, 191,  57],  # 103
    [242,  72, 145],  # 104
    [ 57, 142, 191],  # 105
    [242, 230,  72],  # 106
    [155,  57, 191],  # 107
    [ 72, 242, 197],  # 108
    [191, 116,  57],  # 109
    [ 98,  72, 242],  # 110
    [ 76, 191,  57],  # 111
    [242,  72, 145],  # 112
    [ 57, 154, 191],  # 113
    [188, 242,  72],  # 114
    [191,  57, 187],  # 115
    [ 72, 242, 197],  # 116
    [191, 116,  57],  # 117
    [ 98,  72, 242],  # 118
    [ 76, 191,  57],  # 119
], dtype=np.uint8).tolist()





