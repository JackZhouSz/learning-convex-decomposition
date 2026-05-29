# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import os, math, tempfile, argparse
import numpy as np
import trimesh
import skimage.measure
import vtk
import mesh2sdf
import tetgen


def load_mesh(path):
    m = trimesh.load(path, force='mesh', process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return m


def normalize_mesh_to_unit_box(mesh):
    # normalize to [-1, 1]^3 using bbox
    bmin, bmax = mesh.bounds
    center = 0.5 * (bmin + bmax)
    scale = np.max(bmax - bmin)
    mesh.vertices = (mesh.vertices - center) * (2.0 / scale)
    return mesh


def water_tight_remesh(mesh):
    size = 256
    level = 2.0 / size

    sdf = mesh2sdf.core.compute(mesh.vertices, mesh.faces, size)
    udf = np.abs(sdf)
    v, f, _, _ = skimage.measure.marching_cubes(udf, level)

    parts = trimesh.Trimesh(v, f, process=False).split(only_watertight=False)
    for p in parts:
        p.fix_normals()
    surf = trimesh.util.concatenate(parts)

    tet = tetgen.TetGen(surf.vertices, surf.faces)
    tet.tetrahedralize(
        plc=True, nobisect=1.0, quality=True, fixedvolume=True,
        maxvolume=math.sqrt(2)/12*(2/size)**3
    )

    tmp_vtk = tempfile.NamedTemporaryFile(suffix=".vtk", delete=True)
    tet.grid.save(tmp_vtk.name)

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(tmp_vtk.name)
    reader.Update()

    surf_f = vtk.vtkDataSetSurfaceFilter()
    surf_f.SetInputConnection(reader.GetOutputPort())
    surf_f.Update()

    tmp_obj = tempfile.NamedTemporaryFile(suffix=".obj", delete=True)
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(tmp_obj.name)
    writer.SetInputData(surf_f.GetOutput())
    writer.Update()

    m = load_mesh(tmp_obj.name)

    # keep output in [-1, 1]
    m.vertices = m.vertices * (2.0 / size) - 1.0
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exts = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}
    files = [f for f in os.listdir(args.input_dir)
             if os.path.splitext(f)[1].lower() in exts]

    for f in sorted(files):
        in_path = os.path.join(args.input_dir, f)
        out_path = os.path.join(
            args.output_dir, os.path.splitext(f)[0] + ".ply"
        )

        print("Processing:", f)
        mesh = load_mesh(in_path)
        mesh = normalize_mesh_to_unit_box(mesh)   # ← NEW
        mesh = water_tight_remesh(mesh)
        mesh.export(out_path)


if __name__ == "__main__":
    main()