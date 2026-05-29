# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import argparse
import math
import os
import random
import tempfile

import mesh2sdf
import numpy as np
import pymeshlab as ml
import skimage
import skimage.measure as skm
import tetgen
import trimesh
import vtk

def simplify_trimesh_preserve_topology(mesh, target_faces):
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces), "mesh")
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=True,
        preserveboundary=True,
        optimalplacement=True,
        planarquadric=True,
    )
    m = ms.current_mesh()
    V = m.vertex_matrix()
    F = m.face_matrix()
    return trimesh.Trimesh(vertices=V, faces=F, process=False)

def normalize_mesh(mesh, mesh_scale=0.95):
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    mesh.vertices = (vertices - center) * scale
    return mesh


def process_one(input_path, output_mc_dir, output_qem_dir, size=256, target_faces=18000):
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_mc = os.path.join(output_mc_dir, f"{name}_mc.ply")
    out_qem = os.path.join(output_qem_dir, f"{name}_mc.ply")
    if os.path.exists(out_mc) and os.path.exists(out_qem):
        print(f"skip {name}: outputs exist")
        return

    print(f"processing {name}")
    mesh_scale = 0.95
    level = 2 / size

    mesh = trimesh.load(input_path, force="mesh")
    mesh = normalize_mesh(mesh, mesh_scale=mesh_scale)
    vertices = mesh.vertices
    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)

    udf = np.abs(sdf)
    vertices, faces, _, _ = skimage.measure.marching_cubes(udf, level)
    components = trimesh.Trimesh(vertices, faces).split(only_watertight=False)
    new_mesh = []
    if len(components) > 100000:
        raise NotImplementedError
    for c in components:
        c.fix_normals()
        new_mesh.append(c)
    new_mesh = trimesh.util.concatenate(new_mesh)

    tet = tetgen.TetGen(new_mesh.vertices, new_mesh.faces)
    tet.tetrahedralize(
        plc=True,
        nobisect=1,
        quality=True,
        fixedvolume=True,
        maxvolume=math.sqrt(2) / 12 * (2 / size) ** 3,
    )
    tmp_vtk = tempfile.NamedTemporaryFile(suffix='.vtk', delete=True)
    tet.grid.save(tmp_vtk.name)

    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(tmp_vtk.name)
    reader.Update()
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())
    surface_filter.Update()
    polydata = surface_filter.GetOutput()
    writer = vtk.vtkOBJWriter()
    tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=True)
    writer.SetFileName(tmp_obj.name)
    writer.SetInputData(polydata)
    writer.Update()

    # Use the watertight tet-reconstructed surface directly as mc mesh.
    mc_mesh = trimesh.load(tmp_obj.name, force="mesh")
    mc_mesh = trimesh.Trimesh(mc_mesh.vertices, mc_mesh.faces, process=False)
    mc_mesh.export(out_mc)

    qem_mesh = simplify_trimesh_preserve_topology(mc_mesh, target_faces)
    qem_mesh.export(out_qem)
    print(f"done {name}: {out_mc} | {out_qem}")


def list_inputs(input_dir):
    valid_ext = {".ply", ".obj", ".stl", ".glb", ".off"}
    names = []
    for fn in os.listdir(input_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext in valid_ext:
            names.append(fn)
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_mc_dir", required=True)
    ap.add_argument("--output_qem_dir", required=True)
    ap.add_argument("--name", default="", help="Optional single filename in input_dir")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--target_faces", type=int, default=18000)
    args = ap.parse_args()

    os.makedirs(args.output_mc_dir, exist_ok=True)
    os.makedirs(args.output_qem_dir, exist_ok=True)

    if args.name:
        names = [args.name]
    else:
        names = list_inputs(args.input_dir)
        random.shuffle(names)
    print("num inputs:", len(names))

    for name in names:
        process_one(
            input_path=os.path.join(args.input_dir, name),
            output_mc_dir=args.output_mc_dir,
            output_qem_dir=args.output_qem_dir,
            size=args.size,
            target_faces=args.target_faces,
        )


if __name__ == "__main__":
    main()
