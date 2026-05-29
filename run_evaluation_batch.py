# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import random
from multiprocessing import Pool



###run original old evaluation script
# datasets = [
#     "benchmark_0.1"# "VHACD_data",
# ]
# def run_eval(task):
#     dataset, name = task
#     path = f"/lustre/fs12/portfolios/nvr/users/yuezhiy/release_repo/exp_results/partfield_convex_reg_mc_regonly_v2/{dataset}"
#     # if os.path.exists(f"{path}/metric/{name}_OURS_NEW_metrics.txt"):
#     #     print(f"Skip {name}")
#     #     return
#     print(f"Run {name}")
#     os.chdir("/lustre/fs12/portfolios/nvr/users/yuezhiy/partfield_simp/convex_exp")
#     os.system(f"python evaluate_metric.py --dataset VHACD_data --convex_path ../../release_repo/exp_results/partfield_convex_reg_mc_regonly_v2/{dataset} --mesh {name} --flag OURS_NEW")

# tasks = []
# for d in datasets:
#     names = os.listdir(f"/lustre/fs12/portfolios/nvr/users/yuezhiy/release_repo/exp_results/partfield_convex_reg_mc_regonly_v2/{d}")
#     random.shuffle(names)
#     tasks += [(d, n) for n in names if "Teapot" in n]


# tasks = tasks[:1]
# run_eval(tasks[0])
# with Pool(8) as p:  # change 8 for number of processes
#     p.map(run_eval, tasks)





####run the current evaluation script
DEFAULT_DATASETS = [
    "benchmark_195_eps_0.05",
    "benchmark_195_eps_0.1",
]

script_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset folders under --base_path",
    )
    ap.add_argument(
        "--mesh_path",
        default=os.path.join(script_dir, "data", "mc_data"),
        help="Directory containing GT meshes (*.ply)",
    )
    ap.add_argument(
        "--base_path",
        default=os.path.join(script_dir, "exp_results", "example"),
        help="Root directory containing per-dataset result folders",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel processes",
    )
    return ap.parse_args()


def run_eval(task):
    dataset, name = task
    os.chdir(script_dir)
    os.system(
        f"python evaluation.py "
        f"--mesh {run_eval.mesh_path}/{name}.ply "
        f"--hulls {run_eval.base_path}/{dataset}/{name}/{name}_hull.ply "
        f"--out_dir {run_eval.base_path}/{dataset}/metric_new"
    )


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise ValueError("No datasets provided after parsing --datasets")

    mesh_path = os.path.abspath(args.mesh_path)
    base_path = os.path.abspath(args.base_path)
    if not os.path.isdir(mesh_path):
        raise FileNotFoundError(f"mesh_path not found: {mesh_path}")
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"base_path not found: {base_path}")

    tasks = []
    gt_names = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(mesh_path)
        if fn.endswith(".ply")
    }

    for d in datasets:
        dataset_dir = os.path.join(base_path, d)
        if not os.path.isdir(dataset_dir):
            print(f"[skip] dataset dir not found: {dataset_dir}")
            continue
        names = [
            n for n in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, n)) and n in gt_names
        ]
        random.shuffle(names)
        tasks += [(d, n) for n in names]

    print(f"Total tasks: {len(tasks)}")
    if len(tasks) == 0:
        return

    run_eval.mesh_path = mesh_path
    run_eval.base_path = base_path
    with Pool(args.workers) as p:
        p.map(run_eval, tasks)


if __name__ == "__main__":
    main()
