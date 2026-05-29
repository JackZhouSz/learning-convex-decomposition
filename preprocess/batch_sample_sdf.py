# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from tqdm import tqdm
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import random
import argparse

def process(filename):
    name = filename.split(".")[0]
    if os.path.exists(os.path.join(args.output_dir, f"{name}.h5")):
        print(f"already processed {name}")
        return
    os.system(f"python3 sample_sdf_single.py --name {filename} --input_dir {args.input_dir} --output_dir {args.output_dir}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="")
    ap.add_argument("--output_dir", default="")
    args = ap.parse_args()
    name_list = os.listdir(args.input_dir)
    random.shuffle(name_list)

    process_map(process, name_list, max_workers=20)

