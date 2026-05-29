# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

dir_names = os.listdir(".")
for dir_name in dir_names:
    if os.path.isdir(dir_name):
        file_name = os.listdir(os.path.join(".", dir_name))[0]
        new_name = dir_name+"_hull.ply"
        os.rename(os.path.join(".", dir_name, file_name), os.path.join(".", dir_name, new_name))
        # os.rename(os.path.join(".", dir_name), os.path.join(".", new_name))