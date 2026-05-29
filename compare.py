# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import re, argparse, os

def parse_table(txt_path):
    """
    Parses rows like:
      name |  # | 0.1234/0.0567
    Returns dict:
      key -> {"max": float, "mean": float, "h": int}
    """
    out = {}

    row_pat = re.compile(
        r"^(.*?)\s*\|\s*(\d+)\s*\|\s*([0-9eE+.\-]+)/([0-9eE+.\-]+)"
    )

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = row_pat.match(line.strip())
            if not m:
                continue
            name, h, vmax, vmean = m.groups()
            out[name.strip()] = {
                "h": int(h),
                "max": float(vmax),
                "mean": float(vmean),
            }
    return out


def normalize_name(name):
    """
    Make baseline / ours names match.
    Example:
      table_mc.ply__table_mc_hull.ply_per_hull_concavity
      table_mc_OURS_NEW_metrics
    -> table_mc
    """
    # strip OURS suffix
    name = name.replace("_OURS_NEW_metrics", "")
    # baseline pattern: xxx__xxx_hull...
    if "__" in name:
        name = name.split("__")[0]
    return name


def compare_tables(baseline_txt, ours_txt, out_txt):
    B = parse_table(baseline_txt)
    O = parse_table(ours_txt)

    rows = []

    for bname, b in B.items():
        key = normalize_name(bname)
        for oname, o in O.items():
            if normalize_name(oname) == key:
                diff = o["max"] - b["max"]
                rows.append({
                    "name": key,
                    "bmax": b["max"],
                    "omax": o["max"],
                    "diff": diff,
                })
                break

    # sort by absolute max-concavity difference
    rows.sort(key=lambda r: abs(r["diff"]), reverse=True)

    # formatting
    name_w = max(len("name"), max(len(r["name"]) for r in rows))
    sep = " | "

    lines = []
    header = (
        f"{'name'.ljust(name_w)}{sep}"
        f"{'baseline_max'.rjust(14)}{sep}"
        f"{'ours_max'.rjust(10)}{sep}"
        f"{'diff'.rjust(10)}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(
            f"{r['name'].ljust(name_w)}{sep}"
            f"{r['bmax']:14.4f}{sep}"
            f"{r['omax']:10.4f}{sep}"
            f"{r['diff']:10.4f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline summary txt")
    ap.add_argument("--ours", required=True, help="OURS summary txt")
    ap.add_argument("--out", required=True, help="output comparison txt")
    args = ap.parse_args()

    compare_tables(args.baseline, args.ours, args.out)
    print("Wrote comparison to:", args.out)
    
    
## python compare.py --baseline ./old_VHACD_data_0.1_forward_normalized.txt --ours ./benchmark_0.1_v2_metric_new.txt --out ./compare1.txt