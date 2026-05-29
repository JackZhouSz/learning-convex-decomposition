# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import os, re, argparse

# ---------------------- tiny utils ----------------------
def f6(x):
    if x is None:
        x = 0.0
    if hasattr(x, "item"):
        x = x.item()
    return f"{float(x):.4f}"

def pair(maxv, meanv):
    return f"{f6(maxv)}/{f6(meanv)}"

def pad(s, w, align="<"):
    s = str(s)
    return s.ljust(w) if align == "<" else s.rjust(w)

# ---------------------- parsing ----------------------
def parse_txt(txt_path):
    """
    ONLY supports the VH/CO/CA format:

      # Per-hull concavity metrics (VH/CO/CA):
      hull[000]  VH/CO/CA = 0/0/0.0863
      ...
      # Max concavity across hulls:
      max_COACD_APPROXIMATE = 0.112

    We treat CA (third value) as concavity.

    Returns dict with:
      num_hulls, max_CA, mean_CA
    """
    ca_list = []
    num_hulls = 0
    max_ca = None

    trip_pat = re.compile(
        r"^hull\[\s*\d+\s*\].*VH/CO/CA\s*=\s*([0-9eE+.\-]+)/([0-9eE+.\-]+)/([0-9eE+.\-]+)"
    )
    max_approx_pat = re.compile(r"max_COACD_APPROXIMATE\s*=\s*([0-9eE+.\-]+)")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            ls = line.strip()
            if not ls:
                continue

            m = trip_pat.search(ls)
            if m:
                num_hulls += 1
                ca_list.append(float(m.group(3)))
                continue

            m = max_approx_pat.search(ls)
            if m:
                max_ca = float(m.group(1))
                continue

    mean_ca = (sum(ca_list) / len(ca_list)) if len(ca_list) > 0 else None
    if max_ca is None and len(ca_list) > 0:
        max_ca = max(ca_list)

    return {
        "num_hulls": num_hulls,
        "max_CA": max_ca,
        "mean_CA": mean_ca,
    }

# ---------------------- IO helpers ----------------------
def infer_name_from_filename(fn):
    """
    Keep your heuristic:
      name = fn[:fn.find("_2025")] if present else stem (no .txt).
    """
    stem = fn[:-4] if fn.endswith(".txt") else fn
    cut = stem.find("_2025")
    if cut >= 0:
        return stem[:cut]
    return stem

def collect_metrics_dir(metric_dir):
    out = {}
    if not os.path.exists(metric_dir):
        return out
    for fn in os.listdir(metric_dir):
        if not fn.endswith(".txt"):
            continue
        name = infer_name_from_filename(fn)
        out[name] = parse_txt(os.path.join(metric_dir, fn))
    return out

# ---------------------- baseline-only grid ----------------------
def save_baseline_grid(out_path, result, sort_on="mean_CA", reverse=False):
    """
    Baseline-only table:
    Row1: name | BASELINE
    Row2:   # | concavity_CA (max/mean)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    rows = []
    for name, b in result.items():
        if sort_on == "num_hulls":
            keyv = b.get("num_hulls", 0)
        else:
            keyv = b.get(sort_on)
        if keyv is None:
            continue

        rows.append({
            "name": name,
            "h": int(b.get("num_hulls", 0)),
            "ca": pair(b.get("max_CA"), b.get("mean_CA")),
            "_nums": {
                "h": b.get("num_hulls", 0),
                "ca_max": b.get("max_CA"),
                "ca_mean": b.get("mean_CA"),
                "_sort": keyv,
            }
        })

    rows.sort(key=lambda r: float(r["_nums"]["_sort"]), reverse=reverse)

    name_w = max([len("name")] + [len(r["name"]) for r in rows]) if rows else len("name")
    h_strs = [str(r["h"]) for r in rows] + ["#"]
    ca_strs = [r["ca"] for r in rows] + ["concavity_CA (max/mean)"]

    w_cnt = max(len("#"), max((len(s) for s in h_strs), default=1))
    w_ca  = max(len("concavity_CA (max/mean)"), max((len(s) for s in ca_strs), default=22))

    sep = " | "
    base_row2_block = pad("#", w_cnt, ">") + sep + pad("concavity_CA (max/mean)", w_ca)
    base_block_w = len(base_row2_block)

    header_row1 = pad("name", name_w) + sep + pad("BASELINE", base_block_w)
    header_row2 = pad("", name_w) + sep + base_row2_block

    lines = [header_row1, header_row2]

    def acc_init(): return {"sum": 0.0, "n": 0}
    agg = {"h": acc_init(), "ca_max": acc_init(), "ca_mean": acc_init()}

    def add(k, v):
        if v is None:
            return
        agg[k]["sum"] += float(v)
        agg[k]["n"] += 1

    for r in rows:
        lines.append(
            pad(r["name"], name_w) + sep +
            pad(str(r["h"]), w_cnt, ">") + sep +
            pad(r["ca"], w_ca)
        )
        n = r["_nums"]
        for k in ["h", "ca_max", "ca_mean"]:
            add(k, n[k])

    with open(out_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

        def avg(k):
            return (agg[k]["sum"] / max(agg[k]["n"], 1)) if agg[k]["n"] > 0 else 0.0

        f.write("\n")
        f.write(f"Averages over {len(rows)} shapes (baseline only)\n")

        labels = [
            "baseline #hulls",
            "baseline concavity_CA (max/mean)",
        ]
        label_w = max(len(s) for s in labels)

        def pl(label, val):
            f.write(pad(label + ":", label_w + 1) + " " + val + "\n")

        pl("baseline #hulls", f6(avg("h")))
        pl("baseline concavity_CA (max/mean)", f"{f6(avg('ca_max'))}/{f6(avg('ca_mean'))}")

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric_dir", required=True, help="Directory containing *.txt metric files")
    ap.add_argument("--out_txt", required=True, help="Output summary txt path")
    ap.add_argument("--sort_on", default="max_CA", choices=["mean_CA", "max_CA", "num_hulls"])
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    R = collect_metrics_dir(args.metric_dir)
    print("total number of shapes:", len(R))

    save_baseline_grid(args.out_txt, R, sort_on=args.sort_on, reverse=args.reverse)
    print("Wrote (baseline only):", args.out_txt)

###python gather_eval.py --metric_dir ./exp_results/partfield_convex_reg_mc_regonly_v2/benchmark_0.1_v2_raw/metric_new/ --out_txt ./benchmark_0.1_v2_raw.txt  --sort_on max_CA


# python gather_eval.py --metric_dir ./exp_results/partfield_convex_reg_mc_regonly_v2/baseline_0.1_01_28/metric_new/ --out_txt ./baseline_0.1_01_28.txt  --sort_on max_CA