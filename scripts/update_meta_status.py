#!/usr/bin/env python3
import json
import argparse
import os

def update_meta(meta_path, ready_range, running_range, completed_range):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} does not exist")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images = meta.get("images", [])
    for i, img_entry in enumerate(images):
        idx = i + 1  # 从1开始计数
        if ready_range[0] <= idx < ready_range[1]:
            img_entry["status"] = "ready"
        elif running_range[0] <= idx < running_range[1]:
            img_entry["status"] = "running"
        elif completed_range[0] <= idx < completed_range[1]:
            img_entry["status"] = "completed"
        else:
            img_entry.pop("status", None)  # 删除原有status字段

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {len(images)} images in {meta_path}.")

def parse_range(s):
    """解析区间参数，比如 200-600 -> (200,600)"""
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid range format: {s}")
    return int(parts[0]), int(parts[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch update status in meta.json")
    parser.add_argument("--meta", type=str, required=True, help="Path to meta.json")
    parser.add_argument("--ready", type=parse_range, default=(1,1), help="Range for ready (e.g., 1-200)")
    parser.add_argument("--running", type=parse_range, required=True, help="Range for running (e.g., 200-600)")
    parser.add_argument("--completed", type=parse_range, required=True, help="Range for completed (e.g., 600-900)")

    args = parser.parse_args()
    update_meta(args.meta, args.ready, args.running, args.completed)
