# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.34", "pandas", "pyarrow"]
# ///
"""Stage a per-task, RGB-only subset of behavior-1k/2026-challenge-demos.

The challenge repo is a LeRobot v3 dataset where meta/episodes/chunk-NNN/file-000.parquet
holds exactly the 200 episodes of task NNN. This script downloads, for one task:
  - the shared meta files (info.json, stats.json, tasks.jsonl/parquet)
  - that task's episodes parquet
  - only the data parquets and *RGB* video files its episodes reference
    (depth/seg streams are the bulk of the 3.27 TB repo and are skipped)
into a directory that is itself a valid LeRobot v3 tree, readable by
cosmos-framework's ActionBaseDataset (which globs meta/episodes/chunk-*/file-*.parquet).

Usage: python stage_behavior1k_task.py --task 0 --out /path/to/dataset_root
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "behavior-1k/2026-challenge-demos"
RGB_KEYS = [
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
]
META_FILES = ["meta/info.json", "meta/stats.json", "meta/tasks.jsonl", "meta/tasks.parquet"]


def fetch(rel: str, out: Path) -> Path:
    print(f"  {rel}", flush=True)
    return Path(
        hf_hub_download(repo_id=REPO, repo_type="dataset", filename=rel, local_dir=out)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, help="task index (0-99)")
    ap.add_argument("--out", type=Path, required=True, help="dataset root to create")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("meta files:")
    for rel in META_FILES:
        fetch(rel, out)
    ep_rel = f"meta/episodes/chunk-{args.task:03d}/file-000.parquet"
    ep_path = fetch(ep_rel, out)

    df = pd.read_parquet(ep_path)
    tasks = sorted({t for ts in df["tasks"] for t in ts})
    print(
        f"task {args.task}: {len(df)} episodes "
        f"({df.episode_index.min()}-{df.episode_index.max()}), "
        f"{df.length.sum()} frames, tasks={tasks}"
    )

    data_refs = sorted(set(zip(df["data/chunk_index"], df["data/file_index"])))
    print("data parquets:")
    for chunk, file in data_refs:
        fetch(f"data/chunk-{chunk:03d}/file-{file:03d}.parquet", out)

    for key in RGB_KEYS:
        refs = sorted(
            set(zip(df[f"videos/{key}/chunk_index"], df[f"videos/{key}/file_index"]))
        )
        print(f"videos/{key}: {len(refs)} files")
        for chunk, file in refs:
            fetch(f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4", out)

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
