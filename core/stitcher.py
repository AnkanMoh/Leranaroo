from __future__ import annotations

import os
import subprocess
from typing import List


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def stitch_with_ffmpeg(clip_paths: List[str], out_path: str) -> None:
    """
    Robust concatenation:
    - Strip audio from each clip to avoid DTS / timestamp issues
    - Concat video-only files using concat demuxer
    """
    if not clip_paths:
        raise RuntimeError("No clips to stitch.")

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    mute_paths: List[str] = []
    for p in clip_paths:
        mute_p = os.path.splitext(p)[0] + "_mute.mp4"
        # Strip audio, re-encode lightly to normalize timestamps (fast via videotoolbox if available)
        _run([
            "ffmpeg", "-y",
            "-i", p,
            "-an",  # remove audio
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            mute_p
        ])
        mute_paths.append(mute_p)

    list_path = os.path.join(out_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in mute_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    _run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        out_path
    ])
