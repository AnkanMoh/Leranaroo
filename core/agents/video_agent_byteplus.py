# core/agents/video_agent_byteplus.py
from __future__ import annotations

import os
from typing import Dict, Optional

from tools.genai_client import GenAIClient


def generate_beat_video(
    client: GenAIClient,
    *,
    visual_prompt: str,
    out_path: str,
    duration_s: int,
    seed: Optional[int] = None,
    timeout_s: int = 600,
) -> Dict:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    task_id = client.create_video_task(prompt=visual_prompt, duration_s=duration_s, seed=seed)
    st = client.poll_video_task(task_id, timeout_s=timeout_s)

    # Expected final JSON:
    # st["content"]["video_url"]
    content = st.get("content") or {}
    url = content.get("video_url")
    if not url:
        raise RuntimeError(f"BytePlus succeeded but no video_url found. Status JSON: {st}")

    # Download the mp4
    import requests

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return st
