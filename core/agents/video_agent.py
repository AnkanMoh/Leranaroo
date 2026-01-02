# core/agents/video_agent.py
from __future__ import annotations

import os
from typing import Callable, List, Optional, Union

from core.schemas import LessonPlan
from tools.genai_client import GenAIClient

# Backend selection:
# VIDEO_BACKEND=byteplus   -> uses BytePlus
# VIDEO_BACKEND=vertex/veo -> you can later add a veo version again
VIDEO_BACKEND = (os.getenv("VIDEO_BACKEND") or "byteplus").strip().lower()

if VIDEO_BACKEND == "byteplus":
    from core.agents.video_agent_byteplus import run as _run_impl
else:
    # fallback: still use byteplus impl to avoid breaking
    from core.agents.video_agent_byteplus import run as _run_impl


def run(
    client: GenAIClient,
    plan: LessonPlan,
    audio_paths: List[str],
    run_dir: str,
    report_cb: Optional[Callable[[float, str], None]] = None,
) -> Union[str, List[str]]:
    return _run_impl(client, plan, audio_paths, run_dir, report_cb=report_cb)
