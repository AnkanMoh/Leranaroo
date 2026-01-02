# core/director.py
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Dict, Optional, List

from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips  # moviepy v2 API

from tools.genai_client import GenAIClient
from core.schemas import LessonPlan, Category
from core.agents import scriptwriter, audio_agent
from core.agents.video_agent_byteplus import generate_beat_video


RUNS_DIR = "runs"


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _run_dir(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def _lesson_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "lesson.json")


def _beat_audio_path(run_id: str, scene_idx: int, beat_idx: int) -> str:
    return os.path.join(_run_dir(run_id), "audio", f"s{scene_idx:02d}_b{beat_idx:02d}.wav")


def _beat_video_path(run_id: str, scene_idx: int, beat_idx: int) -> str:
    return os.path.join(_run_dir(run_id), "video", f"s{scene_idx:02d}_b{beat_idx:02d}.mp4")


def _final_path(run_id: str) -> str:
    return os.path.join(_run_dir(run_id), "final_synced.mp4")


def _build_visual_prompt(category: str, title: str, visual_prompt: str) -> str:
    # Keep it short; BytePlus seems to like concise prompts
    s = f"{category} kids 2D animation. Scene: {title}. {visual_prompt}".strip()
    return s[:240]


def _loop_audio_to_duration(aclip: AudioFileClip, target_s: float) -> AudioFileClip:
    """
    MoviePy-version-safe audio looping:
    - If audio shorter than target, concatenate it with itself until >= target
    - Then trim to exactly target
    """
    if target_s <= 0:
        return aclip

    adur = float(aclip.duration)
    if adur <= 0:
        return aclip

    if adur >= target_s:
        return aclip.subclipped(0, target_s)

    clips = [aclip]
    total = adur
    # cap iterations so we never infinite loop on weird durations
    for _ in range(50):
        if total >= target_s:
            break
        clips.append(aclip)
        total += adur

    joined = concatenate_videoclips(
        [VideoFileClip(os.devnull)] * 0, method="compose"
    )  # dummy, never used

    # MoviePy doesn't have concatenate_audioclips consistently across versions in your env,
    # but AudioFileClip supports `with_audio` on VideoClip, not needed here.
    # Instead, use moviepy.audio.AudioClip.concatenate_audioclips if available.
    try:
        from moviepy.audio.AudioClip import concatenate_audioclips  # type: ignore

        out = concatenate_audioclips(clips)
        return out.subclipped(0, target_s)
    except Exception:
        # Fallback: no concatenate_audioclips — just return trimmed original
        # (won’t loop, but avoids crashing)
        return aclip.subclipped(0, min(adur, target_s))


def stage2_generate_lesson_audio_video(
    topic: str,
    *,
    category: Category,
    run_id: Optional[str] = None,
) -> str:
    _ensure_dir(RUNS_DIR)
    run_id = run_id or _now_id()

    rd = _run_dir(run_id)
    _ensure_dir(rd)
    _ensure_dir(os.path.join(rd, "audio"))
    _ensure_dir(os.path.join(rd, "video"))

    client = GenAIClient.from_env()

    # 1) Scriptwriter → validated LessonPlan
    plan_dict = scriptwriter.run(client, topic, category)
    plan = LessonPlan.model_validate(plan_dict)

    # Save lesson json
    with open(_lesson_path(run_id), "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)

    # 2) Generate per-beat audio + video, then stitch
    stitched_clips: List[VideoFileClip] = []

    for scene in plan.scenes:
        for beat in scene.beats:
            audio_path = _beat_audio_path(run_id, scene.idx, beat.idx)
            video_path = _beat_video_path(run_id, scene.idx, beat.idx)

            # audio (English via macOS say)
            audio_agent.generate_beat_audio(beat.model_dump(), audio_path)

            # video (BytePlus)
            vp = _build_visual_prompt(plan.category, beat.title, beat.visual_prompt)
            generate_beat_video(
                client,
                visual_prompt=vp,
                out_path=video_path,
                duration_s=beat.duration_s,
                seed=None,
                timeout_s=900,
            )

            vclip = VideoFileClip(video_path)
            aclip = AudioFileClip(audio_path)

            vdur = float(vclip.duration)
            aclip2 = _loop_audio_to_duration(aclip, vdur)

            vclip2 = vclip.with_audio(aclip2)
            stitched_clips.append(vclip2)

    if not stitched_clips:
        raise RuntimeError("No beats generated. Check scriptwriter output.")

    final = concatenate_videoclips(stitched_clips, method="compose")
    out_path = _final_path(run_id)

    final.write_videofile(out_path, fps=24, audio=True, logger=None)

    # cleanup
    for c in stitched_clips:
        try:
            c.close()
        except Exception:
            pass
    try:
        final.close()
    except Exception:
        pass

    return run_id
