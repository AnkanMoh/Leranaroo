from __future__ import annotations

import os
import re
import subprocess
import time
import traceback
from typing import Dict, Any, List, Optional, Callable

from PIL import Image, ImageChops

from core.script_groq import GroqScriptConfig, generate_valid_script
from tools.byteplus_client import BytePlusClient

ProgressCB = Optional[Callable[[int, str], None]]


# ----------------------------
# Shell helpers (now include STDOUT too)
# ----------------------------
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n"
            f"STDOUT:\n{(p.stdout or '')[:4000]}\n"
            f"STDERR:\n{(p.stderr or '')[:4000]}"
        )


def _run_out(cmd: List[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n"
            f"STDOUT:\n{(p.stdout or '')[:4000]}\n"
            f"STDERR:\n{(p.stderr or '')[:4000]}"
        )
    return (p.stdout or "").strip()


def _ffprobe_duration_s(path: str) -> float:
    out = _run_out([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path
    ])
    try:
        return float(out)
    except Exception:
        return 0.0


def _nearest_allowed_duration(s: float, allowed: List[int]) -> int:
    return min(allowed, key=lambda x: abs(x - s)) if allowed else max(5, int(round(s)))


# ----------------------------
# Pronouns + naming (simple + safe)
# ----------------------------
THEME_TO_MASCOT = {
    "Superhero": "assets/mascots/suptain_comet.png",
    "Space": "assets/mascots/astro_alex.png",
    "Dinosaur": "assets/mascots/dino_diggs.png",
    "Fairytale": "assets/mascots/rusty_router.png",
    "Robot": "assets/mascots/astro_alex.png",
}

THEME_IDENTITY = {
    "Superhero": {"name": "Suptain Comet", "subject": "he", "object": "him", "poss": "his"},
    "Space": {"name": "Astro Alex", "subject": "he", "object": "him", "poss": "his"},
    "Dinosaur": {"name": "Dino Diggs", "subject": "he", "object": "him", "poss": "his"},
    "Fairytale": {"name": "Rusty Router", "subject": "she", "object": "her", "poss": "her"},
    "Robot": {"name": "Robo Buddy", "subject": "they", "object": "them", "poss": "their"},
}

ALLOWED_DURATIONS = [5, 6, 7, 8]


def _select_mascot_for_theme(theme: str) -> str:
    key = (theme or "").strip().title()
    return THEME_TO_MASCOT.get(key, "assets/mascots/astro_alex.png")


def _identity(theme: str) -> Dict[str, str]:
    key = (theme or "").strip().title()
    return THEME_IDENTITY.get(key, {"name": "our friend", "subject": "they", "object": "them", "poss": "their"})


def _sanitize_character_naming(text: str, theme: str) -> str:
    if not text:
        return text
    ident = _identity(theme)
    name = ident["name"]

    patterns = [
        r"\bgravity girl\b", r"\bgravity-girl\b", r"\bgravitygirl\b",
        r"\b(?:captain|cap)\s+comet\b",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, name, out, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", out).strip()


def _force_pronouns_simple(text: str, theme: str) -> str:
    if not text:
        return text
    ident = _identity(theme)
    subj = ident["subject"]

    out = text
    if subj.lower() == "he":
        out = re.sub(r"\bshe\b", "he", out, flags=re.IGNORECASE)
        out = re.sub(r"\bher\b", "him", out, flags=re.IGNORECASE)
        out = re.sub(r"\bhers\b", "his", out, flags=re.IGNORECASE)
    elif subj.lower() == "she":
        out = re.sub(r"\bhe\b", "she", out, flags=re.IGNORECASE)
        out = re.sub(r"\bhim\b", "her", out, flags=re.IGNORECASE)
        out = re.sub(r"\bhis\b", "her", out, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", out).strip()


# ----------------------------
# Clean reference image (removes top label)
# ----------------------------
def _auto_clean_reference_image(src_path: str, out_path: str) -> str:
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size

    top_cut = int(h * 0.18)
    if 0 < top_cut < h - 10:
        img = img.crop((0, top_cut, w, h))

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox)

    pad = 20
    nw, nh = img.size
    canvas = Image.new("RGBA", (nw + pad * 2, nh + pad * 2), (255, 255, 255, 0))
    canvas.paste(img, (pad, pad))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


# ----------------------------
# macOS TTS (same as your code)
# ----------------------------
def _tts_say_to_wav(text: str, out_wav_path: str, voice: str = "Samantha", rate_wpm: int = 175) -> None:
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    tmp_aiff = out_wav_path + ".tmp.aiff"
    _run(["say", "-v", voice, "-r", str(rate_wpm), "-o", tmp_aiff, text])
    _run(["ffmpeg", "-y", "-i", tmp_aiff, "-ac", "1", "-ar", "44100", out_wav_path])
    try:
        os.remove(tmp_aiff)
    except OSError:
        pass


def _pad_silence_to(in_wav: str, out_wav: str, target_s: int) -> None:
    _run([
        "ffmpeg", "-y",
        "-i", in_wav,
        "-filter_complex", f"apad=pad_dur={max(0, target_s)}",
        "-t", str(int(target_s)),
        "-ac", "1", "-ar", "44100",
        out_wav
    ])


def _fit_audio_soft(text: str, out_wav_path: str, target_s: int) -> None:
    target_s = max(1, int(target_s))
    tmp = out_wav_path + ".base.wav"
    tmp2 = out_wav_path + ".tmp2.wav"

    _tts_say_to_wav(text, tmp)

    d = _ffprobe_duration_s(tmp)
    if d <= 0.05:
        _run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(target_s),
            out_wav_path
        ])
        return

    if d <= target_s:
        _pad_silence_to(tmp, out_wav_path, target_s)
        for p in (tmp,):
            try:
                os.remove(p)
            except OSError:
                pass
        return

    speed = min(1.12, max(1.00, d / float(target_s)))
    _run([
        "ffmpeg", "-y",
        "-i", tmp,
        "-filter:a", f"atempo={speed:.6f}",
        "-t", str(target_s),
        "-ac", "1", "-ar", "44100",
        tmp2
    ])
    _pad_silence_to(tmp2, out_wav_path, target_s)

    for p in (tmp, tmp2):
        try:
            os.remove(p)
        except OSError:
            pass


# ----------------------------
# Main pipeline
# Returns traceback in dict when failing (so UI can show it)
# ----------------------------
def run_pipeline(
    *,
    topic: str,
    theme: str,
    ratio: str = "16:9",
    resolution: str = "480p",
    use_narration_audio: bool = True,
    out_dir: str = "runs/latest",
    progress_cb: ProgressCB = None,
) -> Dict[str, Any]:
    def emit(p: int, msg: str) -> None:
        if progress_cb:
            progress_cb(int(max(0, min(100, p))), msg)

    try:
        os.makedirs(out_dir, exist_ok=True)
        audio_dir = os.path.join(out_dir, "audio")
        video_dir = os.path.join(out_dir, "video")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        ident = _identity(theme)
        mascot_path = _select_mascot_for_theme(theme)
        if not os.path.exists(mascot_path):
            raise RuntimeError(f"Mascot image not found: {mascot_path}")

        clean_ref = os.path.join(out_dir, "_clean_ref.png")
        clean_ref = _auto_clean_reference_image(mascot_path, clean_ref)

        emit(5, "Generating script…")
        groq_cfg = GroqScriptConfig()
        script = generate_valid_script(
            groq_cfg,
            topic=topic,
            theme=theme,
            character_name=ident["name"],
            pronouns=f"{ident['subject']}/{ident['object']}/{ident['poss']}",
        )

        beats = script.get("beats", [])
        if not beats:
            raise RuntimeError("Script generated but has no beats")

        emit(10, f"Script ready: {len(beats)} scenes")

        emit(12, "Initializing BytePlus…")
        bp = BytePlusClient(
            api_key=os.getenv("ARK_API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model=os.getenv("MODEL"),
        )

        STYLE_LOCK = (
            "2D animated cartoon, flat colors, thick outlines, bright kid-friendly look. "
            "NO realism, NO 3D, NO poster/card framing, NO static portrait. "
            "Start with action immediately. Camera shows a full scene, not a character card."
        )

        muxed_paths: List[str] = []
        n = len(beats)

        for idx, beat in enumerate(beats, start=1):
            scene_text = (beat.get("visual_prompt") or "").strip()
            narration_raw = (beat.get("narration") or "").strip()

            narration = _sanitize_character_naming(narration_raw, theme)
            narration = _force_pronouns_simple(narration, theme)

            base_duration = int(beat.get("duration_s") or 6)
            base_duration = max(5, min(8, base_duration))
            duration_s = _nearest_allowed_duration(base_duration, ALLOWED_DURATIONS)

            emit(15 + int((idx - 1) * (70 / max(1, n))), f"Scene {idx}/{n}: audio…")
            audio_wav = os.path.join(audio_dir, f"scene_{idx:02d}.wav")
            if use_narration_audio and narration:
                _fit_audio_soft(narration, audio_wav, duration_s)
            else:
                _run([
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", str(duration_s),
                    audio_wav
                ])

            emit(25 + int((idx - 1) * (70 / max(1, n))), f"Scene {idx}/{n}: video…")

            final_prompt = (
                f"{STYLE_LOCK}\n"
                f"Main character: {ident['name']} ({ident['subject']}/{ident['object']}/{ident['poss']}).\n"
                "Use reference image ONLY for character appearance consistency.\n"
                "Do NOT start from the reference image. Do NOT show the printed name label.\n"
                "No text overlays, no captions, no logos.\n"
                "Include motion: walking, pointing, floating objects, swirling stars, moving props.\n\n"
                f"Scene description:\n{scene_text}"
            )

            # --- THIS is where you are failing. We want full exception and message ---
            vid = bp.generate_video(
                prompt=final_prompt,
                duration_s=duration_s,
                ratio=ratio,
                resolution=resolution,
                reference_image_path=clean_ref,
            )

            video_url = vid["video_url"]
            raw_scene = os.path.join(video_dir, f"raw_{idx:02d}.mp4")
            _run(["curl", "-L", "-o", raw_scene, video_url])

            emit(35 + int((idx - 1) * (70 / max(1, n))), f"Scene {idx}/{n}: mux…")
            scene_mp4 = os.path.join(video_dir, f"scene_{idx:02d}.mp4")
            _run([
                "ffmpeg", "-y",
                "-i", raw_scene,
                "-t", str(duration_s),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                scene_mp4
            ])

            muxed = os.path.join(video_dir, f"mux_{idx:02d}.mp4")
            _run([
                "ffmpeg", "-y",
                "-i", scene_mp4,
                "-i", audio_wav,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                muxed
            ])
            muxed_paths.append(muxed)

        emit(92, "Final stitching…")
        concat_file = os.path.join(out_dir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for p in muxed_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")

        final_out = os.path.join(out_dir, "final.mp4")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", final_out])

        emit(100, "Done ✅")
        return {
            "ok": True,
            "output": final_out,
            "beats": len(beats),
            "theme": theme,
            "mascot": mascot_path,
            "clean_ref": clean_ref,
            "character": ident["name"],
            "pronouns": f"{ident['subject']}/{ident['object']}/{ident['poss']}",
        }

    except Exception as e:
        tb = traceback.format_exc()
        emit(100, "Failed ❌ (open Advanced logs for traceback)")
        return {
            "ok": False,
            "error": str(e),
            "traceback": tb,
        }
