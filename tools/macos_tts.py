# tools/macos_tts.py
from __future__ import annotations

import os
import subprocess
from typing import Optional


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\nSTDERR:\n{p.stderr[:4000]}")


def _ffprobe_duration(path: str) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr[:2000]}")
    try:
        return float((p.stdout or "").strip())
    except Exception:
        return 0.0


def _atempo_chain(speed: float) -> str:
    if speed <= 0:
        speed = 1.0
    parts = []
    while speed > 2.0:
        parts.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        parts.append("atempo=0.5")
        speed /= 0.5
    parts.append(f"atempo={speed:.6f}")
    return ",".join(parts)


def _clean_tts_text(text: str) -> str:
    """
    Make narration feel less robotic:
    - Remove 'X says:' patterns
    - Remove extra quotes
    - Keep it short and clean
    """
    t = (text or "").strip()
    if not t:
        return " "
    # remove "Learnaroo says:" etc.
    t = __import__("re").sub(r"\b\w+\s+says:\s*", "", t, flags=__import__("re").IGNORECASE)
    t = t.replace('"', "").replace("“", "").replace("”", "")
    t = __import__("re").sub(r"\s+", " ", t).strip()
    return t or " "


def synthesize_scene_wav(
    *,
    text: str,
    out_wav: str,
    duration_sec: int,
    voice: Optional[str] = None,
    rate_wpm: Optional[int] = None,
) -> str:
    """
    macOS TTS via `say` → aiff → ffmpeg to wav → time-fit to exact duration.
    rate_wpm uses: say -r <wpm>
    """
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    tmp_aiff = out_wav.replace(".wav", ".aiff")
    tmp_raw = out_wav.replace(".wav", "_raw.wav")

    safe_text = _clean_tts_text(text)

    say_cmd = ["say"]
    if voice:
        say_cmd += ["-v", voice]
    if rate_wpm and isinstance(rate_wpm, int) and rate_wpm > 0:
        say_cmd += ["-r", str(rate_wpm)]
    say_cmd += ["-o", tmp_aiff, safe_text]
    _run(say_cmd)

    _run([
        "ffmpeg", "-y",
        "-i", tmp_aiff,
        "-ac", "1",
        "-ar", "48000",
        tmp_raw,
    ])

    target = float(duration_sec)
    dur = _ffprobe_duration(tmp_raw)

    if dur <= 0.05:
        _run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=48000:cl=mono",
            "-t", str(duration_sec),
            out_wav,
        ])
    else:
        if dur > target:
            speed = dur / target
            chain = _atempo_chain(speed)
            _run([
                "ffmpeg", "-y",
                "-i", tmp_raw,
                "-filter:a", chain + f",atrim=0:{target}",
                "-t", str(duration_sec),
                out_wav,
            ])
        else:
            _run([
                "ffmpeg", "-y",
                "-i", tmp_raw,
                "-filter:a", f"apad,atrim=0:{target}",
                "-t", str(duration_sec),
                out_wav,
            ])

    for p in [tmp_aiff, tmp_raw]:
        try:
            os.remove(p)
        except Exception:
            pass

    return out_wav
