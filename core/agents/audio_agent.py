# core/agents/audio_agent.py
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict


def generate_beat_audio(beat: Dict, out_path: str, *, voice: str = "Samantha") -> None:
    """
    macOS TTS -> WAV.
    Requires macOS 'say' + 'afconvert' (both available by default).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    text = (beat.get("narration") or "").strip()
    if len(text) < 5:
        raise RuntimeError("Beat narration is empty; cannot generate audio.")

    # Create AIFF then convert to WAV
    with tempfile.TemporaryDirectory() as td:
        aiff_path = os.path.join(td, "tmp.aiff")

        # -v voice sets English voice; Samantha is common on macOS
        subprocess.run(
            ["say", "-v", voice, "-o", aiff_path, text],
            check=True,
        )

        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, out_path],
            check=True,
        )
