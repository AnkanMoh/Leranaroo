# tools/groq_tts.py
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class GroqTTS:
    """
    TTS wrapper with SAFE fallback.

    Strategy:
    1) Try Groq TTS endpoint if configured.
    2) If Groq returns "terms required" / bad request / unavailable => fallback to local Mac 'say'.

    This guarantees your pipeline can still run for submission.
    """
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"

    # Default model list you showed often require terms acceptance.
    # Keep it configurable; fallback will handle it anyway.
    model: str = os.getenv("GROQ_TTS_MODEL", "playai-tts")

    voice: str = os.getenv("LEARNAROO_TTS_VOICE", "Samantha")  # macOS voice
    timeout_s: int = 120

    def _run(self, cmd: list[str]) -> None:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\nSTDERR:\n{p.stderr}")

    def _mac_say_to_wav(self, text: str, out_path: str) -> str:
        """
        Uses macOS `say` to create an AIFF then converts to WAV via ffmpeg.
        """
        tmp_aiff = out_path + ".aiff"

        # Create AIFF
        self._run(["say", "-v", self.voice, "-o", tmp_aiff, text])

        # Convert to WAV (48k mono keeps size smaller + clean for mux)
        self._run([
            "ffmpeg", "-y",
            "-i", tmp_aiff,
            "-ar", "44100",
            "-ac", "2",
            out_path,
        ])

        try:
            os.remove(tmp_aiff)
        except Exception:
            pass

        return out_path

    def synthesize_to_file(self, *, text: str, out_path: str) -> str:
        text = (text or "").strip()
        if not text:
            raise RuntimeError("TTS: empty text")

        # If user explicitly disables Groq TTS, go straight to Mac TTS
        if os.getenv("LEARNAROO_FORCE_LOCAL_TTS", "0") == "1":
            return self._mac_say_to_wav(text, out_path)

        # --- Try Groq (best effort) ---
        try:
            url = self.base_url.rstrip("/") + "/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Many errors you saw were because of unsupported fields.
            # Keep request MINIMAL.
            payload = {
                "model": self.model,
                "input": text,
            }

            r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            if r.status_code >= 400:
                # If terms acceptance required or invalid request => fallback
                body = (r.text or "")[:2000]
                raise RuntimeError(f"Groq TTS failed HTTP {r.status_code}: {body}")

            # If Groq returns audio bytes
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.content)

            return out_path

        except Exception:
            # --- fallback ---
            return self._mac_say_to_wav(text, out_path)
