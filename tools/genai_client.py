# tools/genai_client.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


# ----------------------------
# Dotenv loading (Streamlit-safe)
# ----------------------------
def _load_env_safely() -> None:
    """
    Streamlit sometimes breaks python-dotenv's find_dotenv() (frame assertions).
    So we load .env explicitly from project root (same folder as app.py typically).
    """
    try:
        from dotenv import load_dotenv  # python-dotenv
    except Exception:
        return

    # Try common locations: current working dir and repo root relative to this file
    candidates = [
        Path(os.getcwd()) / ".env",
        Path(__file__).resolve().parents[1] / ".env",  # project root if tools/ is one level down
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=str(p), override=False)
            return


# ----------------------------
# Gemini wrapper (google-genai)
# ----------------------------
@dataclass
class GeminiClient:
    api_key: str
    model: str = "gemini-2.0-flash"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Put it in .env as GEMINI_API_KEY=... "
                "and make sure Streamlit is running inside the SAME venv."
            )

        # Lazy import so app can still run non-Gemini parts if needed
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "google-genai not installed. Install with:\n"
                "  pip install google-genai\n"
            ) from e

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate_text(self, *, system: str, prompt: str) -> str:
        """
        Returns plain text. We force a stable usage pattern.
        """
        try:
            # google-genai API: client.models.generate_content(...)
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[
                    {"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]},
                ],
            )
        except Exception as e:
            # Surface the real exception message (instead of Tenacity RetryError hiding it)
            raise RuntimeError(f"Gemini generate_text failed: {e}") from e

        # Extract text robustly
        text = ""
        try:
            text = (resp.text or "").strip()
        except Exception:
            text = ""

        if not text:
            # Print a helpful debug snippet
            raise RuntimeError(
                "Gemini returned empty text. This usually means auth/quota/model error.\n"
                "Double-check GEMINI_API_KEY and model name."
            )
        return text

    def generate_json(self, *, system: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        We ask Gemini for JSON, but still parse from text for maximum compatibility.
        (Structured outputs vary across libs/versions.)
        """
        # We instruct JSON only; parsing happens in scriptwriter with fence stripping anyway.
        txt = self.generate_text(system=system, prompt=prompt + "\n\nReturn JSON only.")
        # Let upstream do the smarter extraction if needed, but provide a best-effort parse here
        try:
            return json.loads(txt)
        except Exception:
            return {"_raw_text": txt}


# ----------------------------
# BytePlus ARK wrapper
# ----------------------------
@dataclass
class BytePlusClient:
    api_key: str
    base_url: str = "https://ark.ap-southeast.bytepluses.com/api/v3"
    model: str = "seedance-1-5-pro-251215"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "Missing ARK_API_KEY. Put it in .env as ARK_API_KEY=... "
                "and ensure Streamlit is using the same venv."
            )

    def _headers(self) -> Dict[str, str]:
        # IMPORTANT: BytePlus ARK uses Bearer in most examples; if your account uses a different header,
        # adjust here.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_video_task(self, *, prompt: str, duration_s: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Uses the WORKING endpoint you proved in Colab:
        POST /contents/generations/tasks
        """
        url = f"{self.base_url}/contents/generations/tasks"

        # This is the payload shape that worked for you (Variant 3).
        payload: Dict[str, Any] = {
            "model": self.model,
            "content": {
                "prompt": prompt,
            },
            "duration": int(duration_s),
        }
        if seed is not None:
            payload["seed"] = int(seed)

        r = requests.post(url, headers=self._headers(), json=payload, timeout=60)

        # If non-json, show raw
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            raise RuntimeError(
                "BytePlus ARK non-JSON response\n"
                f"URL: {url}\nStatus: {r.status_code}\nContent-Type: {r.headers.get('Content-Type')}\n"
                f"Body (first 500): {r.text[:500]!r}"
            )

        data = r.json()
        if r.status_code >= 400:
            raise RuntimeError(f"BytePlus ARK error {r.status_code}: {json.dumps(data, indent=2)[:1200]}")
        return data

    def poll_task(self, *, task_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/contents/generations/tasks/{task_id}"
        r = requests.get(url, headers=self._headers(), timeout=60)

        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            raise RuntimeError(
                "BytePlus ARK poll non-JSON response\n"
                f"URL: {url}\nStatus: {r.status_code}\nContent-Type: {r.headers.get('Content-Type')}\n"
                f"Body (first 500): {r.text[:500]!r}"
            )
        data = r.json()
        if r.status_code >= 400:
            raise RuntimeError(f"BytePlus ARK poll error {r.status_code}: {json.dumps(data, indent=2)[:1200]}")
        return data


# ----------------------------
# Unified client used across project
# ----------------------------
class GenAIClient:
    def __init__(self, *, gemini: GeminiClient, byteplus: BytePlusClient):
        self.gemini = gemini
        self.byteplus = byteplus

    @classmethod
    def from_env(cls) -> "GenAIClient":
        _load_env_safely()

        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        ark_key = os.getenv("ARK_API_KEY", "").strip()

        # Optional overrides
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
        ark_base = os.getenv("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3").strip()
        ark_model = os.getenv("ARK_MODEL", "seedance-1-5-pro-251215").strip()

        gemini = GeminiClient(api_key=gemini_key, model=gemini_model)
        byteplus = BytePlusClient(api_key=ark_key, base_url=ark_base, model=ark_model)
        return cls(gemini=gemini, byteplus=byteplus)

    # Convenience passthroughs (optional)
    def generate_text(self, *, system: str, prompt: str) -> str:
        return self.gemini.generate_text(system=system, prompt=prompt)

    def generate_json(self, *, system: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return self.gemini.generate_json(system=system, prompt=prompt, schema=schema)

    def create_video_task(self, *, prompt: str, duration_s: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
        return self.byteplus.create_video_task(prompt=prompt, duration_s=duration_s, seed=seed)

    def poll_video_task(self, *, task_id: str) -> Dict[str, Any]:
        return self.byteplus.poll_task(task_id=task_id)
