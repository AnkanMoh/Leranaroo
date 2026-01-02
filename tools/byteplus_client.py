# tools/byteplus_client.py
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Union

import requests


Json = Dict[str, Any]


class BytePlusError(RuntimeError):
    pass


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _as_data_url(path: str) -> str:
    # best-effort mime
    ext = (os.path.splitext(path)[1] or "").lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    b64 = _b64_file(path)
    return f"data:{mime};base64,{b64}"


def _deep_find_url(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        if obj.startswith("http://") or obj.startswith("https://"):
            return obj
        return None
    if isinstance(obj, dict):
        for k in ("video_url", "url", "download_url", "output_url"):
            v = obj.get(k)
            if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
                return v
        for v in obj.values():
            found = _deep_find_url(v)
            if found:
                return found
        return None
    if isinstance(obj, list):
        for it in obj:
            found = _deep_find_url(it)
            if found:
                return found
        return None
    return None


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


@dataclass
class BytePlusClient:
    api_key: str
    base_url: str
    model: str

    timeout_s: int = 60
    poll_interval_s: float = 2.0
    max_poll_s: int = 180

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or "").strip()
        self.base_url = (self.base_url or "").strip().rstrip("/")
        self.model = (self.model or "").strip()

        if not self.api_key:
            raise BytePlusError("Missing ARK_API_KEY (api_key).")
        if not self.base_url:
            raise BytePlusError("Missing BASE_URL (base_url).")
        if not self.model:
            raise BytePlusError("Missing MODEL (model).")

        # Normalize to always include /api/v3
        if not self.base_url.endswith("/api/v3"):
            self.base_url = self.base_url + "/api/v3"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, url: str, payload: Json) -> Json:
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        if r.status_code not in (200, 201, 202):
            raise BytePlusError(f"BytePlus HTTP {r.status_code} at {url}. Body: {r.text[:1200]}")
        try:
            return r.json()
        except Exception:
            raise BytePlusError(f"BytePlus returned non-JSON at {url}. Body: {r.text[:1200]}")

    def _get_json(self, url: str) -> Json:
        r = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
        if r.status_code != 200:
            raise BytePlusError(f"BytePlus HTTP {r.status_code} at {url}. Body: {r.text[:1200]}")
        try:
            return r.json()
        except Exception:
            raise BytePlusError(f"BytePlus returned non-JSON at {url}. Body: {r.text[:1200]}")

    def generate_video(
        self,
        *,
        prompt: str,
        duration_s: int,
        ratio: str,
        resolution: str,
        reference_image_path: Optional[str] = None,
        camera_fixed: bool = False,
    ) -> Dict[str, Any]:
        """
        Matches the API example you shared:
        POST /api/v3/contents/generations/tasks
        {
          "model": "...",
          "content": [
            {"type":"text","text":"... --duration 5 --camerafixed false"},
            {"type":"image_url","image_url":{"url":"..."}}
          ]
        }
        """
        create_url = f"{self.base_url}/contents/generations/tasks"

        # Seedance prefers duration inside text flags (per your example)
        # Keep your prompt, append required flags.
        flags = f"--duration {int(duration_s)} --camerafixed {'true' if camera_fixed else 'false'}"
        # ratio/resolution flags may or may not be supported; keep them out unless you know they work.
        text = (prompt or "").strip()
        if flags not in text:
            text = (text + " " + flags).strip()

        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

        if reference_image_path:
            if reference_image_path.startswith("http://") or reference_image_path.startswith("https://"):
                img_url = reference_image_path
            else:
                if not os.path.exists(reference_image_path):
                    raise BytePlusError(f"reference_image_path not found: {reference_image_path}")
                # Try data-url. If ARK rejects it, we’ll switch to their accepted base64 field.
                img_url = _as_data_url(reference_image_path)

            content.append({"type": "image_url", "image_url": {"url": img_url}})

        payload: Dict[str, Any] = {
            "model": self.model,
            "content": content,
        }

        created = self._post_json(create_url, payload)

        # Task id extraction (robust)
        task_id = (
            _first_present(created, ["id", "task_id"])
            or _first_present(created.get("data") or {}, ["id", "task_id"])
            or (created.get("data") or {}).get("task", {}).get("id")
        )
        if not task_id:
            raise BytePlusError(f"Task created but no task_id found. URL={create_url} Resp={json.dumps(created)[:1200]}")

        task_id = str(task_id)
        status_url = f"{self.base_url}/contents/generations/tasks/{task_id}"

        deadline = time.time() + float(self.max_poll_s)
        last_state = None
        last_resp: Optional[Json] = None

        while time.time() < deadline:
            status = self._get_json(status_url)
            last_resp = status

            state = (
                _first_present(status, ["status", "state"])
                or _first_present(status.get("data") or {}, ["status", "state"])
                or (status.get("data") or {}).get("task", {}).get("status")
                or (status.get("data") or {}).get("task", {}).get("state")
            )
            last_state = str(state).lower().strip() if state is not None else None

            video_url = _deep_find_url(status)
            if video_url:
                return {
                    "task_id": task_id,
                    "video_url": video_url,
                    "debug": {"create_url": create_url, "status_url": status_url, "state": last_state},
                }

            if last_state in ("succeeded", "success", "done", "completed", "finish", "finished"):
                raise BytePlusError(
                    f"Task finished but no video_url found. status_url={status_url} resp={json.dumps(status)[:1200]}"
                )
            if last_state in ("failed", "error", "canceled", "cancelled"):
                raise BytePlusError(
                    f"Task failed. status_url={status_url} resp={json.dumps(status)[:1200]}"
                )

            time.sleep(self.poll_interval_s)

        raise BytePlusError(
            f"Timed out polling task {task_id}. last_state={last_state} last_resp={json.dumps(last_resp)[:900] if last_resp else None}"
        )
