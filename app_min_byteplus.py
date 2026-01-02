# app_min_byteplus.py
from __future__ import annotations

import os
import time
import uuid
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.pipeline import run_pipeline

ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

THEMES = ["Superhero", "Dinosaur", "Space", "Fairytale", "Robot"]


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _env_ready() -> bool:
    # Keep minimal requirement, but also surface missing base_url/model in logs
    return bool((os.getenv("ARK_API_KEY") or "").strip())


st.set_page_config(page_title="Learnaroo", layout="centered")

st.markdown(
    """
    <div style="text-align:center; margin-bottom: 1.2rem;">
        <div style="font-size:38px;">🦘</div>
        <div style="font-size:32px; font-weight:800;">Learnaroo</div>
        <div style="opacity:0.75; font-size:16px;">
            Kid-friendly lesson → short animated video
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

st.subheader("Topic")
topic = st.text_input("Topic", value="Gravity", label_visibility="collapsed")

st.subheader("Theme")
theme = st.selectbox("Theme", THEMES, index=0, label_visibility="collapsed")

st.divider()
generate = st.button("🎬 Generate", type="primary", use_container_width=True)


if generate:
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    if not _env_ready():
        st.error("App is not configured. Please add ARK_API_KEY to .env.")
        st.stop()

    run_id = _now_id()
    out_dir = os.path.join("runs", run_id)

    st.subheader("Progress")
    progress_bar = st.progress(0, text="Starting…")
    status_box = st.empty()

    # Advanced logs (now we will ALWAYS show error+traceback here on failure)
    exp = st.expander("Advanced (backend logs)", expanded=False)
    with exp:
        env_box = st.empty()
        log_box = st.empty()
        err_box = st.empty()
        tb_box = st.empty()

    # show env diagnostics (safe, no keys printed)
    env_diag = {
        "ARK_API_KEY_set": bool((os.getenv("ARK_API_KEY") or "").strip()),
        "BASE_URL": (os.getenv("BASE_URL") or "").strip() or "(missing)",
        "MODEL": (os.getenv("MODEL") or "").strip() or "(missing)",
    }
    env_box.code("ENV (sanity):\n" + "\n".join([f"- {k}: {v}" for k, v in env_diag.items()]), language="text")

    logs: list[str] = []

    def progress_cb(p: int, msg: str) -> None:
        progress_bar.progress(p, text=msg)
        status_box.markdown(f"**{msg}**")
        logs.append(f"{p:>3}%  {msg}")
        log_box.code("\n".join(logs[-120:]), language="text")

    # Run pipeline and show full errors if any
    result = None
    try:
        result = run_pipeline(
            topic=topic.strip(),
            theme=theme,
            ratio="16:9",
            resolution="480p",
            use_narration_audio=True,
            out_dir=out_dir,
            progress_cb=progress_cb,
        )
    except Exception as e:
        progress_bar.progress(0, text="Failed ❌")
        st.error("Something went wrong while generating the video.")
        # Force open the expander (Streamlit can’t programmatically expand reliably),
        # but we can at least dump the traceback into it.
        err_box.code(f"UNCAUGHT EXCEPTION:\n{repr(e)}", language="text")
        tb_box.code(traceback.format_exc(), language="text")
        st.stop()

    # If pipeline returns structured failure (recommended with updated core/pipeline.py)
    if isinstance(result, dict) and not result.get("ok", True):
        progress_bar.progress(0, text="Failed ❌")
        st.error("Something went wrong while generating the video.")

        err = result.get("error") or "(no error string returned)"
        tb = result.get("traceback") or "(no traceback returned)"
        err_box.code(f"PIPELINE ERROR:\n{err}", language="text")
        tb_box.code(tb, language="text")
        st.stop()

    final_path = (result or {}).get("output")
    if final_path and os.path.exists(final_path):
        st.divider()
        st.subheader("Generated Video")
        st.video(final_path)

        with open(final_path, "rb") as f:
            st.download_button(
                "⬇️ Download video",
                data=f,
                file_name=os.path.basename(final_path),
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.warning("Video generation completed, but output file was not found.")
