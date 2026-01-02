from __future__ import annotations
import os
from dotenv import load_dotenv

from core.agents.video_agent_byteplus import generate_video_from_text

load_dotenv()

prompt = (
    "A cute baby dinosaur superhero waves to the camera in a bright classroom, "
    "gentle smile, soft lighting. "
    "--dur 5 --rs 480p --rt 16:9 --fps 24 --cf false --wm false"
)

out = generate_video_from_text(prompt, out_path="runs/seedance_local.mp4")
print("Saved:", out)
