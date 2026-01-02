# core/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, conint, constr


Category = Literal["superhero", "dinosaur", "space battle", "robot lab", "fairytale"]


class Character(BaseModel):
    name: constr(min_length=1, max_length=40)
    role: constr(min_length=1, max_length=80)
    visual_style: constr(min_length=1, max_length=80)


class Beat(BaseModel):
    # idx is 1-based (pydantic errors you saw required >=1)
    idx: conint(ge=1)
    title: constr(min_length=1, max_length=80)

    # English narration, 5–8 seconds worth typically 1–3 sentences
    narration: constr(min_length=10, max_length=420)

    # Prompt for BytePlus video
    visual_prompt: constr(min_length=10, max_length=240)

    # must be 5–8 for Seedance free tier
    duration_s: conint(ge=5, le=8) = 6

    # optional on-screen lines for this beat (keep short)
    on_screen_text: Optional[List[constr(min_length=1, max_length=40)]] = None


class Scene(BaseModel):
    idx: conint(ge=1)
    title: constr(min_length=1, max_length=80)
    beats: List[Beat] = Field(min_length=1)


class LessonPlan(BaseModel):
    title: constr(min_length=1, max_length=80)
    category: Category

    # Single learning objective sentence (you had max 220 error)
    learning_objective: constr(min_length=10, max_length=220)

    # Characters are required by your earlier errors
    characters: List[Character] = Field(min_length=2, max_length=6)

    # Scenes are the structure we generate and stitch
    scenes: List[Scene] = Field(min_length=4, max_length=12)

    # derived sanity info
    total_duration_s: Optional[int] = None
