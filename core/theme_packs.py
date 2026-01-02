# core/theme_packs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ThemePack:
    theme_name: str
    # Characters allowed for this theme (script must never introduce others)
    cast: List[str]
    # Canonical descriptions for consistent looks
    character_bibles: Dict[str, str]
    # A recurring prop that ties the whole story together
    story_prop: str
    # Global visual style directive
    visual_style: str
    # Tone directive
    narration_style: str


def get_theme_pack(theme: str, topic: str) -> ThemePack:
    """
    Returns a ThemePack. Cast + styles adapt to theme.
    You can extend this without touching the rest of the pipeline.
    """

    t = (theme or "").strip().lower()
    topic_l = (topic or "").strip().lower()

    # Core mascot ALWAYS present
    mascot = "Learnaroo"
    mascot_bible = (
        "Learnaroo: an animated kangaroo teacher mascot with big expressive eyes, "
        "a small backpack, calm teacher-like gestures, and the SAME look in every scene."
    )

    # If topic is gravity, we prefer Gravity Girl + Newton Boy in superhero theme
    gravity_bonus_cast = ["Gravity Girl", "Newton Boy"] if "gravity" in topic_l else ["Science Sidekick", "Newton Boy"]

    if "superhero" in t:
        cast = [mascot] + gravity_bonus_cast
        character_bibles = {
            mascot: mascot_bible + " Learnaroo wears a bright superhero cape and a lightning badge.",
            "Gravity Girl": (
                "Gravity Girl: a friendly kid superhero with a purple cape, a tiny planet badge, "
                "and a confident teacher vibe. Always looks the same."
            ),
            "Newton Boy": (
                "Newton Boy: a clever kid inventor with round glasses, a small notebook, "
                "and a calm ‘explain-it’ style. Always looks the same."
            ),
            "Science Sidekick": (
                "Science Sidekick: a cheerful helper kid with a utility belt of science tools, "
                "always looks the same."
            ),
        }
        story_prop = "a glowing gravity badge that tugs gently toward the ground"
        visual_style = (
            "Simple 2D kids cartoon, flat colors, clean shapes, gentle motion. "
            "Same character design every scene. Minimal text."
        )
        narration_style = "Teacher-like, warm, curious, 1–2 short sentences per beat. English only."
        return ThemePack(theme_name=theme, cast=cast, character_bibles=character_bibles, story_prop=story_prop,
                         visual_style=visual_style, narration_style=narration_style)

    if "dinosaur" in t:
        cast = [mascot, "Dino Doc", "Explorer Newton"]
        character_bibles = {
            mascot: mascot_bible + " Learnaroo wears a safari hat and a small compass badge.",
            "Dino Doc": (
                "Dino Doc: a friendly baby dinosaur scientist wearing a tiny lab coat and goggles. "
                "Cute, not scary. Always looks the same."
            ),
            "Explorer Newton": (
                "Explorer Newton: a kid explorer with a notebook and magnifying glass; calm explainer. "
                "Always looks the same."
            ),
        }
        story_prop = "a bouncy ‘gravity rock’ that always drops straight down"
        visual_style = (
            "Simple 2D kids cartoon, bright jungle colors, clean shapes, gentle motion. "
            "No scary elements. Minimal text."
        )
        narration_style = "Story-first, learning-focused, English only."
        return ThemePack(theme_name=theme, cast=cast, character_bibles=character_bibles, story_prop=story_prop,
                         visual_style=visual_style, narration_style=narration_style)

    if "space" in t:
        cast = [mascot, "Captain Orbit", "Engineer Newton"]
        character_bibles = {
            mascot: mascot_bible + " Learnaroo wears a small space helmet and star patch.",
            "Captain Orbit": (
                "Captain Orbit: a brave kid astronaut with a navy suit and a ringed-planet logo. "
                "Always looks the same."
            ),
            "Engineer Newton": (
                "Engineer Newton: a kid engineer with a tool belt and tablet; calm explainer. "
                "Always looks the same."
            ),
        }
        story_prop = "a tiny planet marble that pulls objects into orbit paths"
        visual_style = "Simple 2D kids cartoon, space colors, clean shapes, gentle motion. Minimal text."
        narration_style = "Clear, curious, English only."
        return ThemePack(theme_name=theme, cast=cast, character_bibles=character_bibles, story_prop=story_prop,
                         visual_style=visual_style, narration_style=narration_style)

    if "robot" in t:
        cast = [mascot, "Robo Buddy", "Newton Engineer"]
        character_bibles = {
            mascot: mascot_bible + " Learnaroo wears a small lab apron and safety goggles.",
            "Robo Buddy": (
                "Robo Buddy: a cute round robot with a screen face showing friendly expressions. "
                "Always looks the same."
            ),
            "Newton Engineer": (
                "Newton Engineer: a kid engineer with a wrench icon badge; calm explainer. "
                "Always looks the same."
            ),
        }
        story_prop = "a magnet-tether demo ball that shows ‘pull’ visually"
        visual_style = "Simple 2D kids cartoon, lab setting, clean shapes, gentle motion. Minimal text."
        narration_style = "Step-by-step, English only."
        return ThemePack(theme_name=theme, cast=cast, character_bibles=character_bibles, story_prop=story_prop,
                         visual_style=visual_style, narration_style=narration_style)

    # Default: fairytale / storybook
    cast = [mascot, "Wise Newton", "Fairy Gravity"]
    character_bibles = {
        mascot: mascot_bible + " Learnaroo wears a tiny storybook cape and a star pin.",
        "Wise Newton": (
            "Wise Newton: a gentle storyteller kid with a scroll and glasses; calm explainer. "
            "Always looks the same."
        ),
        "Fairy Gravity": (
            "Fairy Gravity: a friendly fairy with a small planet wand; not scary; always looks the same."
        ),
    }
    story_prop = "a magic feather that still falls because of gravity"
    visual_style = "Simple 2D kids cartoon, storybook vibe, clean shapes, gentle motion. Minimal text."
    narration_style = "Warm, storybook, English only."
    return ThemePack(theme_name=theme, cast=cast, character_bibles=character_bibles, story_prop=story_prop,
                     visual_style=visual_style, narration_style=narration_style)
