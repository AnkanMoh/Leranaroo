# core/agents/scriptwriter.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Union

from core.schemas import LessonPlan, Category
from tools.genai_client import GenAIClient


SYSTEM_PROMPT = """
You are a world-class children's educator and storyteller.

STRICT RULES:
- Output MUST be in clear, simple ENGLISH only.
- Audience: 6–9 year old children.
- Tone: fun, theme-based storytelling (based on the chosen category).
- No Chinese, no Hinglish, no mixed languages.
- Return JSON only (no markdown fences).
"""

USER_PROMPT_TEMPLATE = """
Topic: {topic}
Theme / Style: {category}

TASK:
Create a short animated educational story for kids explaining the topic.

STRUCTURE REQUIRED:
- Total duration: ~60–90 seconds
- Split into multiple SCENES
- Each scene has 1–2 BEATS
- Each beat lasts 5–8 seconds

For EACH BEAT provide:
- idx (starting from 1)
- title
- narration (what narrator says, child-friendly ENGLISH)
- visual_prompt (what should be animated visually)
- duration_s (number between 5 and 8)

For EACH SCENE provide:
- idx (starting from 1)
- title
- beats (list of beats)

Ensure:
- Language is ENGLISH
- Content is age-appropriate
- Story flows naturally
- Concept is fully explained by the end

OUTPUT:
Return a JSON object matching the LessonPlan schema.
JSON ONLY. No markdown.
"""


# ----------------------------
# Helpers
# ----------------------------
def _allowed_categories() -> set[str]:
    try:
        args = getattr(Category, "__args__", None)
        if args:
            return {str(x).strip().lower() for x in args if isinstance(x, str)}
    except Exception:
        pass
    return {"superhero", "dinosaur", "space battle", "robot lab", "fairytale"}


def _normalize_category(cat: Union[str, Any]) -> str:
    allowed = _allowed_categories()

    # enum-like: has .value
    try:
        if hasattr(cat, "value"):
            v = getattr(cat, "value")
            if isinstance(v, str):
                c = v.strip().lower()
                if c in allowed:
                    return c
                cat = c
    except Exception:
        pass

    if isinstance(cat, str):
        c = cat.strip().lower()
        if c in allowed:
            return c

        alias = {
            "space": "space battle",
            "spacebattle": "space battle",
            "robot": "robot lab",
            "robots": "robot lab",
            "fairy": "fairytale",
            "fairy tale": "fairytale",
            "dinos": "dinosaur",
            "super hero": "superhero",
            "super-hero": "superhero",
        }
        c2 = alias.get(c, c)
        if c2 in allowed:
            return c2

    return "superhero"


def _looks_english(text: str) -> bool:
    if not text:
        return False
    s = text.strip()
    if len(s) < 40:
        return False
    ascii_ok = sum(1 for ch in s if ord(ch) < 128)
    return (ascii_ok / max(1, len(s))) > 0.92


def _extract_any_text(obj: Any) -> str:
    try:
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return " ".join(_extract_any_text(v) for v in obj.values())
        if isinstance(obj, list):
            return " ".join(_extract_any_text(x) for x in obj)
    except Exception:
        return ""
    return ""


def _strip_code_fences(s: str) -> str:
    # Remove ```json ... ``` or ``` ... ```
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s2)
        s2 = re.sub(r"\s*```$", "", s2)
    return s2.strip()


def _extract_json_object(s: str) -> str:
    """
    If Gemini returns extra text, try to extract the first top-level JSON object.
    """
    s = _strip_code_fences(s)

    # If it's already valid JSON, return as-is
    if s.startswith("{") and s.endswith("}"):
        return s

    # Find first '{' and last '}' to slice out a JSON object.
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()

    # If it's something else (empty or plain text), return original for debugging
    return s


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if raw is None:
        raise RuntimeError("Gemini returned None (no output). Check GEMINI_API_KEY / quota / network.")

    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        # try stringifying
        raw = str(raw)

    txt = raw.strip()
    if not txt:
        raise RuntimeError("Gemini returned an empty string. Check GEMINI_API_KEY / quota / model call errors.")

    candidate = _extract_json_object(txt)

    try:
        return json.loads(candidate)
    except Exception as e:
        raise RuntimeError(
            "Scriptwriter returned non-JSON (or wrapped JSON that couldn't be parsed).\n"
            f"First 500 chars:\n{txt[:500]}\n\n"
            f"Candidate JSON slice (first 500):\n{candidate[:500]}"
        ) from e


def _call_gemini_json(client: GenAIClient, *, system: str, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter: tries multiple possible GenAIClient APIs.
    Always returns a dict, or raises a useful error.
    """
    # Preferred: true structured output methods
    if hasattr(client, "generate_json") and callable(getattr(client, "generate_json")):
        out = client.generate_json(system=system, prompt=prompt, schema=schema)  # type: ignore
        return _safe_json_loads(out)

    if hasattr(client, "generate_structured") and callable(getattr(client, "generate_structured")):
        out = client.generate_structured(system=system, prompt=prompt, schema=schema)  # type: ignore
        return _safe_json_loads(out)

    # Generic generate that might return dict or json-string
    if hasattr(client, "generate") and callable(getattr(client, "generate")):
        out = client.generate(system=system, prompt=prompt, schema=schema)  # type: ignore
        return _safe_json_loads(out)

    # Fallback: plain text generation (often returns ```json fenced output)
    if hasattr(client, "generate_text") and callable(getattr(client, "generate_text")):
        out = client.generate_text(system=system, prompt=prompt)  # type: ignore
        return _safe_json_loads(out)

    # Some older wrappers expose .gemini
    if hasattr(client, "gemini"):
        g = getattr(client, "gemini")
        if hasattr(g, "generate_json") and callable(getattr(g, "generate_json")):
            out = g.generate_json(system=system, prompt=prompt, schema=schema)
            return _safe_json_loads(out)
        if hasattr(g, "generate_text") and callable(getattr(g, "generate_text")):
            out = g.generate_text(system=system, prompt=prompt)
            return _safe_json_loads(out)

    raise AttributeError(
        "GenAIClient has no compatible Gemini method. Expected: "
        "generate_json / generate_structured / generate / generate_text (or gemini.*)."
    )


# ----------------------------
# Main
# ----------------------------
def run(client: GenAIClient, topic: str, category: Union[str, Any]) -> Dict[str, Any]:
    """
    Generates a LessonPlan-compatible dict using Gemini with retries + self-correction.
    """
    category_str = _normalize_category(category)
    schema = LessonPlan.model_json_schema()

    last_err: Exception | None = None

    for attempt in range(3):
        try:
            resp = _call_gemini_json(
                client,
                system=SYSTEM_PROMPT,
                prompt=USER_PROMPT_TEMPLATE.format(topic=topic, category=category_str),
                schema=schema,
            )

            plan = LessonPlan.model_validate(resp)

            total_duration = sum(int(b.duration_s or 0) for s in plan.scenes for b in s.beats)
            if total_duration < 50:
                raise ValueError(f"Story too short ({total_duration}s). Need ~60–90s.")

            text_blob = _extract_any_text(plan.model_dump())
            if not _looks_english(text_blob):
                raise ValueError("Output doesn't look English enough. Regenerating.")

            return plan.model_dump()

        except Exception as e:
            last_err = e
            # try again (Gemini sometimes returns fenced JSON or partial output)
            continue

    raise RuntimeError(f"Script generation failed after retries. Last error: {last_err}") from last_err
