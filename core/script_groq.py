from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------
# Public constants expected by core/pipeline.py
# ---------------------------------------------------------------------
SCENE_DURATION = 6
MAX_SCENES = 6  # backward compat; we generate 4 beats

# Seedance-friendly duration bounds
_MIN_DUR = 5
_MAX_DUR = 8


# ---------------------------------------------------------------------
# Config (BACKWARD COMPATIBLE)
# ---------------------------------------------------------------------
@dataclass
class GroqScriptConfig:
    api_key: Optional[str] = None

    # You can override at runtime:
    # export GROQ_MODEL="llama-3.3-70b-versatile"
    # export GROQ_CREATIVE_MODEL="llama-3.3-70b-versatile"
    model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    creative_model: str = os.getenv("GROQ_CREATIVE_MODEL", "llama-3.3-70b-versatile")

    temperature: float = 0.7
    creative_temperature: float = 0.95

    max_tokens: int = 1900
    timeout_s: int = 60
    retries: int = 3

    # 4 scenes MVP
    min_scenes: int = 4
    max_scenes: int = 4
    scene_duration_s: int = SCENE_DURATION

    age_range: str = "7-9"
    english_only: bool = True
    strict_json: bool = True

    # narration rhythm (for ~6s macOS TTS)
    min_words: int = 14
    max_words: int = 26
    max_sentences: int = 4


# ---------------------------------------------------------------------
# Soft banned/filler removal + hard forbidden starters
# ---------------------------------------------------------------------
_BANNED_PHRASES = [
    "let's dive in", "lets dive in",
    "in conclusion", "to sum up",
    "as an ai", "as a language model",
    "um", "uh", "you know", "basically", "actually", "literally",
    "subscribe", "like and", "smash that", "click the bell",
    "welcome back", "hello kids",
]
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in _BANNED_PHRASES) + r")\b", re.IGNORECASE)

_FORBIDDEN_STARTERS: Tuple[str, ...] = (
    "now", "let's", "lets", "here we", "today we", "in this lesson",
    "we will", "we are going to", "i will", "let me",
    "as you can see", "this shows", "we can see", "welcome",
)
_FORBIDDEN_START_RE = re.compile(
    r"^\s*(?:[\"'“”‘’\-–—\(\)\[\]\s]+)?("
    + "|".join(re.escape(x) for x in _FORBIDDEN_STARTERS)
    + r")\b[\s,:\-–—]+",
    re.IGNORECASE,
)

# Beat keys your pipeline expects downstream
_BEAT_KEYS = {"idx", "title", "visual_prompt", "narration", "duration_s"}

# meta/lesson-plan smell
_GENERIC_SMELL = re.compile(
    r"\b(learn|lesson|today|now|we will|let'?s|here we|in this|explain|definition|means)\b",
    re.IGNORECASE,
)

# A light “real moment” cue list (we want action-y words early)
_ACTION_OPENERS = re.compile(
    r"^\s*(Thump|Crash|Whoosh|Pop|Zap|Boom|Swoosh|Clink|Plop|Tap|Snap|Wham|Whirr|Spill|Drop|Bounce|Dash|Spin|Flip|Grab|Catch|Kick|Pull|Push)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Groq client (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------
def _groq_chat(*, messages: List[Dict[str, str]], cfg: GroqScriptConfig, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    api_key = os.getenv("GROQ_API_KEY") or cfg.api_key
    if not api_key:
        return ""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": model or cfg.model,
        "messages": messages,
        "temperature": cfg.temperature if temperature is None else float(temperature),
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    if cfg.strict_json:
        payload["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
        if r.status_code != 200:
            return ""
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_object(s: str) -> Optional[str]:
    s = _strip_code_fences(s)
    if not s:
        return None
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1].strip()
    return None


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    blob = _extract_first_object(s)
    if not blob:
        return None
    blob = _strip_code_fences(blob)

    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fixed = blob.replace("“", '"').replace("”", '"').replace("’", "'")
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    if "'" in fixed and fixed.count('"') < 2:
        fixed = re.sub(r"'", '"', fixed)

    try:
        obj = json.loads(fixed)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------
# Normalization + narration quality guards
# ---------------------------------------------------------------------
def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = _BANNED_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r"\?{2,}", "?", s)
    return s.strip()


def _clamp_duration(d: Any, default: int) -> int:
    try:
        x = int(d)
    except Exception:
        x = int(default)
    return max(_MIN_DUR, min(_MAX_DUR, x))


def _basic_english_heuristic(text: str) -> bool:
    if not text:
        return True
    bad = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        if 32 <= o <= 126:
            continue
        if 160 <= o <= 255:
            continue
        bad += 1
    if total == 0:
        return True
    return (bad / total) < 0.03


def _words(s: str) -> List[str]:
    return [w for w in re.split(r"\s+", (s or "").strip()) if w]


def _count_sentences(s: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?]+", (s or "").strip()) if p.strip()]
    return len(parts)


def _starts_with_forbidden(narr: str) -> bool:
    return bool(_FORBIDDEN_START_RE.search((narr or "").strip()))


def _strip_forbidden_starter(narr: str) -> str:
    n = (narr or "").strip()
    n = _FORBIDDEN_START_RE.sub("", n).strip()
    return n


def _too_generic(narr: str) -> bool:
    n = (narr or "").strip()
    if len(_words(n)) < 10:
        return True
    smells = len(_GENERIC_SMELL.findall(n))
    return smells >= 2


def _ensure_action_opener(narr: str) -> str:
    """
    If narration doesn't start with an action-y beat, nudge it.
    We do NOT hard-force a sound effect, but we prefer it.
    """
    n = (narr or "").strip()
    if not n:
        return n
    if _ACTION_OPENERS.search(n):
        return n
    # If starts with a pronoun/name, prepend a quick action cue.
    return _clean_text(f"Whoosh! {n}")


def _enforce_word_window(narr: str, *, min_words: int, max_words: int) -> str:
    narr = _clean_text(narr)
    narr = _strip_forbidden_starter(narr)
    ws = _words(narr)

    if len(ws) > max_words:
        narr = " ".join(ws[:max_words]).strip()
        if narr and narr[-1] not in ".!?":
            narr += "."
    return narr


def _normalize_lesson_points(obj: Any, topic: str) -> List[Dict[str, str]]:
    t = topic.strip() or "this topic"

    if not isinstance(obj, list):
        return [
            {"concept": f"Big idea #1 about {t}", "kid_example": f"A quick test for {t}", "common_mistake": f"A common mix-up in {t}"},
            {"concept": f"Big idea #2 about {t}", "kid_example": f"Compare two outcomes in {t}", "common_mistake": f"Guessing instead of checking in {t}"},
            {"concept": f"Big idea #3 about {t}", "kid_example": f"Use evidence to explain {t}", "common_mistake": f"Believing a myth about {t}"},
        ]

    out: List[Dict[str, str]] = []
    for it in obj[:3]:
        if not isinstance(it, dict):
            continue
        concept = _clean_text(str(it.get("concept", "")))
        kid_example = _clean_text(str(it.get("kid_example", "")))
        common_mistake = _clean_text(str(it.get("common_mistake", "")))

        if not concept or _too_generic(concept):
            concept = f"Big idea about {t}"
        if not kid_example:
            kid_example = f"A simple test for {t}"
        if not common_mistake:
            common_mistake = f"A common confusion about {t}"

        out.append({"concept": concept, "kid_example": kid_example, "common_mistake": common_mistake})

    while len(out) < 3:
        out.append({"concept": f"Big idea about {t}", "kid_example": f"Example of {t}", "common_mistake": f"Common confusion about {t}"})
    return out[:3]


def _ensure_takeaway(beats: List[Dict[str, Any]], lesson_points: List[Dict[str, str]]) -> None:
    if not beats:
        return
    last = beats[-1]
    narr = (last.get("narration") or "").strip()

    lp = lesson_points[:3]
    c1 = (lp[0].get("concept") or "").strip()
    c2 = (lp[1].get("concept") or "").strip()
    c3 = (lp[2].get("concept") or "").strip()

    desired = (
        f"TAKEAWAY: First, {c1}. Second, {c2}. Third, {c3}."
        if (c1 and c2 and c3)
        else "TAKEAWAY: First, spot the cause. Second, test your idea. Third, trust the evidence."
    )

    if "TAKEAWAY:" not in narr.upper():
        last["narration"] = _clean_text((narr + " " + desired).strip() if narr else desired)
        return

    if not (re.search(r"\bFirst\b", narr, re.I) and re.search(r"\bSecond\b", narr, re.I) and re.search(r"\bThird\b", narr, re.I)):
        parts = re.split(r"(?i)\bTAKEAWAY:\b", narr, maxsplit=1)
        prefix = parts[0].strip()
        last["narration"] = _clean_text((prefix + " " + desired).strip() if prefix else desired)


def _fallback_beats(*, topic: str, theme: str, character_name: str, lesson_points: List[Dict[str, str]], dur: int) -> List[Dict[str, Any]]:
    dur = _clamp_duration(dur, SCENE_DURATION)
    lp = lesson_points[:3]
    t = topic.strip() or "this mystery"
    prop = "a surprising clue"

    beats: List[Dict[str, Any]] = [
        {
            "idx": 1,
            "title": "Hook + Mission",
            "visual_prompt": f"{theme} 2D kid cartoon. {character_name} sees {prop} about {t} while a curious kid points. Fast action, full scene. No text overlays.",
            "narration": _clean_text(
                f"Thump! {character_name} spots {prop}, and a curious kid points—why is {t} acting so weird today?"
            ),
            "duration_s": dur,
        },
        {
            "idx": 2,
            "title": "Lesson 1",
            "visual_prompt": f"{theme} 2D kid cartoon. Same {character_name}. Mini experiment showing: {lp[0]['kid_example']}. No text overlays.",
            "narration": _clean_text(
                f"Whoosh! {character_name} tests it—{lp[0]['kid_example']}. The kid laughs. First discovery: {lp[0]['concept']}."
            ),
            "duration_s": dur,
        },
        {
            "idx": 3,
            "title": "Lesson 2",
            "visual_prompt": f"{theme} 2D kid cartoon. Same {character_name}. Compare outcomes: {lp[1]['kid_example']}. No text overlays.",
            "narration": _clean_text(
                f"Zap! {character_name} tries two ways—{lp[1]['kid_example']}. One works, one flops. Second discovery: {lp[1]['concept']}."
            ),
            "duration_s": dur,
        },
        {
            "idx": 4,
            "title": "Lesson 3 + TAKEAWAY",
            "visual_prompt": f"{theme} 2D kid cartoon. Same {character_name}. Victory moment using: {lp[2]['kid_example']}. No text overlays.",
            "narration": _clean_text(
                f"Boom! {character_name} proves it with {lp[2]['kid_example']}. TAKEAWAY: First, {lp[0]['concept']}. Second, {lp[1]['concept']}. Third, {lp[2]['concept']}."
            ),
            "duration_s": dur,
        },
    ]
    _ensure_takeaway(beats, lp)
    return beats


def _needs_rewrite_beat(narr: str, *, cfg: GroqScriptConfig, beat_idx: int) -> bool:
    n = _clean_text(narr)
    if not n:
        return True
    if _starts_with_forbidden(n):
        return True

    ws = _words(n)
    if len(ws) < max(10, cfg.min_words - 2):
        return True
    if len(ws) > cfg.max_words + 6:
        return True
    if _count_sentences(n) > cfg.max_sentences + 1:
        return True
    if _too_generic(n):
        return True
    if beat_idx == 1 and "?" not in n:
        return True
    if beat_idx == 4 and "TAKEAWAY" not in n.upper():
        return True
    return False


# ---------------------------------------------------------------------
# Two-pass generation: (A) story blueprint (creative) -> (B) strict script JSON
# ---------------------------------------------------------------------
def _generate_story_blueprint(*, topic: str, theme: str, character_name: str, cfg: GroqScriptConfig) -> Optional[Dict[str, Any]]:
    """
    A small creative blueprint so Groq doesn't default to lesson-plan voice.
    MUST be topic-agnostic: create a real-world moment + simple experiments.
    """
    system = (
        "You are a creative kids cartoon writer.\n"
        "Return ONLY JSON. No markdown.\n"
        "Your job: invent ONE real-world moment that sparks curiosity about the TOPIC.\n"
        "Keep it kid-friendly, visual, and grounded.\n"
        "Allowed extra character: an unnamed 'curious kid' (NO other named characters).\n"
        "Avoid meta lesson talk.\n"
    )
    user = (
        f"TOPIC: {topic}\nTHEME: {theme}\nMAIN HERO (only named character): {character_name}\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "setup_moment":"<a real-world moment that happens instantly, like a drop/spill/bounce/surprise>",\n'
        '  "curiosity_question":"<a kid asks a why/how question about the topic>",\n'
        '  "experiments":[\n'
        '     "<simple test #1 the hero tries>",\n'
        '     "<simple test #2 the hero tries>",\n'
        '     "<simple test #3 / proof moment>"\n'
        "  ],\n"
        '  "props":["<3-6 concrete props/objects seen on screen>"]\n'
        "}\n"
        "Rules:\n"
        "- setup_moment must be ACTION, not explanation\n"
        "- experiments must be physical and visual\n"
        "- no definitions, no 'lesson' words\n"
        "- no extra names besides the hero\n"
    )

    raw = _groq_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        model=cfg.creative_model,
        temperature=cfg.creative_temperature,
    )
    parsed = _try_parse_json(raw)
    if not parsed:
        return None

    # quick sanity
    if not isinstance(parsed.get("experiments"), list) or len(parsed["experiments"]) < 2:
        return None
    return parsed


def _generate_strict_script(*, topic: str, theme: str, character_name: str, blueprint: Dict[str, Any], cfg: GroqScriptConfig) -> str:
    """
    Final strict JSON generation, grounded by blueprint.
    """
    system = (
        "You write kid-friendly mini-stories for text-to-speech as STRICT JSON.\n"
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Narration style: cartoon narrator, ACTION-FIRST, story-driven.\n"
        "BANNED starters: Now/Let's/Here we/Today we/We will/In this lesson.\n"
        "BANNED tone: textbook definitions, robotic repetition.\n"
        "Mission must feel continuous across all 4 beats.\n"
        "Allowed extra character: 'a curious kid' (unnamed). No other names.\n"
        "Beat 1 ends with a curiosity question.\n"
        "Beat 4 includes TAKEAWAY and recaps 3 lessons using First/Second/Third.\n"
    )

    # Provide blueprint as grounding
    safe_blueprint = json.dumps(blueprint, ensure_ascii=False)

    user = (
        f"TOPIC: {topic}\n"
        f"THEME: {theme}\n"
        f"HERO NAME (only named character): {character_name}\n\n"
        f"STORY BLUEPRINT (use this, do not ignore):\n{safe_blueprint}\n\n"
        "Return ONLY JSON with EXACT schema:\n"
        "{\n"
        '  "lesson_points": [\n'
        '    {"concept":"...","kid_example":"...","common_mistake":"..."},\n'
        '    {"concept":"...","kid_example":"...","common_mistake":"..."},\n'
        '    {"concept":"...","kid_example":"...","common_mistake":"..."}\n'
        "  ],\n"
        '  "beats": [\n'
        '    {"idx":1,"title":"Hook + Mission","visual_prompt":"...","narration":"... ?","duration_s":6},\n'
        '    {"idx":2,"title":"Lesson 1","visual_prompt":"...","narration":"...","duration_s":6},\n'
        '    {"idx":3,"title":"Lesson 2","visual_prompt":"...","narration":"...","duration_s":6},\n'
        '    {"idx":4,"title":"Lesson 3 + TAKEAWAY","visual_prompt":"...","narration":"... TAKEAWAY: First..., Second..., Third...","duration_s":6}\n'
        "  ]\n"
        "}\n\n"
        "HARD narration rules:\n"
        f"- {cfg.min_words}–{cfg.max_words} words each\n"
        f"- 1–{cfg.max_sentences} short sentences each\n"
        "- Start with an action moment (sound effect / action verb)\n"
        "- Describe ACTION + DISCOVERY (not explaining)\n"
        "- Beat 1 includes the curiosity question\n"
        "- Beat 4 TAKEAWAY with First/Second/Third referencing the 3 lesson_points\n\n"
        "HARD visual rules:\n"
        "- 2D kid cartoon, full scene, motion\n"
        "- no text overlays/logos/captions\n"
        "- include props from blueprint\n"
    )

    return _groq_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        model=cfg.model,
        temperature=cfg.temperature,
    )


def _rewrite_narrations_only(
    *,
    topic: str,
    theme: str,
    character_name: str,
    lesson_points: List[Dict[str, str]],
    beats: List[Dict[str, Any]],
    cfg: GroqScriptConfig,
    reason: str,
) -> Optional[Dict[str, Any]]:
    safe_in = json.dumps({"lesson_points": lesson_points[:3], "beats": beats}, ensure_ascii=False)

    system = (
        "You are a STRICT JSON editor that rewrites narration for kid-friendly cartoon TTS.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Rewrite narrations to be ACTION-FIRST, story-driven, vivid, spoken-friendly.\n"
        "BANNED starters: Now/Let's/Here we/Today we/We will/In this lesson.\n"
        "Allowed extra character: 'a curious kid' (unnamed). No other names.\n"
        "Keep visuals/titles/durations unchanged.\n"
        "Beat 1 ends with a curiosity question.\n"
        "Beat 4 includes TAKEAWAY with First/Second/Third.\n"
    )

    user = (
        f"TOPIC: {topic}\nTHEME: {theme}\nHERO: {character_name}\nREASON: {reason}\n\n"
        "Rewrite narrations ONLY. Keep JSON shape.\n"
        f"Each narration: {cfg.min_words}–{cfg.max_words} words, 1–{cfg.max_sentences} short sentences.\n"
        "Start with action (sound effect/action verb). No meta talk.\n\n"
        f"JSON TO EDIT:\n{safe_in}"
    )

    raw = _groq_chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        cfg=cfg,
        model=cfg.model,
        temperature=cfg.temperature,
    )
    return _try_parse_json(raw)


def _normalize_to_output(
    obj: Dict[str, Any],
    *,
    topic: str,
    theme: str,
    character_name: str,
    cfg: GroqScriptConfig,
) -> Dict[str, Any]:
    lesson_points = _normalize_lesson_points(obj.get("lesson_points"), topic)

    beats = obj.get("beats")
    scenes = obj.get("scenes")
    if beats is None and isinstance(scenes, list):
        beats = scenes

    if not isinstance(beats, list) or not beats:
        beats = _fallback_beats(topic=topic, theme=theme, character_name=character_name, lesson_points=lesson_points, dur=cfg.scene_duration_s)
        return {"lesson_points": lesson_points, "beats": beats, "scenes": beats}

    beats = beats[:4]

    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(beats, start=1):
        raw = raw if isinstance(raw, dict) else {}

        title = _clean_text(str(raw.get("title") or f"Beat {i}"))
        visual = _clean_text(str(raw.get("visual_prompt") or raw.get("image_prompt") or ""))
        narration = _clean_text(str(raw.get("narration") or raw.get("voiceover") or ""))

        dur = _clamp_duration(raw.get("duration_s") or cfg.scene_duration_s, cfg.scene_duration_s)

        if not visual:
            visual = _clean_text(f"{theme} 2D kid cartoon. Same {character_name}. Full scene with motion about {topic}. No text overlays.")

        if cfg.english_only and not _basic_english_heuristic(narration):
            narration = _clean_text(f"Whoosh! {character_name} tackles a clue about {topic} and reacts with a surprised grin!")

        narration = _strip_forbidden_starter(narration)
        narration = _ensure_action_opener(narration)
        narration = _enforce_word_window(narration, min_words=cfg.min_words, max_words=cfg.max_words + 6)

        out.append({"idx": i, "title": title, "visual_prompt": visual, "narration": narration, "duration_s": dur})

    _ensure_takeaway(out, lesson_points)
    return {"lesson_points": lesson_points, "beats": out, "scenes": out}


def _soft_validate(script: Dict[str, Any], *, cfg: GroqScriptConfig) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    lp = script.get("lesson_points")
    if not isinstance(lp, list) or len(lp) < 3:
        issues.append("Missing lesson_points (need 3)")

    beats = script.get("beats")
    if not isinstance(beats, list) or len(beats) != 4:
        issues.append("beats must be exactly 4")

    if not beats:
        return False, ["Missing/empty beats"]

    for b in beats:
        if not isinstance(b, dict):
            issues.append("Beat not dict")
            continue

        missing = _BEAT_KEYS - set(b.keys())
        if missing:
            issues.append(f"Missing keys: {sorted(missing)}")

        d = b.get("duration_s", cfg.scene_duration_s)
        if not isinstance(d, int) or d < _MIN_DUR or d > _MAX_DUR:
            issues.append("Bad duration_s")

        idx = int(b.get("idx") or 0) or 1
        narr = (b.get("narration") or "").strip()

        if _needs_rewrite_beat(narr, cfg=cfg, beat_idx=idx):
            issues.append(f"Beat {idx} narration needs rewrite")

        # Prefer action-y start
        if narr and not _ACTION_OPENERS.search(narr):
            issues.append(f"Beat {idx} lacks action opener")

    last = (beats[-1].get("narration") or "")
    if "TAKEAWAY:" not in last.upper():
        issues.append("Missing TAKEAWAY")

    if not (re.search(r"\bFirst\b", last, re.I) and re.search(r"\bSecond\b", last, re.I) and re.search(r"\bThird\b", last, re.I)):
        issues.append("TAKEAWAY missing First/Second/Third recap")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------
# PUBLIC FUNCTION
# ---------------------------------------------------------------------
def generate_valid_script(
    cfg: GroqScriptConfig,
    *,
    topic: str,
    theme: Optional[str] = None,
    theme_pack: Optional[Dict[str, Any]] = None,
    character_name: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    theme_name = ""
    if isinstance(theme_pack, dict):
        theme_name = str(theme_pack.get("name") or theme_pack.get("theme") or "").strip()
    if not theme_name:
        theme_name = str(theme or "Theme").strip()

    hero = (character_name or "").strip() or "Suptain Comet"

    # --- Attempt loop ---
    last_raw = ""
    for attempt in range(1, max(1, cfg.retries) + 1):
        # Pass A: blueprint (creative)
        blueprint = _generate_story_blueprint(topic=topic, theme=theme_name, character_name=hero, cfg=cfg)
        if not blueprint:
            blueprint = {
                "setup_moment": "Thump! Something surprising happens near a tree and a curious kid reacts fast.",
                "curiosity_question": f"Why is {topic} behaving like that?",
                "experiments": [f"Try a quick test related to {topic}", f"Compare two outcomes about {topic}", f"Prove it with a simple visual clue about {topic}"],
                "props": ["tree", "ball", "leaf", "clue-card", "magnifying glass"],
            }

        # Pass B: strict script
        last_raw = _generate_strict_script(topic=topic, theme=theme_name, character_name=hero, blueprint=blueprint, cfg=cfg)

        parsed = _try_parse_json(last_raw)
        if parsed is None:
            # Hard fallback parse failure -> fallback beats
            lp = _normalize_lesson_points(None, topic)
            beats = _fallback_beats(topic=topic, theme=theme_name, character_name=hero, lesson_points=lp, dur=cfg.scene_duration_s)
            return {"lesson_points": lp, "beats": beats, "scenes": beats}

        normalized = _normalize_to_output(parsed, topic=topic, theme=theme_name, character_name=hero, cfg=cfg)

        ok, issues = _soft_validate(normalized, cfg=cfg)
        if ok:
            # final tighten pass
            beats = normalized["beats"]
            for b in beats:
                b["narration"] = _strip_forbidden_starter(_clean_text(b.get("narration", "")))
                b["narration"] = _ensure_action_opener(b["narration"])
                b["narration"] = _enforce_word_window(b["narration"], min_words=cfg.min_words, max_words=cfg.max_words)
            _ensure_takeaway(beats, normalized["lesson_points"])
            normalized["scenes"] = beats
            return normalized

        # narration-only rewrite if needed
        rewritten = _rewrite_narrations_only(
            topic=topic,
            theme=theme_name,
            character_name=hero,
            lesson_points=normalized.get("lesson_points") or _normalize_lesson_points(None, topic),
            beats=normalized.get("beats") or [],
            cfg=cfg,
            reason="; ".join(issues[:8]) or "narration quality",
        )
        if rewritten:
            normalized2 = _normalize_to_output(rewritten, topic=topic, theme=theme_name, character_name=hero, cfg=cfg)
            ok2, issues2 = _soft_validate(normalized2, cfg=cfg)
            if ok2:
                beats2 = normalized2["beats"]
                for b in beats2:
                    b["narration"] = _strip_forbidden_starter(_clean_text(b.get("narration", "")))
                    b["narration"] = _ensure_action_opener(b["narration"])
                    b["narration"] = _enforce_word_window(b["narration"], min_words=cfg.min_words, max_words=cfg.max_words)
                _ensure_takeaway(beats2, normalized2["lesson_points"])
                normalized2["scenes"] = beats2
                return normalized2

        time.sleep(min(0.6 * attempt, 2.0))

    # Final fallback never breaks pipeline
    lp = _normalize_lesson_points(None, topic)
    beats = _fallback_beats(topic=topic, theme=theme_name, character_name=hero, lesson_points=lp, dur=cfg.scene_duration_s)
    for b in beats:
        b["narration"] = _strip_forbidden_starter(_clean_text(b.get("narration", "")))
        b["narration"] = _ensure_action_opener(b["narration"])
        b["narration"] = _enforce_word_window(b["narration"], min_words=cfg.min_words, max_words=cfg.max_words)
    _ensure_takeaway(beats, lp)
    return {"lesson_points": lp, "beats": beats, "scenes": beats}
