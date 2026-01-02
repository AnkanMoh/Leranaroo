# core/coherence_agent.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from groq import Groq
from core.theme_packs import ThemePack

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _looks_english(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if _CJK_RE.search(s):
        return False
    ascii_count = sum(1 for ch in s if ord(ch) < 128)
    return (ascii_count / max(1, len(s))) >= 0.92


def _contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


def _safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    # strip ```json fences
    t2 = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t2 = re.sub(r"\s*```$", "", t2).strip()
    try:
        return json.loads(t2)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    return None


@dataclass
class CoherenceConfig:
    api_key: str
    model: str = "llama-3.1-8b-instant"
    max_rewrite_attempts: int = 1


BANNED_WORDS = [
    "kill", "blood", "gun", "knife", "weapon", "murder", "drugs",
    "sex", "nude", "porn", "hate", "terror",
    "yay", "woo", "let's go", "subscribe", "smash that", "epic", "lit",
]


def score_and_issues(
    script: Dict[str, Any],
    *,
    topic: str,
    pack: ThemePack,
    scene_duration: int,
    max_scenes: int
) -> Tuple[int, List[str]]:
    issues: List[str] = []
    beats = script.get("beats") or []

    if len(beats) != max_scenes:
        issues.append(f"Beat count {len(beats)} != required {max_scenes}.")

    allowed = set(pack.cast)
    mascot = pack.cast[0]  # Learnaroo

    mascot_mentions = 0
    topic_hits = 0

    for i, b in enumerate(beats, start=1):
        nar = (b.get("narration") or "").strip()
        lp = (b.get("learning_point") or "").strip()
        cq = (b.get("check_question") or "").strip()
        vp = (b.get("visual_prompt") or "").strip()
        chars = b.get("characters_on_screen") or []

        if not isinstance(chars, list) or not chars:
            issues.append(f"Beat {i}: characters_on_screen missing or not a list.")
        else:
            for c in chars:
                if c not in allowed:
                    issues.append(f"Beat {i}: introduces forbidden character '{c}'. Allowed: {sorted(allowed)}")

        if mascot.lower() in nar.lower():
            mascot_mentions += 1

        if int(b.get("duration_sec", 0)) != scene_duration:
            issues.append(f"Beat {i}: duration_sec must be {scene_duration}.")

        if not lp:
            issues.append(f"Beat {i}: missing learning_point.")
        if not cq:
            issues.append(f"Beat {i}: missing check_question.")
        if not nar:
            issues.append(f"Beat {i}: missing narration.")
        if not vp:
            issues.append(f"Beat {i}: missing visual_prompt.")

        if nar and not _looks_english(nar):
            issues.append(f"Beat {i}: narration not English-only.")
        if _contains_any((nar + " " + vp), BANNED_WORDS):
            issues.append(f"Beat {i}: contains banned/filler/unsafe language.")

        # topic anchoring (simple)
        if topic and nar:
            anchor = topic.lower()
            if "gravity" in anchor:
                if any(k in nar.lower() for k in ["gravity", "mass", "pull", "fall", "attract", "orbit", "weight"]):
                    topic_hits += 1
            else:
                if anchor.split()[0] in nar.lower():
                    topic_hits += 1

        # story_prop must appear in each visual prompt
        key = pack.story_prop.lower().split()[0]
        if key not in vp.lower():
            issues.append(f"Beat {i}: story_prop not clearly present in visual_prompt (missing '{key}').")

        # each beat must name at least one allowed character bible name in visual_prompt
        # (prevents character drift)
        if not any(c.lower() in vp.lower() for c in allowed):
            issues.append(f"Beat {i}: visual_prompt does not clearly name allowed characters.")

    if mascot_mentions < max(6, max_scenes - 2):
        issues.append("Continuity: too few mentions of Learnaroo in narration (character drift risk).")

    if topic_hits < max(6, max_scenes - 2):
        issues.append("Learning: narration not consistently anchored to the topic.")

    if beats:
        last = (beats[-1].get("narration") or "").lower()
        if not any(x in last for x in ["takeaway", "remember", "today we learned", "so now you know"]):
            issues.append("Ending: final beat must include an explicit takeaway line.")

    score = 100
    score -= 8 * sum(1 for x in issues if "forbidden character" in x.lower())
    score -= 6 * sum(1 for x in issues if "not english" in x.lower())
    score -= 4 * (len(issues) - sum(1 for x in issues if "forbidden character" in x.lower()) - sum(1 for x in issues if "not english" in x.lower()))
    score = max(0, min(100, score))
    return score, issues


def rewrite_script_if_needed(
    script: Dict[str, Any],
    *,
    topic: str,
    pack: ThemePack,
    scene_duration: int,
    max_scenes: int,
    cfg: CoherenceConfig,
) -> Dict[str, Any]:
    score, issues = score_and_issues(script, topic=topic, pack=pack, scene_duration=scene_duration, max_scenes=max_scenes)
    if score >= 90 and not issues:
        script["coherence"] = {"score": score, "issues": []}
        return script

    client = Groq(api_key=cfg.api_key)
    last_issues = issues

    for _ in range(cfg.max_rewrite_attempts):
        system = f"""
You are a strict educational script fixer for kids (age 7–9).

HARD RULES:
- Output valid JSON only. No markdown. No commentary.
- English only. No CJK characters.
- ONLY these characters are allowed: {pack.cast}
- Learnaroo MUST appear in EVERY beat (in narration and visual_prompt).
- Keep theme: {pack.theme_name}
- Ensure story_prop in EVERY visual_prompt: {pack.story_prop}
- Each beat must teach ONE clear learning_point.
- Final beat must include a TAKEAWAY sentence that starts with "Takeaway:".

VISUAL STYLE:
{pack.visual_style}

NARRATION STYLE:
{pack.narration_style}

CHARACTER BIBLES (keep consistent):
{json.dumps(pack.character_bibles, ensure_ascii=False)}
""".strip()

        user = f"""
Topic: {topic}

Fix this draft JSON (do not add extra text):
{json.dumps(script, ensure_ascii=False)}

Fix these issues:
{chr(10).join(["- " + x for x in last_issues])}

Return JSON matching this schema exactly:
{{
  "title": "string",
  "beats": [
    {{
      "idx": 1,
      "stage": "string",
      "duration_sec": {scene_duration},
      "characters_on_screen": ["Learnaroo", "..."],
      "story_prop": "{pack.story_prop}",
      "learning_point": "string",
      "check_question": "string",
      "visual_prompt": "string",
      "narration": "string",
      "keywords": ["..."]
    }}
  ]
}}
Beats length must be EXACTLY {max_scenes}.
""".strip()

        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.05,
            max_tokens=2300,
        )

        raw = (resp.choices[0].message.content or "").strip()
        fixed = _safe_extract_json(raw)
        if not fixed:
            last_issues = ["Rewrite failed: model did not return valid JSON."]
            continue

        score2, issues2 = score_and_issues(fixed, topic=topic, pack=pack, scene_duration=scene_duration, max_scenes=max_scenes)
        if score2 >= 90 and not issues2:
            fixed["coherence"] = {"score": score2, "issues": []}
            return fixed

        last_issues = issues2

    fixed_fallback = script.copy()
    fixed_fallback["coherence"] = {"score": score, "issues": issues}
    return fixed_fallback
