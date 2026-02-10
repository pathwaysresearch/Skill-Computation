from __future__ import annotations

import ast
import json
import re
from typing import Any

from .models import SkillItem

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _extract_array_candidate(text: str) -> str:
    match = _ARRAY_RE.search(text)
    if match:
        return match.group(0)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _normalize_quotes(text: str) -> str:
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\x00", "")
    return text


def _load_candidate(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    return None


def _coerce_list(parsed: Any) -> list[Any] | None:
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return None


def _clean_mapping_similarity(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_fragment_object(fragment: str) -> dict[str, Any] | None:
    parsed = _load_candidate(fragment)
    if isinstance(parsed, dict):
        return parsed

    # Last-resort regex extraction for malformed dict fragments.
    skill_match = re.search(
        r"(?:'|\")?skill(?:'|\")?\s*:\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\}\]]+))",
        fragment,
        re.IGNORECASE,
    )
    bucket_match = re.search(
        r"(?:'|\")?bucket(?:'|\")?\s*:\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\}\]]+))",
        fragment,
        re.IGNORECASE,
    )
    sim_match = re.search(
        r"(?:'|\")?mapping_similarity(?:'|\")?\s*:\s*([0-9]*\.?[0-9]+)",
        fragment,
        re.IGNORECASE,
    )
    if not skill_match:
        return None

    skill = next((g for g in skill_match.groups() if g), "").strip()
    bucket = None
    if bucket_match:
        bucket = next((g for g in bucket_match.groups() if g), "").strip()
    similarity = float(sim_match.group(1)) if sim_match else None

    return {
        "skill": skill,
        "bucket": bucket,
        "mapping_similarity": similarity,
    }


def parse_skills_cell(raw_value: Any, row_id: int) -> tuple[list[SkillItem], str | None]:
    if raw_value is None:
        return [], None

    raw_text = _string(raw_value)
    if not raw_text or raw_text.lower() in {"nan", "none", "null"}:
        return [], None

    normalized = _normalize_quotes(raw_text)
    candidates = [normalized, _extract_array_candidate(normalized)]

    parsed_items: list[Any] | None = None
    for candidate in candidates:
        loaded = _load_candidate(candidate)
        parsed_items = _coerce_list(loaded)
        if parsed_items is not None:
            break

    if parsed_items is None:
        fragments = _OBJECT_RE.findall(normalized)
        recovered: list[dict[str, Any]] = []
        for fragment in fragments:
            parsed = _parse_fragment_object(fragment)
            if parsed:
                recovered.append(parsed)
        if recovered:
            parsed_items = recovered
        else:
            return [], "unable_to_parse_skills"

    results: list[SkillItem] = []
    for item in parsed_items:
        if not isinstance(item, dict):
            continue
        skill_text = _string(item.get("skill") or item.get("name") or item.get("skill_name"))
        if not skill_text:
            continue
        bucket = item.get("bucket")
        bucket_text = _string(bucket) if bucket is not None else None
        similarity = _clean_mapping_similarity(item.get("mapping_similarity"))

        extra = {
            key: value
            for key, value in item.items()
            if key not in {"skill", "name", "skill_name", "bucket", "mapping_similarity"}
        }
        results.append(
            SkillItem(
                skill_text_raw=skill_text,
                bucket_raw=bucket_text,
                mapping_similarity=similarity,
                source_row_index=row_id,
                extra=extra,
            )
        )

    if not results and parsed_items:
        return [], "parsed_but_no_valid_skill_objects"

    return results, None
