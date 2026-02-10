from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from .config import BUCKET_SCORE_MAP

_MULTI_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!\?\-_/]+$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def snake_case(value: str) -> str:
    value = value.strip().replace("\u2013", "-")
    value = re.sub(r"[\-/]+", "_", value)
    value = re.sub(r"[\s]+", "_", value)
    value = re.sub(r"[^0-9a-zA-Z_]+", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def standardize_columns(columns: list[str]) -> list[str]:
    standardized: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        base = snake_case(str(col))
        if base in seen:
            seen[base] += 1
            standardized.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            standardized.append(base)
    return standardized


def normalize_skill_text(skill_text: str | None) -> str:
    if skill_text is None:
        return ""
    text = str(skill_text).strip().lower()
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _TRAILING_PUNCT_RE.sub("", text)
    return text


def pretty_skill_label(skill_normalized: str) -> str:
    if not skill_normalized:
        return "Unknown Skill"
    return " ".join(piece.capitalize() for piece in skill_normalized.split(" "))


def normalize_bucket(bucket_raw: str | None) -> tuple[str, int]:
    if bucket_raw is None:
        return "unknown", 0

    text = str(bucket_raw).strip().lower()
    text = text.replace("_", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    if text in BUCKET_SCORE_MAP:
        return text, BUCKET_SCORE_MAP[text]

    stripped = _NON_ALNUM_RE.sub(" ", text)
    stripped = _MULTI_SPACE_RE.sub(" ", stripped).strip()
    if stripped in BUCKET_SCORE_MAP:
        return stripped, BUCKET_SCORE_MAP[stripped]

    if "mission" in stripped or "critical" in stripped:
        return stripped, 4
    if "advanced" in stripped:
        return stripped, 3
    if "proficient" in stripped:
        return stripped, 2
    if "working knowledge" in stripped:
        return stripped, 1
    if "familiar" in stripped:
        return stripped, 0
    return stripped or "unknown", 0


def clip_similarity(value: Any) -> float:
    try:
        parsed = float(value)
        if np.isnan(parsed):
            return float("nan")
        return float(min(1.0, max(0.0, parsed)))
    except (TypeError, ValueError):
        return float("nan")


def compute_edge_weight(
    mapping_similarity: Any,
    bucket_score: int,
    default_similarity: float = 0.5,
) -> float:
    bucket_score_norm = min(4, max(0, int(bucket_score))) / 4.0
    similarity = clip_similarity(mapping_similarity)
    if np.isnan(similarity):
        return round((0.5 * bucket_score_norm) + (0.5 * default_similarity), 6)
    return round((0.7 * similarity) + (0.3 * bucket_score_norm), 6)


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def deterministic_id(prefix: str, raw_value: str, length: int = 16) -> str:
    digest = hashlib.sha1(raw_value.encode("utf-8", errors="ignore")).hexdigest()[:length]
    return f"{prefix}::{digest}"


def deterministic_job_id(
    job_title: str | None,
    company_name: str | None,
    posted_at: str | None,
    row_index: int,
) -> str:
    title = (job_title or "").strip().lower()
    company = (company_name or "").strip().lower()
    posted = (posted_at or "").strip().lower()
    seed = "|".join([title, company, posted])
    if not seed.replace("|", ""):
        seed = f"row_{row_index}"
    return deterministic_id("job_fallback", seed, length=20)
