from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParseError:
    row_id: int
    reason: str
    raw_skills_snippet: str


@dataclass(slots=True)
class SkillItem:
    skill_text_raw: str
    bucket_raw: str | None
    mapping_similarity: float | None
    source_row_index: int
    extra: dict[str, Any]


@dataclass(slots=True)
class ProfileRecord:
    stage: str
    seconds: float
    details: dict[str, Any]
