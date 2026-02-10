from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from .config import DEFAULT_JOB_COLUMNS
from .normalization import standardize_columns

_SKILL_LIKE_RE = re.compile(
    r'"skill"\s*:|\'skill\'\s*:|"bucket"\s*:|\'bucket\'\s*:|\[\s*\{',
    re.IGNORECASE,
)


@dataclass(slots=True)
class ColumnResolution:
    canonical_to_actual: dict[str, str]
    missing_required: list[str]


def detect_encoding(path: Path, candidates: Iterable[str]) -> str:
    raw = path.read_bytes()[:512_000]
    for encoding in candidates:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _canonical_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in DEFAULT_JOB_COLUMNS.items():
        for alias in aliases:
            alias_map[alias] = canonical
    return alias_map


def resolve_columns(columns: list[str]) -> ColumnResolution:
    alias_map = _canonical_alias_map()
    standardized = standardize_columns(columns)

    canonical_to_actual: dict[str, str] = {}
    for original, normalized in zip(columns, standardized):
        mapped = alias_map.get(normalized)
        if mapped and mapped not in canonical_to_actual:
            canonical_to_actual[mapped] = original

    missing_required = [
        required for required in ("job_title", "skills") if required not in canonical_to_actual
    ]

    return ColumnResolution(canonical_to_actual=canonical_to_actual, missing_required=missing_required)


def standardize_chunk_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    renamed = chunk.copy()
    renamed.columns = standardize_columns([str(c) for c in renamed.columns])
    return renamed


def read_csv_chunks(
    path: Path,
    chunksize: int,
    encoding: str,
) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        chunksize=chunksize,
        encoding=encoding,
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )


def pick_column(chunk: pd.DataFrame, canonical: str, default: Any = None) -> pd.Series:
    aliases = [canonical, *DEFAULT_JOB_COLUMNS.get(canonical, ())]
    normalized_aliases = set(standardize_columns([str(a) for a in aliases]))

    for col in chunk.columns:
        if col in normalized_aliases:
            return chunk[col]
    return pd.Series([default] * len(chunk), index=chunk.index)


def _is_blank_series(series: pd.Series) -> bool:
    if series.empty:
        return True
    text = series.fillna("").astype(str).str.strip()
    return bool(text.eq("").all())


def infer_skills_column(chunk: pd.DataFrame, sample_size: int = 3000) -> str | None:
    best_col: str | None = None
    best_score = 0.0

    for col in chunk.columns:
        series = chunk[col]
        if series.dropna().empty:
            continue

        sampled = series.dropna().astype(str).head(sample_size)
        if sampled.empty:
            continue

        score = sampled.str.contains(_SKILL_LIKE_RE, regex=True, na=False).mean()
        if score > best_score:
            best_score = float(score)
            best_col = col

    if best_col is None:
        return None
    if best_score < 0.2:
        return None
    return best_col


def canonicalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    standardized = standardize_chunk_columns(chunk)
    output = pd.DataFrame(index=standardized.index)
    canonical_fields = sorted(DEFAULT_JOB_COLUMNS.keys())
    for field in canonical_fields:
        output[field] = pick_column(standardized, field, default=None)

    if _is_blank_series(output["skills"]):
        inferred = infer_skills_column(standardized)
        if inferred is not None:
            output["skills"] = standardized[inferred]

    return output
