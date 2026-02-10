from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .config import PipelineConfig
from .ingest import canonicalize_chunk, detect_encoding, read_csv_chunks
from .normalization import normalize_skill_text, parse_numeric
from .skill_parser import parse_skills_cell

ProgressFn = Callable[[str, float], None] | None


@dataclass(slots=True)
class SubsetResult:
    sample_input_path: Path
    selected_rows: int
    total_rows: int
    target_rows: int
    summary: dict[str, Any]



def _progress(progress_fn: ProgressFn, message: str, fraction: float) -> None:
    if progress_fn:
        progress_fn(message, max(0.0, min(1.0, fraction)))



def _hash01(seed: int, row_idx: int) -> float:
    digest = hashlib.sha1(f"{seed}:{row_idx}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF



def _clean_str(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"



def _salary_band(value: Any) -> str:
    parsed = parse_numeric(value)
    if parsed is None:
        return "unknown"
    if parsed < 10_000:
        return "lt_10k"
    if parsed < 20_000:
        return "10k_20k"
    if parsed < 50_000:
        return "20k_50k"
    return "ge_50k"



def _stratum_key(row: dict[str, Any]) -> str:
    occupation = _clean_str(row.get("assigned_occupation_group"))
    if occupation == "unknown":
        occupation = _clean_str(row.get("group"))
    geography = _clean_str(row.get("district"))
    salary_band = _salary_band(row.get("salary_mean_inr_month"))
    return f"{occupation}||{geography}||{salary_band}"



def _row_size_estimate(row: dict[str, Any]) -> int:
    return sum(len(str(v)) for v in row.values() if v is not None) + max(1, len(row) - 1)



def _compute_quotas(stratum_counts: dict[str, int], target_rows: int) -> dict[str, int]:
    total_rows = sum(stratum_counts.values())
    if total_rows == 0 or target_rows <= 0:
        return {key: 0 for key in stratum_counts}

    quota_float = {k: (count / total_rows) * target_rows for k, count in stratum_counts.items()}
    quotas = {k: int(v) for k, v in quota_float.items()}
    assigned = sum(quotas.values())

    remainder = target_rows - assigned
    if remainder > 0:
        ranked = sorted(
            quota_float.items(),
            key=lambda kv: (kv[1] - int(kv[1]), stratum_counts[kv[0]], kv[0]),
            reverse=True,
        )
        for stratum, _ in ranked[:remainder]:
            quotas[stratum] += 1

    return quotas



def create_deterministic_subset(
    config: PipelineConfig,
    output_dir: Path,
    progress_fn: ProgressFn = None,
) -> SubsetResult:
    input_csv = Path(config.input_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_input_path = output_dir / "sample_input.csv"
    if sample_input_path.exists():
        sample_input_path.unlink()

    encoding = detect_encoding(input_csv, config.encoding_candidates)
    _progress(progress_fn, f"Subset pass 1/3: profiling source ({encoding})", 0.02)

    stratum_counts: dict[str, int] = {}
    row_metadata: list[tuple[int, str]] = []
    skill_freq: Counter[str] = Counter()
    total_rows = 0
    total_size_estimate = 0

    row_offset = 0
    for chunk in read_csv_chunks(input_csv, chunksize=config.chunksize, encoding=encoding):
        canonical = canonicalize_chunk(chunk)
        records = canonical.to_dict(orient="records")
        for idx, row in enumerate(records):
            row_idx = row_offset + idx
            stratum = _stratum_key(row)
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
            row_metadata.append((row_idx, stratum))
            total_size_estimate += _row_size_estimate(row)

            parsed_skills, _ = parse_skills_cell(row.get("skills"), row_idx)
            unique_skills = {
                normalize_skill_text(item.skill_text_raw)
                for item in parsed_skills
                if normalize_skill_text(item.skill_text_raw)
            }
            skill_freq.update(unique_skills)

        row_offset += len(canonical)
        total_rows += len(canonical)

    if total_rows == 0:
        empty = pd.DataFrame()
        empty.to_csv(sample_input_path, index=False)
        return SubsetResult(
            sample_input_path=sample_input_path,
            selected_rows=0,
            total_rows=0,
            target_rows=0,
            summary={"message": "empty input"},
        )

    avg_row_bytes = max(1, int(total_size_estimate / total_rows))
    if config.subset_target_rows and config.subset_target_rows > 0:
        target_rows = min(total_rows, config.subset_target_rows)
    else:
        target_bytes = int(config.subset_target_size_mb * 1024 * 1024)
        target_rows = min(total_rows, max(1, target_bytes // avg_row_bytes))

    quotas = _compute_quotas(stratum_counts, target_rows)

    all_freq = sorted(skill_freq.values())
    if all_freq:
        rare_cutoff = max(1, all_freq[max(0, int(len(all_freq) * 0.1) - 1)])
        common_cutoff = max(1, all_freq[max(0, int(len(all_freq) * 0.9) - 1)])
    else:
        rare_cutoff = 1
        common_cutoff = 999999999

    rare_skills = {skill for skill, count in skill_freq.items() if count <= rare_cutoff}
    common_skills = {skill for skill, count in skill_freq.items() if count >= common_cutoff}

    _progress(progress_fn, "Subset pass 2/3: deterministic stratified selection", 0.35)

    stratum_heaps: dict[str, list[tuple[float, int]]] = {
        stratum: [] for stratum, quota in quotas.items() if quota > 0
    }
    global_heap: list[tuple[float, int]] = []

    row_offset = 0
    for chunk in read_csv_chunks(input_csv, chunksize=config.chunksize, encoding=encoding):
        canonical = canonicalize_chunk(chunk)
        records = canonical.to_dict(orient="records")

        for idx, row in enumerate(records):
            row_idx = row_offset + idx
            stratum = _stratum_key(row)

            parsed_skills, _ = parse_skills_cell(row.get("skills"), row_idx)
            unique_skills = {
                normalize_skill_text(item.skill_text_raw)
                for item in parsed_skills
                if normalize_skill_text(item.skill_text_raw)
            }
            rare_hit = any(skill in rare_skills for skill in unique_skills)
            common_hit = any(skill in common_skills for skill in unique_skills)
            degree = len(unique_skills)

            score = _hash01(config.subset_seed, row_idx)
            if rare_hit:
                score += 0.3
            if common_hit:
                score += 0.1
            score += min(degree, 20) / 100.0

            if stratum in stratum_heaps:
                quota = quotas.get(stratum, 0)
                heap = stratum_heaps[stratum]
                entry = (score, row_idx)
                if len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)

            entry_global = (score, row_idx)
            if len(global_heap) < target_rows:
                heapq.heappush(global_heap, entry_global)
            elif entry_global > global_heap[0]:
                heapq.heapreplace(global_heap, entry_global)

        row_offset += len(canonical)

    selected_rows = {row_idx for heap in stratum_heaps.values() for _, row_idx in heap}
    if len(selected_rows) < target_rows:
        global_best = sorted(global_heap, reverse=True)
        for _, row_idx in global_best:
            if row_idx not in selected_rows:
                selected_rows.add(row_idx)
            if len(selected_rows) >= target_rows:
                break

    selected_sorted = sorted(selected_rows)
    selected_set = set(selected_sorted)

    _progress(progress_fn, "Subset pass 3/3: writing sample_input.csv", 0.72)

    wrote_header = False
    row_offset = 0
    for chunk in read_csv_chunks(input_csv, chunksize=config.chunksize, encoding=encoding):
        if len(chunk) == 0:
            continue
        row_ids = list(range(row_offset, row_offset + len(chunk)))
        mask = [row_id in selected_set for row_id in row_ids]
        sampled = chunk.loc[mask]
        if not sampled.empty:
            sampled.to_csv(sample_input_path, index=False, mode="a", header=not wrote_header)
            wrote_header = True
        row_offset += len(chunk)

    actual_mb = sample_input_path.stat().st_size / (1024 * 1024) if sample_input_path.exists() else 0.0

    selected_strata = Counter()
    for row_idx, stratum in row_metadata:
        if row_idx in selected_set:
            selected_strata[stratum] += 1

    summary = {
        "seed": config.subset_seed,
        "total_rows": total_rows,
        "target_rows": target_rows,
        "selected_rows": len(selected_rows),
        "avg_row_bytes_estimate": avg_row_bytes,
        "target_size_mb": config.subset_target_size_mb,
        "actual_sample_size_mb": round(actual_mb, 3),
        "num_strata": len(stratum_counts),
        "selected_strata": len(selected_strata),
        "rare_skill_cutoff_freq": rare_cutoff,
        "common_skill_cutoff_freq": common_cutoff,
    }

    _progress(progress_fn, "Subset generation complete", 1.0)

    return SubsetResult(
        sample_input_path=sample_input_path,
        selected_rows=len(selected_rows),
        total_rows=total_rows,
        target_rows=target_rows,
        summary=summary,
    )
