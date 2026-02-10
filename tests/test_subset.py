from __future__ import annotations

from pathlib import Path

from isb_igraph.config import PipelineConfig
from isb_igraph.subset import create_deterministic_subset



def test_subset_reproducibility(tmp_path: Path) -> None:
    sample_input = Path("sample_data/jobs_sample.csv").resolve()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    config = PipelineConfig(
        input_csv=sample_input,
        output_dir=out1,
        subset_mode=True,
        subset_target_rows=5,
        subset_seed=42,
    )

    res1 = create_deterministic_subset(config=config, output_dir=out1)

    config2 = PipelineConfig(
        input_csv=sample_input,
        output_dir=out2,
        subset_mode=True,
        subset_target_rows=5,
        subset_seed=42,
    )
    res2 = create_deterministic_subset(config=config2, output_dir=out2)

    bytes1 = res1.sample_input_path.read_bytes()
    bytes2 = res2.sample_input_path.read_bytes()
    assert bytes1 == bytes2


def test_subset_overwrites_existing_sample_file(tmp_path: Path) -> None:
    sample_input = Path("sample_data/jobs_sample.csv").resolve()
    output_dir = tmp_path / "same_dir"

    config = PipelineConfig(
        input_csv=sample_input,
        output_dir=output_dir,
        subset_mode=True,
        subset_target_rows=4,
        subset_seed=42,
    )

    first = create_deterministic_subset(config=config, output_dir=output_dir)
    first_bytes = first.sample_input_path.read_bytes()

    second = create_deterministic_subset(config=config, output_dir=output_dir)
    second_bytes = second.sample_input_path.read_bytes()

    assert first_bytes == second_bytes
