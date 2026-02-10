# ISB iGraph MVP

End-to-end MVP pipeline to convert large job posting CSV data into an igraph-based Job-Skill bipartite graph, compute Job-Job closeness, and export analysis-ready outputs.

## What this MVP does

- Ingests large CSV in chunks (encoding-aware)
- Standardizes varying column names to canonical schema
- Robustly parses malformed `skills` JSON-like payloads
- Normalizes skill text + bucket labels deterministically
- Builds Job-Skill bipartite graph with `python-igraph`
- Computes bipartite metrics + projected Job-Job metrics
- Produces top-K nearest jobs using weighted/unweighted overlap
- Exports mandatory output artifacts
- Includes deterministic subset builder (~100MB target by default)
- Includes Streamlit UI for upload, run, metrics, downloads, and lookup

## Project structure

```text
isb-igraph/
  app.py
  requirements.txt
  README.md
  sample_data/
    jobs_sample.csv
  isb_igraph/
    __init__.py
    __main__.py
    cli.py
    lookup.py
    skill_lookup.py
    config.py
    ingest.py
    skill_parser.py
    normalization.py
    entities.py
    graph.py
    projection.py
    subset.py
    export.py
    qa.py
    pipeline.py
  tests/
    test_skill_parser.py
    test_normalization.py
    test_entities.py
    test_subset.py
    test_qa.py
    test_pipeline_integration.py
    test_lookup.py
    test_skill_lookup.py
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core Similarity Logic (Deterministic, Not Random)

Job-to-job closeness is deterministic and based on shared skills.

For each pair of jobs, scores come from the Job-Skill bipartite edges:

- `unweighted similarity`:
  - `shared_skill_count`
  - Count of shared normalized skills between two jobs
- `weighted similarity`:
  - `sum(min(edge_weight_job_a, edge_weight_job_b))` over shared skills

Ranking is descending by score, then deterministic tie-breaks by job IDs.

By default, lookup now uses `unweighted` (shared-skill-count-first behavior).

## Run CLI Pipeline

```bash
python -m isb_igraph \
  --input sample_data/jobs_sample.csv \
  --output-dir output
```

## Closest Job Lookup Utility

Query `job_similarity_topk.csv` by `job_id` or title:

```bash
python -m isb_igraph.lookup \
  --nodes-csv output/nodes.csv \
  --similarity-csv output/job_similarity_topk.csv \
  --job-id job_001 \
  --method unweighted \
  --limit 20
```

By title:

```bash
python -m isb_igraph.lookup \
  --nodes-csv output/nodes.csv \
  --similarity-csv output/job_similarity_topk.csv \
  --title "4 Wheeler Mechanic" \
  --method weighted \
  --limit 10
```

With explainability columns (`shared_skill_count`, `shared_skills_preview`):

```bash
python -m isb_igraph.lookup \
  --nodes-csv output/nodes.csv \
  --similarity-csv output/job_similarity_topk.csv \
  --edges-csv output/edges.csv \
  --job-id job_001 \
  --method unweighted \
  --limit 20
```

Notes:
- If a title matches multiple jobs, the utility prints candidate `job_id` rows and exits for disambiguation.
- Use `--json` for JSON output.

## Skill-to-Job OR Lookup Utility

### Mode 1: Skill(s) -> Closest Jobs (OR)

```bash
python -m isb_igraph.skill_lookup \
  --nodes-csv output/nodes.csv \
  --edges-csv output/edges.csv \
  --skills "problem solving, analytical thinking, machine operation" \
  --limit 20
```

Scoring for this mode:
- Primary: `matched_skill_count` (how many query skills are matched)
- Tie-break 1: `matched_weight_sum`
- Tie-break 2: `job_id` ascending

### Mode 2: Job -> Top Skills

```bash
python -m isb_igraph.skill_lookup \
  --nodes-csv output/nodes.csv \
  --edges-csv output/edges.csv \
  --job-id job_001 \
  --limit 20
```

Or by title:

```bash
python -m isb_igraph.skill_lookup \
  --nodes-csv output/nodes.csv \
  --edges-csv output/edges.csv \
  --title "4 Wheeler Mechanic" \
  --limit 20
```

Skill resolution behavior:
- Exact normalized match first
- For unresolved skills, fuzzy suggestions are generated using `rapidfuzz`
- Use `--fuzzy-cutoff` and `--suggestion-limit` to tune

## CLI options (important)

- `--similarity-method both|weighted|unweighted`
- `--top-k 20`
- `--similarity-threshold 0.0`
- `--subset-mode`
- `--subset-target-rows 100000`
- `--subset-target-size-mb 100`
- `--subset-seed 42`
- `--projection-max-skill-degree 2000`
- `--disable-betweenness`
- `--disable-closeness`
- `--compute-betweenness-max-vertices` and `--compute-betweenness-max-edges`
- `--compute-closeness-max-vertices` and `--compute-closeness-max-edges`

## Run Streamlit UI

```bash
streamlit run app.py
```

UI supports:
- Pipeline tab:
  - CSV upload
  - weighted/unweighted/both mode
  - top-k and threshold
  - computation profiles: fast, balanced, deep
  - advanced centrality guards (betweenness/closeness enable/disable + size thresholds)
  - subset mode and sample sizing
  - live progress and metrics
  - per-file download buttons
- Subgraph Visualizer tab:
  - ego graph around a selected job (by ID/title filter)
  - top connected component rendering
  - global overview mode (node-type counts, degree distribution, component sizes)
  - global sampled rendering and safe full-render attempt
  - one-click visualization diagnostics
  - node/edge render caps to keep UI responsive
- Lookup tab:
  - Job -> Closest Jobs (default unweighted, optional weighted/all)
  - Skill(s) -> Closest Jobs OR
  - Job -> Top Skills

## Graph Visualization (MVP)

- Current MVP visualization is Streamlit + Altair for filtered subgraphs.
- For large graphs (hundreds of thousands of nodes/edges), rendering the entire graph interactively on a laptop is usually not practical.
- Recommended approach is visualizing filtered views:
  - subset-mode graph (`sample_nodes.csv`, `sample_edges.csv`)
  - ego networks around one job and its nearest neighbors
  - top connected components/high-degree skill neighborhoods

## Output files

### Core outputs

- `nodes.csv` columns:
  - `node_id, node_type(job|skill), label, attributes_json`
- `edges.csv` columns:
  - `source, target, relation, weight, attributes_json`
- `job_similarity_topk.csv` columns:
  - `job_id, neighbor_job_id, similarity_score, rank, method`
- `graph_summary.json`
- `parse_errors.csv`

### Additional metrics/QA outputs

- `job_node_metrics.csv`
- `skill_node_metrics.csv`
- `projected_job_metrics.csv`
- `profiling_summary.csv`
- `validation_report.json`
- `qa_report.json`

### Subset mode outputs

- `sample_input.csv`
- `sample_nodes.csv`
- `sample_edges.csv`
- `sample_job_similarity_topk.csv`
- `sample_graph_summary.json`
- `sample_summary.json`

## Deterministic normalization and weighting

- Skill normalization:
  - lowercase
  - trim + collapse whitespace
  - strip trailing punctuation
- Bucket normalization map:
  - Mission-Critical / mission critical / critical -> 4
  - Advanced / 4: Advanced -> 3
  - Proficient / 3: Proficient -> 2
  - Working Knowledge -> 1
  - Familiarity / 1: Familiarity -> 0
- Edge weight:
  - `0.7 * mapping_similarity_clipped_0_1 + 0.3 * (bucket_score / 4)`
  - fallback if similarity missing:
    - `0.5 * (bucket_score / 4) + 0.5 * default_similarity(0.5)`

## Quick output preview (sample)

Example `job_similarity_topk.csv` rows:

```text
job_id,neighbor_job_id,similarity_score,rank,method
job_001,job_002,1.0,1,unweighted
job_001,job_006,0.92,1,weighted
```

## Tests

```bash
pytest -q
```

Covers:
- parser robustness
- bucket normalization
- dedup behavior
- edge weight computation
- QA checks
- subset reproducibility
- lookup determinism and ranking behavior
- end-to-end pipeline smoke test

## Scaling beyond MVP

- Add optional Parquet intermediate storage to reduce CSV overhead.
- Add ANN indexing on projected Job-Job edges for low-latency lookup at very large scale.
- Add approximate similarity mode (e.g., MinHash/LSH) for very high-degree skill expansion.
- Move heavy centrality jobs to offline batch with configurable scheduling.
