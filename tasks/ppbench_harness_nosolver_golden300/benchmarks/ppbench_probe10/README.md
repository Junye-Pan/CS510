# PPBench Probe10

This directory is a self-contained 10-puzzle probe subset derived from
`ppbench_repr50`. It is intended for cheap, feedback-oriented harness checks
before spending a full evaluator run on the representative 50-puzzle split.

## Layout

```text
benchmarks/ppbench_probe10/
  manifest.json                 benchmark metadata and selection rule
  benchmark.py                  local helper for listing, showing, and validating
  public/
    manifest.json               public-safe benchmark metadata
    puzzles.json                public puzzle records as one JSON payload
    puzzles.jsonl               public puzzle records as JSONL
    puzzle_ids.txt              selected puzzle ids in probe-rank order
    puzzles/                    one JSON file per puzzle
    prompts/                    one prompt-ready Markdown file per puzzle
    ppbench_site/               local no-answer mirror of play pages and assets
  private/
    answer_key.jsonl            evaluator-only official answer material
  provenance/
    selected_puzzles.jsonl      selected-source matching metadata
    selection_method.json       deterministic selection details
    sources.json                source URLs and generation metadata
```

## Selection

`probe10` is a strict subset of `ppbench_repr50`.

Selection procedure:

1. Start from the 50 public puzzles in `ppbench_repr50`.
2. Group puzzles by type.
3. For each type, keep the single puzzle with the highest public solve rate.
4. Sort those 20 type representatives by solve rate descending, using earlier
   source page order as the deterministic tie-break.
5. Take the top 10.

The result contains 10 distinct types and is intentionally easier than
`ppbench_repr50`, so probe runs are more likely to provide nonzero feedback.

## Public Use

Use the helper script from the owning task root:

```bash
python3 benchmarks/ppbench_probe10/benchmark.py validate
python3 benchmarks/ppbench_probe10/benchmark.py list
python3 benchmarks/ppbench_probe10/benchmark.py list --format jsonl
python3 benchmarks/ppbench_probe10/benchmark.py show 1 --format prompt
python3 benchmarks/ppbench_probe10/benchmark.py show nurimisaki_113a07915a22f92f3d82fe15979ce794
```

Experiment code that gives instances to agents should read only `public/`.
The prompt files are convenience views over the same records in
`public/puzzles.jsonl`. Each public record also has `local_play_path`, which
points to a sanitized local play page under `public/ppbench_site/` for offline
inspection.

## Private Use

`private/answer_key.jsonl` is for host-side evaluator code only. Do not copy it
into agent workspaces, shared task context, or any filesystem root that a
solving agent can browse.

## Sources

- PPBench home: https://ppbench.com/
- Puzzle browser: https://ppbench.com/puzzles.html
- Repository: https://github.com/approximatelabs/pencil-puzzle-bench
- Parent benchmark: `benchmarks/ppbench_repr50`
