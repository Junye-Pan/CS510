# PPBench Repr50

This directory is a self-contained 50-puzzle subset of the public PPBench
`golden_300` split. It is intended as a more representative development
benchmark than `ppbench_hard30`, while still fitting inside a smaller evaluator
budget.

## Layout

```text
benchmarks/ppbench_repr50/
  manifest.json                 benchmark metadata and selection rule
  benchmark.py                  local helper for listing, showing, and validating
  public/
    manifest.json               public-safe benchmark metadata
    puzzles.json                public puzzle records as one JSON payload
    puzzles.jsonl               public puzzle records as JSONL
    puzzle_ids.txt              selected puzzle ids in subset-rank order
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

The subset was generated on 2026-04-23 from the 300 puzzles visible in the
public PPBench puzzle browser at https://ppbench.com/puzzles.html.

The full 300-puzzle set contains 20 puzzle types with 15 puzzles per type. The
subset keeps type coverage broad and keeps the global solve-rate distribution
close to the full split.

Selection procedure:

1. For each type, sort its 15 puzzles by public solve rate ascending.
2. Select two base puzzles per type at fixed within-type ranks 5 and 11
   (1-indexed). This yields a 40-puzzle skeleton.
3. Define the 50-puzzle target solve-rate histogram:
   - `0`: 23
   - `(0,1]`: 1
   - `(1,2.5]`: 7
   - `(2.5,5]`: 6
   - `(5,10]`: 5
   - `(10,20]`: 5
   - `(20,100]`: 3
4. Add one extra puzzle to 10 of the 20 types with a dynamic-programming pass
   that exactly fills the histogram deficit while minimizing type-level
   distortion relative to each type's original 15-puzzle distribution.

The result contains 10 types with 3 puzzles and 10 types with 2 puzzles.

## Public Use

Use the helper script from the owning task root:

```bash
python3 benchmarks/ppbench_repr50/benchmark.py validate
python3 benchmarks/ppbench_repr50/benchmark.py list
python3 benchmarks/ppbench_repr50/benchmark.py list --format jsonl
python3 benchmarks/ppbench_repr50/benchmark.py show 1 --format prompt
python3 benchmarks/ppbench_repr50/benchmark.py show nurimisaki_113a07915a22f92f3d82fe15979ce794
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
- Bundled source dataset: `ppbench/bundled/golden_300.jsonl`
