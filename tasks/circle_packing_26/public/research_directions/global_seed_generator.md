# Global Seed Generator

## Scope

Search for strong new basins from fresh global starts rather than polishing the current incumbent.

## Bias

Favor genuinely new seed families, continuation methods, penalty objectives, and broad restart diversity.

## Deprioritize

- tiny shrink tuning on the current incumbent
- long repair-only work on a known basin

## Strong Signals

- a seed family that repeatedly lands near competitive scores
- layouts with visibly different global structure from the current best
- new basins that survive verify or probe without heavy repair

## Representative Moves

- global penalty search
- fresh stochastic restart families
- continuation from loose feasible scaffolds
- structured random seeds that do not inherit the incumbent directly

## Cross-Direction Snapshot Use

Treat other directions' snapshots as comparison or control material only. Do not adopt them as the main working incumbent. If you inspect another direction's candidate, return to fresh seed families rather than continuing its basin.
