# Incumbent Repair

## Scope

Work from the strongest known valid or nearly valid incumbent layouts.

## Bias

Favor repair, feasibility recovery, active-constraint analysis, and score recovery on incumbent basins that already look strong.

## Deprioritize

- brand-new constructive families with no relation to the incumbent
- broad random restarts that do not reuse current structure

## Strong Signals

- tighter valid exports from the same geometric basin
- better per-circle repair than uniform shrink
- local geometric changes that preserve most of the incumbent score

## Representative Moves

- LP-based or non-uniform repair
- active-constraint graph analysis
- conservative local surgery when the incumbent is close to valid
- score recovery after feasibility repair

## Cross-Direction Snapshot Use

This direction may import strong snapshots from other directions when the goal is repair, validation, or comparison. Prefer strict-safe recovery and score preservation on a basin you already own; the dedicated long-horizon public-incumbent follower is `tolerance_edge_polish`, not this slot.
