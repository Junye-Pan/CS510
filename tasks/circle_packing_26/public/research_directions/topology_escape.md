# Topology Escape

## Scope

Escape the current basin by making discrete structural changes rather than only smooth continuous refinement.

## Bias

Favor contact-graph changes, deletion/reinsertion, nonlocal surgery, and structural jumps that could reach a different basin.

## Deprioritize

- pure shrink retuning
- long continuous polishing with unchanged topology

## Strong Signals

- a modified contact structure that can still be repaired to near-validity
- nonlocal edits that recover most of the original score while changing geometry meaningfully
- repeated evidence that one discrete edit family opens new basins

## Representative Moves

- contact surgery
- remove-and-reinsert moves
- local role swaps between circles
- topology-changing perturb-and-polish branches

## Cross-Direction Snapshot Use

Other directions' snapshots may be imported as mutation sources, controls, or comparison targets, but not as direct incumbent-polish targets. A checked-out foreign snapshot should undergo a real discrete structural edit before it becomes a serious candidate again.
