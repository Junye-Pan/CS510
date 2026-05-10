# Tolerance Edge Polish

## Scope

Run long-horizon endgame optimization on the strongest known incumbent basin.

## Bias

Favor incumbent following, joint centers-and-radii polish, fixed-center LP diagnostics, tolerance-edge exports, and iterated perturbation chains that stay near the same strong basin.

## Deprioritize

- broad fresh-seed exploration
- structural novelty for its own sake

## Strong Signals

- verified evaluator gains on the incumbent basin
- strict-safe score that remains close to the raw evaluator score
- repeated local improvement from perturbation chains or higher-precision polish

## Representative Moves

- follow-best incumbent continuation
- joint continuous polish on centers and radii
- fixed-center LP rescoring and export cleanup
- long perturb-reoptimize chains around a strong incumbent

## Cross-Direction Snapshot Use

This is the dedicated endgame slot. It may follow the strongest public incumbent directly and does not need to preserve basin diversity. Use the exploration slots to search for new structures; use this slot to extract the best score from the strongest known basin.
