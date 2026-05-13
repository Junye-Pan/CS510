"""Optional RMSNorm implementation entrypoint for the public seed candidate.

The seed manifest does not declare this file. Agents can add an implementation
entry pointing to ``kernels/rmsnorm.py::run`` and replace this placeholder with
a real destination-passing kernel.
"""


def run(hidden_states, weight, output):
    raise NotImplementedError("The baseline seed candidate uses framework fallback.")
