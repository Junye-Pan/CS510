from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PPBENCH_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PPBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PPBENCH_ROOT))

from baseline_common import build_arg_parser, run_private50_baseline, write_json  # noqa: E402
from strategy import CodexBasicAgenticSolve  # noqa: E402


async def main() -> None:
    parser = build_arg_parser("Run the official-agentic PPBench private50 baseline.")
    args = parser.parse_args()
    output_dir = Path(__file__).resolve().parent / "results"
    summary = await run_private50_baseline(
        strategy_factory=CodexBasicAgenticSolve,
        strategy_name="official_agentic",
        output_dir=output_dir,
        args=args,
    )
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
