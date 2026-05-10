from __future__ import annotations

"""Direct model baseline adapted from PPBench's official DirectAskStrategy.

Source: https://github.com/approximatelabs/pencil-puzzle-bench
Adaptation: use pydantic-ai `instructions=` so the Codex backend receives the
required top-level Responses API instructions field.
"""

import json
import re
from typing import Any

from pydantic_ai import Agent
from ppbench.puzzle import Puzzle
from ppbench.benchmarks.strategy import AgentConfig, Strategy, StrategyResult
from ppbench.benchmarks.strategies._helpers import (
    example_of_inputs,
    get_example_move_context,
    get_rules_for_puzzle,
)


class CodexDirectAskStrategy(Strategy):
    requires_tools = False

    def build_agent(self, puzzle: Any, model_obj: Any, model_name: str) -> AgentConfig:
        ticks = "```"
        pre_post = get_example_move_context(puzzle)
        instructions = (
            "Solve the puzzle!!\n\n"
            "Answer with a list of moves you would like to make that solve the puzzle "
            "as json in a markdown json code block\n\n"
            f"{ticks}json\n"
            '["mouse,left,1,1", "mouse,right,3,1", ...]\n'
            f"{ticks}"
        )
        prompt = f"""
Puzzle Type: {puzzle.pid}
Puzzle Rules:
{ticks}
{get_rules_for_puzzle(puzzle)}
{ticks}

Here is an example of inputs / a solved puzzle (lots of context for you)
{ticks}
{example_of_inputs(puzzle)}
{ticks}

Here's some more:
{pre_post}

Note specifically how the coordinate systems work (for the puzzle vs. the inputs).
For the puzzle you are working on, ensure you fully understand from the example of
input above, that the move is exactly where you expect.

==== ==== ==== ==== ====
Here is the puzzle you are to solve:
{puzzle.get_string_repr()}
==== ==== ==== ==== ====

Please now solve it.
"""
        agent = Agent(model_obj, instructions=instructions, retries=2)
        return AgentConfig(agent=agent, prompt=prompt)

    def extract_result(self, puzzle: Any, deps: Any, output: str) -> StrategyResult:
        list_of_moves: list[str] = []
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", output, re.DOTALL)
        raw_json = match.group(1) if match else output
        try:
            data = json.loads(raw_json)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                list_of_moves = data
        except json.JSONDecodeError:
            pass

        fresh = Puzzle.from_url(puzzle.url)
        for move in list_of_moves:
            fresh.send_move(move)
        is_success = fresh.isComplete()
        return StrategyResult(
            is_success=is_success,
            parsed_moves=list_of_moves,
            raw_output=output,
        )
