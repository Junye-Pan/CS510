# Seed Baseline

The task-provided `initial.py` is a measured nonzero baseline, not an empty
stub. Do not rerun this baseline just to discover its score; use this artifact
as the public control result for this task version.

## Configuration

- entrypoint: `initial.py`
- harness style: `minimum_streamed_direct_v1`
- model: `gpt-5.2`
- backend: `https://chatgpt.com/backend-api/codex`
- transport: `pydantic_ai_streamed_responses`
- recorded private evaluator concurrency: 4
- current default private evaluator concurrency: 10

The API transport is part of the task environment. Keep the task-provided
`pydantic_ai` streamed `OpenAIResponsesModel` calling path. Do not replace it
with hand-written `urllib`, `requests`, `httpx`, direct `openai`, or raw
`/responses` HTTP clients.

## Recorded Scores

| split | resolved | total | legal | invalid | timeout | model calls | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `probe10` | 2 | 10 | 10 | 0 | 0 | 10 | 1577.3s |
| `private50` | 6 | 50 | 50 | 0 | 0 | 50 | 3831.4s |

Solved public probe types:

- `lightup`: 1
- `nurimisaki`: 1

Solved private evaluator types:

- `hitori`: 1
- `lightup`: 1
- `norinori`: 3
- `nurimisaki`: 1

## Optimization Implication

The optimization target is to improve on this harness's agentic behavior:
prompt construction, context shaping, model-call policy, parsing, repair, and
fallback logic. Avoid changes that make the harness faster by reducing model
reasoning time or context so much that solve rate collapses.

Before a full private50 submit, use `ve verify` and `ve probe` to check for
regression against the public baseline. A candidate that scores below 2/10 on
`probe10` is normally not ready for a private submit.
