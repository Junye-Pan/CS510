from __future__ import annotations

import os


if os.environ.get("AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION") == "1":
    try:
        from tasks.llm_inference_qwen3_1_7b_sglang_4090.private import sglang_injection

        sglang_injection.install()
    except Exception as exc:  # pragma: no cover - import-time safety net
        import sys

        sys.stderr.write(
            "[ao-qwen3-inject] failed to install sitecustomize hook: "
            f"{type(exc).__name__}: {exc}\n"
        )
        sys.stderr.flush()
