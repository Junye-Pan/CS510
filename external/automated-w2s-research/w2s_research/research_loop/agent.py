"""
Autonomous W2S research agent using Claude Agent SDK.

Merges agent_loop, base_agent, and stop_conditions into a single file.
"""

import asyncio
import json
import os
import shutil
import time
import traceback
import uuid
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from w2s_research.config import (
    WORKSPACE_DIR,
    S3_BUCKET,
    S3_IDEAS_PREFIX,
    LOGS_DIR,
    FULL_AUTO_MAX_RUNTIME_SECONDS,
    LOCAL_FINDINGS_DIR,
    FINDINGS_POLL_INTERVAL,
    TARGET_IDEA_FILE,
    DATASET_NAME,
    DATA_DIR,
    WEAK_MODEL,
    STRONG_MODEL,
    SERVER_URL,
    AAR_MODE,
)
from .tools.server_api_tools import create_server_api_tools_server
from .tools.prior_work_tools import create_prior_work_tools_server
from .tools.findings_sync import FindingsSync


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def resolve_prompt(template_path: str | Path, output_path: str | Path) -> str:
    """Render Jinja2 prompt template with config variables."""
    from jinja2 import Template

    template_path = Path(template_path)
    output_path = Path(output_path)

    with open(template_path, 'r') as f:
        template = Template(f.read())

    target_idea_content = ""
    if Path(TARGET_IDEA_FILE).exists():
        with open(TARGET_IDEA_FILE, 'r') as f:
            try:
                idea_data = json.load(f)
                target_idea_content = idea_data.get("Description", json.dumps(idea_data, indent=2))
            except json.JSONDecodeError:
                target_idea_content = f.read()
    else:
        print(f"WARNING: Target idea file not found: {TARGET_IDEA_FILE}")

    local_mode = os.getenv("LOCAL_MODE", "false").lower() in ("1", "true", "yes")

    content = template.render(
        workspace_dir=WORKSPACE_DIR,
        dataset_name=DATASET_NAME,
        data_dir=DATA_DIR,
        weak_model=WEAK_MODEL,
        strong_model=STRONG_MODEL,
        s3_bucket=S3_BUCKET,
        s3_ideas_prefix=S3_IDEAS_PREFIX,
        logs_dir=LOGS_DIR,
        server_url=SERVER_URL,
        aar_mode=str(AAR_MODE).lower(),
        local_mode=str(local_mode).lower(),
        target_idea_content=target_idea_content,
    )

    with open(output_path, 'w') as f:
        f.write(content)

    return content


# ---------------------------------------------------------------------------
# Stop condition (just timeout)
# ---------------------------------------------------------------------------

class StopReason(Enum):
    TIMEOUT = "timeout"
    USER_INTERRUPT = "user_interrupt"


class _StopChecker:
    """Simple timeout-only stop checker."""

    def __init__(self, max_runtime: float):
        self.max_runtime = max_runtime
        self.start_time = time.time()
        self.consecutive_errors = 0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def check(self) -> Optional[StopReason]:
        if self.elapsed_time >= self.max_runtime:
            return StopReason.TIMEOUT
        return None

    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    success: bool
    output: Dict[str, Any]
    duration: float
    iteration_count: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Base agent (wraps Claude Agent SDK)
# ---------------------------------------------------------------------------

class BaseAgent:
    """Wraps ClaudeSDKClient for research tasks."""

    def __init__(
        self,
        name: str,
        allowed_tools: List[str],
        workspace: Path,
        mcp_servers: Dict[str, Any],
        model: str = "claude-opus-4-6",
        permission_mode: str = "bypassPermissions",
        cli_path: Optional[str] = None,
        message_callback: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
    ):
        self.name = name
        self.allowed_tools = allowed_tools
        self.workspace = workspace
        self.mcp_servers = mcp_servers
        self.model = model
        self.permission_mode = permission_mode
        self.cli_path = cli_path
        self.message_callback = message_callback
        self.system_prompt = system_prompt

    async def execute(self, task: str) -> AgentResult:
        """Execute agent task. Returns AgentResult."""
        start_time = time.time()
        iteration_count = 0
        messages = []

        try:
            options_dict = {
                "allowed_tools": self.allowed_tools,
                "system_prompt": self.system_prompt,
                "permission_mode": self.permission_mode,
                "cwd": str(self.workspace),
                "model": self.model,
                "mcp_servers": self.mcp_servers,
                "setting_sources": ["project"],
                "betas": ["context-1m-2025-08-07"],
            }
            if self.cli_path:
                options_dict["cli_path"] = self.cli_path

            options = ClaudeAgentOptions(**options_dict)

            async with ClaudeSDKClient(options=options) as client:
                await client.query(task)

                async for message in client.receive_response():
                    messages.append(message)

                    # Log
                    log_msg = self._format_message(message)
                    if log_msg:
                        print(log_msg)

                    if self.message_callback:
                        self.message_callback(message)

                    if isinstance(message, ResultMessage):
                        break

                    if isinstance(message, AssistantMessage):
                        for content in message.content:
                            if isinstance(content, ToolUseBlock):
                                iteration_count += 1

                return AgentResult(
                    success=True,
                    output=self._extract_output(messages),
                    duration=time.time() - start_time,
                    iteration_count=iteration_count,
                )

        except Exception as e:
            return AgentResult(
                success=False,
                output={},
                duration=time.time() - start_time,
                iteration_count=iteration_count,
                error=str(e),
            )

    def _format_message(self, message) -> Optional[str]:
        ts = datetime.now().strftime("%H:%M:%S")
        if isinstance(message, AssistantMessage):
            parts = []
            for content in message.content:
                if isinstance(content, TextBlock):
                    parts.append(f"[{ts}] [{self.name}] {content.text[:200]}")
                elif isinstance(content, ToolUseBlock):
                    tool_input = content.input or {}
                    detail = ""
                    if content.name == "Bash":
                        detail = f" {tool_input.get('command', '')[:100]}"
                    elif content.name in ("Read", "Write", "Edit"):
                        detail = f" {tool_input.get('file_path', '')}"
                    parts.append(f"[{ts}] [{self.name}] -> {content.name}{detail}")
            return "\n".join(parts) if parts else None
        elif isinstance(message, ResultMessage):
            return f"[{ts}] [{self.name}] Done"
        return None

    def _extract_output(self, messages) -> Dict[str, Any]:
        output = {"text_outputs": [], "tool_uses": []}
        for msg in messages:
            if isinstance(msg, AssistantMessage):
                for content in msg.content:
                    if isinstance(content, TextBlock):
                        output["text_outputs"].append(content.text)
                    elif isinstance(content, ToolUseBlock):
                        output["tool_uses"].append({"tool": content.name, "input": content.input})
            if isinstance(msg, ResultMessage):
                output["result_message"] = msg.result
        return output


# ---------------------------------------------------------------------------
# Main autonomous loop
# ---------------------------------------------------------------------------

class AutonomousAgentLoop:
    """
    Main agent loop for autonomous W2S research.

    Each iteration is a fresh Claude session. Agent reads/writes findings.json
    directly. Only stop condition: timeout.
    """

    def __init__(
        self,
        idea_uid: str,
        idea_name: str,
        workspace: Optional[Path] = None,
        max_runtime_seconds: Optional[int] = None,
        logs_dir: Optional[Path] = None,
        s3_bucket: Optional[str] = None,
        model: str = "claude-opus-4-6",
        local_mode: bool = False,
    ):
        if not idea_uid:
            raise ValueError("idea_uid is required")

        self.idea_uid = idea_uid
        self.idea_name = idea_name
        self.local_mode = local_mode
        self.run_id = os.getenv("RUN_ID") or str(uuid.uuid4())

        self.workspace = Path(workspace) if workspace else Path(WORKSPACE_DIR)
        self.logs_dir = Path(logs_dir) if logs_dir else Path(LOGS_DIR)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.findings_path = self.logs_dir.parent / "findings.json"

        self.max_runtime_seconds = max_runtime_seconds or FULL_AUTO_MAX_RUNTIME_SECONDS
        self.s3_bucket = s3_bucket or S3_BUCKET
        self.s3_prefix = f"{S3_IDEAS_PREFIX}{idea_uid}/{self.run_id}/"
        self.model = model

        self.stop_checker = _StopChecker(max_runtime=self.max_runtime_seconds)

        # Create MCP servers (server API tools always needed — in local mode, server runs on localhost)
        self.mcp_servers = {}
        try:
            self.mcp_servers["server-api-tools"] = create_server_api_tools_server()
        except Exception as e:
            print(f"[Init] Warning: server API tools unavailable: {e}")
        if not local_mode:
            try:
                self.mcp_servers["prior-work-tools"] = create_prior_work_tools_server()
            except Exception as e:
                print(f"[Init] Warning: prior work tools unavailable: {e}")

        # Findings sync: disabled in local mode (no other workers)
        self.findings_sync = None
        if not local_mode:
            self.findings_sync = FindingsSync(
                findings_dir=Path(LOCAL_FINDINGS_DIR),
                poll_interval=FINDINGS_POLL_INTERVAL,
            )

        self.session_count = 0
        self._prompt: Optional[str] = None

    def _get_prompt(self) -> str:
        if self._prompt is None:
            template_path = self.workspace / "w2s_research" / "research_loop" / "prompt.jinja2"
            if not template_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {template_path}")
            resolved_path = self.logs_dir / "prompt_resolved.md"
            self._prompt = resolve_prompt(template_path, resolved_path)
        return self._prompt

    def _create_agent(self, session_id: str, message_callback=None) -> BaseAgent:
        allowed_tools = [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebSearch", "WebFetch",
            "mcp__server-api-tools__evaluate_predictions",
            "mcp__server-api-tools__share_finding",
            "mcp__server-api-tools__get_leaderboard",
        ]
        if not self.local_mode:
            allowed_tools.append("mcp__prior-work-tools__download_snapshot")

        return BaseAgent(
            name=f"autonomous-{session_id}",
            allowed_tools=allowed_tools,
            workspace=self.workspace,
            mcp_servers=self.mcp_servers,
            model=self.model,
            cli_path=shutil.which("claude"),
            message_callback=message_callback,
        )

    async def run(self) -> Dict[str, Any]:
        """Run the autonomous agent loop."""
        mode_str = "local" if self.local_mode else "server"
        print(f"\n{'='*60}")
        print(f"Autonomous Agent Loop ({mode_str} mode)")
        print(f"  Run ID: {self.run_id}")
        print(f"  Idea: {self.idea_name} ({self.idea_uid})")
        print(f"  Max Runtime: {self.max_runtime_seconds/3600:.1f}h")
        print(f"{'='*60}\n")

        # Initial findings sync (disabled in local mode)
        if self.findings_sync:
            try:
                count = self.findings_sync.sync_once()
                print(f"[Loop] Fetched {count} finding(s) from server")
            except Exception as e:
                print(f"[Loop] Warning: initial findings fetch failed: {e}")

            try:
                self.findings_sync.start()
            except Exception as e:
                print(f"[Loop] Warning: could not start findings sync: {e}")

        stop_reason = None

        while True:
            stop_reason = self.stop_checker.check()
            if stop_reason:
                print(f"\n[Loop] Stopping: {stop_reason.value}")
                break

            try:
                await self._run_session()
                self.session_count += 1
                self.stop_checker.record_success()
                if not self.local_mode:
                    await self._sync_to_s3()

            except KeyboardInterrupt:
                print("\n[Loop] Interrupted")
                stop_reason = StopReason.USER_INTERRUPT
                break

            except Exception as e:
                print(f"\n[Loop] Session error: {e}")
                traceback.print_exc()
                self.stop_checker.record_error()
                await asyncio.sleep(30)

        if self.findings_sync:
            try:
                self.findings_sync.stop()
            except Exception:
                pass

        if not self.local_mode:
            await self._sync_to_s3()

        result = {
            "run_id": self.run_id,
            "idea_uid": self.idea_uid,
            "idea_name": self.idea_name,
            "sessions": self.session_count,
            "duration_seconds": self.stop_checker.elapsed_time,
            "stop_reason": stop_reason.value if stop_reason else "unknown",
        }

        print(f"\n{'='*60}")
        print(f"Done: {result['sessions']} sessions, "
              f"{result['duration_seconds']/3600:.1f}h, "
              f"reason={result['stop_reason']}")
        print(f"{'='*60}\n")

        return result

    async def _run_session(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{self.session_count:03d}_{timestamp}"
        log_file = self.logs_dir / f"{session_id}.log"

        print(f"\n[Session {self.session_count}] {session_id}")

        prompt = self._get_prompt()

        with open(log_file, "w") as log_f:
            log_f.write(f"# Session {session_id}\n# Started: {datetime.now().isoformat()}\n\n")

            def message_callback(message):
                ts = datetime.now().strftime("%H:%M:%S")
                log_f.write(f"\n[{ts}] {type(message).__name__}\n")
                if hasattr(message, "content"):
                    for content in message.content:
                        if hasattr(content, "text"):
                            log_f.write(f"{content.text}\n")
                        elif hasattr(content, "name"):
                            log_f.write(f"Tool: {content.name}\n")
                            if hasattr(content, "input"):
                                input_str = json.dumps(content.input, indent=2)
                                if len(input_str) > 1000:
                                    input_str = input_str[:1000] + "... [truncated]"
                                log_f.write(f"Input: {input_str}\n")
                log_f.flush()

            try:
                agent = self._create_agent(session_id, message_callback=message_callback)
                result = await agent.execute(task=prompt)

                log_f.write(f"\n# Result: success={result.success}, "
                           f"duration={result.duration:.1f}s, "
                           f"tools={result.iteration_count}\n")
                if result.error:
                    log_f.write(f"# Error: {result.error}\n")

                if not result.success:
                    raise RuntimeError(f"Agent failed: {result.error}")

            except Exception as e:
                log_f.write(f"\n# ERROR: {e}\n{traceback.format_exc()}")
                raise
            finally:
                log_f.write(f"\n# Ended: {datetime.now().isoformat()}\n")

        print(f"[Session {self.session_count}] Completed")

    async def _sync_to_s3(self):
        try:
            from w2s_research.infrastructure.s3_utils import upload_file_to_s3
        except ImportError:
            return

        try:
            if self.findings_path.exists():
                upload_file_to_s3(
                    file_path=self.findings_path,
                    s3_key=f"{self.s3_prefix}findings.json",
                    bucket_name=self.s3_bucket,
                    content_type="application/json",
                )
            for log_file in self.logs_dir.glob("session_*.log"):
                upload_file_to_s3(
                    file_path=log_file,
                    s3_key=f"{self.s3_prefix}logs/{log_file.name}",
                    bucket_name=self.s3_bucket,
                    content_type="text/plain",
                )
        except Exception as e:
            print(f"[Sync] Failed: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def main():
    import sys
    idea_uid = os.getenv("IDEA_UID") or (sys.argv[1] if len(sys.argv) > 1 else None)
    idea_name = os.getenv("IDEA_NAME", "unknown")
    local_mode = os.getenv("LOCAL_MODE", "false").lower() in ("1", "true", "yes")

    if not idea_uid:
        print("Usage: python -m w2s_research.research_loop.agent <idea_uid>")
        print("   Or: IDEA_UID=... python -m w2s_research.research_loop.agent")
        sys.exit(1)

    loop = AutonomousAgentLoop(idea_uid=idea_uid, idea_name=idea_name, local_mode=local_mode)
    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
