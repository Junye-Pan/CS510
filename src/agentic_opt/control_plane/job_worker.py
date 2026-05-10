from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .repository import ControlPlaneRepository


def run_job(*, db_path: Path, job_id: str) -> int:
    repository = ControlPlaneRepository(db_path)
    job = repository.get_job(job_id)
    if job is None:
        raise KeyError(job_id)
    inputs = job.get("inputs") or {}
    outputs = job.get("outputs") or {}
    command = inputs.get("command")
    if not command:
        raise ValueError(f"job {job_id} has no inputs.command")
    stdout_path = Path(outputs.get("stdout_path") or Path.cwd() / f"{job_id}.stdout.log")
    stderr_path = Path(outputs.get("stderr_path") or Path.cwd() / f"{job_id}.stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path(inputs.get("cwd") or os.getcwd()).resolve()
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in (inputs.get("env") or {}).items()})

    repository.update_job(job_id, {"status": "running", "pid": os.getpid()})
    started_at = time.time()
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=stdout,
                stderr=stderr,
                shell=isinstance(command, str),
                text=False,
            )
            repository.update_job(job_id, {"status": "running", "pid": process.pid})
            returncode = process.wait()
            elapsed_s = time.time() - started_at
            current = repository.get_job(job_id) or {}
            final_status = "cancelled" if current.get("status") == "cancelled" else "completed" if returncode == 0 else "failed"
            repository.update_job(
                job_id,
                {
                    "status": final_status,
                    "outputs": {
                        **outputs,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "returncode": returncode,
                        "elapsed_s": elapsed_s,
                    },
                },
            )
            return 0 if returncode == 0 else returncode
        except BaseException as exc:
            stderr.write(traceback.format_exc().encode("utf-8", errors="replace"))
            repository.update_job(
                job_id,
                {
                    "status": "failed",
                    "outputs": {
                        **outputs,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                        "returncode": None,
                        "error": str(exc),
                    },
                },
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentic_opt.control_plane.job_worker")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_job(db_path=args.db, job_id=args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
