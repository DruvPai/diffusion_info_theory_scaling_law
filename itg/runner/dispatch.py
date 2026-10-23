"""Run experiment cases across GPUs, one subprocess per job."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    command: list[str]


@dataclass
class Running:
    job: Job
    gpu: str
    process: subprocess.Popen
    started: float


def discover_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.lower() not in {"-1", "none"}:
        return [part.strip() for part in visible.split(",") if part.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["0"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] or ["0"]


def jobs(case_names: list[str], output_dir: Path, variants: list[list[str]]) -> list[Job]:
    """One subprocess per (case, variant): this script again with `--case` and
    the variant's own flags.

    No spec file and no round-trip -- the arguments that defined the grid are
    exactly the arguments each case is re-run with.
    """
    return [
        Job(
            name=" ".join([name, *extra]),
            command=[
                sys.executable, *sys.argv,
                "--case", name,
                "--output-dir", str(output_dir),
                *extra,
            ],
        )
        for name in case_names
        for extra in variants
    ]


def dispatch(queue: list[Job], *, gpus: list[str] | None = None, poll_seconds: float = 10.0) -> None:
    gpus = gpus or discover_gpus()
    if not queue:
        return
    next_job = 0
    running: list[Running] = []
    failed: list[str] = []
    while next_job < len(queue) or running:
        while next_job < len(queue) and len(running) < len(gpus):
            gpu = next(gpu for gpu in gpus if gpu not in {item.gpu for item in running})
            job = queue[next_job]
            print(f"[dispatch] {job.name} -> GPU {gpu}", flush=True)
            process = subprocess.Popen(
                job.command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            )
            running.append(Running(job, gpu, process, time.monotonic()))
            next_job += 1

        still_running = []
        for item in running:
            code = item.process.poll()
            duration = time.monotonic() - item.started
            if code is None:
                still_running.append(item)
            elif code:
                print(f"[dispatch] {item.job.name} failed code={code} after {duration:.0f}s", flush=True)
                failed.append(item.job.name)
            else:
                print(f"[dispatch] {item.job.name} finished in {duration:.0f}s", flush=True)
        running = still_running
        if running or next_job < len(queue):
            time.sleep(poll_seconds)

    if failed:
        raise RuntimeError(f"failed jobs: {failed}")
