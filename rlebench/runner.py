"""harbor argv construction, device placement and execution.

One shape for every family: `harbor run -p <path> [-i <instance>] -a <agent>
[-m <model>] -o <jobs>/<target>/<agent> --job-name <model>` plus what harbor
cannot know about a task -- the agent's install and API hosts, resume support,
closed-book switch, dataset mount and GPU card -- read from the family manifest
and the target's task.toml.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .providers import Resolved
from .repo import REPO, Family, Target, harbor_bin, read_task_toml, robocasa_asset_dir, task05_data_dirs


def model_slug(model: str) -> str:
    return model.replace("/", "_") if model else "oracle"


def _env(base: dict[str, str]) -> dict[str, str]:
    """The harbor subprocess env. The RoboCasa compose overlays bind-mount
    $ROBOCASA_ASSET_DIR and task05's bind-mount its NANOVLA_* data; unset, each is
    what the checkout's own `make sim-robocasa` / `make task05-data` downloaded."""
    env = dict(base)
    if not env.get("ROBOCASA_ASSET_DIR"):
        env["ROBOCASA_ASSET_DIR"] = str(robocasa_asset_dir())
    for variable, default in task05_data_dirs().items():
        if not env.get(variable):
            env[variable] = str(default)
    return env


def _task_environment(target: Target) -> dict:
    task_dir = REPO / target.path / (target.instance or "")
    return read_task_toml(task_dir / "task.toml").get("environment", {})


def task_gpus(target: Target) -> int:
    """`[environment] gpus` of the target's task.toml (0 when absent)."""
    return int(_task_environment(target).get("gpus") or 0)


def task_requests_gpu(target: Target) -> bool:
    return task_gpus(target) > 0


@dataclass
class Invocation:
    argv: list[str]  # harbor argv, with argv[0] == "harbor"
    env: dict[str, str]
    label: str = ""
    console_log: Path | None = None

    def display(self) -> str:
        return " ".join(self.argv)


@dataclass
class RunOptions:
    jobs_dir: str = "jobs"
    open_book: bool = False
    resume: bool = True
    force_build: bool = False
    max_retries: int | None = None
    extra: list[str] = field(default_factory=list)  # after `--`, verbatim


# --- devices ----------------------------------------------------------------------
# `--device cpu` or `--device cuda:N ...`. Harbor reserves GPUs by count only and
# defers to a compose device reservation, so each GPU family's overlay pins
# `device_ids: ['${RLEBENCH_GPU}']` and a device is an env var per cell.


@dataclass
class Placement:
    env: dict[str, str]
    label: str


def parse_devices(values: list[str] | None, targets: list[Target]) -> list[str] | None:
    """Device ids, or None for cpu. Default: cuda:0 when any target wants a GPU."""
    if not values:
        return ["0"] if any(task_requests_gpu(t) for t in targets) else None
    if values == ["cpu"]:
        gpu_targets = [t.name for t in targets if task_requests_gpu(t)]
        if gpu_targets:
            raise SystemExit(
                "error: --device cpu, but these targets declare GPUs (their compose overlays "
                f"attach one unconditionally): {', '.join(gpu_targets)}"
            )
        return None
    ids = []
    for v in values:
        if not v.startswith("cuda:") or not v[5:].isdigit():
            raise SystemExit(f"error: --device takes `cpu` or `cuda:<index>`, not {v!r}")
        ids.append(v[5:])
    return ids


def place(targets: list[Target], devices: list[str] | None, cpu_jobs: int) -> tuple[list[Placement], int]:
    """Per-target device env plus the worker count."""
    if devices is None:
        return [Placement({}, "cpu") for _ in targets], cpu_jobs
    out: list[Placement] = []
    slot = 0
    for target in targets:
        if task_gpus(target) == 0:
            out.append(Placement({}, "cpu"))
            continue
        gpu = devices[slot % len(devices)]
        slot += 1
        out.append(Placement({"RLEBENCH_GPU": gpu}, f"cuda:{gpu}"))
    return out, len(devices)


# --- one cell -------------------------------------------------------------------


def build(target: Target, family: Family, resolved: Resolved, opts: RunOptions,
          placement: Placement, *, log_to_file: bool) -> Invocation:
    run_cfg = family.run_cfg
    slug = model_slug(resolved.model)
    out_dir = f"{opts.jobs_dir}/{target.name}/{resolved.agent_dir}"

    argv = ["harbor", "run", "-p", target.path]
    if target.instance:
        argv += ["-i", target.instance]
    argv += ["-a", resolved.agent]
    if resolved.model:
        argv += ["-m", resolved.harbor_model or resolved.model]
    argv += ["-o", out_dir, "--job-name", slug]

    retries = opts.max_retries if opts.max_retries is not None else resolved.retries_default
    if retries is not None:
        argv += ["--max-retries", str(retries), "--retry-include", "ApiRateLimitError"]

    # The agent phase is the task's `[agent]` allowlist plus the model API. A
    # restricted environment also needs the hosts the agent installs from; a public
    # one ignores them.
    for host in resolved.api_hosts:
        argv += ["--allow-agent-host", host]
    if _task_environment(target).get("network_mode", "public") != "public":
        for host in resolved.setup_hosts + [h for h in resolved.api_hosts if h not in resolved.setup_hosts]:
            argv += ["--allow-environment-host", host]
    for value in resolved.agent_envs:
        argv += ["--agent-env", value]
    for value in resolved.agent_kwargs:
        argv += ["--ak", value]
    for value in resolved.open_book if opts.open_book else resolved.closed_book:
        argv += ["--ak", value]

    argv += ["--yes"]  # a task.toml `${VAR}` env would otherwise prompt on stdin
    if run_cfg.get("resume_trajectory") and opts.resume and resolved.agent != "oracle":
        argv += ["--resume-trajectory"]
    if opts.force_build:
        argv += ["--force-build"]
    argv += opts.extra

    env = _env(resolved.subprocess_env())
    env.update(placement.env)
    return Invocation(
        argv, env,
        label=f"[{placement.label}] {target.name} -> {out_dir}/{slug}/",
        console_log=REPO / out_dir / f"{slug}.console.log" if log_to_file else None,
    )


# --- execution ------------------------------------------------------------------


def execute(inv: Invocation, *, cwd: Path = REPO) -> int:
    argv = [harbor_bin()] + inv.argv[1:]
    if inv.console_log:
        inv.console_log.parent.mkdir(parents=True, exist_ok=True)
        with open(inv.console_log, "w") as log:
            return subprocess.run(argv, cwd=cwd, env=inv.env, stdout=log, stderr=log).returncode
    return subprocess.run(argv, cwd=cwd, env=inv.env).returncode


def execute_pool(invocations: list[Invocation], *, workers: int, cwd: Path = REPO) -> int:
    """At most `workers` cells in flight, submitted in order."""
    sem = threading.Semaphore(workers)
    codes: list[int] = []
    lock = threading.Lock()

    def one(inv: Invocation) -> None:
        try:
            code = execute(inv, cwd=cwd)
            with lock:
                codes.append(code)
        finally:
            sem.release()

    threads = []
    for inv in invocations:
        sem.acquire()
        print(inv.label)
        t = threading.Thread(target=one, args=(inv,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return max(codes, default=0)


def require_harbor() -> None:
    if not (REPO / ".venv/bin/harbor").exists() and not shutil.which("harbor"):
        print("error: harbor not on PATH", file=sys.stderr)
        raise SystemExit(1)

