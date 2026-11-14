"""rlebench — one entry point for preparing and running the benchmark.

  rlebench list                                    what exists, what is prepared
  rlebench prepare task01                          simulator layer + generated trees + images
  rlebench run task08 -a claude-code -m anthropic/claude-opus-5
  rlebench run task08 -a agy -e gemini/api -m gemini-3.7-flash
  rlebench run task01/L1 -a codex -m openai/gpt-5.6-sol --device cuda:0 cuda:1
  rlebench run task06/rgb-only -a oracle           reference solution end-to-end
  rlebench run task02/06-setting-the-table -a cc -e kimi/kimi-code -m 'kimi/k3[1m]'
  rlebench clean task08 -a claude-code -m anthropic/claude-opus-5   the job dirs a run would write
  rlebench check                                   harbor pin + generator self-checks
  rlebench summarize jobs/task01                   job statuses, scores and group means
  rlebench doctor                                  environment health
  rlebench view [jobs]                      browse a jobs tree in the browser

A target is a family (task01: every cell), a level or group (task01/L1,
task02/06-setting-the-table) or one cell (task01/L1/01-open-fridge). `-a` is
the scaffold, `-m <provider>/<model_name>` the model, and `-e <vendor>/<lane>`
the endpoint and credentials (`rlebench run --help` lists them). rlebench adds what harbor
cannot know: the agent's install and API hosts,
`--resume-trajectory` where supported, the closed-book switch, the dataset
mount and the device placement (`--device cpu | cuda:N ...`). Anything after `--`
reaches harbor verbatim.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import providers, runner
from .repo import REPO, Target, families, read_task_toml, resolve
from .runner import Invocation


def _print_dry(invocations: list[Invocation]) -> None:
    for inv in invocations:
        print(inv.label)
        print(f"  {inv.display()}")


def cmd_list(_: argparse.Namespace) -> int:
    from .prepare import status

    fams = families()
    print(f"{'family':10s} {'targets':>7s}  {'gpu':3s}  {'agent(s)':>9s}  {'verifier(s)':>11s}  prepared")
    for name, fam in fams.items():
        targets = fam.targets()
        gpu = "yes" if fam.run_cfg.get("gpu") else "no"
        agent_t = verif_t = "?"
        if targets:
            toml_path = REPO / targets[0].path / (
                f"{targets[0].instance}/task.toml" if targets[0].instance else "task.toml"
            )
            if toml_path.exists():
                cfg = read_task_toml(toml_path)
                if cfg.get("steps"):
                    agent_t = "+".join(str(int(s.get("agent", {}).get("timeout_sec", 0))) for s in cfg["steps"])
                    verif_t = "+".join(str(int(s.get("verifier", {}).get("timeout_sec", 0))) for s in cfg["steps"])
                else:
                    agent_t = str(int(cfg.get("agent", {}).get("timeout_sec", 0)))
                    verif_t = str(int(cfg.get("verifier", {}).get("timeout_sec", 0)))
        st = status(fam)
        ready = f"{sum(st.values())}/{len(st)}" if st else "-"
        print(f"{name:10s} {len(targets):7d}  {gpu:3s}  {agent_t:>9s}  {verif_t:>11s}  {ready}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .prepare import prepare

    fams = families()
    code = 0
    for name in args.family:
        fam = fams.get(name.split("/")[0])
        if fam is None:
            print(f"error: unknown family {name!r}", file=sys.stderr)
            return 2
        print(f"=== prepare {fam.name}")
        code = prepare(fam, dry_run=args.dry_run) or code
    return code


def build_run(args: argparse.Namespace, extra: list[str]) -> tuple[list[Target], list[Invocation], int]:
    """Targets x (agent, config, model) -> invocations and the worker count."""
    resolved = providers.resolve(args.agent, args.model, args.endpoint)
    for w in resolved.warnings:
        print(f"warning: {w}", file=sys.stderr)
    fams = families()
    targets = [t for name in args.task for t in resolve(name)]
    devices = runner.parse_devices(args.device, targets)
    placements, workers = runner.place(targets, devices, args.jobs)
    opts = runner.RunOptions(
        jobs_dir=args.jobs_dir, open_book=args.open_book, resume=not args.no_resume,
        force_build=args.force_build, max_retries=args.max_retries, extra=extra,
    )
    invocations = [
        runner.build(t, fams[t.name.split("/")[0]], resolved, opts, p, log_to_file=len(targets) > 1)
        for t, p in zip(targets, placements)
    ]
    return targets, invocations, workers


def cmd_run(args: argparse.Namespace) -> int:
    targets, invocations, workers = build_run(args, args.extra)
    if args.dry_run:
        _print_dry(invocations)
        return 0
    runner.require_harbor()
    fams = families()
    for target, inv in zip(targets, invocations):
        family = target.name.split("/")[0]
        for var in fams[family].run_cfg.get("requires_env", []):
            value = inv.env.get(var, "")
            if not os.path.isdir(value):
                raise SystemExit(
                    f"error: {var}={value!r} is not a directory; run `make sim-robocasa` or "
                    f"export it (see tasks/{family}/README.md)"
                )
    return runner.execute_pool(invocations, workers=workers)


def job_dirs(args: argparse.Namespace) -> list[Path]:
    """The job dirs `run` with the same selectors writes; without -a, every agent's."""
    if args.agent:
        _, invocations, _ = build_run(args, [])
        return [REPO / inv.argv[inv.argv.index("-o") + 1] / inv.argv[inv.argv.index("--job-name") + 1]
                for inv in invocations]
    return [REPO / args.jobs_dir / t.name for name in args.task for t in resolve(name)]


def cmd_clean(args: argparse.Namespace) -> int:
    for job_dir in job_dirs(args):
        if not job_dir.exists():
            continue
        print(f"{'would remove' if args.dry_run else 'removing'} {job_dir}")
        if not args.dry_run:
            shutil.rmtree(job_dir)
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    steps = [["make", "check-harbor"], ["make", "check-taskgen"]]
    for argv in steps:
        print(f"$ {' '.join(argv)}")
        code = subprocess.run(argv, cwd=REPO).returncode
        if code != 0:
            return code
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from .doctor import report

    return report()


def cmd_summarize(args: argparse.Namespace) -> int:
    from .summarize import summarize

    return summarize(Path(args.folder))


def cmd_view(args: argparse.Namespace) -> int:
    from .view.server import parse_ports, serve

    return serve(Path(args.folder), args.host, parse_ports(args.port))


class _SingleAgent(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error("agent configs are duplicated; specify -a/--agent only once before `--`")
        setattr(namespace, self.dest, values)


def parse(argv: list[str]) -> argparse.Namespace:
    """argv -> namespace; `run` keeps everything after `--` in `.extra`."""
    argv = list(argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        agent_parser = argparse.ArgumentParser(prog="rlebench run", add_help=False, allow_abbrev=False)
        agent_parser.add_argument("-a", "--agent", action=_SingleAgent)
        agent_parser.parse_known_args(extra, namespace=argparse.Namespace(agent=args.agent))
        args.extra = extra
    elif extra:
        parser.error("`--` is only meaningful for `run`")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(sys.argv[1:] if argv is None else argv)
    return args.fn(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlebench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="families, targets, timeouts, prepared status").set_defaults(fn=cmd_list)

    p = sub.add_parser("prepare", help="simulator layer + generated trees + docker images for a family")
    p.add_argument("family", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser(
        "run", help="run targets through harbor with the task-side extras applied",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="endpoints (-e): " + ", ".join(providers.CONFIGS)
        + "\ndefaults: " + ", ".join(f"{a} -> {c}" for a, c in providers.DEFAULT_CONFIG.items())
        + "\nagent aliases: " + ", ".join(f"{a} -> {b}" for a, b in providers.AGENT_ALIASES.items())
        + "\nanything after `--` is passed to harbor verbatim",
    )
    p.add_argument("task", nargs="+", help="task01 | task01/L1 | task01/L1/01-open-fridge | rgb-only ...")
    p.add_argument("-a", "--agent", action=_SingleAgent, required=True, metavar="SCAFFOLD",
                   help="scaffold: claude-code, codex, agy, grok-build (oracle needs no -m/-e)")
    p.add_argument("-m", "--model", metavar="MODEL",
                   help="provider/model; bare model names also work with gemini/api or a custom Codex/Claude Code API base URL")
    p.add_argument("-e", "--endpoint", help="<vendor>/<lane> endpoint + credentials (gemini/api reads GEMINI_API_KEY)")
    p.add_argument("--device", nargs="+", metavar="DEV", help="cpu | cuda:N [cuda:M ...] (a pool)")
    p.add_argument("--jobs", type=int, default=1, help="concurrent cells on cpu (GPU runs use the pool size)")
    p.add_argument("-o", "--jobs-dir", default="jobs")
    p.add_argument("--max-retries", type=int)
    p.add_argument("--open-book", action="store_true", help="leave the agent's web tools enabled")
    p.add_argument("--no-resume", action="store_true", help="skip --resume-trajectory where the family uses it")
    p.add_argument("--force-build", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("clean", help="remove the job dirs a run with the same selectors writes")
    p.add_argument("task", nargs="+")
    p.add_argument("-a", "--agent", action=_SingleAgent, help="only this scaffold's dirs (omit: every scaffold's)")
    p.add_argument("-m", "--model", metavar="MODEL",
                   help="provider/model; bare model names also work with gemini/api or a custom Codex/Claude Code API base URL")
    p.add_argument("-e", "--endpoint")
    p.add_argument("-o", "--jobs-dir", default="jobs")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_clean, device=None, jobs=1, open_book=False, no_resume=False,
                   force_build=False, max_retries=None)

    p = sub.add_parser("check", help="harbor pin + generator self-checks")
    p.set_defaults(fn=cmd_check)

    sub.add_parser("doctor", help="environment health checks").set_defaults(fn=cmd_doctor)

    p = sub.add_parser("summarize", help="job statuses, scores, costs and group/agent/model statistics; writes summary.json and report.md")
    p.add_argument("folder", nargs="?", default="jobs", help="jobs directory, one job, or saved summary.json (default: jobs)")
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("view", help="browse a jobs tree in the browser: rewards, trajectories, verifier media")
    p.add_argument("folder", nargs="?", default="jobs", help="a jobs directory or one job (default: jobs)")
    p.add_argument("-p", "--port", default="8080-8089", help="port or range, first free one is used")
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(fn=cmd_view)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
