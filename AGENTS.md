# AGENTS.md — RLE-Bench

Instructions for Claude Code working in this repo. Read this before editing.

## What this is

RLE-Bench is a full-stack robotics engineering benchmark for coding agents,
packaged as [Harbor](https://github.com/laude-institute/harbor) tasks. Nine
families (task01–09) each hand the agent a real engineering problem —
learn policies under a metered simulator, engineer a VLA training recipe,
train a humanoid tracking controller, solve tabletop puzzles, design a robot
from stock parts, co-design teleop hardware, build a perception stack, ship a
bin-clearing policy — and score the submission with a verifier the agent never sees.

You are building the **harness** (golden models, metrics, scenario runners,
scorers), NOT solving the tasks. Keep each harness and its agent-facing task
cleanly separated.

## Hard invariants — do not violate

1. **Score against the harness's own instrumentation, never the agent's artifacts.**
   Checkpoints read the harness's own sim state. The agent-under-test's self-reported
   CSVs/plots are graded *separately* (checkpoints S6.*) for consistency only.
   Nothing in `tasks/task08/harness/base_design/` may trust files the agent produced as truth.
2. **The agent's environment must never contain ground truth.** Nothing under
   `tasks/task08/environment/` or in `instruction.md` may include or import
   `harness.base_design`, `tasks/task08/harness/assets/golden/`, or the reference design. Ground truth lives
   only in the verifier image (`tasks/task08/tests/`), which Harbor runs as a *separate*
   environment (`[verifier] environment_mode = "separate"` in `task.toml`). Keep
   this import/content boundary absolute.
3. **Determinism.** Pin MuJoCo version (`requirements.txt` and both Dockerfiles),
   timestep, solver iterations, and all seeds. No wall-clock, no unseeded RNG.
   Same input => same reward.
4. **Golden-first.** Never implement a stability check without a matching analytic
   unit test. The golden model is ground truth: it must score ~100 and never tip.

## Harbor task layout — follow exactly

Each task is a Harbor task directory. Its generated content (environment/assets,
tests/harness, tests/models, solution/payload) is GITIGNORED and rebuilt by
`make task-assets` (or `make taskNN`, which also builds the family's images) --
a fresh clone must run one of them before `harbor run`. Task 08:

```
tasks/task08/
├── task.toml              # metadata, timeouts, resources; [verifier] environment_mode = "separate"
├── instruction.md         # agent-facing prompt: spec, budgets, envelope (qualitative), I/O contract
├── environment/           # agent container — NO ground truth (invariant #2)
│   ├── Dockerfile         # pinned Python + MuJoCo, headless; same pins as requirements.txt
│   └── assets/            # component library + shelf scene + agent-facing harness API
├── solution/             # optional
│   └── solve.sh           # Harbor Oracle: reference solution or documented protocol smoke check
└── tests/                 # the verifier — scoring runs here, not in the agent container
    ├── Dockerfile         # installs the harness, golden controller, reference design (must contain /tests/test.sh)
    └── test.sh            # re-runs scenarios on the agent's submitted model, invokes the scorer,
                           # writes /logs/verifier/reward.json
```

Harbor rules to respect when touching anything under `tasks/`:

- **Reward file:** `tests/test.sh` must write `/logs/verifier/reward.json` —
  `{"reward": <total in [0,1]>, ...}` plus per-checkpoint breakdown fields
  (Harbor reads `reward.json` first, falls back to `reward.txt`). The total
  encodes the weighted stage sum with the Stage-1 gate already applied.
- **Artifact handoff:** the agent's deliverables (submitted model, controller,
  analysis CSVs) reach the separate verifier via `/logs/artifacts/`
  and the `artifacts = [...]` list in `task.toml`. The verifier treats them as
  untrusted input (invariant #1).
- **Absolute paths** in `test.sh` and any `solve.sh` (Harbor copies them to `/tests/`
  and `/solution/` at runtime).
- **Network:** default `network_mode = "no-network"` for both agent and verifier;
  everything needed is baked into the images.
- **Oracle solutions are optional.** A task may omit `solution/`, including
  per-step solutions. If provided, document whether the Oracle solves the task
  or only exercises the protocol. Validate full solutions against their expected
  scores; a protocol smoke check need not solve the task and does not establish
  solvability. Run `-a oracle` only where a runnable script is available.
  Existing golden-model, metric, and verifier checks remain required.

## Repo layout

One rule places every file: what consumes it, and whether it is code, data,
or a build input.

```
rlebench/            ONE Python package. core/: the task-agnostic spine
                     (scoring aggregate + gate, model validity) that families
                     import as rlebench.core and stage into their images.
                     The rest is host-only: the operations CLI
                     (list/prepare/run/clean/oracle/sweep/check/inspect/doctor/view)
                     and the Python it runs (taskgen, the inspectors).
                     agents/: host-side Harbor agent adapters. Ships into no image.
assets/robots/       vendored robot descriptions shared by families (franka,
                     ur5e, xarm7; LICENSE + VENDORED markers). Data, not code:
                     stagers copy them into image contexts by path.
sim/<layer>/         one directory per pinned external stack -- robocasa,
                     perception, motiontrack, libero, robotwin -- whether one family or
                     five use it: pins.env (the ONE source of truth), the
                     layer's entry script (`<layer>.sh`; make targets are a
                     thin facade), base Dockerfile(s), README. Never
                     task-specific content: a task image derives FROM these.
tasks/taskNN/        one Harbor task family, self-contained; nothing else at
                     this level:
  harness/           the family's shipped code + private assets, staged into
                     its images as the uniformly-named `harness` package
                     (in-image imports are harness.* + rlebench.core.*).
                     Exactly one family dir may sit on sys.path per process --
                     RLEBENCH_TEST_TASK selects it for tests
  dev/               what never ships: calibrators (`python -m dev.calibrate`),
                     generators, oracle references, audit data, ablations.
                     Stagers never copy it; that is the image boundary
  build_assets.py    stages the gitignored generated trees (environment
                     payloads, tests/harness, tests/models, solution/payload)
  task.toml, instruction.md, environment/, tests/, manifest.toml
  solution/          optional Oracle scripts and payloads
tasks/task01|02|03|04|05/  additionally emit their task matrices from _template/
                     via build_levels.py / build_groups.py / build_tasks.py /
                     build_subtasks.py.
                     task01/02/03 each own their harness/runtime/ source.
                     task03 stages its domain physics from tabletop/ into
                     the private harness.tabletop package.
third_party/         GITIGNORED vendoring, one subdir per sim layer: robosuite
                     + robocasa + the ~15 GB dataset, perception, motiontrack,
                     libero (the encoder bundles), robotwin, task05 (the data
                     its cells mount).
                     Never check in an absolute path
tests/               host-side pytest — NOT the Harbor verifiers
```

Ground truth (golden robots, hidden seeds) lives only in each family's
harness/ and its verifier image; agent environments get curated copies
staged by build_assets.py, never a path into the harness.

## Commands (keep these green)

The Makefile has one scheme: `sim-<layer>` builds a shared simulator layer
(image, sources, dataset, dev venv; downloads land in `third_party/`) and
`sim-<layer>-clean` tears it down,
`taskNN` / `taskNN-assets` / `taskNN-clean` build, stage and tear down one
family, and everything else is `test-*`, `check-*` or `calibrate-*` keyed by
family id.

- `make install` — pinned deps (uv-managed .venv, includes harbor + tooling)
- `make test TASK=<family>` / `make test-all` — per-family and full dev
  suites; the feedback loop, must always pass
- `make test-task08-metrics` — fast pure-math checks
- `make test-task08-golden` — regression guard: the task08 reference design scores
  at its bar, aces every bounded checkpoint and never tips
- `make calibrate-task08` — sweep/pin numeric thresholds against the golden model
- `make task-assets` — rebuild every task's generated tree from its harness/;
  required after a fresh clone and after relevant changes under that package
- `make sim-robocasa` / `make sim-motiontrack` / ... — a shared simulator layer
- `rlebench run task08 -a oracle` — end-to-end check: reference solution through
  the real container + verifier path (`harbor run -p tasks/task08 -a oracle`)
- `rlebench run <target>... -a <agent> -m <model> [-e vendor/lane] [--device cuda:N ...]`
  — evaluate agents; targets are families, levels/groups or single cells, and
  rlebench adds only the task-side extras harbor cannot know (egress lane,
  `--override-gpus 0`, resume, closed-book, dataset mount, device placement)
- `rlebench view [jobs]` — browse a jobs tree in the browser: harbor view's
  hierarchy and tabs plus the verifiers' media (`verifier/media/`)
- `rlebench doctor` / `list` / `prepare <family>` — environment health, the
  task registry, one-command per-family preparation (tasks/*/manifest.toml)

## Physics notes

- **FASM is the primary dynamic criterion** (Papadopoulos & Rey 1996): valid on
  slopes/uneven contact. **ZMP is flat-ground cross-check only** — do not use it on
  terrain tests.
- Continuous margins score with a **saturating reward capped at threshold**, so
  over-building a huge/heavy base earns nothing extra. Pair with the footprint/mass
  budget checkpoint.
- Stage-1 validity is a **gate**: if the model is invalid, cap the total (GATE_CAP);
  downstream stability numbers are meaningless otherwise.

## Conventions

- Python, numpy, MuJoCo 3.x. Pure functions for metrics (easy to unit-test).
- Every numeric threshold lives in one place (emitted by `calibrate`), never
  hardcoded across files.
- The reward file is written only by the verifier (each task's `tests/test.sh`);
  no other code path writes to `/logs/verifier/`.

## For code testing
Task 01 and Task 02 ship a suite of tasks which barely differ each other. Therefore, if you want to do make test, only select a few subtasks to build, rather than the full suite. 

## IMPORTANT
+ Avoid being too specific and redundant in code, comment, and README. Be concise. 
+ Avoid development narrative in agent-readable scripts.
+ For tasks that use filesystem privilege to separate agent and the simulator, always check that the agent cannot read priviledged information.
