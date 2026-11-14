# Task 05 — nanoVLA on LIBERO-10 and RoboTwin 2.0

The agent designs a vision-language-action policy end to end: model, training recipe and
serving code, in one `solution.py`, from demonstrations and a pretrained encoder bundle.
No baseline is supplied. The verifier replays the recipe under a fixed budget and scores
the served policy on episodes the agent never sees.

Four independent Harbor tasks, generated from one template:

| task                      | simulator    | encoders     | scored on                                        |
| ------------------------- | ------------ | ------------ | ------------------------------------------------ |
| `01-libero-open-design`   | LIBERO-10    | DINOv2-base  | the 500 official initial states                  |
| `02-libero-robustness`    | LIBERO-10    | open bundle  | 500 LIBERO-plus perturbed episodes               |
| `03-robotwin-open-design` | RoboTwin 2.0 | DINOv2-base  | 300 held-out clean-scene episodes                |
| `04-robotwin-robustness`  | RoboTwin 2.0 | open bundle  | 225 domain-randomised episodes, unseen phrasings |

Every cell gives the agent four hours on one H100, a 1,800 s replay budget, and reward
equal to success rate. The robustness cells develop on the clean distribution only, so the
shift they are scored under stays unseen.

## How it works

The agent container holds the demonstration shards, the encoder bundle at `/assets/hf`
and three helpers under `/workspace/nanovla`: the policy socket protocol, loaders for the
bundle models and a client for the evaluation service. It holds no simulator. An
evaluation service runs in a second container, reachable only through a shared socket
directory; it rolls out the agent's policy on fixed seeded development episodes and
meters a per-session episode budget. `nanovla-stage` copies a candidate to
`/logs/artifacts/submission/solution.py`, the only file that reaches the verifier.

The verifier runs the submission twice. `NANOVLA_MODE=train` replays it as uid 65534
under the budget and must write `ckpt.pt`; root freezes that into a read-only snapshot,
sweeps scratch, and starts `NANOVLA_MODE=serve` as uid 65533, which cannot read the
shards. Root then drives the scored episodes through the socket. The simulator, the
episode books and `/tests` are unreadable to both uids.

### Scoring

```
reward = success_rate over the fixed episode set; missing episodes count as failures
```

A run that completes fewer than 95% of its episodes is reported as a failed evaluation
rather than a low score. `reward.json` carries the total; `report.json` carries per-task
rates, the episode count, the checkpoint's parameter count and the replay wall time.
`media/` holds every successful clip among each task's first five scored episodes and
`rollouts.mp4`, one clip per task (`rlebench view`).

## Layout

```
_template/                 task.toml.in, README.md.in, tests/{Dockerfile.in,test.sh}, solution/solve.sh
subtasks.toml              per-subtask scalars: bundle, simulator, resources, timeouts
subtasks/<slug>/           instruction.md, compose overlays, tests/score_task.py,
                           verifier episode books, the Oracle solution.py
build_subtasks.py          the generator: --emit-all, --emit <slug>, --list, --check
harness/                   runtime/ (agent-visible), verifier/ (root-only), the handoff scripts
dev/                       shard preparation and manifest generation; never shipped
NN-slug/                   GENERATED: the Harbor tasks
```

Edit `_template/`, `subtasks.toml`, `subtasks/` or `harness/`, then `make task05-assets`.
The emitted trees are gitignored. Development episode lists live in
`harness/verifier/`; the scored books live under each subtask's `tests/`.

## Running it

```bash
rlebench prepare task05   # the four targets below

make sim-libero      # encoder bundles + the three agent images + the LIBERO verifier base
make sim-robotwin    # third_party/robotwin + the RoboTwin 2.0 simulator image + its verifier base
make task05          # emit the matrix, build the four per-subtask verifier images
make task05-data     # the shards and LIBERO-plus assets the cells mount, into third_party/task05
```

Every download is pinned to a commit and a sha256 (`harness/validate_assets.py`,
`sim/robotwin/pins.env`, `fetch_data.py`); `NANOVLA_HF_{BASE,DINOV2,OPEN}` substitute
prepared bundle directories. Budget roughly 500 GB of Docker storage and 35 GB under
`third_party/task05`.

At run time the cells mount only these read-only directories, readable by uids 1000,
65533 and 65534; `rlebench run` uses the `third_party/task05` copies unless a variable is set:

| variable                     | cells  | content                                                        |
| ---------------------------- | ------ | -------------------------------------------------------------- |
| `NANOVLA_SHARDS_L10`         | 01, 02 | LIBERO-10 shards (`shards_l10_128/`)                           |
| `NANOVLA_ROBOTWIN_SHARDS`    | 03, 04 | RoboTwin clean shards (`shards_rt15_128/`)                     |
| `NANOVLA_LIBERO_PLUS_ASSETS` | 02     | the LIBERO-plus `assets/` (`libero-plus-assets/`), verifier only |
| `NANOVLA_EVAL_LOGS`          | all    | optional host directory for the evaluation service's logs      |
| `NANOVLA_RESUME_DIR`         | 03, 04 | optional: an interrupted run's `ckpt.pt` and rows, to finish it |

`dev/prepare_libero10.py` and `dev/prepare_robotwin.py` record how the shards were cut.

`sim/libero/libero.sh check <slug>` validates the directories a cell mounts.

```bash
rlebench run task05/01-libero-open-design -a <agent> -m <model> -d cuda:0
harbor run -p tasks/task05/03-robotwin-open-design -a oracle
```

`rlebench run` fills `RLEBENCH_GPU` and opens the model API for the agent phase; the
compose overlays attach that one device to the agent container and the evaluation service. The Oracle solves each cell with a reference recipe
under `subtasks/<slug>/solution/`; it is trusted material and never enters an image.

`make test TASK=task05` runs the host-side contract and isolation checks.
