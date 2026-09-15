# RLE-Bench build chain. Four sections, one naming scheme:
#   sim-<layer>[-clean]    a shared simulator layer: image, sources, dataset, dev venv / teardown
#   taskNN[-assets|-clean] one Harbor family: images / the docker-free staging / teardown
#   test-* check-* calibrate-*
# Logic lives in sim/<layer>/<layer>.sh and each family's build_*.py; recipes only
# sequence them. PY is the repo venv; PY_<family> (tests/suites.mk) overrides it for
# the families whose mujoco pin needs its own venv.
PY ?= .venv/bin/python
PYTEST_ARGS ?=
FAMILIES := task01 task02 task03 task04 task05 task06 task07 task08 task09 task10
include tests/suites.mk
include sim/robocasa/pins.env
include sim/perception/pins.env

.PHONY: install sim sim-robocasa sim-perception sim-motiontrack sim-libero sim-robotwin \
        sim-clean sim-robocasa-clean sim-perception-clean sim-motiontrack-clean \
        sim-libero-clean sim-robotwin-clean \
        tasks tasks-clean task-assets $(FAMILIES) $(addsuffix -assets,$(FAMILIES)) $(addsuffix -clean,$(FAMILIES)) \
        test test-all test-host check-harbor \
        check-taskgen lint lint-strict task05-data

# --- host ---------------------------------------------------------------------
# uv, not `python3 -m venv`: host pythons routinely ship without ensurepip.
# Two installs: requirements.txt carries the pytorch extra index, whose stale
# `packaging` would win under uv's first-index strategy and conflict with harbor's.
install:
	uv venv --seed .venv --python 3.12
	uv pip install --python .venv/bin/python -r requirements.txt
	uv pip install --python .venv/bin/python -r requirements-dev.txt
	uv pip install --python .venv/bin/python -e .

# --- simulator layers ---------------------------------------------------------
# One target per pinned external stack, shared by every family that derives from
# it: everything the layer provides, idempotently. Sources and datasets land under
# third_party/ (gitignored), dev venvs at .venv-<layer>. Finer control (assets
# pack/mount/verify, --no-assets, ...) is the layer script: sim/<layer>/<layer>.sh help.
# ARGS reaches the script's `setup`, e.g. `make sim-robocasa ARGS=--no-assets`.
sim: sim-robocasa sim-perception sim-motiontrack sim-libero sim-robotwin

# simulator image + robosuite/robocasa sources + the ~15 GB dataset + .venv-robocasa (task01/02/03)
sim-robocasa:
	sim/robocasa/robocasa.sh image
	sim/robocasa/robocasa.sh setup $(ARGS)

# SAM3 + Contact-GraspNet under third_party/perception; the gated weights need HF_TOKEN (task01 L2/L3)
sim-perception:
	sim/perception/perception.sh setup

# G1 model + LAFAN1 clips + .venv-motiontrack (task04)
sim-motiontrack:
	sim/motiontrack/motiontrack.sh setup $(ARGS)

# the nanoVLA encoder bundles + agent images + LIBERO verifier base (task05 01/02);
# NANOVLA_HF_{BASE,DINOV2,OPEN} name the validated HF_HOME directories to bake in
sim-libero:
	sim/libero/libero.sh bundles
	sim/libero/libero.sh base

# the RoboTwin 2.0 simulator image + its verifier base (task05 03/04)
sim-robotwin:
	sim/robotwin/robotwin.sh images

# sim-<layer>-clean removes what the layer put on this machine: its third_party/
# tree, dev venv and image. Family images are taskNN-clean; run those first.
sim-clean: sim-robocasa-clean sim-perception-clean sim-motiontrack-clean sim-libero-clean sim-robotwin-clean
sim-robocasa-clean sim-perception-clean sim-motiontrack-clean sim-libero-clean sim-robotwin-clean: sim-%-clean:
	sim/$*/$*.sh clean

# --- tasks --------------------------------------------------------------------
# taskNN-assets  the docker-free step: stage the payload trees (06/07/08/09) or emit
#                the task matrix from _template/ (01/02/03/04); all gitignored
# taskNN         assets + every image the family runs from (its sim-* layer first)
# taskNN-clean   remove the generated trees and the images
# task01-L2 / task02-<NN-slug> emit one level / one group (task02: and the shared image).
tasks: $(FAMILIES)
tasks-clean: $(addsuffix -clean,$(FAMILIES))
task-assets: $(addsuffix -assets,$(FAMILIES))

# The agent's uid inside the container. Never 0: the metering guarantee is uid-based.
# 1000 keeps images portable; RLEBENCH_DEBUG=1 bakes in YOUR uid so the agent's
# 0600 session trace is readable while a run is still going.
RLEBENCH_DEBUG ?= 0
ifeq ($(RLEBENCH_DEBUG),0)
AGENT_UID ?= 1000
else
AGENT_UID ?= $(shell u=$$(id -u); if [ "$$u" -eq 0 ]; then echo 1000; else echo $$u; fi)
endif

task01-assets:
	python3 tasks/task01/build_levels.py --emit-all
	python3 tasks/task01/build_levels.py --check
	python3 tasks/task01/build_assets.py --check

task02-assets:
	python3 tasks/task02/build_groups.py --emit-all
	python3 tasks/task02/build_groups.py --check
	python3 tasks/task02/build_assets.py --check

task04-assets:
	python3 tasks/task04/build_tasks.py --emit-all
	python3 tasks/task04/build_tasks.py --check

task05-assets:
	python3 tasks/task05/build_subtasks.py --emit-all
	python3 tasks/task05/build_subtasks.py --check

# the shards and LIBERO-plus assets the task05 cells mount, into third_party/task05
task05-data:
	$(PY) tasks/task05/fetch_data.py

task03-assets:
	python3 tasks/task03/build_levels.py --emit-all
	python3 tasks/task03/build_levels.py --check
	python3 tasks/task03/build_assets.py --check

# Matrix emitters are stdlib-only, so they run on bare python3; the stagers need the venv.
task06-assets task07-assets task08-assets task09-assets task10-assets: task%-assets:
	$(PY) tasks/task$*/build_assets.py

# task01: one image per harness level, rlebench-task01-l{1,2,3}-agent:dev (the level decides
# what the agent's uid can read). L1 is the control condition: it selects the empty
# model stage, so it needs neither the perception models nor the HF token.
TASK01_LEVELS ?= L1 L2 L3
PERCEPTION_VENDOR ?= $(CURDIR)/third_party/perception
task01_models = $(if $(filter L1,$(1)),--build-arg MODELS=models-off,\
    --build-arg MODELS=models-on \
    --build-arg TORCH_VERSION=$(TORCH_VERSION) \
    --build-arg SAM3_SHA=$(SAM3_SHA) \
    --build-arg CGN_SHA=$(CGN_SHA) \
    --build-context perception=$(PERCEPTION_VENDOR))

# The model fetch comes before the loop: a missing token must stop the run in a
# second, not after L1's image is built. `setup` is a no-op once the tree is complete.
task01: task01-assets
	@if [ -n "$(filter-out L1,$(TASK01_LEVELS))" ]; then sim/perception/perception.sh setup; fi
	@for l in $(TASK01_LEVELS); do \
	    echo "==> building rlebench-task01-$$l-agent"; \
	    $(MAKE) --no-print-directory task01-$$l || exit 1; \
	done
	@echo "built every task01 level image, agent uid $(AGENT_UID)"

# One level. The tag is lowercased because a Docker repository name must be.
task01-%:
	python3 tasks/task01/build_levels.py --emit $*
	@if [ "$*" != "L1" ]; then sim/perception/perception.sh setup; fi
	@tag=$$(echo $* | tr A-Z a-z); \
	docker build -t rlebench-task01-$$tag-agent:dev \
	    --build-arg SIM_IMAGE=$(ROBOCASA_SIM_IMAGE) \
	    --build-arg AGENT_UID=$(AGENT_UID) \
	    --build-arg LEVEL=$* \
	    $(call task01_models,$*) \
	    tasks/task01/images/$* \
	&& echo "built rlebench-task01-$$tag-agent:dev, agent uid $(AGENT_UID)"

# task02: one image for every activity group, rlebench-task02-agent:dev; the group is
# selected per task.toml (RLEBENCH_GROUP).
task02: task02-assets
	docker build -t rlebench-task02-agent:dev \
	    --build-arg SIM_IMAGE=$(ROBOCASA_SIM_IMAGE) \
	    --build-arg AGENT_UID=$(AGENT_UID) tasks/task02/image
	@echo "built rlebench-task02-agent:dev from $(ROBOCASA_SIM_IMAGE), agent uid $(AGENT_UID)"

# One group: `make task02-03-setting-the-table` (emits it, rebuilds the shared image).
task02-%:
	python3 tasks/task02/build_groups.py --emit $*
	docker build -t rlebench-task02-agent:dev \
	    --build-arg SIM_IMAGE=$(ROBOCASA_SIM_IMAGE) \
	    --build-arg AGENT_UID=$(AGENT_UID) tasks/task02/image

# task03: standalone robosuite, with no perception models or dataset.
task03: task03-assets
	docker build -t rlebench-task03-agent:dev \
	    --build-arg ROBOSUITE_SHA=$(ROBOSUITE_SHA) \
	    --build-arg AGENT_UID=$(AGENT_UID) \
	    tasks/task03/image
	docker build -t rlebench-task03-pocket-agent:dev tasks/task03/04-rubik-cube/environment
	docker build -t rlebench-task03-hidden-com-agent:dev tasks/task03/05-hidden-center-of-mass/environment
	@echo "built task03 images, agent uid $(AGENT_UID)"

# task04: the agent + verifier pair, built from the repo root with the LAFAN1 clips
# and G1 meshes inside (sim-motiontrack first); the five task directories share them.
task04: task04-assets
	sim/motiontrack/motiontrack.sh images

# task05: the four per-subtask verifier images, on top of sim-libero's bases (01/02)
# and sim-robotwin's (03/04). The three agent images are the sim-libero layer's.
task05: task05-assets
	sim/libero/libero.sh verifiers

# task06/08/09 verifiers build from the REPO ROOT so rlebench.core ships from its one source.
task06: task06-assets
	@for v in rgb-only rgb-depth rgb-depth-model-training method-agnostic; do \
	    docker build -t rlebench-task06-$$v-agent:dev tasks/task06/$$v/environment || exit 1; \
	    docker build -f tasks/task06/$$v/tests/Dockerfile -t rlebench-task06-$$v-verifier:dev . || exit 1; \
	done

task07: task07-assets
	docker build -t rlebench-task07-agent:dev tasks/task07/environment
	docker build -t rlebench-task07-verifier:dev tasks/task07/tests

task10: task10-assets
	docker build -t rlebench-task10-agent:dev tasks/task10/environment
	docker build -t rlebench-task10-verifier:dev tasks/task10/tests

task08 task09: task%: task%-assets
	docker build -t rlebench-task$*-agent:dev tasks/task$*/environment
	docker build -f tasks/task$*/tests/Dockerfile -t rlebench-task$*-verifier:dev .

# The generated trees per family (what .gitignore lists), then the images by tag.
CLEAN_task01 := tasks/task01/L[0-9] tasks/task01/images
CLEAN_task02 := tasks/task02/[0-9][0-9]-* tasks/task02/image
CLEAN_task03 := tasks/task03/[0-9][0-9]-* tasks/task03/image tasks/task03/harness
CLEAN_task04 := tasks/task04/[0-9][0-9]-*
CLEAN_task05 := tasks/task05/[0-9][0-9]-* third_party/task05
# The agent and bundle images belong to sim-libero, not to the family: only the
# per-subtask verifiers are task05-clean's to remove.
IMAGES_task05 := 'rlebench-task05-*-verifier'
CLEAN_task06 := tasks/task06/*/environment/agent tasks/task06/*/environment/private tasks/task06/*/solution/payload \
                tasks/task06/*/tests/assets tasks/task06/*/tests/harness tasks/task06/*/tests/rlebench
CLEAN_task07 := tasks/task07/environment/assets tasks/task07/solution/payload tasks/task07/tests/score_task.py \
                tasks/task07/tests/harness tasks/task07/tests/assets
CLEAN_task08 := tasks/task08/environment/assets tasks/task08/tests/harness tasks/task08/tests/rlebench \
                tasks/task08/tests/models tasks/task08/solution/payload tasks/task08/reference
CLEAN_task09 := tasks/task09/environment/assets tasks/task09/solution/payload tasks/task09/tests/harness \
                tasks/task09/tests/rlebench tasks/task09/tests/models
CLEAN_task10 := tasks/task10/environment/assets tasks/task10/environment/dev_runner.py \
                tasks/task10/solution/payload tasks/task10/tests/harness tasks/task10/tests/assets
$(addsuffix -clean,$(FAMILIES)): task%-clean:
	rm -rf $(CLEAN_task$*)
	docker images -q --filter reference=$(or $(IMAGES_task$*),'rlebench-task$**') | xargs -r docker rmi -f

# --- tests --------------------------------------------------------------------
# One family per pytest process: the per-task `harness` packages share a top-level
# name on purpose, so they can never be imported together. tests/suites.mk owns
# the file lists and each family's interpreter.
#   make test TASK=task09      (or make test-task09)
#   make test-task09-metrics   fast pure-math checks
#   make test-task09-golden    regression guard: the reference scores at its bar
test:
	@if [ -z "$(TASK)" ]; then \
		echo "Choose a task: make test TASK=<task-id>; choices: $(TASK_IDS)"; \
		exit 2; \
	fi
	@if ! echo " $(TASK_IDS) " | grep -q " $(TASK) "; then \
		echo "Unknown TASK=$(TASK); choose one of: $(TASK_IDS)"; \
		exit 2; \
	fi
	RLEBENCH_TEST_TASK=$(TASK) $(or $(PY_$(TASK)),$(PY)) -m pytest -q $(PYTEST_ARGS) $(TASK_TESTS_$(TASK))

# Every .venv family plus the cross-cutting modules. task04 and the simulator suites
# need their own venvs and run through their per-family targets.
test-all:
	@for t in task01 task02 task03 task05 task06 task07 task08 task09 task10; do \
	    $(MAKE) -s test TASK=$$t || exit 1; \
	done
	$(MAKE) -s test-host

test-host:
	RLEBENCH_TEST_TASK= $(PY) -m pytest -q $(PYTEST_ARGS) $(HOST_TESTS)

test-%:
	$(MAKE) test TASK=$*

test-%-metrics:
	@[ -n "$(TASK_METRICS_$*)" ] || { echo "no metrics suite for $*"; exit 2; }
	RLEBENCH_TEST_TASK=$* $(or $(PY_$*),$(PY)) -m pytest -q $(PYTEST_ARGS) $(TASK_METRICS_$*)

test-%-golden:
	@[ -n "$(TASK_GOLDEN_$*)" ] || { echo "no golden suite for $*"; exit 2; }
	RLEBENCH_TEST_TASK=$* $(or $(PY_$*),$(PY)) -m pytest -q $(PYTEST_ARGS) $(TASK_GOLDEN_$*)

# --- checks -------------------------------------------------------------------
# The dev harbor matches the pin.
check-harbor:
	@pin=$$(sed -n 's/^harbor==\(.*\)$$/\1/p' requirements-dev.txt); \
	actual=$$(.venv/bin/python -c "from importlib.metadata import version; print(version('harbor'))" 2>/dev/null); \
	if [ "$$pin" != "$$actual" ]; then echo "harbor '$$actual' != pinned '$$pin' (requirements-dev.txt)"; exit 1; fi; \
	echo "harbor $$actual matches pin"

# Every generator emits and self-checks, and the committed generator-owned trees
# (task06, task07, task09) stay byte-stable. No docker, no GPU, no dataset.
check-taskgen: task-assets
	git diff --exit-code -- tasks/task06 tasks/task07 tasks/task09

lint:
	$(PY) -m rlebench.lint

# Retain the CI entrypoint; local and CI lint enforce the same scope.
lint-strict: lint

# --- calibrate ----------------------------------------------------------------
# Re-pin a family's numeric thresholds against its golden model: `make calibrate-task08`.
CALIBRATE_ARGS_task04 := --emit tasks/task04/dev/oracle_ref
calibrate-%:
	PYTHONPATH=tasks/$* $(or $(PY_$*),$(PY)) -m dev.calibrate $(CALIBRATE_ARGS_$*)
