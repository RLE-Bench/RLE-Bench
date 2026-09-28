"""Persistent workflow and metering, independent of simulator implementation."""
import asyncio
import hashlib
import math
import time

from . import protocol as P
from .process import Worker, WorkerFailure, INITIALIZE_SECONDS, FINISH_SECONDS
from .store import Store


class Engine:
    def __init__(self, root, config, worker_factory=Worker):
        self.root, self.config = root, config
        self.worker_factory = worker_factory
        development = config["mode"] in ("task01", "task02")
        self.store = Store(root / "ledger.sqlite", dict(
            config=config, phase="development" if development else "evaluation",
            trial=-1 if development else 0, dev_steps=0, episode_steps=0,
            episodes=0, episode_over=True, ended=None, ready=True,
            task=config.get("train", [None])[0], results={}, progress={}, pending=None,
            finished_steps=[], last_request=None, failures=0))
        self.s = self.store.state
        if self.s["config"] != config:
            raise ValueError("configuration changed during a run")
        self.worker = None
        self.boot = None
        self.info = {}
        self.cached = {}

    @property
    def mode(self):
        return self.config["mode"]

    def save(self, kind, **data):
        self.store.save(kind, data)

    def status(self):
        development = self.s["phase"] == "development"
        limit = self.config["budget"] if development else self.config["horizon"]
        used = self.s["dev_steps"] if development else self.s["episode_steps"]
        if self.mode == "tabletop":
            used, limit = self.s["dev_steps"], self.config["budget"]
        clock = self.s.get("clock")
        seconds = self.config.get("seconds", {}).get(
            "session" if self.mode == "task01" else "develop" if development else "evaluate")
        remaining = max(0., seconds-(time.time()-clock)) if seconds and clock else seconds
        return dict(phase=self.s["phase"], ready=self.s["ready"],
                    episode_over=self.s["episode_over"], ended=self.s["ended"],
                    steps_used=used, steps_remaining=max(0, limit-used),
                    trial_index=self.s["trial"], total_trials=len(self.config["plan"]),
                    last_request=self.s["last_request"], seconds_remaining=remaining,
                    interaction_budget=self.config["budget"], done=self.s["phase"] == "finished",
                    task=self.s["task"] if self.mode != "task02" or development else None,
                    total_steps=sum(r["steps"] for i,r in self.s["results"].items() if int(i)<self.s["trial"])+self.s["episode_steps"])

    def task_info(self):
        return {**self.status(), **self.config["public"], **self.info,
                "task": self.s["task"] if self.s["phase"] == "development" else None,
                "interaction_budget": self.config["budget"],
                "max_steps_per_trial": self.config["horizon"]}

    async def recover(self):
        if self.s["phase"] == "finished":
            return
        if self.s["pending"] or not self.s["episode_over"] or not self.s["ready"]:
            await self.failed()
        elif self.s["phase"] == "evaluation" and not self.s["results"] and self.s["ended"] is None:
            self.start_trial()

    async def stop(self):
        if self.boot is not None and self.boot is not asyncio.current_task():
            self.boot.cancel()
            await asyncio.gather(self.boot, return_exceptions=True)
            self.boot = None
        if self.worker:
            worker, self.worker = self.worker, None
            await worker.stop()

    def charge(self, operation):
        if self.s["phase"] == "development" or self.mode == "tabletop":
            self.s["dev_steps"] += 1
        if operation != "reset":
            self.s["episode_steps"] += 1
        self.s["pending"] = operation
        self.save("reserved", operation=operation)

    def confirmed(self, evidence):
        self.s["pending"] = None
        self.s["progress"] = evidence
        self.save("confirmed", evidence=evidence)

    async def failed(self, error=None):
        self.s.update(episode_over=True, ready=True, ended="sim_error",
                      failures=self.s["failures"] + 1)
        self.s["pending"] = None
        if self.s["phase"] == "evaluation":
            self.record()
        details = getattr(error, "details", None) or dict(kind="unavailable", operation="recovery")
        self.s.setdefault("failure_details", []).append(dict(trial_index=self.s["trial"], **details))
        self.save("worker_failure", details=details)
        if self.worker:
            worker, self.worker = self.worker, None
            await worker.stop()

    def record(self):
        index = str(self.s["trial"])
        if index not in self.s["results"]:
            self.s["results"][index] = dict(evidence=self.s["progress"],
                steps=self.s["episode_steps"], ended=self.s["ended"])
            self.save("trial_end", index=self.s["trial"])

    def descriptor(self):
        return dict(self.config["plan"][self.s["trial"]], media_name=f'trial-{self.s["trial"]+1:02d}')

    async def open(self, descriptor):
        await self.stop()
        self.s.update(ready=False, episode_over=True, ended=None, episode_steps=0, progress={})
        self.info, self.cached = {}, {}
        self.save("initializing")
        self.worker = self.worker_factory(self.root)
        try:
            result = await self.worker.start(self.config["adapter"], descriptor)
            self.info = result["info"]
            self.confirmed(result["evidence"])
            self.s.update(ready=True, episode_over=False)
            self.save("ready")
        except WorkerFailure as exc:
            await self.failed(exc)

    def start_trial(self):
        self.s.update(ready=False, episode_over=True, ended=None, episode_steps=0, progress={}, pending=None)
        self.save("assigned", index=self.s["trial"])
        self.boot = asyncio.create_task(self.open(self.descriptor()))

    async def wait_ready(self):
        if self.boot and self.boot is not asyncio.current_task():
            await asyncio.shield(self.boot)
            self.boot = None

    async def observe(self, spec):
        await self.wait_ready()
        if not self.worker or self.s["episode_over"]:
            return dict(obs={}, live=False, **self.status())
        try:
            self.cached = await self.worker.call("observe", spec=spec)
            return {**self.cached, "live": True, **self.status()}
        except WorkerFailure as exc:
            await self.failed(exc)
            return dict(obs={}, live=False, **self.status())

    def validate_spec(self, raw):
        if not isinstance(raw, dict) or set(raw) - {"width", "height", "cameras", "depth"}:
            raise P.RequestError("invalid observation specification")
        for key in ("width", "height"):
            if raw.get(key) is not None and (type(raw[key]) is not int or raw[key] <= 0):
                raise P.RequestError("image dimensions must be positive integers")
        cams = raw.get("cameras")
        if cams is not None and (not isinstance(cams, (list, tuple)) or any(
                c not in self.config["public"]["cameras"] for c in cams)):
            raise P.RequestError("unknown camera")
        if type(raw.get("depth", False)) is not bool:
            raise P.RequestError("depth must be boolean")
        if raw.get("depth") and not self.config["public"].get("depth", True):
            raise P.RequestError("depth is unavailable")
        return raw

    def require_live(self):
        if self.s["episode_over"] or self.s["phase"] not in ("development", "evaluation"):
            raise P.RequestError("episode is over; inspect status", "episode_over")

    async def execute_actions(self, actions, spec, move=None, recover=False):
        await self.wait_ready()
        self.require_live()
        if len(actions) > self.status()["steps_remaining"]:
            raise P.RequestError("batch exceeds remaining budget", "budget_exhausted")
        if recover:
            try:
                available = await self.worker.call("recovery_available")
            except WorkerFailure as exc:
                await self.failed(exc)
                return {**await self.observe(spec), "steps": 0, "success": False}
            if not available:
                raise P.RequestError("Recovery is available only after the cube has dropped.")
        applied = 0
        for action in actions:
            self.charge("step")
            try:
                evidence = await self.worker.call("act", action=action, move=move, recover=recover)
                self.confirmed(evidence)
                applied += 1
                if evidence.get("success") and self.mode not in ("tabletop", "hidden_com"):
                    self.s.update(episode_over=True, ended="env_success")
                elif evidence.get("done"):
                    self.s.update(episode_over=True, ended="env_done")
                elif self.status()["steps_remaining"] == 0 or self.s["episode_steps"] >= self.config["horizon"]:
                    if self.mode != "hidden_com":
                        self.s.update(episode_over=True, ended="horizon")
                if self.s["episode_over"]:
                    if self.mode == "tabletop":
                        await self.finish("horizon")
                    elif self.s["phase"] == "evaluation":
                        self.record()
                    self.save("episode_end")
                    break
            except WorkerFailure as exc:
                await self.failed(exc)
                break
        return {**await self.observe(spec), "steps": applied,
                "success": bool(self.s["progress"].get("success", False))}

    async def reset(self, task=None, seed=None):
        if self.s["phase"] != "development" and self.mode != "tabletop":
            raise P.RequestError("reset is only available during development", "wrong_phase")
        if self.mode == "tabletop" and (task is not None or seed is not None):
            raise P.RequestError("this task resets only to its original scene")
        if self.status()["steps_remaining"] <= 0 or self.s["phase"] == "finished":
            raise P.RequestError("development is closed or budget exhausted", "budget_exhausted")
        task = task or self.s["task"]
        if self.mode != "tabletop" and task not in self.config["train"]:
            raise P.RequestError("unknown training task")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
            raise P.RequestError("seed must be an integer in [0, 2**31)")
        self.s["episodes"] += 1
        if seed is None:
            key = f'{self.mode}|{task}|{self.s["episodes"]}'
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (2**31-1)
        descriptor = (self.descriptor() if self.mode == "tabletop" else
                      dict(task=task, seed=seed, split="pretrain", level=self.config.get("level", "L1"),
                           default_resolution=self.config["public"]["default_resolution"], mode=self.mode))
        self.charge("reset")
        if self.worker and task == self.s["task"] and not self.s["ended"] == "sim_error":
            try:
                result = await self.worker.call("reset", timeout=INITIALIZE_SECONDS, seed=descriptor["seed"])
                self.info = result["info"]
                self.s.update(episode_steps=0, episode_over=False, ended=None, ready=True)
                self.confirmed(result["evidence"])
            except WorkerFailure as exc:
                await self.failed(exc)
        else:
            await self.open(descriptor)
        self.s["task"] = task
        if self.status()["steps_remaining"] == 0:
            if self.mode == "tabletop":
                await self.finish("budget_exhausted")
            else:
                self.s.update(episode_over=True, ended="budget_exhausted")
        self.save("reset_end")
        return await self.observe({})

    async def finish(self, reason="agent_finish"):
        await self.wait_ready()
        if self.worker and self.mode in ("tabletop", "pocket") and str(self.s["trial"]) not in self.s["results"]:
            try:
                self.confirmed(await self.worker.call("finish", timeout=FINISH_SECONDS))
            except WorkerFailure as exc:
                await self.failed(exc)
        self.s.update(episode_over=True, ended=self.s["ended"] or reason)
        if self.s["phase"] == "evaluation":
            self.record()
        if self.mode in ("tabletop", "pocket"):
            self.s["phase"] = "finished"
        self.save("finished")
        return self.status()

    async def next_trial(self):
        if self.mode != "task01" or self.s["phase"] != "evaluation":
            raise P.RequestError("Harbor controls the next trial", "wrong_phase")
        if not self.s["episode_over"]:
            raise P.RequestError("finish the current trial first")
        self.s["trial"] += 1
        if self.s["trial"] >= len(self.config["plan"]):
            self.s["phase"] = "finished"
            self.save("evaluation_end")
            await self.stop()
            return self.status()
        await self.open(self.descriptor())
        return await self.observe({})

    async def submit(self, quadrant, trial):
        if self.mode != "hidden_com" or self.s["phase"] != "evaluation":
            raise P.RequestError("submission unavailable")
        await self.wait_ready()
        if trial != self.s["trial"] + 1 or quadrant not in ("A", "B", "C", "D"):
            raise P.RequestError("invalid answer or stale trial")
        if self.s["episode_over"]:
            if self.s["ended"] != "sim_error":
                raise P.RequestError("trial already ended")
            # A failed box accepts no answer, but the caller can advance to the next box.
            quadrant = None
        self.s["progress"]["answer"] = quadrant
        index = self.s["trial"]
        await self.finish("submitted")
        self.s["trial"] += 1
        if self.s["trial"] < len(self.config["plan"]):
            await self.open(self.descriptor())
        else:
            self.s["phase"] = "finished"
            self.save("evaluation_end")
            await self.stop()
        return dict(submitted=True, quadrant=quadrant, trial=index+1,
                    remaining_trials=len(self.config["plan"])-index-1, done=self.s["phase"] == "finished")

    async def finish_step(self, step):
        if step in self.s["finished_steps"]:
            return self.status()
        if self.mode == "task01" and step == "develop":
            if self.s["phase"] == "evaluation":
                await self.finish("step_ended")
            await self.stop()
            self.s.update(phase="finished", episode_over=True, ready=True)
            self.s["finished_steps"].append(step)
            self.save("step_end", step=step)
            return self.status()
        if step == "develop" and self.s["phase"] in ("development", "development_closed"):
            await self.stop()
            self.s.update(phase="evaluation", trial=0)
        elif self.mode == "task02" and step == f'eval_{self.s["trial"]+1:02d}':
            await self.finish("step_ended")
            await self.stop()
            self.s["trial"] += 1
        elif step in ("evaluate", "attempt"):
            await self.finish("step_ended")
            await self.stop()
            self.s["trial"] = len(self.config["plan"])
        else:
            raise P.RequestError("step does not match assigned phase")
        self.s["finished_steps"].append(step)
        self.s["clock"] = None
        if self.s["trial"] >= len(self.config["plan"]):
            self.s.update(phase="finished", episode_over=True, ready=True)
        self.save("step_end", step=step)
        if self.s["phase"] != "finished":
            self.start_trial()
        return self.status()

    async def dispatch(self, op, fields):
        if not self.s.get("clock"):
            self.s["clock"] = time.time()  # Reporting only; never used by score or transition rules.
            self.save("clock_started")
        if self.mode == "hidden_com" and op in ("step", "move"):
            if fields.pop("trial", None) != self.s["trial"]+1:
                raise P.RequestError("stale or missing trial; observe before acting")
        if op == "status":
            return self.status()
        if op == "task_info":
            await self.wait_ready()
            return self.task_info()
        if op == "list_tasks":
            return {"tasks": self.config.get("train", []) if self.s["phase"] == "development" else []}
        if op == "observe":
            return await self.observe(self.validate_spec(fields.get("spec", {})))
        if op == "step":
            spec = self.validate_spec(fields.get("spec", {}))
            batch = P.actions(fields.get("actions"), self.config["public"]["action_dim"])
            if self.mode == "hidden_com" and any(abs(v)>1 for a in fields["actions"] for v in a):
                raise P.RequestError("action values must be in [-1,1]")
            return await self.execute_actions(batch, spec)
        if op == "reset":
            return await self.reset(**fields)
        if op == "end_development":
            if self.mode == "task01":
                if self.s["phase"] == "development":
                    await self.stop()
                    self.s.update(phase="evaluation", trial=0)
                    self.start_trial()
                    self.save("development_end")
                return self.status()
            if self.s["phase"] not in ("development", "development_closed"):
                raise P.RequestError("development already ended", "wrong_phase")
            self.s["phase"] = "development_closed"
            self.save("development_end")
            return self.status()
        if op == "finish_trial":
            if self.s["phase"] != "evaluation":
                raise P.RequestError("no evaluation trial", "wrong_phase")
            return await self.finish()
        if op == "next_trial":
            return await self.next_trial()
        if op == "submit":
            return await self.submit(**fields)
        if op == "recover_drop" and self.mode == "pocket":
            result = await self.execute_actions([[0.]*6 + [-1.] + [0.]*6 + [-1.]], {}, recover=True)
            if result["steps"]:
                result["recovered"] = True
            return result
        if op == "move" and self.mode in ("tabletop", "hidden_com"):
            count = fields.pop("steps", 40)
            if type(count) is not int or not 1 <= count <= P.MAX_BATCH:
                raise P.RequestError("invalid move duration")
            for name, length in (("position", 3), ("quaternion", 4)):
                v = fields.get(name)
                if not isinstance(v, list) or len(v) != length or any(type(x) not in (int, float) or not math.isfinite(x) for x in v):
                    raise P.RequestError("invalid pose")
            if abs(sum(x*x for x in fields["quaternion"])-1) > 1e-4:
                raise P.RequestError("quaternion must be unit xyzw")
            grip = fields.setdefault("gripper", 1)
            if type(grip) not in (int, float) or not -1 <= grip <= 1:
                raise P.RequestError("invalid gripper")
            return await self.execute_actions([None]*count, {}, move=fields)
        raise P.RequestError("unknown operation")
