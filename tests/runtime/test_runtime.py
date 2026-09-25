import asyncio
import json
import socket

import pytest

from rlebench.runtime.engine import Engine
from rlebench.runtime.process import WorkerFailure
from rlebench.runtime.protocol import RequestError
from rlebench.runtime.scoring import score
from rlebench.runtime.server import Server
from rlebench.runtime.store import read_state


def config(mode="task02"):
    return dict(mode=mode, adapter="fake", train=["train"], budget=8, horizon=5,
                plan=[dict(task="secret", seed=i) for i in range(3)],
                public=dict(action_dim=2, cameras=["camera"], default_resolution=128, depth=True))


class FakeWorker:
    failure = None
    starts = 0
    delay = 0

    def __init__(self, root):
        self.n = 0
        self.closed = False

    async def start(self, module, descriptor):
        type(self).starts += 1
        if self.failure == "create":
            raise WorkerFailure()
        return dict(info=dict(instruction="do the task"), evidence=dict(score=0))

    async def call(self, op, **kw):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure == op:
            raise WorkerFailure()
        if op == "act":
            self.n += 1
            return dict(score=.25*self.n, success=self.n >= 4)
        if op == "observe":
            return dict(obs=dict(position=self.n))
        if op == "reset":
            self.n = 0
            return dict(info={}, evidence=dict(score=0))
        if op == "finish":
            return dict(quality=.7)
        raise AssertionError(op)

    async def stop(self):
        self.closed = True


@pytest.fixture(autouse=True)
def clean_fake():
    FakeWorker.failure = None
    FakeWorker.starts = 0
    FakeWorker.delay = 0


def run(fn):
    return asyncio.run(fn())


def test_budget_batch_and_no_free_retry(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        await e.reset()
        assert e.s["dev_steps"] == 1
        await e.dispatch("step", dict(actions=[[0, 0]]*2))
        assert e.s["dev_steps"] == 3
        FakeWorker.failure = "act"
        result = await e.dispatch("step", dict(actions=[[0, 0]]*2))
        assert result["ended"] == "sim_error" and result["steps"] == 0
        assert e.s["dev_steps"] == 4
        assert e.worker is None
        FakeWorker.failure = None
        await e.reset()
        assert e.s["dev_steps"] == 5 and not e.s["episode_over"]
        with pytest.raises(RequestError):
            await e.dispatch("step", dict(actions=[[0, 0]]*4))
        assert e.s["dev_steps"] == 5
        await e.stop()
        e.store.close()
    run(scenario)


def test_failure_keeps_credit_and_next_harbor_step(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        await e.finish_step("develop")
        await e.wait_ready()
        await e.dispatch("step", dict(actions=[[0, 0]]*2))
        FakeWorker.failure = "act"
        await e.dispatch("step", dict(actions=[[0, 0]]))
        assert e.s["results"]["0"]["evidence"]["score"] == .5
        FakeWorker.failure = None
        await e.finish_step("eval_01")
        await e.wait_ready()
        assert e.s["trial"] == 1 and not e.s["episode_over"]
        await e.finish_step("eval_01")
        assert e.s["trial"] == 1  # Collect + verifier is not two transitions.
        assert score(e.s)["reward"] == pytest.approx(.5/3)
        await e.stop()
        e.store.close()
    run(scenario)


def test_startup_failure_is_one_trial(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        FakeWorker.failure = "create"
        await e.finish_step("develop")
        await e.wait_ready()
        assert e.s["ready"] and e.s["ended"] == "sim_error"
        FakeWorker.failure = None
        await e.finish_step("eval_01")
        await e.wait_ready()
        assert e.s["trial"] == 1 and not e.s["episode_over"]
        await e.stop()
        e.store.close()
    run(scenario)


def test_service_restart_does_not_replay(tmp_path):
    async def scenario():
        cfg = config()
        e = Engine(tmp_path, cfg, FakeWorker)
        await e.finish_step("develop")
        await e.wait_ready()
        await e.dispatch("step", dict(actions=[[0, 0]]))
        e.charge("step")
        await e.stop()
        e.store.close()
        recovered = Engine(tmp_path, cfg, FakeWorker)
        await recovered.recover()
        assert recovered.s["ended"] == "sim_error"
        assert recovered.s["results"]["0"]["steps"] == 2
        assert recovered.s["results"]["0"]["evidence"]["score"] == .25
        assert FakeWorker.starts == 1
        recovered.store.close()
    run(scenario)


def test_reset_and_next_trial_are_distinct(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config("task01"), FakeWorker)
        await e.dispatch("end_development", {})
        await e.wait_ready()
        with pytest.raises(RequestError):
            await e.reset()
        with pytest.raises(RequestError):
            await e.next_trial()
        await e.finish()
        await e.next_trial()
        assert e.s["trial"] == 1
        await e.stop()
        e.store.close()
    run(scenario)


def test_wire_exclusive_idle_owner_and_control_preemption(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        server = Server(e, tmp_path / "sim.sock", tmp_path / "control.sock")
        await server.start()
        r1, w1 = await asyncio.open_unix_connection(str(server.public_path))
        assert json.loads(await r1.readline())["ok"]
        r2, w2 = await asyncio.open_unix_connection(str(server.public_path))
        assert json.loads(await r2.readline())["kind"] == "busy"
        w2.close()
        # Idle owner does not prevent Harbor advancing the phase.
        rc, wc = await asyncio.open_unix_connection(str(server.control_path))
        wc.write(b'{"op":"finish_step","step":"develop"}\n')
        reply = json.loads(await asyncio.wait_for(rc.readline(), 1))
        assert reply["ok"] and reply["phase"] == "evaluation"
        assert await r1.readline() == b""
        wc.close(); w1.close()
        await e.wait_ready()
        await server.close()
    run(scenario)


def test_wire_cancels_hung_action_and_advances(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        await e.finish_step("develop")
        await e.wait_ready()
        server = Server(e, tmp_path / "sim.sock", tmp_path / "control.sock")
        # start() includes crash recovery, so only open the listener here.
        listener = await asyncio.start_unix_server(server.public, path=str(server.public_path))
        control = await asyncio.start_unix_server(server.control, path=str(server.control_path))
        server.servers = [listener, control]
        r, w = await asyncio.open_unix_connection(str(server.public_path))
        await r.readline()
        FakeWorker.delay = 100
        w.write(b'{"op":"step","request_id":"a","actions":[[0,0]]}\n')
        for _ in range(50):
            if e.s["pending"]:
                break
            await asyncio.sleep(.01)
        rc, wc = await asyncio.open_unix_connection(str(server.control_path))
        wc.write(b'{"op":"finish_step","step":"eval_01"}\n')
        reply = json.loads(await asyncio.wait_for(rc.readline(), 1))
        assert reply["ok"] and e.s["trial"] == 1
        assert e.s["results"]["0"]["ended"] == "sim_error"
        w.close(); wc.close()
        FakeWorker.delay = 0
        await e.wait_ready()
        await server.close()
    run(scenario)


@pytest.mark.parametrize("spec", [{"width": 0}, {"width": True}, {"depth": 1}, {"cameras": ["secret"]}])
def test_invalid_spec_never_reaches_worker(tmp_path, spec):
    e = Engine(tmp_path, config(), FakeWorker)
    with pytest.raises(RequestError):
        e.validate_spec(spec)
    assert FakeWorker.starts == 0
    e.store.close()


def test_invalid_ledger_is_not_read_as_zero(tmp_path):
    path = tmp_path / "ledger.sqlite"
    path.write_text("not a database")
    with pytest.raises(Exception):
        read_state(path)


def test_wire_deduplicates_mutations_and_survives_invalid_request(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config(), FakeWorker)
        server = Server(e, tmp_path/"sim.sock", tmp_path/"control.sock")
        await server.start()
        reader, writer = await asyncio.open_unix_connection(str(server.public_path))
        await reader.readline()
        async def send(value):
            writer.write(json.dumps(value).encode()+b"\n")
            return json.loads(await reader.readline())
        request = dict(op="reset", request_id="reset-once")
        assert (await send(request))["ok"]
        assert (await send(request))["kind"] == "duplicate_request"
        assert e.s["dev_steps"] == 1
        bad = dict(op="step", request_id="bad", actions=[[float("inf"), 0]])
        assert not (await send(bad))["ok"]
        assert e.s["dev_steps"] == 1
        good = dict(op="step", request_id="good", actions=[[0, 0]])
        assert (await send(good))["steps"] == 1
        writer.close()
        await server.close()
    run(scenario)


def test_task01_continuous_session(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr('rlebench.runtime.engine.time.time', lambda: clock[0])

    async def scenario():
        cfg = config('task01')
        cfg['seconds'] = {'session': 32400}
        e = Engine(tmp_path, cfg, FakeWorker)
        await e.dispatch('reset', {})
        await e.dispatch('step', dict(actions=[[0, 0]]))
        development_steps = e.s['dev_steps']
        clock[0] += 60
        status = await e.dispatch('end_development', {})
        assert status['phase'] == 'evaluation' and not status['ready']
        assert status['seconds_remaining'] == 32340
        await e.dispatch('task_info', {})
        starts = FakeWorker.starts
        await e.dispatch('end_development', {})
        assert FakeWorker.starts == starts and e.s['trial'] == 0
        for i in range(3):
            assert e.s['trial'] == i
            await e.dispatch('step', dict(actions=[[0, 0]]*4))
            clock[0] += 1
            await e.dispatch('next_trial', {})
        assert e.s['phase'] == 'finished'
        assert e.s['dev_steps'] == development_steps
        assert e.status()['seconds_remaining'] == 32337
        assert score(e.s)['success_rate'] == 1
        before = score(e.s)
        await e.dispatch('end_development', {})
        await e.finish_step('develop')
        await e.finish_step('develop')
        assert score(e.s) == before
        e.store.close()
    run(scenario)


@pytest.mark.parametrize('evaluate', [False, True])
def test_task01_session_collection_never_opens_another_trial(tmp_path, evaluate):
    async def scenario():
        e = Engine(tmp_path, config('task01'), FakeWorker)
        await e.dispatch('reset', {})
        if evaluate:
            await e.dispatch('end_development', {})
            await e.dispatch('step', dict(actions=[[0, 0]]*4))
            await e.next_trial()
        starts = FakeWorker.starts
        await e.finish_step('develop')
        await e.finish_step('develop')
        assert e.s['phase'] == 'finished' and e.worker is None
        assert FakeWorker.starts == starts
        assert score(e.s)['success_rate'] == (1/3 if evaluate else 0)
        assert len(e.s['results']) == (2 if evaluate else 0)
        e.store.close()
    run(scenario)


def test_task01_initialization_failure_can_advance(tmp_path):
    async def scenario():
        e = Engine(tmp_path, config('task01'), FakeWorker)
        FakeWorker.failure = 'create'
        await e.dispatch('end_development', {})
        status = await e.dispatch('task_info', {})
        assert status['ended'] == 'sim_error' and e.s['failures'] == 1
        FakeWorker.failure = None
        await e.next_trial()
        assert e.s['trial'] == 1 and not e.s['episode_over']
        await e.finish_step('develop')
        assert e.s['phase'] == 'finished'
        e.store.close()
    run(scenario)
