import json

from harness.runtime.engine import Engine
from harness.runtime import verify
from test_runtime import config, FakeWorker


def test_score_survives_unavailable_control_and_media(tmp_path, monkeypatch):
    engine = Engine(tmp_path, config(), FakeWorker)
    engine.s.update(phase="finished", results={"0": dict(evidence=dict(score=.6), steps=2)})
    engine.save("test")
    engine.store.close()

    def fail(*_):
        raise OSError("unavailable")
    monkeypatch.setattr(verify, "collect", fail)
    monkeypatch.setattr(verify, "export_media", fail)
    output = tmp_path / "output"
    verify.verify(tmp_path, output, "eval_03")
    result = json.loads((output / "reward.json").read_text())
    assert abs(result["reward"]-.2)<1e-12
    assert json.loads((output / "diagnosis.json").read_text())["ledger_ok"]
