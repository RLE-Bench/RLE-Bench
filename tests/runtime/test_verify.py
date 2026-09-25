import json

from rlebench.runtime.engine import Engine
from rlebench.runtime import verify
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


def test_diagnostics_only_export_after_final_collection(tmp_path, monkeypatch):
    e = Engine(tmp_path, config(), FakeWorker)
    e.s['phase'] = 'evaluation'
    e.save('test')
    monkeypatch.setattr(verify, 'collect', lambda *_: None)
    monkeypatch.setattr(verify, 'export_media', lambda *_: None)
    monkeypatch.setattr(verify, 'restore', lambda *_: None)
    from types import SimpleNamespace
    monkeypatch.setattr(verify.pwd, 'getpwnam', lambda _: SimpleNamespace(pw_uid=123, pw_gid=123))
    exports = []
    monkeypatch.setattr(verify, 'export_diagnostics', lambda *args: exports.append(args))
    output = tmp_path / 'output'
    verify.verify(tmp_path, output, 'eval_01')
    assert exports == []
    e.s.update(phase='finished', results={'0': dict(evidence=dict(score=.6), steps=2)})
    e.save('test')
    verify.verify(tmp_path, output, 'eval_03')
    assert len(exports) == 1

    def fail(*_):
        raise OSError('export failed')
    monkeypatch.setattr(verify, 'export_diagnostics', fail)
    verify.verify(tmp_path, output, 'eval_03')
    assert json.loads((output / 'reward.json').read_text())['reward'] == .6/3
    assert json.loads((output / 'diagnosis.json').read_text())['diagnostics_ok'] is False
    exports.clear()
    monkeypatch.setattr(verify, 'export_diagnostics', lambda *args: exports.append(args))
    monkeypatch.setattr(verify, 'collect', fail)
    verify.verify(tmp_path, output, 'eval_03')
    assert exports == []
    e.store.close()
