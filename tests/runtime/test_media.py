import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

from rlebench.runtime import media, verify


@pytest.mark.parametrize('flag,expected', [(None, False), ('false', False), ('0', False),
                                          ('TRUE', True), ('1', True), ('yes', False)])
def test_evaluation_opt_in(monkeypatch, flag, expected):
    monkeypatch.delenv('RLEBENCH_MEDIA', raising=False)
    if flag is not None:
        monkeypatch.setenv('RLEBENCH_MEDIA', flag)
    assert media.evaluation_enabled() is expected


def metadata(video):
    return json.loads(video.path.with_suffix('.json').read_text())


def test_bundled_encoder_produces_playable_video(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', '')
    exe = media.ffmpeg_exe()
    assert exe and Path(exe).is_file()
    video = media.Video(tmp_path / 'trial-01.mp4', fps=10)
    for i in range(5):
        video.add(np.full((256, 768, 3), i * 40, dtype=np.uint8))
    video.close()
    video.close()
    assert not video.thread.is_alive()
    assert metadata(video) == dict(status='complete', reason=None, frames=5)
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(video.path))
    info = next(reader)
    reader.close()
    assert info['fps'] == 10 and info['size'] == (768, 256)
    result = subprocess.run([exe, '-v', 'error', '-i', str(video.path), '-f', 'rawvideo',
                             '-pix_fmt', 'rgb24', 'pipe:1'], capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(5, 256, 768, 3)
    np.testing.assert_allclose(frames[:, 0, 0, 0], np.arange(5)*40, atol=3)


def test_encoder_unavailable_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(media, 'ffmpeg_exe', lambda: None)
    video = media.Video(tmp_path / 'trial-01.mp4')
    video.add(np.zeros((2, 2, 3), dtype=np.uint8))
    video.close()
    assert metadata(video)['status'] == 'incomplete'
    assert 'ffmpeg unavailable' in metadata(video)['reason']


def test_overflow_disables_recording_without_dropping_silently(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(media.Video, '_encode', lambda self: release.wait(5))
    video = media.Video(tmp_path / 'trial-01.mp4')
    try:
        for _ in range(33):
            video.add(np.zeros((2, 2, 3), dtype=np.uint8))
        assert video.queue.qsize() == 32
        assert metadata(video)['reason'] == 'encoder queue full'
        video.add(np.zeros((2, 2, 3), dtype=np.uint8))
        assert video.queue.qsize() == 32
    finally:
        release.set()
        video.close()


@pytest.mark.parametrize('cleanup_delay', [0, .4])
def test_stuck_encoder_is_killed_within_worker_shutdown_bound(tmp_path, monkeypatch, cleanup_delay):
    import sys
    real_popen = subprocess.Popen
    started = threading.Event()
    def stuck(*args, **kwargs):
        proc = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        started.set()
        return proc
    monkeypatch.setattr(media.subprocess, 'Popen', stuck)
    real_fail = media.Video.fail
    def delayed_cleanup(self, reason):
        if reason.startswith('encoding failed:'):
            time.sleep(cleanup_delay)
        real_fail(self, reason)
    monkeypatch.setattr(media.Video, 'fail', delayed_cleanup)
    video = media.Video(tmp_path / 'trial-01.mp4')
    video.add(np.zeros((256, 768, 3), dtype=np.uint8))
    assert started.wait(2)
    start = time.monotonic()
    video.close()
    assert time.monotonic() - start < 3
    assert not video.thread.is_alive()
    assert video.proc.poll() is not None
    assert metadata(video)['status'] == 'incomplete'
    assert metadata(video)['reason'] == 'encoder shutdown timed out'


def test_nonzero_encoder_exit_is_incomplete(tmp_path, monkeypatch):
    import sys
    real_popen = subprocess.Popen
    monkeypatch.setattr(media.subprocess, 'Popen', lambda *a, **kw: real_popen(
        [sys.executable, '-c', 'import sys; sys.stdin.buffer.read(); sys.exit(1)'], **kw))
    video = media.Video(tmp_path / 'trial-01.mp4')
    video.add(np.zeros((2, 2, 3), dtype=np.uint8))
    video.close()
    assert metadata(video)['reason'] == 'encoder exited unsuccessfully'


def test_only_completed_ledger_trials_with_finalized_media_export(tmp_path, monkeypatch):
    monkeypatch.setenv('RLEBENCH_MEDIA', 'true')
    root = tmp_path / 'media'
    root.mkdir()
    for i in (1, 2, 3, 4, 6):
        (root / f'trial-{i:02d}.mp4').write_bytes(b'video')
        (root / f'trial-{i:02d}.json').write_text(json.dumps(dict(status='complete', frames=1)))
    (root / 'trial-02.json').write_text(json.dumps(dict(status='recording')))
    (root / 'trial-03.json').write_text(json.dumps(dict(status='incomplete', reason='encoder queue full')))
    (root / 'trial-06.mp4').unlink()
    (root / 'trial-06.mp4').symlink_to(root / 'trial-01.mp4')
    exports = []
    monkeypatch.setattr(verify, 'export_files', lambda *args: exports.append(args))
    # Trial 4 is already encoding while task02's verifier collects trial 3.
    state = dict(config=dict(mode='task02'), results={str(i): {} for i in (0, 1, 2, 4, 5)})
    verify.export_media(tmp_path, tmp_path / 'output', state)
    _, _, paths, index = exports.pop()
    assert [p.name for p in paths] == ['trial-01.mp4']
    assert index == dict(enabled=True, files=['trial-01.mp4'], skipped=[
        dict(name='trial-02.mp4', reason='interrupted recording'),
        dict(name='trial-03.mp4', reason='encoder queue full'),
        dict(name='trial-05.mp4', reason='missing or interrupted recording'),
        dict(name='trial-06.mp4', reason='finalized video missing or invalid')])
    monkeypatch.delenv('RLEBENCH_MEDIA')
    verify.export_media(tmp_path, tmp_path / 'output', state)
    assert exports.pop()[3] == dict(enabled=False, files=[], skipped=[])
    state['config']['mode'] = 'tabletop'
    verify.export_media(tmp_path, tmp_path / 'output', state)
    assert exports.pop()[3]['files'] == [f'trial-{i:02d}.mp4' for i in (1, 2, 3, 4)]
