"""Run as the agent UID inside a started container."""
import importlib
import os
from pathlib import Path
import socket

assert os.geteuid() != 0
for path in ('/opt/private', '/opt/src', '/var/lib/rlebench'):
    assert not os.access(path, os.R_OK | os.X_OK), path
for path in ('/var/lib/rlebench/ledger.sqlite', '/var/lib/rlebench/worker.log',
             '/var/lib/rlebench/service.log', '/var/lib/rlebench/failures.jsonl', '/var/lib/rlebench/handoff/MANUAL.md'):
    try:
        with open(path, 'rb'):
            raise AssertionError('private file readable')
    except (PermissionError, FileNotFoundError):
        pass
for module in ('robocasa', 'robosuite', 'harness.task', 'harness.adapter', 'harness.config',
               'rlebench.runtime.server', 'rlebench.runtime.store', 'rlebench.runtime.engine',
               'rlebench.runtime.worker', 'rlebench.runtime.scoring', 'rlebench.cli'):
    try:
        importlib.import_module(module)
    except (ImportError, PermissionError):
        pass
    else:
        raise AssertionError('private module importable: '+module)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    try:
        sock.connect('/var/lib/rlebench/control.sock')
    except PermissionError:
        pass
    else:
        raise AssertionError('control socket reachable')
for proc in Path('/proc').glob('[0-9]*/environ'):
    try:
        if proc.stat().st_uid == 0:
            proc.read_bytes()
            raise AssertionError('root process environment readable')
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        pass
from harness.client import SimClient
with SimClient() as sim:
    assert 'phase' in sim.status()
print('isolation passed')
