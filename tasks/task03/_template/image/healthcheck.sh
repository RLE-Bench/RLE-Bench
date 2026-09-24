#!/bin/sh
exec /usr/local/bin/python -I - <<'PY'
import json, os, socket
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
    sock.settimeout(2)
    sock.connect('/run/rlebench/sim.sock')
    with sock.makefile('rb') as stream:
        response = json.loads(stream.readline(4096))
    assert response.get('ok') or response.get('kind') == 'busy'
if os.environ.get('RLEBENCH_FAMILY') == 'task01' and os.environ.get('RLEBENCH_IMAGE_LEVEL') in ('L2','L3'):
    assert os.path.exists('/run/rlebench/perception.sock')
PY
