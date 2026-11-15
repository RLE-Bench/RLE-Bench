#!/bin/bash
set -u
mkdir -p /logs/verifier
# harbor leaves it world-writable; the submission's uids must not write here
chown 0:0 /logs/verifier && chmod 0755 /logs/verifier || exit 1
export PYTHONPATH=/opt/nanovla-verifier
python3 /tests/score_task.py
status=$?
pkill -KILL -u 65534 2>/dev/null || true
if [ ! -f /logs/verifier/reward.json ]; then echo '{"reward": 0.0}' > /logs/verifier/reward.json; fi
exit $status
