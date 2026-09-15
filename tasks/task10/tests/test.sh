#!/bin/bash
set -u
mkdir -p /logs/verifier
chmod -R go-rwx /tests || true
export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi
exit $status
