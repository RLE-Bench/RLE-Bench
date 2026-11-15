#!/bin/bash
# Harbor verifier entry point (task06-rgb-only); the only writer of
# /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

# Lock the verifier context BEFORE any submitted code can run: sandboxed
# estimators execute as uid 65534 and must not be able to read the scoring
# code, thresholds, or evaluation seeds (deterministic rendering would let
# readable seeds reconstruct ground truth), nor write the reward Harbor
# leaves world-writable.
chmod -R o-rwx,g-rwx /tests || true
chown 0:0 /logs/verifier && chmod 0755 /logs/verifier || exit 1

export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
