#!/bin/bash
# Harbor verifier entry point (task07). The ONLY writer of
# /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier

# Lock the verifier context BEFORE any submitted code runs: sandboxed
# policies execute as uid 65534 and must not be able to read the scoring
# code or the evaluation seeds (deterministic piles would let readable
# seeds reconstruct ground truth), nor write the reward Harbor leaves
# world-writable.
chmod -R o-rwx,g-rwx /tests || true
chown 0:0 /logs/verifier && chmod 0700 /logs/verifier || exit 1
chmod 0700 /logs/artifacts 2>/dev/null || true
# Shared writable files or dirs would carry policy state across episodes;
# each policy gets its own private home instead.
find / /dev/shm /dev/mqueue /logs -xdev ! -type l -perm -0002 \
    -exec chmod o-w {} + 2>/dev/null
[ -z "$(find /tmp /var/tmp /dev/shm -maxdepth 0 -perm -0002)" ] || exit 1

export PYTHONPATH=/tests
python3 /tests/score_task.py
status=$?

if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
