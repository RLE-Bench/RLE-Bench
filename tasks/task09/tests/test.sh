#!/bin/bash
# Harbor verifier entry point; runs in the separate verifier container and is
# the only writer of /logs/verifier/reward.json.
set -u

mkdir -p /logs/verifier
# trim.py runs as uid 65534; keep the scoring code, hidden seeds, and
# reward out of its reach. The sandbox refuses to run if this fails.
chmod 700 /tests /logs/verifier || exit 1
chmod 700 /logs/artifacts 2>/dev/null || true
find / /dev/shm /dev/mqueue /logs -xdev ! -type l -perm -0002 \
    -exec chmod o-w {} + 2>/dev/null
[ -z "$(find /tmp /var/tmp /dev/shm -maxdepth 0 -perm -0002)" ] || exit 1
umask 022

export PYTHONPATH=/tests
# software renderer for the post-eval device renders (physics never touches GL)
export MUJOCO_GL=osmesa
python3 /tests/score_task.py
status=$?

# score_task.py always writes reward.json itself, including on scoring
# errors; this fallback only covers a catastrophic interpreter failure.
if [ ! -f /logs/verifier/reward.json ]; then
    echo '{"reward": 0.0, "harness_crash": 1}' > /logs/verifier/reward.json
fi

exit $status
