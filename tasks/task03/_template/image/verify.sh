#!/bin/sh
set -u
cd /
/usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.runtime.verify import main; main()' "$@"
if [ ! -f /logs/verifier/reward.json ]; then
    mkdir -p /logs/verifier
    printf '%s\n' '{"reward":0.0,"infrastructure_failures":1}' > /logs/verifier/reward.json
    printf '%s\n' '{"ledger_ok":false,"error":"verifier unavailable"}' > /logs/verifier/diagnosis.json
fi
exit 0
