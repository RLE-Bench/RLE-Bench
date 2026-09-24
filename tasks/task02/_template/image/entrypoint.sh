#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
export PYTHONPATH=/opt/private
if [ "$RLEBENCH_FAMILY" != task03 ]; then
    /usr/local/bin/python -I /opt/private/verify_assets.py
fi
exec /usr/local/bin/python -m harness.runtime.launcher
