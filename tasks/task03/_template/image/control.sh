#!/bin/sh
set -eu
cd /
exec /usr/local/bin/python -I -c 'import sys; sys.path.insert(0,"/opt/private"); from harness.runtime.control import main; main()' "$@"
