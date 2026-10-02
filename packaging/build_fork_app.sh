#!/bin/sh
# Thin wrapper: builds "oMLX Fork.app" + DMG into packaging/_fork_dist/.
# Uses the repo venv so the dependency pins are read from the fork environment.
# All options are forwarded (see --help). Docs: packaging/FORK_APP.md
set -eu
REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "error: $PY not found (create the venv first)" >&2; exit 1; }
cd "$REPO_ROOT"
exec "$PY" "$REPO_ROOT/packaging/build_fork_app.py" "$@"
