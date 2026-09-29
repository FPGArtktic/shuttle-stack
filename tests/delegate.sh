#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Builds a throwaway environment for the delegate and runs its tests.
# Nothing here talks to a llama-server, so it runs in CI as it does on a
# machine with the stack installed.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../delegate"

if ! type -P uv > /dev/null; then
	printf 'delegate: uv not found; it builds the test environment\n' >&2
	exit 1
fi

# SHUTTLE_PYTHON pins the interpreter, so CI can run the same script
# against every version the package claims to support.
pin=()
if [[ -n ${SHUTTLE_PYTHON:-} ]]; then
	pin=(--python "$SHUTTLE_PYTHON")
fi

exec uv run --quiet "${pin[@]}" python -m unittest discover -s tests -t . "$@"
