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

# The tests must not see this machine's installation: one that reads
# ~/.config/shuttle/stack.env passes for a developer who has the stack
# and fails on a runner that does not.  XDG_CONFIG_HOME points at an
# empty directory so neither can happen.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

XDG_CONFIG_HOME=$scratch \
	uv run --quiet "${pin[@]}" python -m unittest discover -s tests -t . "$@"
