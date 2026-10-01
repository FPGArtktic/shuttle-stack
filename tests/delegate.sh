#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Builds a throwaway environment for the delegate and runs its tests.
# Nothing here talks to a llama-server, so it runs in CI as it does on a
# machine with the stack installed.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../delegate"

die()
{
	printf 'delegate: %s\n' "$*" >&2
	exit 1
}

if ! type -P uv > /dev/null; then
	die "uv not found; it builds the test environment"
fi

# SHUTTLE_PYTHON pins the interpreter, so CI can run the same script
# against every version the package claims to support.
pin=()
if [[ -n ${SHUTTLE_PYTHON:-} ]]; then
	pin=(--python "$SHUTTLE_PYTHON")
fi

# The tests must neither read this machine's installation nor write to
# the tree they run in.  One that reads ~/.config/shuttle/stack.env
# passes for a developer who has the stack and fails on a runner that
# does not; one that reaches the audit log leaves a file behind, which
# in the scratch worktree of tests/commits.sh stops the next checkout.
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

# mypy is what the type hints are for.  It runs from the same locked
# environment as the tests, so it sees the installed mcp package and
# checks the delegate against the real signatures rather than against
# stubs it had to guess.
uv run --quiet "${pin[@]}" mypy shuttle_delegate tests ||
	die "mypy shuttle_delegate tests"
printf 'delegate: mypy: ok\n'

XDG_CONFIG_HOME=$scratch SHUTTLE_HOME=$scratch/state \
	uv run --quiet "${pin[@]}" python -m unittest discover -s tests -t . "$@"
