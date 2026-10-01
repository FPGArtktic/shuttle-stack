#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Checks every commit must pass.  They need neither a GPU nor root: the
# phases run here only read the system.

set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail()
{
	printf 'smoke: FAIL: %s\n' "$*" >&2
	exit 1
}

# Prefer an installed ruff, fall back to uvx; missing both is an error,
# not a reason to skip the check.
ruff()
{
	if type -P ruff > /dev/null; then
		command ruff "$@"
	elif type -P uvx > /dev/null; then
		uvx ruff "$@"
	else
		fail "neither ruff nor uvx found; ruff is to the delegate\
 what shellcheck is to the installer"
	fi
}

# Older commits have no delegate, and there is nothing to lint there.
check_python()
{
	if [[ ! -d delegate ]]; then
		return 0
	fi
	ruff check delegate || fail "ruff check delegate"
	ruff format --check delegate || fail "ruff format --check delegate"
	printf 'smoke: ruff: ok\n'
	tests/delegate.sh || fail "tests/delegate.sh"
	printf 'smoke: delegate tests: ok\n'
}

# The project's MCP registration is committed, so a typo in it would
# reach a reader as a server that silently never starts.
check_mcp_json()
{
	local name

	jq -e '.mcpServers.shuttle.command' .mcp.json > /dev/null ||
		fail ".mcp.json does not describe a shuttle server"
	name=$(jq -r '.mcpServers.shuttle.command' .mcp.json)
	[[ $name == shuttle-delegate ]] ||
		fail ".mcp.json runs '$name', not the installed launcher"
	jq -e '.mcpServers.shuttle.args | index("stdio")' .mcp.json \
		> /dev/null || fail ".mcp.json does not ask for stdio"
	printf 'smoke: .mcp.json: ok\n'
}

check_dry_run()
{
	local distro=$1 out

	out=$(./install.sh detect plan --dry-run --force-distro "$distro") ||
		fail "detect plan --dry-run --force-distro $distro"
	# The fast server does not depend on the machine, so its line is a
	# fixed point of the plan output.
	[[ $out == *$'\nfast.ctx '*' 16384 '* ]] ||
		fail "plan for $distro lacks fast.ctx 16384"
	printf 'smoke: dry run for %s: ok\n' "$distro"
}

main()
{
	local distro

	bash -n install.sh || fail "bash -n install.sh"
	shellcheck install.sh tests/*.sh || fail shellcheck
	printf 'smoke: syntax and shellcheck: ok\n'
	check_python
	check_mcp_json
	for distro in ubuntu arch; do
		check_dry_run "$distro"
	done
}

main "$@"
