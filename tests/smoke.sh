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
	for distro in ubuntu arch; do
		check_dry_run "$distro"
	done
}

main "$@"
