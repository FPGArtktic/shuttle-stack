#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Checks a range of commits against CONTRIBUTING.md.  Each commit is also
# built in a scratch worktree and must pass tests/smoke.sh, so that
# git bisect never stops on a broken tree.
#
# usage: tests/commits.sh BASE..HEAD

set -Eeuo pipefail

readonly SUBSYSTEMS=(install plan quadlet models verify bench delegate
	readme tests build)

errors=0
scratch=""

complain()
{
	printf 'commits: %s: %s\n' "${1:0:12}" "$2" >&2
	errors=$(( errors + 1 ))
}

check_subject()
{
	local commit=$1 subject list

	subject=$(git log -1 --format=%s "$commit")
	list=$(IFS='|'; printf '%s' "${SUBSYSTEMS[*]}")
	if [[ ! $subject =~ ^($list):\ [^[:space:]] ]]; then
		complain "$commit" "subject needs 'subsystem: ' ($list)"
	fi
	if (( ${#subject} > 72 )); then
		complain "$commit" "subject longer than 72 characters"
	fi
	if [[ $subject == *. ]]; then
		complain "$commit" "subject ends with a period"
	fi
}

# Long lines are allowed where the kernel allows them: URLs.
check_body()
{
	local commit=$1 body line

	body=$(git log -1 --format=%b "$commit")
	if [[ $body != *"Signed-off-by: "* ]]; then
		complain "$commit" "no Signed-off-by (git commit -s)"
	fi
	while IFS= read -r line; do
		if (( ${#line} > 72 )) && [[ $line != *://* ]]; then
			complain "$commit" "body line over 72 columns: $line"
		fi
	done <<< "$body"
}

check_tree()
{
	local commit=$1

	git -C "$scratch/tree" checkout -q --detach "$commit"
	if [[ ! -x $scratch/tree/tests/smoke.sh ]]; then
		return 0
	fi
	if ! "$scratch/tree/tests/smoke.sh" > "$scratch/smoke.log" 2>&1 \
			< /dev/null; then
		sed 's/^/    /' "$scratch/smoke.log" >&2
		complain "$commit" "tests/smoke.sh fails"
	fi
}

cleanup()
{
	if [[ -d $scratch/tree ]]; then
		git worktree remove --force "$scratch/tree"
	fi
	rm -rf "$scratch"
}

main()
{
	local range=${1:?usage: tests/commits.sh BASE..HEAD} commit count=0

	scratch=$(mktemp -d)
	trap cleanup EXIT
	git worktree add -q --detach "$scratch/tree"
	while read -r commit; do
		check_subject "$commit"
		check_body "$commit"
		check_tree "$commit"
		count=$(( count + 1 ))
	done < <(git rev-list --reverse --no-merges "$range")
	if (( errors )); then
		printf 'commits: %d problem(s) in %d commit(s)\n' \
			"$errors" "$count" >&2
		exit 1
	fi
	printf 'commits: %d commit(s) ok\n' "$count"
}

main "$@"
