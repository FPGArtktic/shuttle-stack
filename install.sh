#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# SHUTTLE M0 installer: two llama-server instances in rootless Podman, run
# by Quadlet on an internal network without egress.  See --help.

set -Eeuo pipefail
shopt -s nullglob

readonly PODMAN_MIN=409			# 4.9 as encoded by version_code()
readonly PHASES=(detect)


phases=()
dry_run=0
use_gpu=1
force_distro=""


user=${USER:-$(id -un)}
uid=$(id -u)
data_dir=${XDG_DATA_HOME:-$HOME/.local/share}/shuttle
state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/shuttle
models_dir=""
tmp_dir=""
cleanups=()

# Filled by load_facts().
facts=()
blockers=()
facts_loaded=0
distro=""
gpu_found=0
gpu_used=0
vram_total=0
vram_used=0
gpu_driver=""
cores=0
mem_total_kib=0
mem_avail_kib=0

usage()
{
	cat <<EOF
usage: install.sh [phase...] [options]

Phases always run in this order; without any, all of them run:
  detect

Options:
  --dry-run                  show commands and file changes, change nothing
  --no-gpu                   run shuttle-long on the CPU as well
  --force-distro arch|ubuntu skip distribution detection (tests only)
  --prefix DIR               models and KV cache location
                             (default: ~/.local/share/shuttle)
EOF
}

die()
{
	printf 'shuttle: error: %s\n' "$*" >&2
	exit 1
}

usage_die()
{
	printf 'shuttle: %s (see --help)\n' "$*" >&2
	exit 2
}

on_err()
{
	local status=$?

	die "line $1: '$2' failed with status $status"
}

on_exit()
{
	local fn

	for fn in "${cleanups[@]}"; do
		"$fn"
	done
	if [[ -n $tmp_dir ]]; then
		rm -rf "$tmp_dir"
	fi
}

step()
{
	printf '\n==> %s\n' "$*" >&2
}

info()
{
	printf '    %s\n' "$*" >&2
}

quote()
{
	local s

	s=$(printf '%q ' "$@")
	printf '%s\n' "${s% }"
}

# Tab-separated rows on stdin; the arguments are the column widths.
table()
{
	awk -F '\t' -v widths="$*" '
		BEGIN { n = split(widths, w, " ") }
		{
			for (i = 1; i < NF; i++)
				printf(i <= n ? "%-" w[i] "s " : "%s ", $i)
			print $NF
		}'
}

# Everything that changes the system is spelled "run cmd args...", which
# is where --dry-run takes effect and where the log is written.
run()
{
	if (( dry_run )); then
		printf '    + %s\n' "$(quote "$@")" >&2
		return 0
	fi
	command -v "$1" >/dev/null ||
		die "$1: not found (the host phase installs dependencies)"
	mkdir -p "$state_dir"
	printf '%s %s\n' "$(date -Is)" "$(quote "$@")" \
		>> "$state_dir/install.log"
	"$@"
}

set_value()
{
	case $1 in
	--prefix)
		[[ $2 == /* ]] || usage_die "--prefix needs an absolute path"
		data_dir=$2 ;;
	--force-distro)
		[[ $2 == arch || $2 == ubuntu ]] ||
			usage_die "--force-distro takes arch or ubuntu"
		force_distro=$2 ;;
	esac
}

# A long but flat switch, one line per option.
parse_args()
{
	while (( $# )); do
		case $1 in
		detect)
			phases+=("$1") ;;
		--dry-run)		dry_run=1 ;;
		--no-gpu)		use_gpu=0 ;;
		--prefix|--force-distro)
			(( $# >= 2 )) || usage_die "$1 needs a value"
			set_value "$1" "$2"
			shift ;;
		-h|--help)
			usage
			exit 0 ;;
		*)
			usage_die "unknown argument '$1'" ;;
		esac
		shift
	done
}

finish_options()
{
	if (( ${#phases[@]} == 0 )); then
		phases=(detect)
	fi
	models_dir=$data_dir/models
}

wanted()
{
	local p

	for p in "${phases[@]}"; do
		if [[ $p == "$1" ]]; then
			return 0
		fi
	done
	return 1
}

version_code()
{
	local major minor

	IFS=. read -r major minor _ <<< "$1"
	minor=${minor%%[!0-9]*}
	[[ $major =~ ^[0-9]+$ && $minor =~ ^[0-9]+$ ]] ||
		die "cannot parse version '$1'"
	printf '%d\n' $(( 10#$major * 100 + 10#$minor ))
}

os_field()
{
	sed -n "s/^$1=//p" /etc/os-release | tr -d '"'
}

gib()
{
	local kib=$1

	printf '%d.%d GiB' $(( kib / 1048576 )) \
		$(( kib % 1048576 * 10 / 1048576 ))
}

meminfo_kib()
{
	sed -n "s/^$1:[[:space:]]*\([0-9]*\) kB$/\1/p" /proc/meminfo
}

free_kib()
{
	local dir=$1

	while [[ ! -d $dir ]]; do
		dir=$(dirname "$dir")
	done
	df -Pk "$dir" | awk 'NR == 2 { print $4 }'
}

has_subids()
{
	local f

	for f in /etc/subuid /etc/subgid; do
		grep -qE "^($user|$uid):" "$f" 2>/dev/null || return 1
	done
}

linger_state()
{
	loginctl show-user "$user" -p Linger 2>/dev/null |
		sed -n 's/^Linger=//p' || true
}

gpu_query()
{
	nvidia-smi --query-gpu=memory.total,memory.used,driver_version \
		--format=csv,noheader,nounits 2>/dev/null | sed -n 1p
}

fact()
{
	facts+=("$1"$'\t'"$2"$'\t'"$3")
	if [[ $3 == FAIL* ]]; then
		blockers+=("$1: $2")
	fi
}

fact_distro()
{
	local id version

	id=$(os_field ID)
	version=$(os_field VERSION_ID)
	if [[ -n $force_distro ]]; then
		distro=$force_distro
		fact distro "$id $version, forced to $distro" ok
	elif [[ $id == arch ]]; then
		distro=arch
		fact distro "arch (rolling)" ok
	elif [[ $id == ubuntu ]] &&
	     (( $(version_code "$version") >= 2404 )); then
		distro=ubuntu
		fact distro "ubuntu $version" ok
	else
		die "unsupported distribution '$id $version'" \
			"(arch or ubuntu >= 24.04; --force-distro for tests)"
	fi
}

fact_user()
{
	if (( uid == 0 )); then
		fact user root "FAIL: rootless Podman needs a regular user"
	else
		fact user "$user ($uid)" ok
	fi
}

fact_podman()
{
	local version

	if ! command -v podman >/dev/null; then
		fact podman missing "fixed by host"
		return 0
	fi
	version=$(podman --version)
	version=${version##* }
	if (( $(version_code "$version") < PODMAN_MIN )); then
		fact podman "$version" "FAIL: need 4.9 or newer"
	else
		fact podman "$version" ok
	fi
}

fact_cgroup()
{
	local fs

	fs=$(stat -fc %T /sys/fs/cgroup)
	if [[ $fs == cgroup2fs ]]; then
		fact cgroup "v2 ($fs)" ok
	else
		fact cgroup "$fs" "FAIL: need cgroup v2 (cgroup2fs)"
	fi
}

fact_subids()
{
	if has_subids; then
		fact subids "/etc/subuid and /etc/subgid" ok
	else
		fact subids "missing for $user" "fixed by host"
	fi
}

fact_linger()
{
	local state

	state=$(linger_state)
	case $state in
	yes)	fact linger yes ok ;;
	no)	fact linger no "fixed by host" ;;
	*)	fact linger unknown "warn: loginctl did not answer" ;;
	esac
}

fact_gpu()
{
	local line="" verdict=ok

	if command -v nvidia-smi >/dev/null && line=$(gpu_query) &&
	   [[ -n $line ]]; then
		IFS=', ' read -r vram_total vram_used gpu_driver <<< "$line"
		gpu_found=1
	fi
	gpu_used=$(( gpu_found && use_gpu ))
	if (( ! gpu_found )); then
		fact gpu none "ok: CPU only"
		return 0
	fi
	if (( ! use_gpu )); then
		verdict="unused: --no-gpu"
	fi
	fact gpu "$vram_total MiB, $vram_used MiB used" "$verdict"
	fact driver "$gpu_driver" ok
}

fact_cdi()
{
	if (( ! gpu_used )); then
		return 0
	fi
	if command -v nvidia-ctk >/dev/null; then
		fact nvidia-ctk present ok
	else
		fact nvidia-ctk missing "fixed by host"
	fi
	if [[ -f /etc/cdi/nvidia.yaml ]]; then
		fact cdi /etc/cdi/nvidia.yaml ok
	else
		fact cdi missing "fixed by gpu"
	fi
}

fact_cpu()
{
	local flags f simd=""

	cores=$(lscpu -p=CORE,SOCKET | sed '/^#/d' | sort -u | wc -l)
	flags=$(lscpu | sed -n 's/^Flags:[[:space:]]*//p')
	for f in avx2 avx512f; do
		if [[ " $flags " == *" $f "* ]]; then
			simd+=" $f"
		fi
	done
	(( cores > 0 )) || die "lscpu reports no cores"
	fact cpu "$cores physical cores,${simd:- no avx2}" ok
}

fact_memory()
{
	local total avail

	mem_total_kib=$(meminfo_kib MemTotal)
	mem_avail_kib=$(meminfo_kib MemAvailable)
	total=$(gib "$mem_total_kib")
	avail=$(gib "$mem_avail_kib")
	fact memory "$total total, $avail available" ok
}

fact_disk()
{
	fact disk "$(gib "$(free_kib "$models_dir")") free" \
		"checked by models"
}

fact_tools()
{
	local t missing=""

	for t in curl jq; do
		if ! command -v "$t" >/dev/null; then
			missing+=" $t"
		fi
	done
	if [[ -z $missing ]]; then
		fact tools "curl jq" ok
	else
		fact tools "missing:$missing" "fixed by host"
	fi
}

load_facts()
{
	if (( facts_loaded )); then
		return 0
	fi
	fact_distro
	fact_user
	fact_podman
	fact_cgroup
	fact_subids
	fact_linger
	fact_gpu
	fact_cdi
	fact_cpu
	fact_memory
	fact_disk
	fact_tools
	facts_loaded=1
}

phase_detect()
{
	load_facts
	{
		printf 'check\tvalue\tverdict\n'
		printf '%s\n' "${facts[@]}"
	} | table 12 42
	if (( ${#blockers[@]} )); then
		info "${#blockers[@]} blocking problem(s):" \
			"phases that change the system will refuse to run"
	fi
}

main()
{
	local phase

	parse_args "$@"
	finish_options
	tmp_dir=$(mktemp -d)
	for phase in "${PHASES[@]}"; do
		if wanted "$phase"; then
			step "$phase"
			"phase_$phase"
		fi
	done
}

trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT
main "$@"
