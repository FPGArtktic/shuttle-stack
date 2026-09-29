#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# SHUTTLE M0 installer: two llama-server instances in rootless Podman, run
# by Quadlet on an internal network without egress.  See --help.

set -Eeuo pipefail
shopt -s nullglob

readonly PODMAN_MIN=409			# 4.9 as encoded by version_code()
readonly IMAGE_CUDA=ghcr.io/ggml-org/llama.cpp:server-cuda
readonly IMAGE_CPU=ghcr.io/ggml-org/llama.cpp:server
readonly HF_URL=https://huggingface.co
readonly NETWORK=shuttle
readonly NATIVE_CTX=32768		# Qwen3 without YaRN
readonly FAST_CTX=16384
readonly FAST_SLOTS=4
readonly VRAM_RESERVE_MB=768
readonly DRAFT_OVERHEAD_MB=256
# q8_0 KV of Qwen3-8B: 8 KV heads * 128 dims * (K + V) * 1 byte.
readonly KV_BYTES_PER_TOKEN_LAYER=2048
readonly PHASES=(detect plan)

declare -rA SERVER_PORT=([long]=8081 [fast]=8082)

phases=()
dry_run=0
use_gpu=1
use_draft=1
expose_direct=1
opt_ctx=""
opt_ngl=""
opt_threads=""
layers=36
force_distro=""

declare -A model_repo=(
	[long]=unsloth/Qwen3-8B-GGUF
	[fast]=unsloth/Qwen3-1.7B-GGUF
	[draft]=unsloth/Qwen3-0.6B-GGUF
)
declare -A model_file=(
	[long]=Qwen3-8B-Q4_K_M.gguf
	[fast]=Qwen3-1.7B-Q8_0.gguf
	[draft]=Qwen3-0.6B-Q8_0.gguf
)

user=${USER:-$(id -un)}
uid=$(id -u)
data_dir=${XDG_DATA_HOME:-$HOME/.local/share}/shuttle
state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/shuttle
models_dir=""
tmp_dir=""
hf_auth=()
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

# Filled by load_plan().
plan=()
plan_loaded=0
plan_threads=0
long_ctx=0
long_ngl=0
draft_ngl=0
declare -A server_image=()
declare -A server_device=()

usage()
{
	cat <<EOF
usage: install.sh [phase...] [options]

Phases always run in this order; without any, all of them run:
  detect plan

Options:
  --dry-run                  show commands and file changes, change nothing
  --ctx N                    context of shuttle-long (default: from RAM)
  --ngl N                    layers of shuttle-long on the GPU
  --threads N                CPU threads of both servers (default: cores)
  --layers N                 layers of the long model (default: 36)
  --no-gpu                   run shuttle-long on the CPU as well
  --no-draft                 disable speculative decoding on shuttle-long
  --expose-direct            publish 127.0.0.1:8081 and :8082 (default)
  --no-expose-direct         reach the servers only via the shuttle network
  --long-model REPO FILE     Hugging Face GGUF for shuttle-long
  --fast-model REPO FILE     Hugging Face GGUF for shuttle-fast
  --draft-model REPO FILE    Hugging Face GGUF for the draft model
  --force-distro arch|ubuntu skip distribution detection (tests only)
  --prefix DIR               models and KV cache location
                             (default: ~/.local/share/shuttle)

HF_TOKEN in the environment is sent to Hugging Face if set.
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

# Leading zeros are refused because bash arithmetic reads them as octal.
need_number()
{
	[[ $2 =~ ^(0|[1-9][0-9]*)$ ]] ||
		usage_die "$1 expects a number, got '$2'"
	(( $2 >= $3 )) || usage_die "$1 must be at least $3"
}

set_value()
{
	case $1 in
	--ctx)
		need_number "$1" "$2" 512
		opt_ctx=$2 ;;
	--ngl)
		need_number "$1" "$2" 0
		opt_ngl=$2 ;;
	--threads)
		need_number "$1" "$2" 1
		opt_threads=$2 ;;
	--layers)
		need_number "$1" "$2" 1
		layers=$2 ;;
	--prefix)
		[[ $2 == /* ]] || usage_die "--prefix needs an absolute path"
		data_dir=$2 ;;
	--force-distro)
		[[ $2 == arch || $2 == ubuntu ]] ||
			usage_die "--force-distro takes arch or ubuntu"
		force_distro=$2 ;;
	esac
}

set_model()
{
	local role=${1#--}

	role=${role%-model}
	[[ $2 =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] ||
		usage_die "$1: '$2' is not a Hugging Face REPO (owner/name)"
	[[ $3 == *.gguf && $3 != */* ]] ||
		usage_die "$1: '$3' is not a GGUF file name"
	model_repo[$role]=$2
	model_file[$role]=$3
}

# A long but flat switch, one line per option.
parse_args()
{
	while (( $# )); do
		case $1 in
		detect|plan)
			phases+=("$1") ;;
		--dry-run)		dry_run=1 ;;
		--no-gpu)		use_gpu=0 ;;
		--no-draft)		use_draft=0 ;;
		--expose-direct)	expose_direct=1 ;;
		--no-expose-direct)	expose_direct=0 ;;
		--ctx|--ngl|--threads|--layers|--prefix|--force-distro)
			(( $# >= 2 )) || usage_die "$1 needs a value"
			set_value "$1" "$2"
			shift ;;
		--long-model|--fast-model|--draft-model)
			(( $# >= 3 )) || usage_die "$1 needs REPO FILE"
			set_model "$1" "$2" "$3"
			shift 2 ;;
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
		phases=(detect plan)
	fi
	models_dir=$data_dir/models
}

# The token goes into a header file so that it never appears in the
# printed commands or in the log.
setup_hf_auth()
{
	if [[ -z ${HF_TOKEN:-} ]]; then
		return 0
	fi
	printf 'Authorization: Bearer %s\n' "$HF_TOKEN" > "$tmp_dir/hf-auth"
	hf_auth=(-H "@$tmp_dir/hf-auth")
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

hf_entry()
{
	local repo=$1 name=$2 url json

	url=$HF_URL/api/models/$repo/tree/main
	json=$(curl -fsSL "${hf_auth[@]}" "$url") ||
		die "cannot list $repo ($url)"
	# The tree API reports the LFS sha256 as lfs.oid.
	jq -er --arg f "$name" '.[] | select(.path == $f)
		| "\(.lfs.size // .size) \(.lfs.oid)"' <<< "$json" ||
		die "$repo has no file '$name'; not guessing another name"
}

model_bytes()
{
	local role=$1 path=$models_dir/${model_file[$1]} entry

	if [[ -f $path ]]; then
		stat -c %s "$path"
		return 0
	fi
	entry=$(hf_entry "${model_repo[$role]}" "${model_file[$role]}")
	printf '%s\n' "${entry%% *}"
}

plan_row()
{
	plan+=("$1"$'\t'"$2"$'\t'"${*:3}")
}

plan_threads()
{
	if [[ -n $opt_threads ]]; then
		plan_threads=$opt_threads
		plan_row threads "$plan_threads" "--threads"
	else
		plan_threads=$cores
		plan_row threads "$plan_threads" "physical cores for both" \
			"servers; they contend only when both are busy"
	fi
}

plan_images()
{
	server_image=([long]=$IMAGE_CPU [fast]=$IMAGE_CPU)
	server_device=([long]="" [fast]="")
	if (( gpu_used )); then
		server_image[long]=$IMAGE_CUDA
		server_device[long]=nvidia.com/gpu=all
	fi
	plan_row long.image "${server_image[long]##*/}" \
		"${server_device[long]:-no GPU device}"
	plan_row long.model "${model_file[long]}" "${model_repo[long]}"
}

# Thresholds are decimal GB, as in the specification, so that a nominal
# 24 GB machine is not demoted by the memory its firmware reserves.
plan_ctx()
{
	local bytes=$(( mem_total_kib * 1024 ))

	if [[ -n $opt_ctx ]]; then
		long_ctx=$opt_ctx
		plan_row long.ctx "$long_ctx" "--ctx"
	elif (( bytes >= 24000000000 )); then
		long_ctx=$NATIVE_CTX
		plan_row long.ctx "$long_ctx" \
			"RAM >= 24 GB: Qwen3 native maximum without YaRN"
	elif (( bytes >= 12000000000 )); then
		long_ctx=16384
		plan_row long.ctx "$long_ctx" "RAM 12-24 GB"
	else
		long_ctx=8192
		plan_row long.ctx "$long_ctx" "RAM below 12 GB"
	fi
	plan_yarn
}

plan_yarn()
{
	local scale factor

	if (( long_ctx <= NATIVE_CTX )); then
		return 0
	fi
	scale=$(( (long_ctx * 100 + NATIVE_CTX - 1) / NATIVE_CTX ))
	factor=$(printf '%d.%02d' $(( scale / 100 )) $(( scale % 100 )))
	plan_row long.rope "yarn x$factor" \
		"explicit --ctx above $NATIVE_CTX needs YaRN"
}

plan_ngl()
{
	if (( ! gpu_used )); then
		[[ ${opt_ngl:-0} == 0 ]] || die "--ngl $opt_ngl needs a GPU"
		plan_row long.ngl 0 "no GPU in use"
		return 0
	fi
	if (( use_draft )); then
		draft_ngl=all
	fi
	if [[ -n $opt_ngl ]]; then
		long_ngl=$(( opt_ngl < layers ? opt_ngl : layers ))
		plan_row long.ngl "$long_ngl" "--ngl"
		return 0
	fi
	estimate_ngl
}

estimate_ngl()
{
	local bytes model_mib draft_mib usable kv_kib per_layer_kib draft=""

	bytes=$(model_bytes long)
	model_mib=$(( bytes / 1048576 ))
	usable=$(( vram_total - vram_used - VRAM_RESERVE_MB ))
	if (( use_draft )); then
		bytes=$(model_bytes draft)
		draft_mib=$(( bytes / 1048576 ))
		usable=$(( usable - draft_mib - DRAFT_OVERHEAD_MB ))
		draft=" - draft ($draft_mib + $DRAFT_OVERHEAD_MB)"
	fi
	kv_kib=$(( long_ctx * KV_BYTES_PER_TOKEN_LAYER / 1024 ))
	per_layer_kib=$(( model_mib * 1024 / layers + kv_kib ))
	long_ngl=$(( usable > 0 ? usable * 1024 / per_layer_kib : 0 ))
	long_ngl=$(( long_ngl > layers ? layers : long_ngl ))
	plan_row long.ngl "~$long_ngl" "usable VRAM / VRAM per layer"
	plan_row "" "" "usable $usable MiB = $vram_total total -" \
		"$vram_used used - $VRAM_RESERVE_MB reserve$draft"
	plan_row "" "" "per layer $(( per_layer_kib / 1024 )) MiB =" \
		"$model_mib MiB / $layers layers + KV q8_0" \
		"$(( kv_kib / 1024 )) MiB at ctx $long_ctx"
}

plan_draft()
{
	if (( ! use_draft )); then
		plan_row long.draft off "--no-draft"
		return 0
	fi
	plan_row long.draft "${model_file[draft]}" \
		"speculative decoding, draft ngl $draft_ngl"
}

plan_servers()
{
	plan_row long.np 1 "KV q8_0, flash attention, slot save to /cache"
	plan_row fast.model "${model_file[fast]}" "${model_repo[fast]}"
	plan_row fast.image "${IMAGE_CPU##*/}" \
		"CPU only: the GPU belongs to long"
	plan_row fast.np "$FAST_SLOTS" "parallel slots"
	plan_row fast.ctx "$FAST_CTX" \
		"-c is the total; unified KV lets one slot use all of it"
	if (( expose_direct )); then
		plan_row direct "127.0.0.1:${SERVER_PORT[long]} long" \
			"and 127.0.0.1:${SERVER_PORT[fast]} fast"
	else
		plan_row direct off "only via the $NETWORK network"
	fi
}

load_plan()
{
	if (( plan_loaded )); then
		return 0
	fi
	load_facts
	plan_threads
	plan_images
	plan_ctx
	plan_ngl
	plan_draft
	plan_servers
	plan_loaded=1
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

phase_plan()
{
	load_plan
	{
		printf 'setting\tvalue\treason (~ marks an estimate)\n'
		printf '%s\n' "${plan[@]}"
	} | table 12 22
}

main()
{
	local phase

	parse_args "$@"
	finish_options
	tmp_dir=$(mktemp -d)
	setup_hf_auth
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
