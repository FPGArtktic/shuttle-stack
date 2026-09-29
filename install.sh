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
readonly IMAGE_CURL=docker.io/curlimages/curl
readonly CDI_SPEC=/etc/cdi/nvidia.yaml
readonly HF_URL=https://huggingface.co
readonly NETWORK=shuttle
readonly SUBNET=10.89.7.0/24
readonly GATEWAY=10.89.7.1
readonly PROBE=shuttle-probe
readonly NATIVE_CTX=32768		# Qwen3 without YaRN
readonly FAST_CTX=16384
readonly FAST_SLOTS=4
readonly VRAM_RESERVE_MB=768
readonly DRAFT_OVERHEAD_MB=256
# q8_0 KV of Qwen3-8B: 8 KV heads * 128 dims * (K + V) * 1 byte.
readonly KV_BYTES_PER_TOKEN_LAYER=2048
readonly HEALTH_TIMEOUT=900
readonly BENCH_PREDICT=64
readonly PHASES=(detect plan host gpu quadlets models verify bench status)
readonly EXEC_COMMON="--slot-save-path /cache --flash-attn on --no-webui"

declare -rA SERVER_IP=([long]=10.89.7.10 [fast]=10.89.7.11)
declare -rA SERVER_PORT=([long]=8081 [fast]=8082)
declare -rA SERVER_EXEC=(
	[long]="$EXEC_COMMON --cache-type-k q8_0 --cache-type-v q8_0"
	[fast]="$EXEC_COMMON"
)

# Every difference between the supported distributions lives here.
declare -rA DISTRO=(
	[arch.refresh]=""
	[arch.install]="pacman -S --needed --noconfirm"
	[arch.query]=pacman_installed
	[arch.packages]="podman passt jq curl"
	[arch.gpu_packages]="nvidia-container-toolkit"
	[arch.gpu_repo]=""
	[ubuntu.refresh]="apt-get update"
	[ubuntu.install]="apt-get install -y"
	[ubuntu.query]=dpkg_installed
	[ubuntu.packages]="podman passt uidmap jq curl"
	[ubuntu.gpu_packages]="nvidia-container-toolkit"
	[ubuntu.gpu_repo]=nvidia_apt_repo
)

phases=()
dry_run=0
assume_yes=0
use_gpu=1
use_draft=1
expose_direct=1
opt_ctx=""
opt_ngl=""
opt_threads=""
layers=36
draft_layers=28
bench_tokens=(2048 8192)
bench_custom=0
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
declare -A model_size=()
declare -A model_sha=()
models=()

user=${USER:-$(id -un)}
uid=$(id -u)
config_home=${XDG_CONFIG_HOME:-$HOME/.config}
config_dir=$config_home/shuttle
unit_dir=$config_home/containers/systemd
data_dir=${XDG_DATA_HOME:-$HOME/.local/share}/shuttle
state_dir=${XDG_STATE_HOME:-$HOME/.local/state}/shuttle
models_dir=""
cache_dir=""
tmp_dir=""
stamp=""
hf_auth=()
cleanups=()
changed=0
route=()

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
draft_weights_mib=0
draft_kv_mib=0
extra_network=""
declare -A server_image=()
declare -A server_device=()
declare -A server_ctx=()
declare -A server_ngl=()
declare -A server_slots=()
declare -A server_env=()

usage()
{
	cat <<EOF
usage: install.sh [phase...] [options]

Phases always run in this order; without any, all but bench run:
  detect plan host gpu quadlets models verify bench status

Options:
  --dry-run                  show commands and file changes, change nothing
  --yes                      do not ask before commands that run as root
  --ctx N                    context of shuttle-long (default: from RAM)
  --ngl N                    layers of shuttle-long on the GPU
  --threads N                CPU threads of both servers (default: cores)
  --layers N                 layers of the long model (default: 36)
  --draft-layers N           layers of the draft model (default: 28)
  --no-gpu                   run shuttle-long on the CPU as well
  --no-draft                 disable speculative decoding on shuttle-long
  --expose-direct            publish 127.0.0.1:8081 and :8082 (default)
  --no-expose-direct         reach the servers only via the shuttle network
  --long-model REPO FILE     Hugging Face GGUF for shuttle-long
  --fast-model REPO FILE     Hugging Face GGUF for shuttle-fast
  --draft-model REPO FILE    Hugging Face GGUF for the draft model
  --force-distro arch|ubuntu skip distribution detection (tests only)
  --bench-tokens N           prompt size for bench; repeat for several
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

confirm()
{
	local reply

	if (( assume_yes || dry_run )); then
		return 0
	fi
	[[ -t 0 ]] || die "no terminal to confirm '$*'; rerun with --yes"
	read -r -p "    run as root: $* [y/N] " reply
	[[ $reply == [yY] || $reply == [yY][eE][sS] ]] || die "declined: $*"
}

root()
{
	confirm "$(quote "$@")"
	run sudo "$@"
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
	--draft-layers)
		need_number "$1" "$2" 1
		draft_layers=$2 ;;
	--bench-tokens)
		need_number "$1" "$2" 1
		add_bench_size "$2" ;;
	--prefix)
		[[ $2 == /* ]] || usage_die "--prefix needs an absolute path"
		data_dir=$2 ;;
	--force-distro)
		[[ $2 == arch || $2 == ubuntu ]] ||
			usage_die "--force-distro takes arch or ubuntu"
		force_distro=$2 ;;
	esac
}

add_bench_size()
{
	if (( ! bench_custom )); then
		bench_tokens=()
		bench_custom=1
	fi
	bench_tokens+=("$1")
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
		detect|plan|host|gpu|quadlets|models|verify|bench|status)
			phases+=("$1") ;;
		--dry-run)		dry_run=1 ;;
		--yes)			assume_yes=1 ;;
		--no-gpu)		use_gpu=0 ;;
		--no-draft)		use_draft=0 ;;
		--expose-direct)	expose_direct=1 ;;
		--no-expose-direct)	expose_direct=0 ;;
		--ctx|--ngl|--threads|--layers|--draft-layers|\
		--bench-tokens|--prefix|--force-distro)
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
		phases=(detect plan host gpu quadlets models verify status)
	fi
	models=(long fast)
	if (( use_draft )); then
		models+=(draft)
	fi
	models_dir=$data_dir/models
	cache_dir=$data_dir/cache
	stamp=$(date +%Y%m%d-%H%M%S)
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

# A running server holds VRAM, and counting it as taken would lower the
# estimate on every rerun, so what the servers use is measured here and
# left out of the figure the plan works from.
own_vram_mib()
{
	local role pid apps used total=0

	apps=$(nvidia-smi --query-compute-apps=pid,used_memory \
		--format=csv,noheader,nounits 2>/dev/null) || true
	for role in long fast; do
		pid=$(podman inspect --format '{{.State.Pid}}' \
			"shuttle-$role" 2>/dev/null) || continue
		used=$(awk -F', ' -v p="$pid" '$1 == p { s += $2 }
			END { print s + 0 }' <<< "$apps")
		total=$(( total + used ))
	done
	printf '%d\n' "$total"
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
	local line="" verdict=ok own=0

	if command -v nvidia-smi >/dev/null && line=$(gpu_query) &&
	   [[ -n $line ]]; then
		IFS=', ' read -r vram_total vram_used gpu_driver <<< "$line"
		gpu_found=1
		own=$(own_vram_mib)
		vram_used=$(( vram_used > own ? vram_used - own : 0 ))
	fi
	gpu_used=$(( gpu_found && use_gpu ))
	if (( ! gpu_found )); then
		fact gpu none "ok: CPU only"
		return 0
	fi
	if (( ! use_gpu )); then
		verdict="unused: --no-gpu"
	fi
	fact gpu "$vram_total MiB, $vram_used MiB used elsewhere" "$verdict"
	fact driver "$gpu_driver" ok
	if (( own )); then
		fact "own vram" "$own MiB in running shuttle servers" \
			"not counted as taken"
	fi
}

# Every library path in the specification carries the driver version, so
# an upgraded driver leaves it naming files that are no longer there.
cdi_current()
{
	[[ -n $gpu_driver && -f $CDI_SPEC ]] &&
		grep -qF "$gpu_driver" "$CDI_SPEC"
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
	if [[ ! -f $CDI_SPEC ]]; then
		fact cdi missing "fixed by gpu"
	elif cdi_current; then
		fact cdi "$CDI_SPEC" ok
	else
		fact cdi "$CDI_SPEC" "stale: driver is $gpu_driver; fixed by gpu"
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

require_supported()
{
	load_facts
	(( ${#blockers[@]} == 0 )) ||
		die "blocked by: ${blockers[*]} (see the detect phase)"
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
	server_env=([long]="" [fast]="LLAMA_ARG_KV_UNIFIED=1"$'\n')
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
	server_env[long]+="LLAMA_ARG_ROPE_SCALING_TYPE=yarn"$'\n'
	server_env[long]+="LLAMA_ARG_ROPE_SCALE=$factor"$'\n'
	server_env[long]+="LLAMA_ARG_YARN_ORIG_CTX=$NATIVE_CTX"$'\n'
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

# llama.cpp sizes the draft context from the target context and offers no
# option to cap it, so the draft KV follows --ctx and outgrows the draft
# weights.  Every Qwen3 size uses 8 KV heads of 128 dimensions, so a
# draft layer costs what a long layer costs.
draft_budget()
{
	local bytes kib

	bytes=$(model_bytes draft)
	draft_weights_mib=$(( bytes / 1048576 ))
	kib=$(( draft_layers * long_ctx * KV_BYTES_PER_TOKEN_LAYER / 1024 ))
	draft_kv_mib=$(( kib / 1024 ))
}

estimate_ngl()
{
	local bytes model_mib usable kv_kib per_layer_kib draft=""

	bytes=$(model_bytes long)
	model_mib=$(( bytes / 1048576 ))
	usable=$(( vram_total - vram_used - VRAM_RESERVE_MB ))
	if (( use_draft )); then
		draft_budget
		usable=$(( usable - draft_weights_mib - draft_kv_mib ))
		usable=$(( usable - DRAFT_OVERHEAD_MB ))
		draft=" - draft"
	fi
	kv_kib=$(( long_ctx * KV_BYTES_PER_TOKEN_LAYER / 1024 ))
	per_layer_kib=$(( model_mib * 1024 / layers + kv_kib ))
	long_ngl=$(( usable > 0 ? usable * 1024 / per_layer_kib : 0 ))
	long_ngl=$(( long_ngl > layers ? layers : long_ngl ))
	explain_ngl "$usable" "$model_mib" "$per_layer_kib" "$kv_kib" "$draft"
}

explain_ngl()
{
	local usable=$1 model_mib=$2 per_layer_kib=$3 kv_kib=$4 draft=$5

	plan_row long.ngl "~$long_ngl" "usable VRAM / VRAM per layer"
	plan_row "" "" "usable $usable MiB = $vram_total total -" \
		"$vram_used used - $VRAM_RESERVE_MB reserve$draft"
	if (( use_draft )); then
		plan_row "" "" "draft $draft_weights_mib MiB weights +" \
			"$draft_kv_mib MiB KV over $draft_layers layers +" \
			"$DRAFT_OVERHEAD_MB MiB overhead; its context" \
			"follows --ctx and cannot be capped"
	fi
	plan_row "" "" "per layer $(( per_layer_kib / 1024 )) MiB =" \
		"$model_mib MiB / $layers layers + KV q8_0" \
		"$(( kv_kib / 1024 )) MiB at ctx $long_ctx"
}

plan_draft()
{
	local draft=/models/${model_file[draft]}

	if (( ! use_draft )); then
		plan_row long.draft off "--no-draft"
		return 0
	fi
	server_env[long]+="LLAMA_ARG_SPEC_TYPE=draft-simple"$'\n'
	server_env[long]+="LLAMA_ARG_SPEC_DRAFT_MODEL=$draft"$'\n'
	server_env[long]+="LLAMA_ARG_N_GPU_LAYERS_DRAFT=$draft_ngl"$'\n'
	server_env[long]+="LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K=q8_0"$'\n'
	server_env[long]+="LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V=q8_0"$'\n'
	plan_row long.draft "${model_file[draft]}" \
		"speculative decoding, draft ngl $draft_ngl"
	plan_row "" "" "spec type draft-simple: a draft model on its own" \
		"leaves speculative decoding off"
}

plan_servers()
{
	server_ctx=([long]=$long_ctx [fast]=$FAST_CTX)
	server_ngl=([long]=$long_ngl [fast]=0)
	server_slots=([long]=1 [fast]=$FAST_SLOTS)
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

# NVIDIA's documented apt source, with the armored key kept as .asc so
# that gpg is not needed on the host.
nvidia_apt_repo()
{
	local base=https://nvidia.github.io/libnvidia-container
	local key=/usr/share/keyrings/nvidia-container-toolkit-keyring.asc
	local list=/etc/apt/sources.list.d/nvidia-container-toolkit.list

	run curl -fsSL -o "$tmp_dir/nvidia.asc" "$base/gpgkey"
	run curl -fsSL -o "$tmp_dir/nvidia.list" \
		"$base/stable/deb/nvidia-container-toolkit.list"
	run sed -i "s#^deb https://#deb [signed-by=$key] https://#" \
		"$tmp_dir/nvidia.list"
	root install -m 0644 "$tmp_dir/nvidia.asc" "$key"
	root install -m 0644 "$tmp_dir/nvidia.list" "$list"
}

pacman_installed()
{
	pacman -Qq "$1" > /dev/null 2>&1
}

# dpkg still knows a package that was removed but kept its configuration
# files, so the status is compared instead of the exit code.
dpkg_installed()
{
	local status

	status=$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null) || return 1
	[[ $status == "install ok installed" ]]
}

missing_packages()
{
	local query=${DISTRO[$distro.query]} p

	for p in "$@"; do
		if ! "$query" "$p"; then
			printf '%s\n' "$p"
		fi
	done
}

host_packages()
{
	local -a refresh install packages gpu missing
	local repo=${DISTRO[$distro.gpu_repo]}

	read -ra packages <<< "${DISTRO[$distro.packages]}"
	if (( gpu_used )); then
		read -ra gpu <<< "${DISTRO[$distro.gpu_packages]}"
		packages+=("${gpu[@]}")
	fi
	mapfile -t missing < <(missing_packages "${packages[@]}")
	if (( ${#missing[@]} == 0 )); then
		info "packages: ${#packages[@]} present"
		return 0
	fi
	if (( gpu_used )) && [[ -n $repo ]]; then
		"$repo"
	fi
	read -ra refresh <<< "${DISTRO[$distro.refresh]}"
	read -ra install <<< "${DISTRO[$distro.install]}"
	if (( ${#refresh[@]} )); then
		root "${refresh[@]}"
	fi
	root "${install[@]}" "${missing[@]}"
}

# The fixed range must not overlap another user's, or two users would
# share container UIDs.
host_subids()
{
	local f

	if has_subids; then
		info "subuid/subgid: present"
		return 0
	fi
	for f in /etc/subuid /etc/subgid; do
		if awk -F: '$2 < 165536 && $2 + $3 > 100000 { found = 1 }
			    END { exit !found }' "$f" 2>/dev/null; then
			die "$f already uses part of 100000-165535;" \
				"add a free range for $user by hand"
		fi
	done
	root usermod --add-subuids 100000-165535 \
		--add-subgids 100000-165535 "$user"
	run podman system migrate
}

host_linger()
{
	if [[ $(linger_state) == yes ]]; then
		info "linger: enabled"
		return 0
	fi
	root loginctl enable-linger "$user"
}

phase_host()
{
	require_supported
	host_packages
	host_subids
	host_linger
}

phase_gpu()
{
	require_supported
	if (( ! gpu_used )); then
		info "skipped: no GPU in use"
		return 0
	fi
	if cdi_current; then
		info "cdi: $CDI_SPEC matches driver $gpu_driver"
	else
		root nvidia-ctk cdi generate --output="$CDI_SPEC"
	fi
	run podman run --rm --device nvidia.com/gpu=all \
		--entrypoint nvidia-smi "$IMAGE_CUDA"
}

pull_images()
{
	local image images=("${server_image[@]}")
	local -A seen=()

	if (( ! expose_direct )); then
		images+=("$IMAGE_CURL")
	fi
	for image in "${images[@]}"; do
		if [[ -z ${seen[$image]:-} ]]; then
			seen[$image]=1
			run podman pull "$image"
		fi
	done
}

probe_cleanup()
{
	podman rm -f "$PROBE" >/dev/null 2>&1 || true
	podman network rm -f "$PROBE" >/dev/null 2>&1 || true
}

probe_wait_internal()
{
	local i code

	for (( i = 0; i < 30; i++ )); do
		code=$(podman run --rm --network "$PROBE" \
			--entrypoint curl "$IMAGE_CPU" -s -o /dev/null \
			-w '%{http_code}' "http://$PROBE:8080/health") || true
		if [[ $code == [1-5][0-9][0-9] ]]; then
			return 0
		fi
		sleep 1
	done
	die "PublishPort probe inconclusive: nothing answered inside" \
		"the internal network within 30 s"
}

# Whether rootless PublishPort reaches a container on an Internal=true
# network depends on the Podman and netavark versions, so it is measured
# on the installed Podman.  llama-server without a model starts in router
# mode, which is enough of a listener.
probe_publish()
{
	local address code

	extra_network=""
	if (( ! expose_direct )); then
		return 0
	fi
	if (( dry_run )); then
		info "dry-run: PublishPort probe not run; units assume it works"
		return 0
	fi
	cleanups+=(probe_cleanup)
	probe_cleanup
	run podman network create --internal "$PROBE" >/dev/null
	run podman run -d --name "$PROBE" --network "$PROBE" \
		-p 127.0.0.1::8080 "$IMAGE_CPU" --host 0.0.0.0 --port 8080 \
		>/dev/null
	probe_wait_internal
	address=$(podman port "$PROBE" 8080/tcp)
	code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
		"http://$address/health") || true
	probe_cleanup
	if [[ $code != 000 ]]; then
		info "PublishPort reaches the internal network"
		return 0
	fi
	extra_network=podman
	info "PublishPort does not reach Internal=true networks here;" \
		"adding Network=podman, which gives the containers egress"
}

# A renamed variable is silently ignored by llama-server (the draft model
# already moved to LLAMA_ARG_SPEC_DRAFT_MODEL), so the image is asked.
check_env_names()
{
	local role=$1 help name device=()

	if (( dry_run )); then
		info "dry-run: $role.env not checked against the image --help"
		return 0
	fi
	if [[ -n ${server_device[$role]} ]]; then
		device=(--device "${server_device[$role]}")
	fi
	help=$(podman run --rm "${device[@]}" "${server_image[$role]}" \
		--help 2>&1) || die "${server_image[$role]} --help failed"
	while IFS='=' read -r name _; do
		if [[ $name == LLAMA_ARG_* && $help != *"(env: $name)"* ]]; then
			die "${server_image[$role]} does not know $name"
		fi
	done < "$tmp_dir/$role.env"
}

render_env()
{
	local role=$1

	cat <<EOF
# Generated by shuttle install.sh; local edits are overwritten.
LLAMA_ARG_MODEL=/models/${model_file[$role]}
LLAMA_ARG_ALIAS=shuttle-$role
LLAMA_ARG_CTX_SIZE=${server_ctx[$role]}
LLAMA_ARG_N_GPU_LAYERS=${server_ngl[$role]}
LLAMA_ARG_THREADS=$plan_threads
LLAMA_ARG_N_PARALLEL=${server_slots[$role]}
LLAMA_ARG_HOST=0.0.0.0
LLAMA_ARG_PORT=8080
EOF
	printf '%s' "${server_env[$role]}"
}

render_stack_env()
{
	local role name

	printf '# Generated by shuttle install.sh; local edits are'
	printf ' overwritten.\n'
	printf 'SHUTTLE_NETWORK=%s\n' "$NETWORK"
	for role in long fast; do
		name=SHUTTLE_${role^^}
		printf '%s_URL=http://shuttle-%s:8080\n' "$name" "$role"
		printf '%s_IP=%s\n' "$name" "${SERVER_IP[$role]}"
		if (( expose_direct )); then
			printf '%s_DIRECT_PORT=%s\n' "$name" \
				"${SERVER_PORT[$role]}"
		fi
	done
}

render_network()
{
	cat <<EOF
# Generated by shuttle install.sh; local edits are overwritten.
[Unit]
Description=SHUTTLE internal network (no egress)

[Network]
NetworkName=$NETWORK
Subnet=$SUBNET
Gateway=$GATEWAY
Internal=true
EOF
}

# The address is given as a network option rather than IP= because
# Podman refuses --ip once a second network is attached.
render_container()
{
	local role=$1

	cat <<EOF
# Generated by shuttle install.sh; local edits are overwritten.
[Unit]
Description=SHUTTLE $role llama-server

[Container]
Image=${server_image[$role]}
ContainerName=shuttle-$role
Network=$NETWORK.network:ip=${SERVER_IP[$role]}
EnvironmentFile=$config_dir/$role.env
Volume=$models_dir:/models:ro,z
Volume=$cache_dir/$role:/cache:rw,Z
Exec=${SERVER_EXEC[$role]}
HealthCmd=curl -fsS -o /dev/null http://127.0.0.1:8080/health
HealthInterval=30s
HealthStartPeriod=${HEALTH_TIMEOUT}s
EOF
	if [[ -n $extra_network ]]; then
		printf 'Network=%s\n' "$extra_network"
	fi
	if [[ -n ${server_device[$role]} ]]; then
		printf 'AddDevice=%s\n' "${server_device[$role]}"
	fi
	if (( expose_direct )); then
		printf 'PublishPort=127.0.0.1:%s:8080\n' "${SERVER_PORT[$role]}"
	fi
	cat <<EOF

[Service]
Restart=on-failure
TimeoutStartSec=$HEALTH_TIMEOUT

[Install]
WantedBy=default.target
EOF
}

show_diff()
{
	local old=$1

	if [[ ! -f $old ]]; then
		old=/dev/null
	fi
	diff -u --label "$1" --label "$1 (new)" "$old" "$2" > "$2.diff" ||
		[[ $? -eq 1 ]]
	sed 's/^/    /' "$2.diff" >&2
}

# Files are rewritten only when their content changes, and the previous
# version is kept, so a rerun is cheap and every change is reversible.
install_file()
{
	local dst=$1 new

	new=$(mktemp -p "$tmp_dir")
	cat > "$new"
	if [[ -f $dst ]] && cmp -s "$new" "$dst"; then
		info "unchanged: $dst"
		return 0
	fi
	show_diff "$dst" "$new"
	if [[ -f $dst ]]; then
		run cp -p "$dst" "$dst.bak.$stamp"
	fi
	run install -D -m 0644 "$new" "$dst"
	changed=1
}

apply_changes()
{
	if (( ! changed )); then
		return 0
	fi
	run systemctl --user daemon-reload
	run systemctl --user try-restart shuttle-long.service \
		shuttle-fast.service
}

phase_quadlets()
{
	local role

	require_supported
	load_plan
	pull_images
	probe_publish
	for role in long fast; do
		render_env "$role" > "$tmp_dir/$role.env"
		check_env_names "$role"
	done
	run mkdir -p "$models_dir" "$cache_dir/long" "$cache_dir/fast"
	install_file "$config_dir/long.env" < "$tmp_dir/long.env"
	install_file "$config_dir/fast.env" < "$tmp_dir/fast.env"
	install_file "$config_dir/stack.env" < <(render_stack_env)
	install_file "$unit_dir/$NETWORK.network" < <(render_network)
	for role in long fast; do
		install_file "$unit_dir/shuttle-$role.container" \
			< <(render_container "$role")
	done
	apply_changes
}

model_lookup()
{
	local role=$1 entry

	entry=$(hf_entry "${model_repo[$role]}" "${model_file[$role]}")
	read -r "model_size[$role]" "model_sha[$role]" <<< "$entry"
	[[ ${model_sha[$role]} =~ ^[0-9a-f]{64}$ ]] ||
		die "${model_repo[$role]}: no sha256 for ${model_file[$role]}"
	info "${model_file[$role]}: $(gib $(( model_size[$role] / 1024 )))" \
		"sha256 ${model_sha[$role]:0:16}"
}

model_valid()
{
	local role=$1 path=$models_dir/${model_file[$1]} sum

	if [[ ! -f $path ]]; then
		return 1
	fi
	info "checking $path"
	sum=$(sha256sum "$path")
	[[ ${sum%% *} == "${model_sha[$role]}" ]]
}

part_bytes()
{
	local part=$models_dir/${model_file[$1]}.part

	if [[ -f $part ]]; then
		stat -c %s "$part"
	else
		echo 0
	fi
}

verify_sha()
{
	local sum

	sum=$(sha256sum "$1")
	sum=${sum%% *}
	if [[ $sum != "$2" ]]; then
		rm -f "$1"
		die "sha256 mismatch for $1 (got $sum, want $2);" \
			"the file was removed, rerun to download it again"
	fi
}

check_space()
{
	local need_kib=$(( $1 / 1024 )) free

	free=$(free_kib "$models_dir")
	(( free > need_kib + 1048576 )) ||
		die "models need $(gib "$need_kib") and 1 GiB spare," \
			"but $models_dir has $(gib "$free") free"
}

fetch_model()
{
	local role=$1 dst=$models_dir/${model_file[$1]}
	local url=$HF_URL/${model_repo[$1]}/resolve/main/${model_file[$1]}

	if (( $(part_bytes "$role") < model_size[$role] )); then
		run curl -fL -C - --retry 5 "${hf_auth[@]}" \
			-o "$dst.part" "$url"
	fi
	run verify_sha "$dst.part" "${model_sha[$role]}"
	run mv -f "$dst.part" "$dst"
}

phase_models()
{
	local role part need=0 pending=()

	require_supported
	for role in "${models[@]}"; do
		model_lookup "$role"
		if model_valid "$role"; then
			info "ok: ${model_file[$role]}"
		else
			pending+=("$role")
			part=$(part_bytes "$role")
			need=$(( need + model_size[$role] - part ))
		fi
	done
	check_space "$need"
	run mkdir -p "$models_dir"
	for role in "${pending[@]}"; do
		fetch_model "$role"
	done
}

# Sets route to the curl command that reaches server $1 at path $2: the
# host port, or a curl container on the internal network.
set_route()
{
	local role=$1 path=$2

	shift 2
	if (( expose_direct )); then
		route=(curl -fsS "$@"
			"http://127.0.0.1:${SERVER_PORT[$role]}$path")
	else
		route=(podman run --rm -i --network "$NETWORK" "$IMAGE_CURL"
			-fsS "$@" "http://shuttle-$role:8080$path")
	fi
}

api()
{
	set_route "$@"
	run "${route[@]}"
}

wait_healthy()
{
	local role=$1 unit=shuttle-$1.service waited=0

	info "waiting for $unit to load its model (up to ${HEALTH_TIMEOUT}s)"
	until api "$role" /health < /dev/null > /dev/null 2>&1; do
		run systemctl --user is-active --quiet "$unit" ||
			die "$unit is not running;" \
				"see journalctl --user -u $unit"
		(( waited < HEALTH_TIMEOUT )) ||
			die "$unit: no /health after ${HEALTH_TIMEOUT}s"
		sleep 5
		waited=$(( waited + 5 ))
	done
}

smoke_completion()
{
	local role=$1 body out

	body=$(jq -n '{prompt: "The capital of Poland is", n_predict: 32}')
	out=$(api "$role" /completion --json @- <<< "$body") ||
		die "shuttle-$role: /completion failed"
	jq -r --arg s "shuttle-$role" '[$s, "ok",
		(.timings.prompt_per_second * 10 | round / 10),
		(.timings.predicted_per_second * 10 | round / 10)] | @tsv' \
		<<< "$out"
}

slot_action()
{
	api long "/slots/0?action=$1" --json @- <<< '{"filename": "smoke.bin"}'
}

kv_roundtrip()
{
	local file=$cache_dir/long/smoke.bin out saved restored

	out=$(slot_action save) || die "shuttle-long: slot save failed"
	saved=$(jq -r '.n_saved // empty' <<< "$out")
	run test -f "$file" || die "slot save did not create $file"
	out=$(slot_action restore) || die "shuttle-long: slot restore failed"
	restored=$(jq -r '.n_restored // empty' <<< "$out")
	run rm -f "$file"
	printf 'kv save/restore\tshuttle-long\tok\t%s saved\t%s restored\n' \
		"${saved:--}" "${restored:--}"
}

phase_verify()
{
	local role rows=()

	require_supported
	run systemctl --user daemon-reload
	run systemctl --user start shuttle-long.service shuttle-fast.service
	for role in long fast; do
		wait_healthy "$role"
	done
	rows+=("test"$'\t'"server"$'\t'"result"$'\t'"pp tok/s"$'\t'"tg tok/s")
	for role in long fast; do
		rows+=("completion"$'\t'"$(smoke_completion "$role")")
	done
	rows+=("$(kv_roundtrip)")
	printf '%s\n' "${rows[@]}" | table 16 14 8 12
}

# A prompt of exactly n tokens: every number is at least one token, so
# tokenizing 1..n gives enough, and the ids are cut to length.
bench_prompt()
{
	local role=$1 n=$2 body

	body=$(seq -s ' ' 1 "$n" | jq -Rs '{content: .}')
	api "$role" /tokenize --json @- <<< "$body" |
		jq -c --argjson n "$n" --argjson p "$BENCH_PREDICT" '
			if (.tokens | length) < $n then error("too few tokens")
			else {prompt: .tokens[:$n], n_predict: $p,
			      cache_prompt: false} end'
}

bench_one()
{
	local role=$1 n=$2 ctx=$3 body

	if [[ -n $ctx ]] && (( n + BENCH_PREDICT > ctx )); then
		info "shuttle-$role: $n tokens skipped, slot context is $ctx"
		return 0
	fi
	info "shuttle-$role: $n-token prompt"
	body=$(bench_prompt "$role" "$n")
	api "$role" /completion --json @- <<< "$body" |
		jq -c --arg s "shuttle-$role" --argjson n "$n" '{server: $s,
			tokens: $n, prompt_n: .timings.prompt_n,
			pp: (.timings.prompt_per_second * 10 | round / 10),
			tg: (.timings.predicted_per_second * 10 | round / 10)}'
}

env_json()
{
	local file=$config_dir/$1.env

	if [[ ! -f $file ]]; then
		echo '{}'
		return 0
	fi
	jq -Rn '[inputs | select(startswith("LLAMA_ARG_"))
		| capture("^(?<key>[^=]+)=(?<value>.*)$")] | from_entries' \
		< "$file"
}

bench_report()
{
	local file=$state_dir/bench-$stamp.json json

	json=$(printf '%s\n' "$@" | jq -s --arg date "$(date -Is)" \
		--argjson long "$(env_json long)" \
		--argjson fast "$(env_json fast)" \
		'{date: $date, config: {long: $long, fast: $fast},
		  results: .}')
	printf '%s\n' "$json" > "$tmp_dir/bench.json"
	run install -D -m 0644 "$tmp_dir/bench.json" "$file"
	{
		printf 'server\ttokens\tpp tok/s\ttg tok/s\n'
		jq -r '.results[] | [.server, .tokens, .pp, .tg] | @tsv' \
			<<< "$json"
	} | table 14 8 10
}

phase_bench()
{
	local role n ctx results=()

	require_supported
	for role in long fast; do
		ctx=$(api "$role" /props < /dev/null |
			jq -r '.default_generation_settings.n_ctx // empty') ||
			die "shuttle-$role: /props failed; run verify first"
		for n in "${bench_tokens[@]}"; do
			results+=("$(bench_one "$role" "$n" "$ctx")")
		done
	done
	bench_report "${results[@]}"
}

status_units()
{
	local unit state

	for unit in shuttle-network shuttle-long shuttle-fast; do
		state=$(systemctl --user is-active "$unit.service" \
			2>/dev/null) || true
		printf 'unit\t%s\t%s\n' "$unit" "${state:-unknown}"
	done
}

status_health()
{
	local role

	# Read-only, so it runs even under --dry-run.
	for role in long fast; do
		set_route "$role" /health
		if "${route[@]}" < /dev/null > /dev/null 2>&1; then
			printf 'health\tshuttle-%s\tok\n' "$role"
		else
			printf 'health\tshuttle-%s\tdown\n' "$role"
		fi
	done
}

status_models()
{
	local role path

	for role in "${models[@]}"; do
		path=$models_dir/${model_file[$role]}
		if [[ -f $path ]]; then
			printf 'model\t%s\t%s\n' "${model_file[$role]}" \
				"$(gib $(( $(stat -c %s "$path") / 1024 )))"
		else
			printf 'model\t%s\tmissing\n' "${model_file[$role]}"
		fi
	done
	printf 'disk\t%s\t%s free\n' "$data_dir" \
		"$(gib "$(free_kib "$data_dir")")"
}

status_bench()
{
	local files=("$state_dir"/bench-*.json)

	if (( ${#files[@]} == 0 )); then
		printf 'bench\tnone\trun ./install.sh bench\n'
		return 0
	fi
	jq -r --arg f "${files[-1]##*/}" '.results[] | ["bench", $f,
		"\(.server) \(.tokens) tok: pp \(.pp) tg \(.tg) tok/s"]
		| @tsv' "${files[-1]}"
}

phase_status()
{
	{
		status_units
		status_health
		status_models
		status_bench
	} | table 8 22
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
