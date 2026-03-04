#!/usr/bin/env bash
# ==============================================================================
# Infinite Orbits MLOps Pipeline Orchestrator
#
# Runs the four pipeline steps in sequence using either local Poetry commands
# or Docker containers. Each step must succeed before the next one begins.
#
# Usage:
#   ./orchestrate.sh --config pipeline_config.yaml [OPTIONS]
#
# Options:
#   --config PATH        Path to the pipeline YAML config (required)
#   --mode local|docker  Execution mode (default: local)
#   --image-prefix STR   Docker image prefix (default: io)
#   --skip-build         Skip Docker image builds (docker mode)
#   --dry-run            Print commands without executing
#   --step STEP          Run only a single step (config_validation, dataset_loading,
#                        model_training, model_registration)
#   --from STEP          Start from a specific step (inclusive)
#   --verbose            Enable debug-level logging
#   -h, --help           Show this help message
# ==============================================================================

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS=(config_validation dataset_loading model_training model_registration)
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_PATH=""
MODE="local"
IMAGE_PREFIX="io"
SKIP_BUILD=false
DRY_RUN=false
SINGLE_STEP=""
FROM_STEP=""
VERBOSE=false

# ---------------------------------------------------------------------------
# Colors / formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${CYAN}  STEP: $*${NC}"; \
              echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════${NC}\n"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_fatal() { log_error "$@"; exit 1; }

run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
        return 0
    fi
    if [[ "$VERBOSE" == true ]]; then
        log_info "exec: $*"
    fi
    eval "$@"
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
    sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)       CONFIG_PATH="$2"; shift 2 ;;
            --mode)         MODE="$2"; shift 2 ;;
            --image-prefix) IMAGE_PREFIX="$2"; shift 2 ;;
            --skip-build)   SKIP_BUILD=true; shift ;;
            --dry-run)      DRY_RUN=true; shift ;;
            --step)         SINGLE_STEP="$2"; shift 2 ;;
            --from)         FROM_STEP="$2"; shift 2 ;;
            --verbose)      VERBOSE=true; shift ;;
            -h|--help)      usage ;;
            *)              log_fatal "Unknown option: $1" ;;
        esac
    done

    # Validate required args
    [[ -z "$CONFIG_PATH" ]] && log_fatal "Missing required --config argument. Use -h for help."
    [[ ! -f "$CONFIG_PATH" ]] && log_fatal "Config file not found: $CONFIG_PATH"
    [[ "$MODE" != "local" && "$MODE" != "docker" ]] && log_fatal "Invalid --mode: $MODE (must be 'local' or 'docker')"

    # Validate --step if provided
    if [[ -n "$SINGLE_STEP" ]]; then
        local valid=false
        for s in "${STEPS[@]}"; do [[ "$s" == "$SINGLE_STEP" ]] && valid=true; done
        [[ "$valid" == false ]] && log_fatal "Invalid --step: $SINGLE_STEP (valid: ${STEPS[*]})"
    fi

    # Validate --from if provided
    if [[ -n "$FROM_STEP" ]]; then
        local valid=false
        for s in "${STEPS[@]}"; do [[ "$s" == "$FROM_STEP" ]] && valid=true; done
        [[ "$valid" == false ]] && log_fatal "Invalid --from: $FROM_STEP (valid: ${STEPS[*]})"
    fi

    # Make config path absolute
    CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
}

# ---------------------------------------------------------------------------
# YAML parser (portable, no external deps)
#
# Reads flat and one-level-nested keys from the pipeline YAML.
# Sets variables like: CFG_pipeline_name, CFG_dataset_source_path, etc.
# ---------------------------------------------------------------------------
parse_yaml() {
    local yaml_file="$1"
    local prefix="CFG"
    local section=""

    # Helper: strip trailing whitespace, then strip surrounding quotes
    _clean_val() {
        local v="$1"
        # Strip trailing whitespace
        v="${v%"${v##*[! ]}"}"
        # Strip surrounding double quotes
        v="${v%\"}" ; v="${v#\"}"
        # Strip surrounding single quotes
        v="${v%\'}" ; v="${v#\'}"
        echo "$v"
    }

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Strip comments and trailing whitespace
        line="${line%%#*}"
        [[ -z "${line// /}" ]] && continue

        # Detect section headers (e.g., "dataset:")
        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*):[[:space:]]*$ ]]; then
            section="${BASH_REMATCH[1]}"
            continue
        fi

        # Detect top-level key: value
        if [[ "$line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*):[[:space:]]+(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local val
            val="$(_clean_val "${BASH_REMATCH[2]}")"
            declare -g "${prefix}_${key}=${val}"
            section=""
            continue
        fi

        # Detect nested key: value (indented under a section)
        if [[ -n "$section" && "$line" =~ ^[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*):[[:space:]]+(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local val
            val="$(_clean_val "${BASH_REMATCH[2]}")"
            declare -g "${prefix}_${section}_${key}=${val}"
            continue
        fi

        # Detect list items under a section (e.g., "  - v1")
        if [[ -n "$section" && "$line" =~ ^[[:space:]]+-[[:space:]]+(.+)$ ]]; then
            local val
            val="$(_clean_val "${BASH_REMATCH[1]}")"
            local list_var="${prefix}_${section}_tags"
            local existing="${!list_var:-}"
            if [[ -z "$existing" ]]; then
                declare -g "${list_var}=${val}"
            else
                declare -g "${list_var}=${existing},${val}"
            fi
            continue
        fi
    done < "$yaml_file"
}

# ---------------------------------------------------------------------------
# Resolve which steps to run
# ---------------------------------------------------------------------------
resolve_steps() {
    local result=()

    if [[ -n "$SINGLE_STEP" ]]; then
        result=("$SINGLE_STEP")
    elif [[ -n "$FROM_STEP" ]]; then
        local found=false
        for s in "${STEPS[@]}"; do
            [[ "$s" == "$FROM_STEP" ]] && found=true
            [[ "$found" == true ]] && result+=("$s")
        done
    else
        result=("${STEPS[@]}")
    fi

    echo "${result[@]}"
}

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Poetry install helper (local mode)
# ---------------------------------------------------------------------------
poetry_install() {
    local step_dir="$1"
    log_info "Installing dependencies for ${step_dir##*/}"
    run_cmd "(cd '${step_dir}' && poetry install --no-interaction --quiet)"
}

docker_build() {
    local step="$1"
    local image="${IMAGE_PREFIX}-${step//_/-}"
    local context="${REPO_ROOT}/${step}"

    log_info "Building image: ${image}:latest"
    run_cmd "docker build -t '${image}:latest' -t '${image}:${TIMESTAMP}' '${context}'"
}

docker_run() {
    local image="$1"; shift
    local args=("$@")

    run_cmd "docker run --rm \
        -v '${CONFIG_PATH}:/data/pipeline_config.yaml:ro' \
        -v '${REPO_ROOT}/artifacts:/artifacts' \
        '${image}:latest' \
        ${args[*]}"
}

# ---------------------------------------------------------------------------
# Helper: append a flag only if the variable is non-empty and not "null"
# ---------------------------------------------------------------------------
append_optional() {
    local flag="$1"
    local value="$2"
    if [[ -n "$value" && "$value" != "null" ]]; then
        echo " ${flag} '${value}'"
    fi
}

# ---------------------------------------------------------------------------
# Helper: convert a boolean value to Typer flag style
# Usage: $(bool_flag --cos-lr "$cos_lr")
# Outputs "--cos-lr" when true, "--no-cos-lr" when false
# ---------------------------------------------------------------------------
bool_flag() {
    local flag="$1"
    local value="$2"
    if [[ "$value" == "true" || "$value" == "True" || "$value" == "1" ]]; then
        echo "$flag"
    else
        echo "--no-${flag#--}"
    fi
}

# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------
run_config_validation() {
    log_step "1/4 — Config Validation"

    # ---- Required fields ----
    local experiment_name="${CFG_experiment_name:?Missing experiment.name in config}"
    local dataset_version="${CFG_dataset_version:?Missing dataset.version in config}"
    local dataset_source="${CFG_dataset_source:?Missing dataset.source in config}"
    local model_variant="${CFG_model_variant:?Missing model.variant in config}"
    local training_epochs="${CFG_training_epochs:?Missing training.epochs in config}"
    local training_batch_size="${CFG_training_batch_size:?Missing training.batch_size in config}"
    local training_image_size="${CFG_training_image_size:?Missing training.image_size in config}"
    local training_learning_rate="${CFG_training_learning_rate:?Missing training.learning_rate in config}"
    local training_optimizer="${CFG_training_optimizer:?Missing training.optimizer in config}"
    local checkpointing_interval_epochs="${CFG_checkpointing_interval_epochs:?Missing checkpointing.interval_epochs in config}"
    local checkpointing_storage_path="${CFG_checkpointing_storage_path:?Missing checkpointing.storage_path in config}"
    local early_stopping_patience="${CFG_early_stopping_patience:?Missing early_stopping.patience in config}"

    # ---- Optional ----
    local experiment_description="${CFG_experiment_description:-}"
    local dataset_path_override="${CFG_dataset_path_override:-}"
    local dataset_sample_size="${CFG_dataset_sample_size:-}"
    local dataset_seed="${CFG_dataset_seed:-42}"
    local model_pretrained_weights="${CFG_model_pretrained_weights:-}"
    local checkpointing_resume_from="${CFG_checkpointing_resume_from:-}"
    local training_freeze="${CFG_training_freeze:-}"

    # ---- Training defaults ----
    local training_cos_lr="${CFG_training_cos_lr:-true}"
    local training_lrf="${CFG_training_lrf:-0.01}"
    local training_momentum="${CFG_training_momentum:-0.937}"
    local training_weight_decay="${CFG_training_weight_decay:-0.0005}"
    local training_warmup_epochs="${CFG_training_warmup_epochs:-3.0}"
    local training_warmup_momentum="${CFG_training_warmup_momentum:-0.8}"
    local training_dropout="${CFG_training_dropout:-0.0}"
    local training_label_smoothing="${CFG_training_label_smoothing:-0.0}"
    local training_nbs="${CFG_training_nbs:-64}"
    local training_amp="${CFG_training_amp:-true}"
    local training_close_mosaic="${CFG_training_close_mosaic:-10}"
    local training_seed="${CFG_training_seed:-0}"
    local training_deterministic="${CFG_training_deterministic:-true}"

    # ---- Pose gains ----
    local training_pose="${CFG_training_pose:-12.0}"
    local training_kobj="${CFG_training_kobj:-2.0}"
    local training_box="${CFG_training_box:-7.5}"
    local training_cls="${CFG_training_cls:-0.5}"
    local training_dfl="${CFG_training_dfl:-1.5}"

    # ---- Augmentation ----
    local aug_hsv_h="${CFG_augmentation_hsv_h:-0.015}"
    local aug_hsv_s="${CFG_augmentation_hsv_s:-0.7}"
    local aug_hsv_v="${CFG_augmentation_hsv_v:-0.4}"
    local aug_degrees="${CFG_augmentation_degrees:-0.0}"
    local aug_translate="${CFG_augmentation_translate:-0.1}"
    local aug_scale="${CFG_augmentation_scale:-0.5}"
    local aug_shear="${CFG_augmentation_shear:-0.0}"
    local aug_perspective="${CFG_augmentation_perspective:-0.0}"
    local aug_flipud="${CFG_augmentation_flipud:-0.0}"
    local aug_fliplr="${CFG_augmentation_fliplr:-0.0}"
    local aug_mosaic="${CFG_augmentation_mosaic:-1.0}"
    local aug_mixup="${CFG_augmentation_mixup:-0.0}"
    local aug_copy_paste="${CFG_augmentation_copy_paste:-0.0}"
    local aug_erasing="${CFG_augmentation_erasing:-0.4}"
    local aug_bgr="${CFG_augmentation_bgr:-0.0}"

    # ---- Output path ----
    local output_path
    if [[ "$MODE" == "docker" ]]; then
        output_path="/artifacts/validated_config.json"
    else
        output_path="${REPO_ROOT}/artifacts/validated_config.json"
    fi

    # ---- Build args array (SAFE) ----
    local args=(
        --experiment-name "$experiment_name"
        --dataset-version "$dataset_version"
        --dataset-source "$dataset_source"
        --dataset-seed "$dataset_seed"
        --model-variant "$model_variant"
        --training-epochs "$training_epochs"
        --training-batch-size "$training_batch_size"
        --training-image-size "$training_image_size"
        --training-learning-rate "$training_learning_rate"
        --training-optimizer "$training_optimizer"
        --training-lrf "$training_lrf"
        --training-momentum "$training_momentum"
        --training-warmup-epochs "$training_warmup_epochs"
        --training-warmup-momentum "$training_warmup_momentum"
        --training-weight-decay "$training_weight_decay"
        --training-dropout "$training_dropout"
        --training-label-smoothing "$training_label_smoothing"
        --training-nbs "$training_nbs"
        --training-close-mosaic "$training_close_mosaic"
        --training-seed "$training_seed"
        --training-pose "$training_pose"
        --training-kobj "$training_kobj"
        --training-box "$training_box"
        --training-cls "$training_cls"
        --training-dfl "$training_dfl"
        --checkpointing-interval-epochs "$checkpointing_interval_epochs"
        --checkpointing-storage-path "$checkpointing_storage_path"
        --early-stopping-patience "$early_stopping_patience"
        --aug-hsv-h "$aug_hsv_h"
        --aug-hsv-s "$aug_hsv_s"
        --aug-hsv-v "$aug_hsv_v"
        --aug-degrees "$aug_degrees"
        --aug-translate "$aug_translate"
        --aug-scale "$aug_scale"
        --aug-shear "$aug_shear"
        --aug-perspective "$aug_perspective"
        --aug-flipud "$aug_flipud"
        --aug-fliplr "$aug_fliplr"
        --aug-mosaic "$aug_mosaic"
        --aug-mixup "$aug_mixup"
        --aug-copy-paste "$aug_copy_paste"
        --aug-erasing "$aug_erasing"
        --aug-bgr "$aug_bgr"
        --output-path "$output_path"
    )

    # ---- Boolean flags (Typer uses --flag / --no-flag, not --flag true) ----
    [[ "$training_cos_lr" == "true" ]] && args+=(--training-cos-lr) || args+=(--no-training-cos-lr)
    [[ "$training_amp" == "true" ]] && args+=(--training-amp) || args+=(--no-training-amp)
    [[ "$training_deterministic" == "true" ]] && args+=(--training-deterministic) || args+=(--no-training-deterministic)

    # ---- Append optional safely ----
    [[ -n "$experiment_description" && "$experiment_description" != "null" ]] && args+=(--experiment-description "$experiment_description")
    [[ -n "$dataset_path_override" && "$dataset_path_override" != "null" ]] && args+=(--dataset-path-override "$dataset_path_override")
    [[ -n "$dataset_sample_size" && "$dataset_sample_size" != "null" ]] && args+=(--dataset-sample-size "$dataset_sample_size")
    [[ -n "$model_pretrained_weights" && "$model_pretrained_weights" != "null" ]] && args+=(--model-pretrained-weights "$model_pretrained_weights")
    [[ -n "$checkpointing_resume_from" && "$checkpointing_resume_from" != "null" ]] && args+=(--checkpointing-resume-from "$checkpointing_resume_from")
    [[ -n "$training_freeze" && "$training_freeze" != "null" ]] && args+=(--training-freeze "$training_freeze")

    # ---- Execute ----
    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build config_validation
        docker_run "${IMAGE_PREFIX}-config-validation" "${args[@]}"
    else
        poetry_install "${REPO_ROOT}/config_validation"
        (cd "${REPO_ROOT}/config_validation" && poetry run config-validation "${args[@]}")
    fi

    log_ok "Config validation passed"
}

run_dataset_loading() {
    log_step "2/4 — Dataset Loading"

    local version="${CFG_dataset_version:?Missing dataset.version in config}"
    local source="${CFG_dataset_source:?Missing dataset.source in config}"
    local seed="${CFG_dataset_seed:-42}"
    local output_dir="${REPO_ROOT}/artifacts/dataset"

    # Optional fields
    local path_override="${CFG_dataset_path_override:-}"
    local sample_size="${CFG_dataset_sample_size:-}"

    local optional_flags=""
    optional_flags+="$(append_optional --path-override "$path_override")"
    [[ -n "$sample_size" && "$sample_size" != "null" ]] && \
        optional_flags+=" --sample-size ${sample_size}"

    # S3 streaming mode selection:
    #   "manifest_only" — only data.yaml + S3 key listing (labels streamed too)
    #   "labels_only"   — download labels + data.yaml (images streamed)
    #   default for s3  — labels_only (backward compatible)
    local streaming_mode="${CFG_dataset_streaming_mode:-}"
    if [[ "$streaming_mode" == "manifest_only" ]]; then
        optional_flags+=" --manifest-only"
    elif [[ "$streaming_mode" == "labels_only" ]] || [[ "$source" == "s3" && -z "$streaming_mode" ]]; then
        optional_flags+=" --labels-only"
    fi

    mkdir -p "$output_dir"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build dataset_loading
        docker_run "${IMAGE_PREFIX}-dataset-loading" \
            "--version '${version}'" \
            "--source '${source}'" \
            "--output-dir /artifacts/dataset" \
            "--seed ${seed}" \
            "${optional_flags}"
    else
        poetry_install "${REPO_ROOT}/dataset_loading"
        run_cmd "cd '${REPO_ROOT}/dataset_loading' && \
            poetry run dataset-loading \
                --version '${version}' \
                --source '${source}' \
                --output-dir '${output_dir}' \
                --seed ${seed} \
                ${optional_flags}"
    fi

    log_ok "Dataset loading complete -> ${output_dir}"
}

run_model_training() {
    log_step "3/4 — Model Training"

    # ---- Identity ----
    local experiment_name="${CFG_experiment_name:?Missing experiment.name in config}"
    local model_variant="${CFG_model_variant:?Missing model.variant in config}"
    local dataset_dir="${REPO_ROOT}/artifacts/dataset"
    local output_dir="${REPO_ROOT}/artifacts/training"

    # ---- Weight init / resume ----
    local pretrained_weights="${CFG_model_pretrained_weights:-}"
    local resume_from="${CFG_checkpointing_resume_from:-}"

    # ---- Core schedule ----
    local epochs="${CFG_training_epochs:-100}"
    local batch_size="${CFG_training_batch_size:-16}"
    local image_size="${CFG_training_image_size:-640}"

    # ---- Learning rate ----
    local learning_rate="${CFG_training_learning_rate:-0.01}"
    local cos_lr="${CFG_training_cos_lr:-true}"
    local lrf="${CFG_training_lrf:-0.01}"

    # ---- Optimizer ----
    local optimizer="${CFG_training_optimizer:-SGD}"
    local momentum="${CFG_training_momentum:-0.937}"
    local weight_decay="${CFG_training_weight_decay:-0.0005}"

    # ---- Warmup ----
    local warmup_epochs="${CFG_training_warmup_epochs:-3.0}"
    local warmup_momentum="${CFG_training_warmup_momentum:-0.8}"

    # ---- Regularization ----
    local dropout="${CFG_training_dropout:-0.0}"
    local label_smoothing="${CFG_training_label_smoothing:-0.0}"

    # ---- Training efficiency ----
    local nbs="${CFG_training_nbs:-64}"
    local freeze="${CFG_training_freeze:-}"
    local amp="${CFG_training_amp:-true}"
    local close_mosaic="${CFG_training_close_mosaic:-10}"
    local seed="${CFG_training_seed:-0}"
    local deterministic="${CFG_training_deterministic:-true}"

    # ---- Pose loss gains ----
    local pose_gain="${CFG_training_pose:-12.0}"
    local kobj="${CFG_training_kobj:-2.0}"
    local box="${CFG_training_box:-7.5}"
    local cls_gain="${CFG_training_cls:-0.5}"
    local dfl="${CFG_training_dfl:-1.5}"

    # ---- Early stopping ----
    local patience="${CFG_early_stopping_patience:-50}"

    # ---- Checkpointing ----
    # Parse s3://bucket/prefix from checkpointing.storage_path
    local storage_path="${CFG_checkpointing_storage_path:-s3://temp-mlops/checkpoints}"
    local s3_path="${storage_path#s3://}"
    local checkpoint_bucket="${s3_path%%/*}"
    local checkpoint_prefix="${s3_path#*/}"
    local checkpoint_interval="${CFG_checkpointing_interval_epochs:-10}"

    # ---- Augmentation ----
    local hsv_h="${CFG_augmentation_hsv_h:-0.015}"
    local hsv_s="${CFG_augmentation_hsv_s:-0.7}"
    local hsv_v="${CFG_augmentation_hsv_v:-0.4}"
    local degrees="${CFG_augmentation_degrees:-0.0}"
    local translate="${CFG_augmentation_translate:-0.1}"
    local scale="${CFG_augmentation_scale:-0.5}"
    local shear="${CFG_augmentation_shear:-0.0}"
    local perspective="${CFG_augmentation_perspective:-0.0}"
    local flipud="${CFG_augmentation_flipud:-0.0}"
    local fliplr="${CFG_augmentation_fliplr:-0.0}"
    local mosaic="${CFG_augmentation_mosaic:-1.0}"
    local mixup="${CFG_augmentation_mixup:-0.0}"
    local copy_paste="${CFG_augmentation_copy_paste:-0.0}"
    local erasing="${CFG_augmentation_erasing:-0.4}"
    local bgr="${CFG_augmentation_bgr:-0.0}"

    # ---- Dataset source (s3 streaming) ----
    local dataset_source="${CFG_dataset_source:-local}"

    # ---- Optional flags ----
    local optional_flags=""
    optional_flags+="$(append_optional --pretrained-weights "$pretrained_weights")"
    optional_flags+="$(append_optional --resume-from "$resume_from")"
    [[ -n "$freeze" && "$freeze" != "null" ]] && \
        optional_flags+=" --freeze ${freeze}"

    # S3 streaming: pass bucket/prefix so training streams images directly
    if [[ "$dataset_source" == "s3" ]]; then
        local dataset_path_override="${CFG_dataset_path_override:-}"
        local s3_bucket s3_prefix
        if [[ -n "$dataset_path_override" && "$dataset_path_override" != "null" ]]; then
            # Parse s3://bucket/prefix from the override
            local s3_path="${dataset_path_override#s3://}"
            s3_bucket="${s3_path%%/*}"
            s3_prefix="${s3_path#*/}"
        else
            s3_bucket="temp-mlops"
            s3_prefix="datasets/speedplus_yolo/${CFG_dataset_version:-v1}/"
        fi
        optional_flags+=" --source s3"
        optional_flags+=" --s3-bucket '${s3_bucket}'"
        optional_flags+=" --s3-prefix '${s3_prefix}'"
    fi

    mkdir -p "$output_dir"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build model_training
        docker_run "${IMAGE_PREFIX}-model-training" \
            "--model-variant '${model_variant}'" \
            "--experiment-name '${experiment_name}'" \
            "--dataset-dir /artifacts/dataset" \
            "--output-dir /artifacts/training" \
            "--epochs ${epochs}" \
            "--batch-size ${batch_size}" \
            "--image-size ${image_size}" \
            "--learning-rate ${learning_rate}" \
            "$(bool_flag --cos-lr "$cos_lr")" \
            "--lrf ${lrf}" \
            "--optimizer '${optimizer}'" \
            "--momentum ${momentum}" \
            "--weight-decay ${weight_decay}" \
            "--warmup-epochs ${warmup_epochs}" \
            "--warmup-momentum ${warmup_momentum}" \
            "--dropout ${dropout}" \
            "--label-smoothing ${label_smoothing}" \
            "--nbs ${nbs}" \
            "$(bool_flag --amp "$amp")" \
            "--close-mosaic ${close_mosaic}" \
            "--seed ${seed}" \
            "$(bool_flag --deterministic "$deterministic")" \
            "--pose ${pose_gain}" \
            "--kobj ${kobj}" \
            "--box ${box}" \
            "--cls ${cls_gain}" \
            "--dfl ${dfl}" \
            "--patience ${patience}" \
            "--checkpoint-interval ${checkpoint_interval}" \
            "--checkpoint-bucket '${checkpoint_bucket}'" \
            "--checkpoint-prefix '${checkpoint_prefix}'" \
            "--hsv-h ${hsv_h}" \
            "--hsv-s ${hsv_s}" \
            "--hsv-v ${hsv_v}" \
            "--degrees ${degrees}" \
            "--translate ${translate}" \
            "--scale ${scale}" \
            "--shear ${shear}" \
            "--perspective ${perspective}" \
            "--flipud ${flipud}" \
            "--fliplr ${fliplr}" \
            "--mosaic ${mosaic}" \
            "--mixup ${mixup}" \
            "--copy-paste ${copy_paste}" \
            "--erasing ${erasing}" \
            "--bgr ${bgr}" \
            "${optional_flags}"
    else
        poetry_install "${REPO_ROOT}/model_training"
        run_cmd "cd '${REPO_ROOT}/model_training' && \
            poetry run model-training \
                --model-variant '${model_variant}' \
                --experiment-name '${experiment_name}' \
                --dataset-dir '${dataset_dir}' \
                --output-dir '${output_dir}' \
                --epochs ${epochs} \
                --batch-size ${batch_size} \
                --image-size ${image_size} \
                --learning-rate ${learning_rate} \
                $(bool_flag --cos-lr "$cos_lr") \
                --lrf ${lrf} \
                --optimizer '${optimizer}' \
                --momentum ${momentum} \
                --weight-decay ${weight_decay} \
                --warmup-epochs ${warmup_epochs} \
                --warmup-momentum ${warmup_momentum} \
                --dropout ${dropout} \
                --label-smoothing ${label_smoothing} \
                --nbs ${nbs} \
                $(bool_flag --amp "$amp") \
                --close-mosaic ${close_mosaic} \
                --seed ${seed} \
                $(bool_flag --deterministic "$deterministic") \
                --pose ${pose_gain} \
                --kobj ${kobj} \
                --box ${box} \
                --cls ${cls_gain} \
                --dfl ${dfl} \
                --patience ${patience} \
                --checkpoint-interval ${checkpoint_interval} \
                --checkpoint-bucket '${checkpoint_bucket}' \
                --checkpoint-prefix '${checkpoint_prefix}' \
                --hsv-h ${hsv_h} \
                --hsv-s ${hsv_s} \
                --hsv-v ${hsv_v} \
                --degrees ${degrees} \
                --translate ${translate} \
                --scale ${scale} \
                --shear ${shear} \
                --perspective ${perspective} \
                --flipud ${flipud} \
                --fliplr ${fliplr} \
                --mosaic ${mosaic} \
                --mixup ${mixup} \
                --copy-paste ${copy_paste} \
                --erasing ${erasing} \
                --bgr ${bgr} \
                ${optional_flags}"
    fi

    log_ok "Model training complete -> ${output_dir}"
}

run_model_registration() {
    log_step "4/4 — Model Registration"

    # ---- Read training result artifact ----
    local result_json
    if [[ "$MODE" == "docker" ]]; then
        result_json="/artifacts/training/training_result.json"
    else
        result_json="${REPO_ROOT}/artifacts/training/training_result.json"
    fi

    local mlflow_run_id best_checkpoint_s3 final_map50 model_variant_result

    if [[ "$DRY_RUN" == true ]] && [[ ! -f "$result_json" ]]; then
        log_warn "No training_result.json found (dry-run) — using placeholders"
        mlflow_run_id="dry-run-placeholder"
        best_checkpoint_s3="s3://temp-mlops/checkpoints/placeholder/best.pt"
        final_map50="0.0"
        model_variant_result="${CFG_model_variant:-yolov8n-pose.pt}"
    else
        [[ ! -f "$result_json" ]] && log_fatal "No training_result.json at ${result_json}. Did model_training run?"
        mlflow_run_id="$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['mlflow_run_id'])")"
        best_checkpoint_s3="$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['best_checkpoint_s3'])")"
        final_map50="$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['final_map50'])")"
        model_variant_result="$(python3 -c "import json; d=json.load(open('${result_json}')); print(d['model_variant'])")"
    fi

    log_info "MLflow run ID:   ${mlflow_run_id}"
    log_info "Best checkpoint: ${best_checkpoint_s3}"

    # ---- Optional fields ----
    local registered_model_name="${CFG_registration_registered_model_name:-}"
    local promote_to="${CFG_registration_promote_to:-}"
    local dataset_version="${CFG_dataset_version:-}"
    local dataset_sample_size="${CFG_dataset_sample_size:-}"
    local git_commit
    git_commit="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"

    local extra_flags=""
    extra_flags+="$(append_optional --registered-model-name "$registered_model_name")"
    extra_flags+="$(append_optional --promote-to "$promote_to")"
    extra_flags+="$(append_optional --dataset-version "$dataset_version")"
    [[ -n "$dataset_sample_size" && "$dataset_sample_size" != "null" ]] && \
        extra_flags+=" --dataset-sample-size ${dataset_sample_size}"
    extra_flags+="$(append_optional --git-commit "$git_commit")"
    extra_flags+="$(append_optional --model-variant "$model_variant_result")"
    [[ -n "$final_map50" && "$final_map50" != "0.0" ]] && \
        extra_flags+=" --best-map50 ${final_map50}"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build model_registration
        docker_run "${IMAGE_PREFIX}-model-registration" \
            "--mlflow-run-id '${mlflow_run_id}'" \
            "--best-checkpoint-path '${best_checkpoint_s3}'" \
            "${extra_flags}"
    else
        poetry_install "${REPO_ROOT}/model_registration"
        run_cmd "cd '${REPO_ROOT}/model_registration' && \
            poetry run model-registration \
                --mlflow-run-id '${mlflow_run_id}' \
                --best-checkpoint-path '${best_checkpoint_s3}' \
                ${extra_flags}"
    fi

    log_ok "Model registration complete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"

    echo -e "\n${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║        Infinite Orbits MLOps Pipeline Orchestrator          ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}\n"

    log_info "Config:    ${CONFIG_PATH}"
    log_info "Mode:      ${MODE}"
    log_info "Timestamp: ${TIMESTAMP}"
    [[ "$DRY_RUN" == true ]] && log_warn "DRY-RUN mode enabled — no commands will be executed"
    echo ""

    # Parse the pipeline YAML into shell variables
    parse_yaml "$CONFIG_PATH"

    log_info "Experiment: ${CFG_experiment_name:-unknown}"

    # Create shared artifacts directory
    mkdir -p "${REPO_ROOT}/artifacts"

    # Resolve and run steps
    local steps_to_run
    read -ra steps_to_run <<< "$(resolve_steps)"

    log_info "Steps:     ${steps_to_run[*]}"
    echo ""

    local start_time=$SECONDS
    local failed=false

    for step in "${steps_to_run[@]}"; do
        local step_start=$SECONDS

        case "$step" in
            config_validation)  run_config_validation ;;
            dataset_loading)    run_dataset_loading ;;
            model_training)     run_model_training ;;
            model_registration) run_model_registration ;;
            *)                  log_fatal "Unknown step: $step" ;;
        esac

        local step_elapsed=$(( SECONDS - step_start ))
        log_info "Step duration: ${step_elapsed}s"
    done

    local total_elapsed=$(( SECONDS - start_time ))

    echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║           Pipeline completed successfully                   ║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    log_info "Total duration: ${total_elapsed}s"
    log_info "Artifacts:      ${REPO_ROOT}/artifacts/"
}

main "$@"
