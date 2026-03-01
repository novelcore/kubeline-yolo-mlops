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
            local val="${BASH_REMATCH[2]}"
            val="${val%\"}" ; val="${val#\"}"  # strip quotes
            val="${val%\'}" ; val="${val#\'}"
            declare -g "${prefix}_${key}=${val}"
            section=""
            continue
        fi

        # Detect nested key: value (indented under a section)
        if [[ -n "$section" && "$line" =~ ^[[:space:]]+([a-zA-Z_][a-zA-Z0-9_]*):[[:space:]]+(.+)$ ]]; then
            local key="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"
            val="${val%\"}" ; val="${val#\"}"
            val="${val%\'}" ; val="${val#\'}"
            declare -g "${prefix}_${section}_${key}=${val}"
            continue
        fi

        # Detect list items under a section (e.g., "  - v1")
        if [[ -n "$section" && "$line" =~ ^[[:space:]]+-[[:space:]]+(.+)$ ]]; then
            local val="${BASH_REMATCH[1]}"
            val="${val%\"}" ; val="${val#\"}"
            val="${val%\'}" ; val="${val#\'}"
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
        run ${args[*]}"
}

# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------
run_config_validation() {
    log_step "1/4 — Config Validation"

    # Required fields
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

    # Optional fields
    local experiment_description="${CFG_experiment_description:-}"
    local dataset_path_override="${CFG_dataset_path_override:-}"
    local dataset_sample_size="${CFG_dataset_sample_size:-}"
    local dataset_seed="${CFG_dataset_seed:-42}"
    local model_pretrained_weights="${CFG_model_pretrained_weights:-}"
    local training_warmup_epochs="${CFG_training_warmup_epochs:-3.0}"
    local training_warmup_momentum="${CFG_training_warmup_momentum:-0.8}"
    local training_weight_decay="${CFG_training_weight_decay:-0.0005}"
    local checkpointing_resume_from="${CFG_checkpointing_resume_from:-}"

    # Build optional flags
    local optional_flags=""
    [[ -n "$experiment_description" ]]    && optional_flags+=" --experiment-description '${experiment_description}'"
    [[ -n "$dataset_path_override" ]]     && optional_flags+=" --dataset-path-override '${dataset_path_override}'"
    [[ -n "$dataset_sample_size" ]]       && optional_flags+=" --dataset-sample-size ${dataset_sample_size}"
    [[ -n "$model_pretrained_weights" ]]  && optional_flags+=" --model-pretrained-weights '${model_pretrained_weights}'"
    [[ -n "$checkpointing_resume_from" ]] && optional_flags+=" --checkpointing-resume-from '${checkpointing_resume_from}'"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build config_validation
        docker_run "${IMAGE_PREFIX}-config-validation" \
            "--experiment-name '${experiment_name}'" \
            "--dataset-version '${dataset_version}'" \
            "--dataset-source '${dataset_source}'" \
            "--dataset-seed ${dataset_seed}" \
            "--model-variant '${model_variant}'" \
            "--training-epochs ${training_epochs}" \
            "--training-batch-size ${training_batch_size}" \
            "--training-image-size ${training_image_size}" \
            "--training-learning-rate ${training_learning_rate}" \
            "--training-optimizer '${training_optimizer}'" \
            "--training-warmup-epochs ${training_warmup_epochs}" \
            "--training-warmup-momentum ${training_warmup_momentum}" \
            "--training-weight-decay ${training_weight_decay}" \
            "--checkpointing-interval-epochs ${checkpointing_interval_epochs}" \
            "--checkpointing-storage-path '${checkpointing_storage_path}'" \
            "--early-stopping-patience ${early_stopping_patience}" \
            "--output-path /artifacts/validated_config.json" \
            "${optional_flags}"
    else
        run_cmd "cd '${REPO_ROOT}/config_validation' && \
            poetry run config-validation run \
                --experiment-name '${experiment_name}' \
                --dataset-version '${dataset_version}' \
                --dataset-source '${dataset_source}' \
                --dataset-seed ${dataset_seed} \
                --model-variant '${model_variant}' \
                --training-epochs ${training_epochs} \
                --training-batch-size ${training_batch_size} \
                --training-image-size ${training_image_size} \
                --training-learning-rate ${training_learning_rate} \
                --training-optimizer '${training_optimizer}' \
                --training-warmup-epochs ${training_warmup_epochs} \
                --training-warmup-momentum ${training_warmup_momentum} \
                --training-weight-decay ${training_weight_decay} \
                --checkpointing-interval-epochs ${checkpointing_interval_epochs} \
                --checkpointing-storage-path '${checkpointing_storage_path}' \
                --early-stopping-patience ${early_stopping_patience} \
                --output-path '${REPO_ROOT}/.tmp/validated_config.json' \
                ${optional_flags}"
    fi

    log_ok "Config validation passed"
}

run_dataset_loading() {
    log_step "2/4 — Dataset Loading"

    local source_path="${CFG_dataset_source_path:?Missing dataset.source_path in config}"
    local format="${CFG_dataset_format:?Missing dataset.format in config}"
    local train_split="${CFG_dataset_train_split:-0.8}"
    local val_split="${CFG_dataset_val_split:-0.1}"
    local test_split="${CFG_dataset_test_split:-0.1}"
    local output_dir="${REPO_ROOT}/artifacts/dataset"

    mkdir -p "$output_dir"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build dataset_loading
        docker_run "${IMAGE_PREFIX}-dataset-loading" \
            "--source-path '${source_path}'" \
            "--output-dir /artifacts/dataset" \
            "--format '${format}'" \
            "--train-split ${train_split}" \
            "--val-split ${val_split}" \
            "--test-split ${test_split}"
    else
        run_cmd "cd '${REPO_ROOT}/dataset_loading' && \
            poetry run dataset-loading run \
                --source-path '${source_path}' \
                --output-dir '${output_dir}' \
                --format '${format}' \
                --train-split ${train_split} \
                --val-split ${val_split} \
                --test-split ${test_split}"
    fi

    log_ok "Dataset loading complete -> ${output_dir}"
}

run_model_training() {
    log_step "3/4 — Model Training"

    local model_name="${CFG_training_model_name:?Missing training.model_name in config}"
    local epochs="${CFG_training_epochs:-10}"
    local batch_size="${CFG_training_batch_size:-32}"
    local learning_rate="${CFG_training_learning_rate:-0.0001}"
    local optimizer="${CFG_training_optimizer:-adamw}"
    local dataset_dir="${REPO_ROOT}/artifacts/dataset"
    local cfg_output="${CFG_training_output_dir:-}"
    local output_dir

    # Resolve output_dir: if the config path starts with /artifacts (Docker convention),
    # map it under REPO_ROOT for local runs; otherwise use as-is or fall back to default.
    if [[ "$MODE" == "local" && "$cfg_output" == /artifacts* ]]; then
        output_dir="${REPO_ROOT}${cfg_output}"
    elif [[ -n "$cfg_output" ]]; then
        output_dir="$cfg_output"
    else
        output_dir="${REPO_ROOT}/artifacts/checkpoints"
    fi

    mkdir -p "$output_dir"

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build model_training
        docker_run "${IMAGE_PREFIX}-model-training" \
            "--model-name '${model_name}'" \
            "--dataset-dir /artifacts/dataset" \
            "--output-dir /artifacts/checkpoints" \
            "--epochs ${epochs}" \
            "--batch-size ${batch_size}" \
            "--learning-rate ${learning_rate}" \
            "--optimizer '${optimizer}'"
    else
        run_cmd "cd '${REPO_ROOT}/model_training' && \
            poetry run model-training run \
                --model-name '${model_name}' \
                --dataset-dir '${dataset_dir}' \
                --output-dir '${output_dir}' \
                --epochs ${epochs} \
                --batch-size ${batch_size} \
                --learning-rate ${learning_rate} \
                --optimizer '${optimizer}'"
    fi

    log_ok "Model training complete -> ${output_dir}"
}

run_model_registration() {
    log_step "4/4 — Model Registration"

    local model_name="${CFG_registration_model_name:?Missing registration.model_name in config}"
    local registry_url="${CFG_registration_registry_url:?Missing registration.registry_url in config}"
    local promote_to="${CFG_registration_promote_to:-}"
    local tags="${CFG_registration_tags:-}"

    # Find the latest checkpoint
    local cfg_ckpt_dir="${CFG_training_output_dir:-}"
    local checkpoint_dir

    if [[ "$MODE" == "local" && "$cfg_ckpt_dir" == /artifacts* ]]; then
        checkpoint_dir="${REPO_ROOT}${cfg_ckpt_dir}"
    elif [[ -n "$cfg_ckpt_dir" ]]; then
        checkpoint_dir="$cfg_ckpt_dir"
    else
        checkpoint_dir="${REPO_ROOT}/artifacts/checkpoints"
    fi
    local checkpoint_path=""

    if [[ -d "$checkpoint_dir" ]]; then
        checkpoint_path="$(find "${checkpoint_dir}" -maxdepth 1 -name '*.pt' -type f 2>/dev/null | sort | tail -1)"
    fi

    if [[ -z "$checkpoint_path" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            checkpoint_path="${checkpoint_dir}/model_latest.pt"
            log_warn "No checkpoint found (dry-run) — using placeholder: ${checkpoint_path}"
        else
            log_fatal "No checkpoint found in ${checkpoint_dir}. Did model_training run?"
        fi
    fi

    log_info "Using checkpoint: ${checkpoint_path}"

    # Build optional flags
    local extra_flags=""
    [[ -n "$promote_to" ]] && extra_flags+=" --promote-to '${promote_to}'"

    if [[ -n "$tags" ]]; then
        IFS=',' read -ra tag_arr <<< "$tags"
        for t in "${tag_arr[@]}"; do
            extra_flags+=" --tags '${t}'"
        done
    fi

    if [[ "$MODE" == "docker" ]]; then
        [[ "$SKIP_BUILD" == false ]] && docker_build model_registration
        docker_run "${IMAGE_PREFIX}-model-registration" \
            "--model-name '${model_name}'" \
            "--checkpoint-path /artifacts/checkpoints/$(basename "${checkpoint_path}")" \
            "--registry-url '${registry_url}'" \
            "${extra_flags}"
    else
        run_cmd "cd '${REPO_ROOT}/model_registration' && \
            poetry run model-registration run \
                --model-name '${model_name}' \
                --checkpoint-path '${checkpoint_path}' \
                --registry-url '${registry_url}' \
                ${extra_flags}"
    fi

    log_ok "Model registered: ${model_name} -> ${registry_url}"
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
