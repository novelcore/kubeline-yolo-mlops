#!/usr/bin/env bash
# orchestrate.sh — local pipeline orchestrator for the YOLO MLOps pipeline.
#
# Runs the four pipeline steps in order, propagating artefact values
# (config_hash, lakefs_commit, dataset_hash) across step boundaries.
#
# Usage:
#   ./orchestrate.sh --config pipeline_config.yaml [options]
#
# Options:
#   --config <path>   Pipeline config YAML (default: pipeline_config.yaml)
#   --mode local      Execution mode — only 'local' is supported right now
#   --step <name>     Run only this step and exit
#   --from <name>     Skip all steps before <name>; requires prior artefacts
#   --dry-run         Print commands but do not execute them
#
# Step names: config-validation | dataset-loading | model-training | model-registration

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG_PATH="pipeline_config.yaml"
MODE="local"
ONLY_STEP=""
FROM_STEP=""
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse CLI args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)   CONFIG_PATH="$2"; shift 2 ;;
        --mode)     MODE="$2";        shift 2 ;;
        --step)     ONLY_STEP="$2";   shift 2 ;;
        --from)     FROM_STEP="$2";   shift 2 ;;
        --dry-run)  DRY_RUN=true;     shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$MODE" != "local" ]]; then
    echo "Only --mode local is supported. In-cluster runs use Argo Workflows." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$REPO_ROOT/artifacts}"

# ---------------------------------------------------------------------------
# Parse pipeline config YAML into shell variables (all in one Python call)
# ---------------------------------------------------------------------------
eval "$(python3 - "$CONFIG_PATH" <<'PYEOF'
import sys, yaml, json

path = sys.argv[1]
with open(path) as f:
    c = yaml.safe_load(f)

def sh(v):
    """Safely quote a value for shell assignment."""
    if v is None:
        return ''
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, list):
        return ','.join(str(i) for i in v)
    return str(v)

exp  = c.get('experiment', {})
ds   = c.get('dataset', {})
mdl  = c.get('model', {})
tr   = c.get('training', {})
ck   = c.get('checkpointing', {})
es   = c.get('early_stopping', {})
aug  = c.get('augmentation', {})
xp   = c.get('export', {})
reg  = c.get('registration', {})

# Parse checkpoint storage_path into bucket + prefix
storage_path = sh(ck.get('storage_path', 's3://temp-mlops/checkpoints'))
ck_bucket = ''
ck_prefix = ''
if storage_path.startswith('s3://'):
    parts = storage_path[5:].split('/', 1)
    ck_bucket = parts[0]
    ck_prefix = parts[1] if len(parts) > 1 else ''

fields = {
    # experiment
    'CFG_EXPERIMENT_NAME':        sh(exp.get('name')),
    'CFG_EXPERIMENT_DESCRIPTION': sh(exp.get('description')),
    # dataset
    'CFG_DATASET_VERSION':        sh(ds.get('version')),
    'CFG_DATASET_SOURCE':         sh(ds.get('source')),
    'CFG_DATASET_LAKEFS_REPO':    sh(ds.get('lakefs_repo')),
    'CFG_DATASET_LAKEFS_BRANCH':  sh(ds.get('lakefs_branch')),
    'CFG_DATASET_PATH_OVERRIDE':  sh(ds.get('path_override')),
    'CFG_DATASET_SAMPLE_SIZE':    sh(ds.get('sample_size')),
    'CFG_DATASET_SEED':           sh(ds.get('seed', 42)),
    'CFG_DATASET_LABELS_ONLY':    sh(ds.get('labels_only', False)),
    'CFG_DATASET_MANIFEST_ONLY':  sh(ds.get('manifest_only', False)),
    # model
    'CFG_MODEL_VARIANT':          sh(mdl.get('variant')),
    'CFG_MODEL_PRETRAINED_WEIGHTS': sh(mdl.get('pretrained_weights')),
    # training
    'CFG_TRAINING_EPOCHS':        sh(tr.get('epochs')),
    'CFG_TRAINING_BATCH_SIZE':    sh(tr.get('batch_size')),
    'CFG_TRAINING_IMAGE_SIZE':    sh(tr.get('image_size')),
    'CFG_TRAINING_LR':            sh(tr.get('learning_rate')),
    'CFG_TRAINING_COS_LR':        sh(tr.get('cos_lr', True)),
    'CFG_TRAINING_LRF':           sh(tr.get('lrf', 0.01)),
    'CFG_TRAINING_OPTIMIZER':     sh(tr.get('optimizer')),
    'CFG_TRAINING_MOMENTUM':      sh(tr.get('momentum', 0.937)),
    'CFG_TRAINING_WEIGHT_DECAY':  sh(tr.get('weight_decay', 0.0005)),
    'CFG_TRAINING_WARMUP_EPOCHS': sh(tr.get('warmup_epochs', 3.0)),
    'CFG_TRAINING_WARMUP_MOMENTUM': sh(tr.get('warmup_momentum', 0.8)),
    'CFG_TRAINING_DROPOUT':       sh(tr.get('dropout', 0.0)),
    'CFG_TRAINING_LABEL_SMOOTHING': sh(tr.get('label_smoothing', 0.0)),
    'CFG_TRAINING_NBS':           sh(tr.get('nbs', 64)),
    'CFG_TRAINING_FREEZE':        sh(tr.get('freeze')),
    'CFG_TRAINING_AMP':           sh(tr.get('amp', True)),
    'CFG_TRAINING_CLOSE_MOSAIC':  sh(tr.get('close_mosaic', 10)),
    'CFG_TRAINING_SEED':          sh(tr.get('seed', 0)),
    'CFG_TRAINING_DETERMINISTIC': sh(tr.get('deterministic', True)),
    'CFG_TRAINING_POSE':          sh(tr.get('pose', 12.0)),
    'CFG_TRAINING_KOBJ':          sh(tr.get('kobj', 2.0)),
    'CFG_TRAINING_BOX':           sh(tr.get('box', 7.5)),
    'CFG_TRAINING_CLS':           sh(tr.get('cls', 0.5)),
    'CFG_TRAINING_DFL':           sh(tr.get('dfl', 1.5)),
    # checkpointing
    'CFG_CK_INTERVAL':            sh(ck.get('interval_epochs')),
    'CFG_CK_STORAGE_PATH':        storage_path,
    'CFG_CK_BUCKET':              ck_bucket,
    'CFG_CK_PREFIX':              ck_prefix,
    'CFG_CK_RESUME_FROM':         sh(ck.get('resume_from')),
    # early stopping
    'CFG_ES_PATIENCE':            sh(es.get('patience')),
    # augmentation
    'CFG_AUG_HSV_H':    sh(aug.get('hsv_h', 0.015)),
    'CFG_AUG_HSV_S':    sh(aug.get('hsv_s', 0.7)),
    'CFG_AUG_HSV_V':    sh(aug.get('hsv_v', 0.4)),
    'CFG_AUG_DEGREES':  sh(aug.get('degrees', 0.0)),
    'CFG_AUG_TRANSLATE':sh(aug.get('translate', 0.1)),
    'CFG_AUG_SCALE':    sh(aug.get('scale', 0.5)),
    'CFG_AUG_SHEAR':    sh(aug.get('shear', 0.0)),
    'CFG_AUG_PERSPECTIVE': sh(aug.get('perspective', 0.0)),
    'CFG_AUG_FLIPUD':   sh(aug.get('flipud', 0.0)),
    'CFG_AUG_FLIPLR':   sh(aug.get('fliplr', 0.0)),
    'CFG_AUG_MOSAIC':   sh(aug.get('mosaic', 1.0)),
    'CFG_AUG_MIXUP':    sh(aug.get('mixup', 0.0)),
    'CFG_AUG_COPY_PASTE': sh(aug.get('copy_paste', 0.0)),
    'CFG_AUG_ERASING':  sh(aug.get('erasing', 0.4)),
    'CFG_AUG_BGR':      sh(aug.get('bgr', 0.0)),
    # export
    'CFG_EXPORT_ENABLED':    sh(xp.get('enabled', False)),
    'CFG_EXPORT_FORMATS':    sh(xp.get('formats', [])),
    'CFG_EXPORT_PRECISIONS': sh(xp.get('precisions', [])),
    # registration
    'CFG_REG_MODEL_NAME': sh(reg.get('registered_model_name')),
    'CFG_REG_PROMOTE_TO': sh(reg.get('promote_to')),
}

for k, v in fields.items():
    print(f"{k}={json.dumps(v)}")
PYEOF
)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Emit --flag-name for True, --no-flag-name for False
bool_flag() {
    local name=$1 val=$2
    [[ "$val" == "True" ]] && echo "--${name}" || echo "--no-${name}"
}

# Run or print a command depending on --dry-run
run() {
    if $DRY_RUN; then
        echo "[dry-run]" "$@"
    else
        "$@"
    fi
}

# Determine whether a step should run given --step / --from flags
should_run_step() {
    local step=$1
    if [[ -n "$ONLY_STEP" ]]; then
        [[ "$step" == "$ONLY_STEP" ]]
    elif [[ -n "$FROM_STEP" ]]; then
        # Mark when we've reached the from-step, then run all remaining
        case "$FROM_STEP" in
            config-validation)  true ;;
            dataset-loading)    [[ "$step" != "config-validation" ]] ;;
            model-training)     [[ "$step" == "model-training" || "$step" == "model-registration" ]] ;;
            model-registration) [[ "$step" == "model-registration" ]] ;;
            *) echo "Unknown step for --from: $FROM_STEP" >&2; exit 1 ;;
        esac
    else
        true
    fi
}

# ---------------------------------------------------------------------------
# Step: config-validation
# ---------------------------------------------------------------------------
CV_ARTIFACT="$ARTIFACTS_DIR/config_validation/validated_config.json"

run_config_validation() {
    echo "==> [1/4] config-validation"
    mkdir -p "$(dirname "$CV_ARTIFACT")"

    local extra_flags=()
    [[ -n "$CFG_EXPERIMENT_DESCRIPTION" ]] && extra_flags+=(--experiment-description "$CFG_EXPERIMENT_DESCRIPTION")
    [[ -n "$CFG_DATASET_LAKEFS_REPO" ]]   && extra_flags+=(--dataset-lakefs-repo "$CFG_DATASET_LAKEFS_REPO")
    [[ -n "$CFG_DATASET_LAKEFS_BRANCH" ]] && extra_flags+=(--dataset-lakefs-branch "$CFG_DATASET_LAKEFS_BRANCH")
    [[ -n "$CFG_DATASET_PATH_OVERRIDE" ]] && extra_flags+=(--dataset-path-override "$CFG_DATASET_PATH_OVERRIDE")
    [[ -n "$CFG_DATASET_SAMPLE_SIZE" ]]   && extra_flags+=(--dataset-sample-size "$CFG_DATASET_SAMPLE_SIZE")
    [[ -n "$CFG_MODEL_PRETRAINED_WEIGHTS" ]] && extra_flags+=(--model-pretrained-weights "$CFG_MODEL_PRETRAINED_WEIGHTS")
    [[ -n "$CFG_TRAINING_FREEZE" ]] && extra_flags+=(--training-freeze "$CFG_TRAINING_FREEZE")
    [[ -n "$CFG_CK_RESUME_FROM" ]]  && extra_flags+=(--checkpointing-resume-from "$CFG_CK_RESUME_FROM")

    run poetry -C "$REPO_ROOT/config_validation" run config-validation run \
        --experiment-name "$CFG_EXPERIMENT_NAME" \
        --dataset-version "$CFG_DATASET_VERSION" \
        --dataset-source  "$CFG_DATASET_SOURCE" \
        --dataset-seed    "$CFG_DATASET_SEED" \
        "$(bool_flag dataset-labels-only "$CFG_DATASET_LABELS_ONLY")" \
        "$(bool_flag dataset-manifest-only "$CFG_DATASET_MANIFEST_ONLY")" \
        --model-variant "$CFG_MODEL_VARIANT" \
        --training-epochs        "$CFG_TRAINING_EPOCHS" \
        --training-batch-size    "$CFG_TRAINING_BATCH_SIZE" \
        --training-image-size    "$CFG_TRAINING_IMAGE_SIZE" \
        --training-learning-rate "$CFG_TRAINING_LR" \
        --training-optimizer     "$CFG_TRAINING_OPTIMIZER" \
        "$(bool_flag training-cos-lr "$CFG_TRAINING_COS_LR")" \
        --training-lrf              "$CFG_TRAINING_LRF" \
        --training-momentum         "$CFG_TRAINING_MOMENTUM" \
        --training-weight-decay     "$CFG_TRAINING_WEIGHT_DECAY" \
        --training-warmup-epochs    "$CFG_TRAINING_WARMUP_EPOCHS" \
        --training-warmup-momentum  "$CFG_TRAINING_WARMUP_MOMENTUM" \
        --training-dropout          "$CFG_TRAINING_DROPOUT" \
        --training-label-smoothing  "$CFG_TRAINING_LABEL_SMOOTHING" \
        --training-nbs              "$CFG_TRAINING_NBS" \
        "$(bool_flag training-amp "$CFG_TRAINING_AMP")" \
        --training-close-mosaic     "$CFG_TRAINING_CLOSE_MOSAIC" \
        --training-seed             "$CFG_TRAINING_SEED" \
        "$(bool_flag training-deterministic "$CFG_TRAINING_DETERMINISTIC")" \
        --training-pose "$CFG_TRAINING_POSE" \
        --training-kobj "$CFG_TRAINING_KOBJ" \
        --training-box  "$CFG_TRAINING_BOX" \
        --training-cls  "$CFG_TRAINING_CLS" \
        --training-dfl  "$CFG_TRAINING_DFL" \
        --checkpointing-interval-epochs "$CFG_CK_INTERVAL" \
        --checkpointing-storage-path    "$CFG_CK_STORAGE_PATH" \
        --early-stopping-patience "$CFG_ES_PATIENCE" \
        --aug-hsv-h    "$CFG_AUG_HSV_H" \
        --aug-hsv-s    "$CFG_AUG_HSV_S" \
        --aug-hsv-v    "$CFG_AUG_HSV_V" \
        --aug-degrees  "$CFG_AUG_DEGREES" \
        --aug-translate "$CFG_AUG_TRANSLATE" \
        --aug-scale    "$CFG_AUG_SCALE" \
        --aug-shear    "$CFG_AUG_SHEAR" \
        --aug-perspective "$CFG_AUG_PERSPECTIVE" \
        --aug-flipud   "$CFG_AUG_FLIPUD" \
        --aug-fliplr   "$CFG_AUG_FLIPLR" \
        --aug-mosaic   "$CFG_AUG_MOSAIC" \
        --aug-mixup    "$CFG_AUG_MIXUP" \
        --aug-copy-paste "$CFG_AUG_COPY_PASTE" \
        --aug-erasing  "$CFG_AUG_ERASING" \
        --aug-bgr      "$CFG_AUG_BGR" \
        --output-path  "$CV_ARTIFACT" \
        "${extra_flags[@]+"${extra_flags[@]}"}"
}

# ---------------------------------------------------------------------------
# Step: dataset-loading
# ---------------------------------------------------------------------------
DATASET_DIR="$ARTIFACTS_DIR/dataset"
DATASET_STATS="$DATASET_DIR/dataset_stats.json"

run_dataset_loading() {
    echo "==> [2/4] dataset-loading"
    mkdir -p "$DATASET_DIR"

    local extra_flags=()
    [[ -n "$CFG_DATASET_LAKEFS_REPO" ]]    && extra_flags+=(--lakefs-repo "$CFG_DATASET_LAKEFS_REPO")
    [[ -n "$CFG_DATASET_LAKEFS_BRANCH" ]]  && extra_flags+=(--lakefs-branch "$CFG_DATASET_LAKEFS_BRANCH")
    [[ -n "$CFG_DATASET_PATH_OVERRIDE" ]]  && extra_flags+=(--path-override "$CFG_DATASET_PATH_OVERRIDE")
    [[ -n "$CFG_DATASET_SAMPLE_SIZE" ]]    && extra_flags+=(--sample-size "$CFG_DATASET_SAMPLE_SIZE")

    run poetry -C "$REPO_ROOT/dataset_loading" run dataset-loading run \
        --version    "$CFG_DATASET_VERSION" \
        --source     "$CFG_DATASET_SOURCE" \
        --output-dir "$DATASET_DIR" \
        --seed       "$CFG_DATASET_SEED" \
        "$(bool_flag labels-only "$CFG_DATASET_LABELS_ONLY")" \
        "$(bool_flag manifest-only "$CFG_DATASET_MANIFEST_ONLY")" \
        "${extra_flags[@]+"${extra_flags[@]}"}"
}

# ---------------------------------------------------------------------------
# Step: model-training
# ---------------------------------------------------------------------------
TRAINING_DIR="$ARTIFACTS_DIR/training"
TRAINING_RESULT="$TRAINING_DIR/training_result.json"

run_model_training() {
    echo "==> [3/4] model-training"
    mkdir -p "$TRAINING_DIR"

    # --- Propagate provenance values (FR-06) ---
    # Read config_hash from config_validation artefact
    local config_hash=""
    if [[ -f "$CV_ARTIFACT" ]]; then
        config_hash=$(python3 -c "
import json, sys
d = json.load(open('$CV_ARTIFACT'))
print(d.get('config_hash') or '', end='')
")
    fi

    # Read dataset provenance from dataset_stats.json
    local lakefs_commit="" dataset_version="" lakefs_branch=""
    if [[ -f "$DATASET_STATS" ]]; then
        lakefs_commit=$(python3 -c "
import json
d = json.load(open('$DATASET_STATS'))
print(d.get('lakefs_commit') or '', end='')
")
        dataset_version=$(python3 -c "
import json
d = json.load(open('$DATASET_STATS'))
print(d.get('version') or '', end='')
")
    fi
    # lakefs_branch comes from the config (dataset_stats has no branch field)
    lakefs_branch="$CFG_DATASET_LAKEFS_BRANCH"

    local provenance_flags=()
    [[ -n "$config_hash" ]]     && provenance_flags+=(--config-hash "$config_hash")
    [[ -n "$lakefs_commit" ]]   && provenance_flags+=(--lakefs-commit "$lakefs_commit")
    [[ -n "$dataset_version" ]] && provenance_flags+=(--dataset-version "$dataset_version")
    [[ -n "$lakefs_branch" ]]   && provenance_flags+=(--lakefs-branch "$lakefs_branch")

    # Optional flags
    local extra_flags=()
    [[ -n "$CFG_MODEL_PRETRAINED_WEIGHTS" ]] && extra_flags+=(--pretrained-weights "$CFG_MODEL_PRETRAINED_WEIGHTS")
    [[ -n "$CFG_CK_RESUME_FROM" ]]           && extra_flags+=(--resume-from "$CFG_CK_RESUME_FROM")
    [[ -n "$CFG_TRAINING_FREEZE" ]]          && extra_flags+=(--freeze "$CFG_TRAINING_FREEZE")
    if [[ "$CFG_EXPORT_ENABLED" == "True" ]]; then
        extra_flags+=("$(bool_flag export "$CFG_EXPORT_ENABLED")")
        [[ -n "$CFG_EXPORT_FORMATS" ]]    && extra_flags+=(--export-formats "$CFG_EXPORT_FORMATS")
        [[ -n "$CFG_EXPORT_PRECISIONS" ]] && extra_flags+=(--export-precisions "$CFG_EXPORT_PRECISIONS")
    else
        extra_flags+=(--no-export)
    fi

    run poetry -C "$REPO_ROOT/model_training" run model-training run \
        --model-variant    "$CFG_MODEL_VARIANT" \
        --experiment-name  "$CFG_EXPERIMENT_NAME" \
        --dataset-dir      "$DATASET_DIR" \
        --output-dir       "$TRAINING_DIR" \
        --source           local \
        --epochs           "$CFG_TRAINING_EPOCHS" \
        --batch-size       "$CFG_TRAINING_BATCH_SIZE" \
        --image-size       "$CFG_TRAINING_IMAGE_SIZE" \
        --learning-rate    "$CFG_TRAINING_LR" \
        --optimizer        "$CFG_TRAINING_OPTIMIZER" \
        "$(bool_flag cos-lr "$CFG_TRAINING_COS_LR")" \
        --lrf              "$CFG_TRAINING_LRF" \
        --momentum         "$CFG_TRAINING_MOMENTUM" \
        --weight-decay     "$CFG_TRAINING_WEIGHT_DECAY" \
        --warmup-epochs    "$CFG_TRAINING_WARMUP_EPOCHS" \
        --warmup-momentum  "$CFG_TRAINING_WARMUP_MOMENTUM" \
        --dropout          "$CFG_TRAINING_DROPOUT" \
        --label-smoothing  "$CFG_TRAINING_LABEL_SMOOTHING" \
        --nbs              "$CFG_TRAINING_NBS" \
        "$(bool_flag amp "$CFG_TRAINING_AMP")" \
        --close-mosaic     "$CFG_TRAINING_CLOSE_MOSAIC" \
        --seed             "$CFG_TRAINING_SEED" \
        "$(bool_flag deterministic "$CFG_TRAINING_DETERMINISTIC")" \
        --pose "$CFG_TRAINING_POSE" \
        --kobj "$CFG_TRAINING_KOBJ" \
        --box  "$CFG_TRAINING_BOX" \
        --cls  "$CFG_TRAINING_CLS" \
        --dfl  "$CFG_TRAINING_DFL" \
        --patience              "$CFG_ES_PATIENCE" \
        --checkpoint-interval   "$CFG_CK_INTERVAL" \
        --checkpoint-bucket     "$CFG_CK_BUCKET" \
        --checkpoint-prefix     "$CFG_CK_PREFIX" \
        --hsv-h    "$CFG_AUG_HSV_H" \
        --hsv-s    "$CFG_AUG_HSV_S" \
        --hsv-v    "$CFG_AUG_HSV_V" \
        --degrees  "$CFG_AUG_DEGREES" \
        --translate "$CFG_AUG_TRANSLATE" \
        --scale    "$CFG_AUG_SCALE" \
        --shear    "$CFG_AUG_SHEAR" \
        --perspective "$CFG_AUG_PERSPECTIVE" \
        --flipud   "$CFG_AUG_FLIPUD" \
        --fliplr   "$CFG_AUG_FLIPLR" \
        --mosaic   "$CFG_AUG_MOSAIC" \
        --mixup    "$CFG_AUG_MIXUP" \
        --copy-paste "$CFG_AUG_COPY_PASTE" \
        --erasing  "$CFG_AUG_ERASING" \
        --bgr      "$CFG_AUG_BGR" \
        "${provenance_flags[@]+"${provenance_flags[@]}"}" \
        "${extra_flags[@]+"${extra_flags[@]}"}"
}

# ---------------------------------------------------------------------------
# Step: model-registration
# ---------------------------------------------------------------------------

run_model_registration() {
    echo "==> [4/4] model-registration"

    if [[ ! -f "$TRAINING_RESULT" ]]; then
        if $DRY_RUN; then
            echo "[dry-run] poetry -C $REPO_ROOT/model_registration run model-registration run <args from training_result.json>"
            return 0
        else
            echo "training_result.json not found at $TRAINING_RESULT — cannot register." >&2
            exit 1
        fi
    fi

    # Parse training result
    local mlflow_run_id best_checkpoint_s3 final_map50 exported_models_json
    mlflow_run_id=$(python3 -c "
import json
d = json.load(open('$TRAINING_RESULT'))
print(d['mlflow_run_id'], end='')
")
    best_checkpoint_s3=$(python3 -c "
import json
d = json.load(open('$TRAINING_RESULT'))
print(d['best_checkpoint_s3'], end='')
")
    final_map50=$(python3 -c "
import json
d = json.load(open('$TRAINING_RESULT'))
print(d['final_map50'], end='')
")
    exported_models_json=$(python3 -c "
import json
d = json.load(open('$TRAINING_RESULT'))
exports = d.get('exported_models', {})
print(json.dumps(exports) if exports else '', end='')
")

    # Derive last.pt path from best.pt path
    local last_checkpoint_s3
    last_checkpoint_s3="${best_checkpoint_s3/best.pt/last.pt}"

    # Re-read provenance from artefacts (same as training step)
    local config_hash="" lakefs_commit="" dataset_version=""
    [[ -f "$CV_ARTIFACT" ]] && config_hash=$(python3 -c "
import json
d = json.load(open('$CV_ARTIFACT'))
print(d.get('config_hash') or '', end='')
")
    [[ -f "$DATASET_STATS" ]] && lakefs_commit=$(python3 -c "
import json
d = json.load(open('$DATASET_STATS'))
print(d.get('lakefs_commit') or '', end='')
") && dataset_version=$(python3 -c "
import json
d = json.load(open('$DATASET_STATS'))
print(d.get('version') or '', end='')
")

    local git_commit
    git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "")

    local extra_flags=()
    [[ -n "$CFG_REG_MODEL_NAME" ]]  && extra_flags+=(--registered-model-name "$CFG_REG_MODEL_NAME")
    [[ -n "$CFG_REG_PROMOTE_TO" ]]  && extra_flags+=(--promote-to "$CFG_REG_PROMOTE_TO")
    [[ -n "$config_hash" ]]         && extra_flags+=(--config-hash "$config_hash")
    [[ -n "$lakefs_commit" ]]       && extra_flags+=(--git-commit "$git_commit")
    [[ -n "$dataset_version" ]]     && extra_flags+=(--dataset-version "$dataset_version")
    [[ -n "$CFG_DATASET_SAMPLE_SIZE" ]] && extra_flags+=(--dataset-sample-size "$CFG_DATASET_SAMPLE_SIZE")
    [[ -n "$final_map50" ]]         && extra_flags+=(--best-map50 "$final_map50")
    [[ -n "$exported_models_json" ]] && extra_flags+=(--exported-models "$exported_models_json")

    run poetry -C "$REPO_ROOT/model_registration" run model-registration run \
        --mlflow-run-id           "$mlflow_run_id" \
        --best-checkpoint-path    "$best_checkpoint_s3" \
        --last-checkpoint-path    "$last_checkpoint_s3" \
        --model-variant           "$CFG_MODEL_VARIANT" \
        "${extra_flags[@]+"${extra_flags[@]}"}"
}

# ---------------------------------------------------------------------------
# Main — run steps
# ---------------------------------------------------------------------------
echo "Pipeline config : $CONFIG_PATH"
echo "Artifacts dir   : $ARTIFACTS_DIR"
echo "Experiment      : $CFG_EXPERIMENT_NAME"
$DRY_RUN && echo "(dry-run mode)"
echo ""

should_run_step config-validation  && run_config_validation
should_run_step dataset-loading    && run_dataset_loading
should_run_step model-training     && run_model_training
should_run_step model-registration && run_model_registration

echo ""
echo "Pipeline complete."
