---
title: "Resumable Training — Pod-Failure-Resilient Pipeline with Checkpoint & Chunk Resume"
document_id: "PRD-KYM-9"
prd_version: 0.1.1
status: draft
audience: technical
difficulty: advanced
topics:
  - argo-workflows
  - checkpoint-resume
  - mlflow-run-continuation
  - dataset-chunking
  - fault-tolerance
phase: "Phase 1 — Pipeline Resilience"
business_gate: "KAOS YOLO training pipeline hardening"
owner: "meter-peter"
github_issue: 9
last_updated: 2026-06-12
extracted_chunks:
  - architectural-decision
  - architectural-rationale
  - feature-requirement
  - constraint
  - acceptance-criterion
  - threat
  - open-question
---

## 1. Business Context

YOLO pose training for the KAOS YOLO project runs as a four-step Argo Workflow (`config_validation` → `dataset_loading` → `model_training` → `model_registration`) on GPU nodes. A full schedule is 100 epochs on A100-class hardware, taking many hours. When a training pod dies — node preemption, OOM kill, network fault, image-pull failure — the workflow fails terminally. The only recovery path today is a manual resubmission, which by default restarts at epoch zero and re-downloads the entire dataset. Periodic checkpoints are already uploaded to S3 every N epochs, but no platform component consumes them automatically: the existing `checkpointing.resume_from` knob requires an engineer to hand-edit the pipeline YAML and resubmit. Every restart-from-zero is directly visible as duplicated GPU spend, and the risk grows linearly with schedule length, blocking the move to larger datasets and preemptible (spot) GPU node groups. This PRD makes resume a first-class workflow capability: automatic on pod failure, and explicitly requestable at submission time.

## 2. Problem Statement

Training continuity is implemented only at the lowest layer (Ultralytics `resume=True`) and is unreachable from the orchestration layer. The Argo workflow has no retry semantics for the training template, no submission parameters expressing "this is a resume of run X", and no validated linkage between a checkpoint, the MLflow run that produced it, and the exact dataset content it was trained on. A resumed run today creates a *new* MLflow run, splitting an experiment's metric history across runs and breaking lineage in model registration.

### 2.1 Identified Gaps

- **No automatic recovery**: the `model_training` Argo template has no `retryStrategy`; any pod failure terminates the workflow even though a usable `last.pt` checkpoint exists in S3 (`model_training/app/services/model_training.py:398-429`).
- **No resume inputs at submission**: the WorkflowTemplate accepts only the config blob (`argo submit -p config=...`); there is no `resume` flag, no MLflow run ID, no source workflow UID parameter, so the Argo UI submission form cannot express resume intent.
- **MLflow run discontinuity**: the Ultralytics MLflow callback always creates a fresh run; `mlflow_run_id` is only harvested *after* training (`model_training.py:451-470`). A crash before completion loses the ID, and a resume produces a second run with restarted metric history.
- **No dataset chunk concept**: `dataset_loading` downloads the full (or sampled) dataset as one monolithic operation; a re-run repeats all downloads, and nothing records which dataset content a checkpoint corresponds to.
- **No working checkpoint discovery**: `resume_from: auto` as currently coded crashes (`AssertionError: nothing to resume`) — S3 checkpoint discovery is net-new functionality (F-06), not a hardening of existing logic. There is also no compatibility check against model variant, image size, or dataset version, and no integrity check against partially-uploaded checkpoints.

## 3. Architectural Rationale — Why We Made These Choices

### 3.1 Two resume paths, one mechanism

Pod failures and operator-initiated continuation are different triggers for the same underlying operation: locate the authoritative run state, validate it, and hand Ultralytics a checkpoint with `resume=True`. We therefore build one resume engine (run-state manifest + checkpoint validation + MLflow run reattachment) and expose it through two doors: an Argo `retryStrategy` on the training template (automatic, same workflow, same pod template re-executed) and explicit submission parameters (`resume`, `mlflow-run-id`, `source-workflow-uid`) for a brand-new workflow that continues an older experiment. Building two separate mechanisms would double validation logic and create divergent failure modes.

### 3.2 S3 run-state manifest as source of truth

Argo retries start a fresh pod with no memory of the previous attempt, and an explicit resume may happen days later from a different workflow. The only storage layer visible to both is S3, where checkpoints already live. A small `run_state.json` per experiment — updated atomically at every checkpoint interval with the MLflow run ID, last completed epoch, checkpoint URIs and content hashes, and the dataset manifest hash — gives every resume path a single, always-current source of truth. This avoids depending on Argo's artifact store (which the pipeline deliberately bypasses for large files) and avoids querying MLflow as a control-plane dependency during recovery.

### 3.3 Run identity: linked-run lineage in v1, single-run continuity as follow-up

The original plan was to reattach every resume to the original MLflow run via `MLFLOW_RUN_ID` for one continuous metric curve. Local verification (Alexa-Aposto, 2026-06-12) showed the Ultralytics MLflow callback *does* respect a pre-set active run (`active_run() or start_run()`), but on resume it re-logs `trainer.args`, in which the `model` parameter has changed (variant name → `last.pt` path); MLflow rejects the changed param, the callback swallows the error and stops logging the resumed segment entirely. Pure env-var injection is therefore not a viable v1 implementation: continuity requires a callback fix (skip/normalize the param re-log on resume) or platform-owned run lifecycle. v1 default is accordingly a **new run per resume attempt**, tagged `resumed_from=<original run id>`, `resume.attempt=N`, `resume.source_workflow_uid` — a queryable lineage chain at the cost of split metric history in the UI (D-03 option B). Single-run continuity (option A) remains the target end-state and graduates to default once the callback fix is proven. Capturing the active run ID *at training start* into the run-state manifest (Section 3.2) is required under both options. Invariant: every resumed run must be reachable from the original via lineage tags; if tagging is ever dropped, model-registration lineage (F-08) breaks under both options.

### 3.4 Chunk-aware dataset loading for idempotent recovery

"Resume at the corresponding chunk" has two halves. First, `dataset_loading` must not redo finished work: the download set is split into deterministic chunks (stable ordering derived from the dataset version and seed), each with a completion marker, so a re-run — whether an Argo retry or an explicit resume — skips completed chunks and finishes only the remainder. Second, training must be able to prove it is resuming against *the same data*: the chunked manifest carries a content hash, recorded in `run_state.json` and in each checkpoint's metadata, and validation refuses a resume whose dataset hash differs from the checkpoint's. Chunking is a loading/integrity concept; the in-epoch sample order remains Ultralytics' responsibility via its restored RNG and dataloader state.

Scope refinement (Q3, 2026-06-12): in-cluster runs stream the dataset — no shared volume exists between the load and train pods — so chunked download with completion markers applies only to the full-download/local modes; streaming and labels-only modes skip chunk markers and enforce `manifest_sha256` alone. Default `chunk_size: 500` is confirmed. Note that the Argo `retryStrategy` (D-01) covers only `model_training` — `dataset_loading` is not retried — so chunk-level resume of the load step pays off only on explicit resubmission. Relatedly, the streaming trainer has *no resume state of its own*: resume is epoch-granular, Ultralytics restores model/optimizer/epoch/scheduler from `last.pt`, and the dataloader is rebuilt every epoch via the `build_dataset()` override (re-streamed, not restored) — verified locally that custom-trainer + `resume=True` composes and resumes at the right epoch.

### 3.5 Validate resume at step 1, fail fast

A resume that will fail (missing checkpoint, deleted MLflow run, incompatible model variant, changed image size, mismatched dataset hash) must be rejected in `config_validation` — seconds into the workflow on a CPU node — not minutes into `model_training` after a GPU node has been provisioned and the dataset materialised. Step 1 already owns config validation and produces the validated-config artifact consumed downstream; extending it with resume validation keeps the established contract that downstream steps trust their inputs.

### 3.6 Retry classification by pod reason, with in-attempt guards

Argo retry expressions see exit codes, but the two highest-volume failure causes — OOMKill and spot preemption/eviction — both surface as exit code 137 and demand opposite handling: OOM at a fixed batch size is deterministic and must not retry, while eviction is the flagship retryable case. Classification therefore keys on the pod's termination *reason*, never the exit code alone. Non-retryable: config/validation errors, image-pull failures, true OOMKilled at fixed batch size, auth failures (MLflow 401/403, storage access denied), corrupt checkpoint/dataset, and exceptions raised by our own code. Retryable: spot eviction/preemption, node loss, network blips, transient upload 5xx. Two guards complement classification. First, a **progress guard**: if the resume epoch does not advance between attempts, stop and alert rather than burn the retry budget on a poison `last.pt` (T-09). Second, because Argo retries re-run only the `model_training` template, F-02's step-1 validation does **not** re-execute — each retry attempt performs a light self-check (checkpoint checksum, run-state sanity) and falls back to a linked new MLflow run if the original is gone (CON-06). Finally, unschedulable GPU (no capacity) produces no failure exit code at all; it is handled by a pending/`activeDeadlineSeconds` timeout on the training template, deliberately separate from retry classification (T-11). Reversing any of this — classifying on exit codes, or trusting step-1 validation to cover retries — reintroduces silent budget burn.

## 4. Goals & Non-Goals

### 4.1 Goals

- A training pod failure at any epoch results in automatic continuation from the most recent valid checkpoint within the same workflow, without operator action.
- An operator can submit a new workflow in resume mode via parameters visible in the Argo UI submission form: a resume toggle plus MLflow run ID, source Argo workflow UID, and optional explicit checkpoint URI.
- A resumed run (automatic or explicit) is linked to the original MLflow run via lineage tags (`resumed_from`, attempt, source workflow UID); v1 ships split-but-linked runs, with single-continuous-run continuity promoted to default once the Ultralytics callback param-collision fix is proven (D-03).
- `dataset_loading` is idempotent and chunk-aware: re-runs skip completed chunks; the dataset content hash is recorded and enforced at resume.
- Resume requests are validated in `config_validation` before any GPU resource is consumed.
- `orchestrate.sh` local mode gains parity flags so the resume path is testable without a cluster.

### 4.2 Non-Goals

- Distributed or multi-node training (single-pod GPU training remains the execution model).
- A custom web UI for submission — the Argo Workflows UI form rendered from WorkflowTemplate parameters is the interface.
- Resume support for non-YOLO kubelines (the run-state manifest format is designed to generalise, but only this pipeline is in scope).
- Checkpoint retention/pruning policy automation (manual S3 lifecycle rules remain acceptable).
- Mid-epoch resume granularity — the unit of recovery is the last completed checkpoint interval, as restored by Ultralytics.

## 5. Constraints

### CON-01 — Ultralytics owns in-training resume state

**Statement**: Restoration of model weights, optimizer state, LR scheduler, EMA, and epoch counter is delegated exclusively to Ultralytics `resume=True`; the platform never re-implements or partially overrides this logic.
**Why it exists**: Ultralytics serialises its full trainer state into `last.pt`, including the original training arguments; duplicating any of it creates silent divergence between framework versions.
**Enforcement**: Code review; the training service may only select *which* checkpoint to hand over, never mutate its contents.
**Refs**: `model_training/app/services/model_training.py:777-778`, D-01.

### CON-02 — Large artifacts stay out of Argo's artifact store

**Statement**: Checkpoints, run-state manifests, and dataset content are exchanged via S3 URIs; Argo artifacts carry only small JSON summaries (≤ a few KB), preserving the existing artifact-passing contract.
**Why it exists**: The pipeline architecture deliberately passes multi-hundred-MB files by reference to keep workflow storage lean and resumes independent of Argo's artifact GC.
**Enforcement**: Validation step rejects configs pointing resume sources at non-S3 URIs; review of WorkflowTemplate changes.
**Refs**: `docs/mlops_technical_architecture.md` §7.3, D-02.

### CON-03 — Resume must be a no-op risk to fresh runs

**Statement**: All new parameters default to fresh-run behaviour (`resume=false`, empty IDs); an unmodified existing submission produces byte-identical behaviour to today.
**Why it exists**: The pipeline is in active use; resilience work must not destabilise the happy path or invalidate existing configs (`pipeline_config.example.yaml`, `pipeline_config.toy.yaml`).
**Enforcement**: Regression test running the toy config without resume parameters; schema defaults in `pipeline_config.py`.
**Refs**: F-01, AC-08.

### CON-04 — Checkpoint writes must be atomic and verifiable

**Statement**: Every checkpoint and run-state upload uses write-then-rename (or S3 multipart completion) semantics with a recorded SHA-256, so a reader never observes a partial object.
**Why it exists**: Pod failure can occur *during* checkpoint upload; auto-resume picking up a truncated `last.pt` would crash the retry loop or, worse, corrupt training silently.
**Enforcement**: Checksum verification before any resume; T-01 mitigation test that kills the uploader mid-transfer.
**Refs**: F-03, T-01.

### CON-05 — Resume validation precedes GPU scheduling

**Statement**: No resume-mode workflow may schedule the `model_training` template before `config_validation` has confirmed checkpoint integrity, MLflow run existence, config compatibility, and dataset-hash match.
**Why it exists**: GPU node provisioning (autoscaled node groups) is the dominant cost; a doomed resume must fail on the CPU-only validation pod.
**Enforcement**: DAG dependency order (already sequential) plus explicit resume checks in step 1; AC-05.
**Refs**: Section 3.5, F-02.

### CON-06 — Retry attempts must self-validate and prove progress

**Statement**: Because Argo `retryStrategy` re-runs only the `model_training` template, every retry attempt performs a light resume self-check (checkpoint SHA-256, run-state sanity, MLflow run reachability with linked-new-run fallback) and must observe the resume epoch advancing across attempts; an attempt that would resume at the same epoch as its predecessor stops the retry loop and alerts.
**Why it exists**: F-02's full validation runs only at step 1 of the original workflow; without an in-attempt guard, a poison checkpoint or vanished MLflow run silently burns the entire retry/GPU budget.
**Enforcement**: Training-entrypoint self-check; AC-10 chaos test; alert on progress-guard stop.
**Refs**: Section 3.6, F-06, T-09.

## 6. Architectural Decisions

### D-01 — Recovery mechanism for in-flight pod failure

| Option | Description | Verdict |
|---|---|---|
| A. Argo `retryStrategy` + checkpoint auto-resume | Retry the training template; each attempt self-locates latest valid checkpoint via run-state manifest | **Chosen** |
| B. Exit-handler resubmits a new workflow | `onExit` template submits a fresh resume-mode workflow | Rejected |
| C. External controller watching for failures | Operator/cron resubmits failed workflows | Rejected |

**Recommendation**: Option A — `retryStrategy` (`retryPolicy: OnFailure`, capped attempts, exponential backoff) on the `model_training` template only, with the training entrypoint performing checkpoint discovery on every start when `run_state.json` exists. Retry classification follows Section 3.6: pod termination *reason*, not exit code alone (OOM and eviction are both 137); deterministic failures non-retryable; progress guard across attempts (CON-06); and a pending/`activeDeadlineSeconds` timeout for unschedulable GPU, which never surfaces as a failing exit code.
**Rationale**: Retries stay inside one workflow UID, preserving Argo UI observability and the existing DAG; no new controller surface. Option B fragments an experiment across workflow objects for the *automatic* case (explicit resume legitimately creates a new workflow — that path exists separately). Option C adds an always-on component for a problem Argo already solves natively.

### D-02 — Authoritative resume state location

| Option | Description | Verdict |
|---|---|---|
| A. `run_state.json` in S3 next to checkpoints | Atomic JSON manifest per experiment, updated at each checkpoint interval | **Chosen** |
| B. MLflow tags/params as control plane | Query MLflow for last checkpoint and epoch | Rejected |
| C. Argo workflow outputs/labels | Persist state on the workflow object | Rejected |

**Recommendation**: Option A, at `s3://{checkpoint_bucket}/{checkpoint_prefix}/{experiment_name}/run_state.json`, containing schema version, MLflow run ID, experiment name, last completed epoch, checkpoint index (URI + SHA-256 + epoch), dataset manifest hash, source workflow UID, and config hash.
**Rationale**: S3 is already the durable layer both resume paths can reach; co-locating state with checkpoints makes the experiment directory self-describing. MLflow (B) becomes a hard availability dependency for recovery and its tag writes are not atomic with checkpoint uploads. Argo objects (C) are garbage-collected and invisible to cross-workflow explicit resume.

### D-03 — MLflow run identity on resume

| Option | Description | Verdict |
|---|---|---|
| A. Reattach to original run via `MLFLOW_RUN_ID` | Same run continues; metrics keyed by epoch extend the series | Follow-up (target end-state) |
| B. New run tagged `resumed_from` | Fresh run per attempt, linked by tags | **Chosen (v1 default)** |
| C. MLflow nested runs (parent/child) | Parent experiment run, child per attempt | Rejected |

**Recommendation**: Option B for v1, promoting Option A once the callback collision is fixed. Local verification (Alexa-Aposto, 2026-06-12) confirmed the Ultralytics MLflow callback honours a pre-set active run (`active_run() or start_run()`), but on resume it re-logs `trainer.args` with a changed `model` param (variant name → `last.pt` path); MLflow rejects the param change and the callback silently stops tracking the resumed segment. v1 therefore creates a new run per resume attempt, tagged `resumed_from=<original run id>`, `resume.attempt=N`, `resume.source_workflow_uid`, and still creates/records the active run ID in `run_state.json` *at training start* — closing today's gap where the ID is only known after completion. Option A needs a small callback fix (skip/normalize the param re-log on resume) or platform-owned run lifecycle, and graduates to default once proven.
**Rationale**: Split-but-linked history is strictly simpler and robust against the *confirmed* param-collision failure; the only cost is a fragmented metric view in the UI, while lineage tags keep registration (F-08) and audit intact. Option A as v1 default would ship a known silent-tracking-loss bug. Nested runs (C) stay rejected — the callback does not manage nesting and registry lineage expects a flat chain.

### D-04 — Dataset chunking and resume correspondence

| Option | Description | Verdict |
|---|---|---|
| A. Chunked download manifest + completion markers + content hash | Deterministic chunks at load time; hash recorded in run state and validated on resume | **Chosen** |
| B. Per-epoch data sharding (train on chunk k at epoch k) | Curriculum-style chunk scheduling inside training | Rejected |
| C. Keep monolithic loading, rely on S3 streaming mode | No chunk concept; streaming cache absorbs re-runs | Rejected |

**Recommendation**: Option A. `dataset_loading` partitions the resolved file list (post-sampling, seed-ordered) into fixed-size chunks (default 500 images, configurable via `dataset.chunk_size`), writes `chunks/chunk_{i:04d}.done` markers and an extended `dataset_manifest.json` carrying per-chunk key lists and a global `manifest_sha256`. Re-runs verify markers and download only incomplete chunks. The manifest hash is the "corresponding chunk" contract: `run_state.json` and checkpoint metadata record it; resume validation requires equality. Scope (Q3 resolution): chunked download + markers apply to the full-download/local modes only — in-cluster streaming and labels-only modes enforce `manifest_sha256` without chunk markers; default 500 confirmed.
**Rationale**: This makes recovery idempotent for both the download step and the training step's data identity without touching the training loop. Option B changes training semantics (data distribution per epoch) — a research decision, not a resilience one. Option C leaves the labels-only and full-download modes unprotected and provides no data-identity proof.

### D-05 — Submission interface for explicit resume

| Option | Description | Verdict |
|---|---|---|
| A. Dedicated WorkflowTemplate parameters with enum/defaults | `resume` (enum "false"/"true"), `mlflow-run-id`, `source-workflow-uid`, `resume-checkpoint` | **Chosen** |
| B. Resume fields only inside the config YAML blob | Extend `checkpointing.*` and keep single `config` parameter | Rejected as sole interface |
| C. Separate resume WorkflowTemplate | A second template just for resume submissions | Rejected |

**Recommendation**: Option A, layered on B: the four parameters are declared in `spec.arguments.parameters` with safe defaults — `resume` as an enum (`"false"`, `"true"`) so the Argo UI renders a constrained toggle/dropdown satisfying the "checkbox" requirement — and `config_validation` merges them into the validated config (parameters override the YAML's `checkpointing.resume_from`). The config YAML keeps equivalent fields so GitOps-style submissions and `orchestrate.sh` work without Argo parameters.
**Rationale**: Top-level parameters are what the Argo UI form renders and what `argo submit -p` can set per-run without editing the config artifact; burying resume intent solely in a pasted YAML blob (B) is invisible and error-prone at submission time. A second template (C) duplicates the DAG and drifts.

## 7. Feature Requirements

| Ref | Feature | Owner | Effort | Priority | Defer? | Decision Refs | Rationale Refs | Description |
|---|---|---|---|---|---|---|---|---|
| F-01 | Resume submission parameters | meter-peter | S | P0 | No | D-05 | 3.1 | Add `resume`, `mlflow-run-id`, `source-workflow-uid`, `resume-checkpoint` WorkflowTemplate parameters with fresh-run defaults; rendered in Argo UI form; merged into validated config by step 1. Lands in the kubecore-operator k8smlapp composition (see §8); also injects `{{workflow.uid}}` (plus `{{workflow.name}}` for readability) into pod env so run_state/heartbeat can record the source workflow — nothing injects a workflow ID today |
| F-02 | Resume validation in config_validation | meter-peter | M | P0 | No | D-02, D-05 | 3.5 | When resume requested: verify `run_state.json` and checkpoint exist, SHA-256 matches, MLflow run exists, config compatibility (model variant, image size, kpt shape), dataset manifest hash match; fail fast with actionable error |
| F-03 | Run-state manifest writer | meter-peter | M | P0 | No | D-02 | 3.2 | Training service writes/updates `run_state.json` atomically at run start and every checkpoint interval: MLflow run ID, last epoch, checkpoint index with hashes, dataset hash, config hash, workflow UID (from the env F-01 injects) + heartbeat timestamp (T-05) |
| F-04 | MLflow run lineage (v1) / continuation (follow-up) | meter-peter | M | P0 | No | D-03 | 3.3 | Own the MLflow run lifecycle: create the run before training and write its ID to run_state at start; v1 creates a new run per resume tagged `resumed_from` + attempt + source workflow UID (D-03 B); single-run continuity (A) follows once the callback param-collision fix (skip/normalize `trainer.args` re-log on resume) is proven |
| F-05 | Chunk-aware idempotent dataset loading | meter-peter | M | P0 | No | D-04 | 3.4 | Deterministic chunked downloads with completion markers, `dataset.chunk_size` config (default 500), extended manifest with per-chunk keys and `manifest_sha256`; re-runs skip complete chunks. Full-download/local modes only — streaming and labels-only enforce `manifest_sha256` without markers |
| F-06 | Argo retryStrategy with auto-resume | meter-peter | M | P0 | No | D-01, D-02 | 3.1, 3.2, 3.6 | `retryStrategy` on `model_training` template only (pod-reason classification per §3.6, limit, backoff, progress guard, `activeDeadlineSeconds` for unschedulable GPU); training entrypoint auto-discovers latest valid checkpoint from `run_state.json` on every attempt — net-new code, `resume_from: auto` currently crashes — plus per-attempt self-check (CON-06). retryStrategy lands in the kubecore-operator k8smlapp composition (see §8) |
| F-07 | orchestrate.sh resume parity | meter-peter | S | P1 | No | D-05 | 4.1 | `--resume`, `--mlflow-run-id`, `--resume-checkpoint` flags in local/docker modes exercising the same code paths |
| F-08 | Registration lineage for resumed runs | meter-peter | S | P1 | No | D-03 | 3.3 | `model_registration` propagates resume lineage tags (attempt count, source workflow UID, original run ID) onto registered model versions |

## 8. Target File / System Structure

```
kubecore-operator/                       # separate repo — operator-side changes (F-01, F-06)
└── compositions/apis/kubeapp/k8smlapp/
    └── composition.yaml                # WorkflowTemplate: resume parameters, retryStrategy,
                                        # {{workflow.uid}}/{{workflow.name}} pod-env injection

kubeline-yolo-mlops/
├── config_validation/app/
│   ├── models/pipeline_config.py       # + dataset.chunk_size, checkpointing resume fields
│   └── services/resume_validation.py   # NEW — F-02 checks
├── dataset_loading/app/services/
│   └── dataset_loading.py              # chunked downloads, markers, manifest hash (F-05)
├── model_training/app/services/
│   ├── model_training.py               # MLflow reattach, auto-discovery on start (F-04, F-06)
│   └── run_state.py                    # NEW — run_state.json schema + atomic S3 writer (F-03)
├── model_registration/app/services/
│   └── model_registration.py           # lineage tags (F-08)
├── orchestrate.sh                      # --resume / --mlflow-run-id / --resume-checkpoint (F-07)
└── docs/mlops_technical_architecture.md  # §9 rewritten for two-path resume
```

S3 experiment directory gains: `run_state.json` (D-02) and `chunks/chunk_NNNN.done` markers under the dataset prefix (D-04).

## 9. Acceptance Criteria

| Ref | Criterion | How to Verify | Status |
|---|---|---|---|
| AC-01 | Killing the training pod mid-epoch leads to automatic continuation from the last checkpoint and a Succeeded workflow with no operator action | Chaos test: `kubectl delete pod` during epoch ≥ checkpoint interval on toy config; inspect retry attempt logs for resume epoch | pending |
| AC-02 | A resumed run (automatic or explicit) is linked to the original MLflow run: the new run carries `resumed_from=<original run id>`, `resume.attempt`, and source-workflow-UID tags, and combined epoch coverage across the chain is complete (single-run continuity graduates this AC once D-03 option A lands) | Query MLflow API: walk the `resumed_from` chain, assert tags present and epoch metric coverage has no gaps | pending |
| AC-03 | Explicit resume via Argo UI: setting `resume=true` + MLflow run ID on the submission form continues a previously stopped experiment | Submit resume workflow against a stopped run; verify training starts at last epoch + 1 | pending |
| AC-04 | Re-running dataset_loading downloads only incomplete chunks | Delete two `.done` markers + corresponding files; re-run; logs show only those chunks fetched; manifest hash unchanged | pending |
| AC-05 | An invalid resume (missing/corrupt checkpoint, deleted MLflow run, changed model variant or dataset hash) fails in config_validation with a specific error, before any GPU pod is scheduled | Submit each invalid variant; assert workflow fails at step 1 with the named reason; no `model_training` pod created | pending |
| AC-06 | `run_state.json` is updated atomically at every checkpoint interval and never observed partial | Integration test polling the object during training; checksum verification on each read | pending |
| AC-07 | Checkpoint-chunk correspondence enforced: resume against a dataset whose manifest hash differs is rejected | Modify one image between runs; attempt resume; validation rejects with hash mismatch | pending |
| AC-08 | Fresh-run behaviour is unchanged: existing configs submitted without resume parameters produce identical results to current main | Run toy config end-to-end on both branches; compare step outputs and MLflow artifacts | pending |
| AC-09 | `orchestrate.sh --resume` reproduces the resume path locally without a cluster | Local run: train 3 epochs, interrupt, resume to completion; verify the MLflow lineage chain (single run once D-03 option A lands) | pending |
| AC-10 | Retry attempts that fail to advance the resume epoch stop the retry loop and alert instead of exhausting the budget | Chaos test with a poisoned/stale `last.pt`: assert at most one wasted attempt, workflow fails with the progress-guard reason, alert fired | pending |

## 10. Threat Register

| ID | Threat | Severity | Status | Features | Mitigation |
|---|---|---|---|---|---|
| T-01 | Pod dies during checkpoint upload, leaving a truncated `.pt`; auto-resume crash-loops on it | High | open | F-03, F-06 | Atomic write-then-rename + SHA-256 in run_state; discovery skips entries failing checksum, falls back to previous checkpoint |
| T-02 | Retry loop on a deterministic failure (bad config, OOM at fixed batch) burns GPU budget | High | open | F-06 | Retry limit (default 3) + exponential backoff; non-retryable classification per §3.6 via pod termination reason; alert on exhausted retries |
| T-03 | Config drift between original and resume submission silently changes training semantics | Medium | open | F-02 | Config hash stored in run_state; validation diffs incompatible fields (variant, imgsz, kpt shape) and rejects; compatible diffs logged |
| T-04 | MLflow tracking server unavailable during recovery blocks an otherwise-valid resume | Medium | open | F-04 | Reattach is best-effort with bounded retries; fallback to linked new run (D-03 option B) so training proceeds |
| T-05 | Concurrent workflows resume the same experiment and corrupt run_state/checkpoints | Medium | open | F-03 | Run-state carries active workflow UID + heartbeat timestamp; validation refuses resume of an experiment with a live heartbeat |
| T-06 | Ultralytics version upgrade changes `last.pt` trainer-state format, breaking old checkpoints | Medium | open | F-02 | Record ultralytics version in run_state; validation warns on mismatch; pin version in step image |
| T-07 | Chunk markers present but underlying S3 objects deleted/modified (marker lies) | Low | open | F-05 | Markers store per-chunk key count + size hash; cheap verification pass before skip; `--force-reload` escape hatch |
| T-08 | Duplicate metric steps if a retried attempt re-runs a partially-logged epoch | Low | open | F-04 | Epoch-keyed metrics make re-logs idempotent per step; resume starts at last *completed* epoch + 1 |
| T-09 | Poison checkpoint: retries resume at the same epoch repeatedly, burning GPU budget on a bad `last.pt` | High | open | F-06 | Progress guard (CON-06): stop retrying + alert when the resume epoch fails to advance between attempts |
| T-10 | OOMKilled and spot eviction both exit 137; exit-code-only classification either retries deterministic OOM or fails recoverable eviction | High | open | F-06 | Classify on pod termination reason (§3.6), never exit code alone; chaos-test both 137 paths |
| T-11 | Unschedulable GPU (no capacity) hangs the workflow with no failure exit code, so retry and alerting never trigger | Medium | open | F-06 | Pending timeout via `activeDeadlineSeconds` on the training template, separate from retry classification |

## 11. Validation Strategy

Three test layers. **Unit**: run-state writer atomicity and schema round-trip, chunk partitioning determinism (same version+seed+chunk_size ⇒ same chunks ⇒ same `manifest_sha256`), resume-validation matrix (each rejection reason). **Integration (local, orchestrate.sh + toy config)**: interrupt-and-resume end-to-end (AC-09), idempotent re-download (AC-04), MLflow continuity against a local tracking server (AC-02). **Cluster (chaos)**: pod-kill during training and during checkpoint upload (AC-01, T-01), retry-exhaustion alerting (T-02), Argo UI explicit-resume walkthrough (AC-03). Regression: AC-08 fresh-run comparison gates the merge. Every cluster test result is captured with workflow UID + MLflow run ID in the test log. Additional cluster obligations from the Q&A round (2026-06-12): one smoke of resume on the *live streaming* path (Q2 was verified locally only), the poison-checkpoint progress-guard test (AC-10), and chaos coverage of both exit-137 paths (OOMKilled vs eviction, T-10).

## 12. Dependency Chain

`run_state.json` (F-03) is the keystone: F-06 (auto-resume) and F-02 (validation) both read it, and F-04 (MLflow continuation) writes the run ID into it at start — so F-03 + F-04 land first, together. F-05 (chunking) is independent of F-03/F-04 but must precede full F-02, whose dataset-hash check consumes the chunked manifest. F-01 (parameters) is mechanically small but only meaningful once F-02 can act on the parameters. F-06 (retryStrategy) is last among P0 — switching it on before atomic checkpoints (CON-04) and discovery logic exist would crash-loop. F-07 and F-08 are P1 polish after the P0 chain: **F-03+F-04 → F-05 → F-02 → F-01 → F-06 → (F-07, F-08)**.

## 13. Effort Summary

Total estimate: **~3–4 engineer-weeks**. F-03 + F-04 (run state + MLflow reattach): ~1 week, highest design risk (atomicity, Ultralytics callback interplay). F-05 (chunked loading): ~3–4 days across the three loading modes (full, labels-only, manifest-only streaming). F-02 (validation): ~3 days including the rejection matrix tests. F-01 + F-06 (WorkflowTemplate parameters + retryStrategy): ~2 days plus cluster chaos testing. F-07 + F-08: ~2 days combined. Add ~3 days for chaos/integration test harness and documentation updates (`docs/mlops_technical_architecture.md` §9, CLAUDE.md).

## 14. Open Questions

| # | Question | Why It Matters | Owner | Due | Status | Resolution |
|---|---|---|---|---|---|---|
| 1 | Does the Ultralytics MLflow callback respect a pre-set `MLFLOW_RUN_ID` / active run, or unconditionally call `start_run()`? | Determines whether F-04 is pure env-var injection or needs a custom callback override | meter-peter | before F-04 | resolved (Alexa-Aposto, 2026-06-12) | Yes — callback uses `active_run() or start_run()` — but on resume it re-logs `trainer.args` with a changed `model` param (variant → `last.pt` path); MLflow rejects the change and the callback silently stops tracking (confirmed locally). Not pure env-var injection: D-03 flipped to option B for v1; option A needs a callback fix or owned run lifecycle (§3.3) |
| 2 | Does Ultralytics `resume=True` fully restore the custom `S3PoseTrainer` streaming dataloader state, or only the standard trainer? | Streaming mode (manifest-only) is the target for large datasets; resume must work there | meter-peter | before F-04 | resolved (Alexa-Aposto, 2026-06-12) | Nothing to restore: resume is epoch-granular; model/optimizer/epoch/scheduler come from `last.pt` and the dataloader is rebuilt each epoch via the `build_dataset()` override (re-streamed). Verified custom-trainer + `resume=True` composes locally; one cluster smoke of the live streaming path still owed (§11) |
| 3 | What is the right default `chunk_size`, and should chunking apply to the labels-only mode where images are streamed? | Chunk size trades marker overhead vs. wasted re-download; streaming mode may need manifest-hash only | meter-peter | before F-05 | resolved (Alexa-Aposto, 2026-06-12) | Default 500 confirmed. Chunk markers only for full-download/local modes; streaming/labels-only enforce `manifest_sha256` alone (no shared volume between load and train pods in-cluster). Auto-retry re-runs only `model_training`, so chunk-resume of the load step helps explicit resubmits only (D-04, §3.4) |
| 4 | Which Argo exit codes/conditions should be non-retryable (OOM, image pull, config error) in the `retryStrategy` expression? | Wrong classification either burns GPU budget or fails recoverable runs | meter-peter | before F-06 | resolved (Alexa-Aposto, 2026-06-12) | Classify on pod termination reason, not exit code (OOM vs eviction are both 137). Non-retryable: config/validation, image pull, true OOM at fixed batch, auth 401/403/storage-denied, corrupt checkpoint/dataset, own-code exceptions. Retryable: preemption/eviction, node/network blips, transient 5xx. Plus progress guard (T-09/CON-06), per-attempt self-check, retry cap+backoff+alert, `activeDeadlineSeconds` for unschedulable GPU (T-11). See §3.6 |
| 5 | Should spot/preemptible GPU node groups be enabled together with this PRD, or as a follow-up once resume is proven? | Spot adoption is the main cost payoff but multiplies failure frequency | meter-peter | post AC-01 | resolved (Alexa-Aposto, 2026-06-12) | Follow-up: prove auto-resume on on-demand first, then flip spot on for the cost win — spot multiplies failure frequency and should not be front-loaded before recovery is proven |
| 6 | Ownership split: F-01/F-06 are operator-repo work (kubecore-operator k8smlapp composition); who implements F-02–F-05/F-07/F-08 in kubeline-yolo-mlops, and in what sequence relative to the operator-side changes? | Two repos, two release trains (operator OCI Configuration package vs pipeline images); the dependency chain (§12) crosses the boundary at F-01/F-06 | meter-peter | before implementation start | open | — |
