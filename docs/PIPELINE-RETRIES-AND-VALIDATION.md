# Pipeline: retries, failures & validation

A plain-language guide to **what happens when a step fails**, **which steps retry automatically (and which don't)**, **how long a step can run**, and **what the pipeline validates before it does any real work**.

The pipeline runs as an Argo Workflow with these steps, in order:

```
config-validation → dataset-loading → model-training → ┬→ model-quantization
                                                       ├→ qat-finetune  (only when mode=qat)
                                                       └→ model-registration
```

Each step waits for the previous one to **succeed** (or be **skipped**) before it starts. A hard failure in any step **stops everything downstream** — you don't get a half-trained, half-registered model.

---

## 1. Retry policy — the one rule to remember

Every step retries **up to 2 times (3 attempts total)**, but **only on infrastructure errors** — never on your data or config being wrong.

| Setting | Value | What it means |
|---|---|---|
| Retry limit | **2** | Up to **3 total attempts** per step (1 + 2 retries) |
| Retry policy | **`OnError`** | Retries **only** on infra/platform errors — not on application failures |
| Retry exclusion | **OOMKilled (exit 137) is NOT retried** | An out-of-memory kill re-runs identically — so it fails fast instead (see below) |
| Backoff | **30s → 60s → 120s** (×2, capped at 5 min) | Waits longer between each retry so a flaky node/service can recover |

### What counts as an "error" (→ **retried automatically**)

These are transient platform problems that a fresh attempt can clear. The pipeline retries them for you:

- **Node lost / evicted** mid-run (autoscaler churn, node repair)
- **Image-pull failures** (registry hiccup, network blip)
- **Spot/preemptible preemption** of the node
- **Argo controller / pod-sandbox errors** — the step never really ran

### What counts as a "failure" (→ **NOT retried, pipeline stops**)

These mean *your* inputs, resources, or the run logic are wrong — retrying would just fail the same way, so the pipeline fails fast and tells you why:

- **OOMKilled (out of memory)** — the step needed more RAM than its node has. Each step runs **whole-node** on the compute class you pick, so retrying on the **same class** just OOMs again → it is **not** retried. **Fix:** pick a **bigger compute class** for that step (the `{step}-class` dropdown — e.g. move `model-training-class` from `cpu-standard` to a larger CPU/GPU class), **or** ask your platform admin to make a bigger node group available. Then re-submit. (There is no per-step memory knob — memory comes from the node the class runs on.)
- **Validation rejected** your config or dataset (see §3)
- **The training/quant script exited non-zero** from its own logic (bad params, corrupt data, an assertion)
- **Auth failed** (e.g. MLflow/lakeFS credentials) — a code/config problem, not a transient one
- **You hit the step's time limit** (see §2)

> **Rule of thumb:** *"The platform broke" (node/network/registry) → retried automatically. "My input, memory, or params were wrong" → stops immediately, fix and re-submit.*

---

## 2. Time limits (a step is killed if it runs too long)

Each step has a hard deadline. If it exceeds it, the step is **killed and marked failed** (and, because it's a timeout not an infra error, it is **not** retried).

| Step | CPU-node deadline | GPU-node deadline |
|---|---|---|
| config-validation | 30 min | 180 min |
| dataset-loading | 30 min | 180 min |
| model-training | 30 min | 180 min |
| qat-finetune | 180 min | 180 min |
| model-quantization | 180 min | 180 min |
| model-registration | 30 min | 180 min |

*(Each step runs on either a CPU or a GPU node depending on the `{step}-class` you pick in the form; the deadline follows the node type. A step routed to CPU that legitimately needs longer should be pointed at a GPU class, or its work reduced.)*

---

## 3. Validation gates — checked **before** any expensive work

Two steps exist purely to **catch bad inputs early**, so you fail in seconds instead of after a 2-hour GPU run. A validation failure **stops the pipeline immediately** (and is **not** retried — the input won't fix itself).

### `config-validation` — checks your submit-form values

Runs first, on CPU, in seconds. Rejects a run when:

- `experiment.name` is empty
- `dataset.source` is not `s3` or `lakefs`
- `dataset.lakefs_repo` / `dataset.lakefs_branch` are empty **when set**
- `sample_size` / other numeric fields are out of range
- *(Note: `dataset.version` is **optional** provenance — leaving it blank is fine.)*

If this fails, **nothing else runs** — nothing is downloaded, no GPU is provisioned.

### `dataset-loading` — checks your dataset actually matches the YOLO-pose contract

Runs next. It reads your dataset from lakeFS and validates the structure **before** training touches it. It rejects a run when:

- `data.yaml` is missing at the dataset root, or is missing required keys (`path`, `train`, `val`, `kpt_shape`, `names`)
- The `train` / `val` split folders (`images/…`, `labels/…`) are missing or empty
- **Image ↔ label mismatch** — an image with no matching `.txt` label (or vice-versa) by filename stem
- Label rows don't match the declared `kpt_shape` (pose keypoint count)

> Tip: run `kubecore-dataset validate ./my_dataset` **locally before uploading** — it runs the *same* checks, so you catch these on your laptop instead of in the pipeline. There's also a manifest-only mode that validates the dataset without pulling every file.

---

## 4. Per-step summary — where retry helps and where it doesn't

| Step | Retries on infra error? | What a **failure** here usually means | Stops downstream? |
|---|---|---|---|
| **config-validation** | ✅ up to 3 attempts | Your form values are invalid (§3) — *fix & re-submit* | ✅ yes — nothing else runs |
| **dataset-loading** | ✅ up to 3 attempts | Dataset doesn't match the YOLO-pose contract (§3), or lakeFS/auth issue | ✅ yes |
| **model-training** | ✅ up to 3 attempts | Bad hyperparameters, corrupt data, or GPU capacity (see below) | ✅ yes |
| **qat-finetune** *(mode=qat only)* | ✅ up to 3 attempts | QAT config/data issue | ✅ yes |
| **model-quantization** | ✅ up to 3 attempts | Export/conversion error on the trained model | ✅ yes |
| **model-registration** | ✅ up to 3 attempts | MLflow auth/connectivity, or missing artifact | ✅ yes |
| **hpc-burst** *(optional)* | ❌ **no retry** | HPC bridge/target error — surfaced directly | — |

**Two caveats on GPU steps:**

- **No GPUs available (cloud stockout):** the training/quant pod can sit **Pending** waiting for a node — this is *not* a step failure and *not* retried (the step hasn't run yet). The pod schedules automatically once GPU capacity returns, or the workflow eventually times out (§2). Cloud-capacity condition, not a pipeline bug.
- **Out of memory (OOMKilled):** treated as a failure, **not** retried (see §1). Pick a **bigger compute class** for that step (`{step}-class`) or ask the admin to add a larger node group — a retry on the same class would just OOM again.

---

## 5. Platform troubleshooting (compute, capacity, quota)

These are **platform/capacity** conditions — not bugs in your run. They show up as a step that's OOMKilled, stuck **Pending**, or failing to get a node.

| Symptom | What it means | What to do |
|---|---|---|
| **Step OOMKilled (exit 137)** | The step needed more RAM than its compute class's node has. Steps run whole-node, so it's the node size, not a per-step limit. | Pick a **bigger compute class** for that step (`{step}-class`), e.g. move `model-training-class` off `cpu-standard` to a larger CPU/GPU class. If no class is big enough, **ask your platform admin to make a bigger node group available**. |
| **Step stuck `Pending` (won't schedule)** | No node of the chosen class is available. Either the node pool is scaling up (wait), it's a GPU stockout, or the **project vCPU/GPU quota is exhausted** by other concurrent runs. | First just wait — pools scale from zero and this usually clears in a few minutes. If it persists, check whether many pipelines are running at once (the project has a **finite vCPU quota**); serialize your run, or **ask the admin to raise the quota / add capacity**. |
| **GPU step Pending for a long time** | GPU (e.g. T4) capacity isn't available in the region right now (a cloud stockout) — the step hasn't run, so it isn't "failed" and isn't retried. | Leave it (it schedules when capacity returns) or re-submit later. Not a pipeline bug. |
| **Autoscaler "backoff after failed scale-up"** | A prior scale-up hit quota/stockout and the autoscaler is in a cool-down. | Self-heals: the platform's autoscaler-healer clears stale backoff automatically once capacity frees. If it lingers, the admin can check the node-pool quota. |

> **Rule of thumb for the team:** OOM or "can't get a node" is a **sizing/capacity** issue — pick a **bigger compute class**, or ask the admin to **make a bigger node group / more quota available**. It is never fixed by re-running on the same class.

---

## 6. What to do when a run fails

1. **Open the failed step in the Argo UI** and read its logs — the failure reason is printed (validation message, traceback, auth error, timeout, OOMKilled).
2. **If it's a validation message** (§3) → fix your config/dataset and re-submit. Run `kubecore-dataset validate` locally first.
3. **If it's OOMKilled or stuck Pending** → it's a compute/capacity issue (§5): pick a bigger `{step}-class`, or ask the admin for more capacity.
4. **If it retried and still failed** → it's not transient; the logs show the real cause (bad params, auth, data).
5. **Re-running is always safe** — submit a fresh run with corrected inputs. Nothing partial is left in a bad state (no half-registered model).

---

*Source of truth: the rendered `WorkflowTemplate` (`retryStrategy: limit=2, retryPolicy=OnError, backoff 30s×2 max 5m`; per-step `activeDeadlineSeconds`), plus the `config_validation` and `dataset_loading` services in your app repo. This document describes the current behavior — if the platform changes retry/timeout defaults, they'll be reflected in the WorkflowTemplate.*
