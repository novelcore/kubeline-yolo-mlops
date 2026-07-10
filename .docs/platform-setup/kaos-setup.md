# Infrastructure Setup

Follow these five steps in order to provision the full ML pipeline stack on KAOS.

!!! warning "Work through each step completely before moving to the next"
    Each layer depends on the previous one. A partially-ready KubePool will cause KubeProject to stall.

---

## Step 1: Create a KubeOrg

!!! warning "Verify GCP field names"
    The exact GCP field names in the KAOS CRDs (e.g. `spec.gcpConfig.*`) depend on your KAOS operator version — confirm them against your operator's CRD schema before applying.

A `KubeOrg` defines your organisation's GCP project, GitHub integration, and network foundations.
Everything else in KAOS is scoped to an organisation.

**What it creates:**

- GCP and GitHub provider credentials in the control plane
- Shared networking defaults (VPC configuration) for clusters in this organisation
- Labels and environment data that downstream resources inherit

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.gcpConfig.project` | Your GCP project ID |
| `spec.gcpConfig.region` | Primary GCP region (e.g., `europe-central2`) |
| `spec.githubConfig.org` | Your GitHub organisation name |
| `spec.network` | VPC CIDR and subnet configuration |

!!! tip "Use the KAOS dashboard or MCP agent"
    You can create a KubeOrg through the KAOS web UI or by asking the KAOS MCP agent: *"Create a KubeOrg for my organisation"*.

**Verify it is ready:**

```bash
kubectl get kubeorg <your-org-name> -o jsonpath='{.status.phase}'
# Expected: Ready
```

---

## Step 2: Create a KubePool

A `KubePool` provisions a GKE cluster and installs the operators required for ML workloads.

**What it creates:**

- A GKE cluster with a system node pool
- GKE node pools + the cluster autoscaler for dynamic GPU and CPU node provisioning
- The NVIDIA GPU operator (for CUDA-enabled training)
- CloudNativePG (for MLflow's PostgreSQL backend)
- Optional: observability stack (VictoriaMetrics + Grafana)

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.cluster.version` | Kubernetes version (e.g., `1.31`) |
| `spec.operators.nvidiaGpu.enabled` | `true` — required for GPU training steps |
| `spec.operators.autoscaler.enabled` | `true` — required for dynamic node scaling |
| `spec.operators.autoscaler.nodeServiceAccount` | GCP IAM service account for autoscaler-managed nodes |
| `spec.operators.autoscaler.controllerServiceAccount` | Workload Identity service account for the autoscaler controller |
| `spec.operators.postgres.enabled` | `true` — required for MLflow backend |
| `spec.features.observability` | `true` — enables cost dashboards (recommended) |

!!! warning "GPU operator is required"
    Without `spec.operators.nvidiaGpu.enabled: true`, the model training step will fail to schedule on GPU nodes.

**Verify it is ready (cluster creation takes 20–30 minutes):**

```bash
kubectl get kubepool <your-pool-name> -o jsonpath='{.status.phase}'
# Expected: Ready
```

---

## Step 3: Create a KubeProject

A `KubeProject` provisions the project-level ML infrastructure: MLflow, LakeFS, and a GCS bucket (object storage).

**What it creates:**

- A dedicated Kubernetes namespace (`ml-<project-name>`)
- A PostgreSQL database for MLflow
- An MLflow tracking server and model registry
- A LakeFS instance for dataset versioning (optional but recommended)
- A GCS bucket (object storage) for datasets, checkpoints, and MLflow artifacts

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.ml.enabled` | `true` |
| `spec.ml.kubePool` | Name of the KubePool created in Step 2 |
| `spec.ml.components.mlflow` | `true` |
| `spec.ml.components.lakefs` | `true` (recommended) |
| `spec.ml.s3.enabled` | `true` |
| `spec.ml.postgres.storageSize` | Database storage, e.g., `20Gi` |

**Verify it is ready:**

```bash
kubectl get kubeproject <your-project-name> -o jsonpath='{.status.mlStack}'
# Expected: all components show Ready
```

After this step, you have your **MLflow URL** and **LakeFS URL**. You can find them in the project status:

```bash
kubectl get kubeproject <your-project-name> -o yaml | grep -A5 "mlStack"
```

---

## Step 4: Register the XKubeAppTemplate

An `XKubeAppTemplate` defines the pipeline template type. For YOLO MLOps, use:

```yaml
apiVersion: platform.kubecore.io/v1beta1
kind: XKubeAppTemplate
metadata:
  name: kubeline-yolo-mlops
spec:
  type: ml-pipeline
  source:
    url: https://github.com/novelcore/kubeline-yolo-mlops
    branch: main
  profiles:
    small:
      gpu: "1"
      cpu: "8"
      memory: "32Gi"
    medium:
      gpu: "2"
      cpu: "16"
      memory: "64Gi"
    large:
      gpu: "4"
      cpu: "32"
      memory: "128Gi"
  environment:
    schema:
      LAKEFS_ENDPOINT:
        type: string
        description: LakeFS server URL
      MLFLOW_TRACKING_URI:
        type: string
        description: MLflow tracking server URL
```

This tells KAOS that the `kubeline-yolo-mlops` template is an ML pipeline type and where to find the source. The `profiles` section defines pre-configured resource tiers (small, medium, large), and the `environment.schema` declares the environment variables that each pipeline instance must provide.

**Verify:**

```bash
kubectl get xkubeapptemplate kubeline-yolo-mlops
# Expected: READY = true
```

---

## Step 5: Create a KubeApp

A `KubeApp` is your specific pipeline instance — it connects the template to your project and configures compute resources.

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.kubeAppTemplateRef` | `kubeline-yolo-mlops` |
| `spec.mlPipeline.gpu.instanceFamilies` | GPU machine families (e.g., `["n1", "a2"]` (n1 hosts T4; a2 hosts A100) or `["*"]` for any) |
| `spec.mlPipeline.gpu.instanceSizes` | GPU instance sizes (e.g., `["xlarge", "2xlarge"]` or `["*"]`) |
| `spec.mlPipeline.gpu.maxResources` | Hard cap on GPU usage (e.g., `nvidia.com/gpu: "4"`) |
| `spec.mlPipeline.cpu.instanceFamilies` | CPU instance families for non-training steps |
| `spec.mlPipeline.cpu.maxResources` | Hard cap on CPU usage |
| `spec.mlPipeline.gpu.diskSizeGi` | Root persistent disk for GPU nodes (e.g., `200`) |

!!! tip "Use wildcards for flexibility"
    Setting `instanceFamilies: ["*"]` and `instanceSizes: ["*"]` lets the cluster autoscaler pick the best available machine type. This is the fastest way to get started.

**What gets created after KubeApp is ready:**

- Per-app GPU and CPU node pools managed by the cluster autoscaler
- An Artifact Registry repository for pipeline Docker images
- A rendered Argo Workflows `WorkflowTemplate` in your GitOps repository
- A Grafana cost dashboard (if observability is enabled)

**Verify:**

```bash
kubectl get kubeapp <your-app-name> -o jsonpath='{.status.mlPipelineStatus}'
# Expected: WorkflowTemplate shown as synced
```

---

## Status Fields to Watch

After creating resources, monitor these status fields during rollout:

```bash
# Project ML stack readiness
kubectl get kubeproject <project> -o jsonpath='{.status.mlStack}'

# App ML pipeline status
kubectl get kubeapp <app> -o jsonpath='{.status.xK8sMLAppRef}'
kubectl get kubeapp <app> -o jsonpath='{.status.mlPipelineStatus}'

# WorkflowTemplate link (where your Argo template lives)
kubectl get xk8smlapps.platform.kubecore.io <app> -o jsonpath='{.status.workflowTemplateLink}'

# Cost tracking dashboard URL
kubectl get xk8smlapps.platform.kubecore.io <app> -o jsonpath='{.status.costTracking.dashboardUrl}'
```

!!! tip
    The `workflowTemplateLink` field tells you the exact path of your rendered WorkflowTemplate in the GitOps repo. The `costTracking.dashboardUrl` gives you a direct link to the Grafana cost dashboard for your pipeline.

---

## All Done

Your environment is fully provisioned. You should now have:

- [x] Argo Workflows UI accessible via SSO
- [x] MLflow URL for your project
- [x] A `WorkflowTemplate` ready to submit

Continue to configure your first experiment:

[:octicons-arrow-right-24: Configure Your Experiment](../configure/index.md)
