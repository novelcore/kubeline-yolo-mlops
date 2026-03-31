# Infrastructure Setup

Follow these five steps in order to provision the full ML pipeline stack on KAOS.

!!! warning "Work through each step completely before moving to the next"
    Each layer depends on the previous one. A partially-ready KubePool will cause KubeProject to stall.

---

## Step 1: Create a KubeOrg

A `KubeOrg` defines your organisation's AWS account, GitHub integration, and network foundations.
Everything else in KAOS is scoped to an organisation.

**What it creates:**

- AWS and GitHub provider credentials in the control plane
- Shared networking defaults (VPC configuration) for clusters in this organisation
- Labels and environment data that downstream resources inherit

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.awsConfig.account` | Your AWS account ID |
| `spec.awsConfig.region` | Primary AWS region (e.g., `eu-central-1`) |
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

A `KubePool` provisions an EKS cluster and installs the operators required for ML workloads.

**What it creates:**

- An EKS cluster with a system node group
- Karpenter for dynamic GPU and CPU node provisioning
- The NVIDIA GPU operator (for CUDA-enabled training)
- CloudNativePG (for MLflow's PostgreSQL backend)
- Optional: observability stack (VictoriaMetrics + Grafana)

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.cluster.version` | Kubernetes version (e.g., `1.31`) |
| `spec.operators.nvidiaGpu.enabled` | `true` — required for GPU training steps |
| `spec.operators.karpenter.enabled` | `true` — required for dynamic node scaling |
| `spec.operators.karpenter.nodeRoleName` | IAM role name for Karpenter-managed nodes |
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

A `KubeProject` provisions the project-level ML infrastructure: MLflow, LakeFS, and an S3 bucket.

**What it creates:**

- A dedicated Kubernetes namespace (`ml-<project-name>`)
- A PostgreSQL database for MLflow
- An MLflow tracking server and model registry
- A LakeFS instance for dataset versioning (optional but recommended)
- An S3 bucket for datasets, checkpoints, and MLflow artifacts

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

## Step 4: Register the KubeAppTemplate

A `KubeAppTemplate` defines the pipeline template type. For YOLO MLOps, use:

```yaml
apiVersion: platform.kubecore.io/v1beta1
kind: KubeAppTemplate
metadata:
  name: kubeline-yolo-mlops
spec:
  type: ml-pipeline
  source:
    url: https://github.com/novelcore/kubeline-yolo-mlops
    branch: main
```

This tells KAOS that the `kubeline-yolo-mlops` template is an ML pipeline type and where to find the source.

**Verify:**

```bash
kubectl get kubeapptemplate kubeline-yolo-mlops
# Expected: READY = true
```

---

## Step 5: Create a KubeApp

A `KubeApp` is your specific pipeline instance — it connects the template to your project and configures compute resources.

**Key fields to set:**

| Field | Description |
| --- | --- |
| `spec.kubeAppTemplateRef` | `kubeline-yolo-mlops` |
| `spec.mlPipeline.gpu.instanceFamilies` | GPU instance families (e.g., `["p3", "p4"]` or `["*"]` for any) |
| `spec.mlPipeline.gpu.instanceSizes` | GPU instance sizes (e.g., `["xlarge", "2xlarge"]` or `["*"]`) |
| `spec.mlPipeline.gpu.maxResources` | Hard cap on GPU usage (e.g., `nvidia.com/gpu: "4"`) |
| `spec.mlPipeline.cpu.instanceFamilies` | CPU instance families for non-training steps |
| `spec.mlPipeline.cpu.maxResources` | Hard cap on CPU usage |
| `spec.mlPipeline.gpu.diskSizeGi` | Root EBS volume for GPU nodes (e.g., `200`) |

!!! tip "Use wildcards for flexibility"
    Setting `instanceFamilies: ["*"]` and `instanceSizes: ["*"]` lets Karpenter pick the best available instance. This is the fastest way to get started.

**What gets created after KubeApp is ready:**

- Per-app GPU and CPU NodePools in Karpenter
- An ECR repository for pipeline Docker images
- A rendered Argo Workflows `WorkflowTemplate` in your GitOps repository
- A Grafana cost dashboard (if observability is enabled)

**Verify:**

```bash
kubectl get kubeapp <your-app-name> -o jsonpath='{.status.mlPipelineStatus}'
# Expected: WorkflowTemplate shown as synced
```

---

## All Done

Your environment is fully provisioned. You should now have:

- [x] Argo Workflows UI accessible via SSO
- [x] MLflow URL for your project
- [x] A `WorkflowTemplate` ready to submit

Continue to configure your first experiment:

[:octicons-arrow-right-24: Configure Your Experiment](../configure/index.md)
