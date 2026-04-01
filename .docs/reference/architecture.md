# Architecture

Technical overview of the Kubeline YOLO MLOps pipeline system.

## System Overview

```mermaid
graph TB
    subgraph Workstation["Your Workstation"]
        Params["Argo Submission Parameters"]
    end

    Params -->|submit via Argo UI / GitHub PR| ArgoAPI["Argo Workflows"]

    subgraph K8s["Kubernetes Cluster (KAOS / EKS)"]
        subgraph Argo["Argo Workflows Controller"]
            S1["Step 1: Config Validation\n(CPU · seconds)"]
            S2["Step 2: Dataset Loading\n(CPU · minutes)"]
            S3["Step 3: Model Training\n(GPU · hours–days)"]
            S4["Step 4: Model Registration\n(CPU · seconds)"]
            S1 --> S2 --> S3 --> S4
        end

        ArgoAPI --> S1

        MLflow["MLflow\nTracking + Registry"]
        S3Storage["S3 / LakeFS\nDatasets + Artifacts"]
    end

    S3 -->|log metrics + artifacts| MLflow
    S4 -->|register model| MLflow
    S3Storage -->|fetch dataset| S2
    S3 -->|save checkpoints| S3Storage
```

## Component Roles

| Component | Role |
| --- | --- |
| **Argo Workflows** | DAG orchestration — sequences steps, manages retries, passes artifacts |
| **MLflow Tracking** | Records all parameters, metrics, and artifacts per run |
| **MLflow Model Registry** | Versions trained model checkpoints with stage labels |
| **S3 / LakeFS** | Stores datasets, checkpoints, and MLflow binary artifacts |
| **Karpenter** | Dynamically provisions GPU nodes when training starts, releases them after |
| **ECR** | Stores Docker images for each pipeline step |

## Data Flow

```mermaid
flowchart LR
    subgraph Storage["S3 / LakeFS"]
        DS["Dataset\n(images + labels)"]
        CK["Checkpoints"]
    end

    subgraph Pipeline["Argo Workflow"]
        S1["Config\nValidation"]
        S2["Dataset\nLoading"]
        S3["Model\nTraining"]
        S4["Model\nRegistration"]
        S1 -->|validated config| S2
        S2 -->|dataset path| S3
        S3 -->|training summary| S4
    end

    subgraph MLflow["MLflow"]
        Track["Tracking"]
        Reg["Registry"]
    end

    Params["Argo Submission Parameters"] --> S1
    DS -->|fetch| S2
    S3 -->|periodic checkpoints| CK
    CK -->|resume| S3
    S3 -->|metrics + artifacts| Track
    S4 -->|register best.pt| Reg
```

## Step Architecture

Every pipeline step follows the same internal pattern (Kubestep Python Template):

```
<step>/
├── app/
│   ├── cli.py          # Typer CLI — exposes the `run` command
│   ├── manager.py      # Wires config and services; calls manager.run()
│   ├── models/
│   │   ├── config.py   # Pydantic BaseSettings — reads env vars
│   │   └── domain.py   # Domain-specific models
│   └── services/
│       └── service.py  # Core business logic
├── Dockerfile          # Python 3.12 Alpine, Poetry install
└── pyproject.toml
```

Each step is a fully self-contained Python package with its own Docker image.

## Compute Allocation

GPU and CPU nodes are provisioned dynamically by Karpenter:

| Step | Node Pool | Why |
| --- | --- | --- |
| Config Validation | CPU | Lightweight validation only |
| Dataset Loading | CPU | Network I/O, no GPU needed |
| Model Training | **GPU** | CUDA-accelerated YOLO training |
| Model Registration | CPU | MLflow API calls only |

GPU nodes are provisioned when the training step starts and released when it finishes, keeping costs minimal.
