# Platform Setup Overview

This track walks you through deploying the Kubeline YOLO MLOps pipeline template on KAOS from scratch.

By the end of this track, you will have:

- A Kubernetes cluster (EKS) with GPU support via Karpenter
- MLflow and LakeFS deployed and connected to S3
- An Argo Workflows WorkflowTemplate ready to submit
- Access URLs for Argo and MLflow

## What Gets Created

The KAOS platform provisions ML infrastructure in five layers, each building on the previous:

```mermaid
graph TD
    A["KubeOrg\nAWS + GitHub foundations"] --> B["KubePool\nEKS cluster + GPU operators"]
    B --> C["KubeProject\nMLflow · LakeFS · S3 bucket"]
    C --> D["KubeAppTemplate\nPipeline template definition"]
    D --> E["KubeApp\nYour running pipeline instance"]

    style A fill:#4a148c,stroke:#7b1fa2,color:#fff
    style B fill:#4a148c,stroke:#7b1fa2,color:#fff
    style C fill:#4a148c,stroke:#7b1fa2,color:#fff
    style D fill:#4a148c,stroke:#7b1fa2,color:#fff
    style E fill:#6a1b9a,stroke:#9c27b0,color:#fff,stroke-width:2px
```

!!! info "Each layer is a prerequisite for the next"
    If a layer is missing or not fully ready, the next layer will not reconcile. Work through the steps in order.

## Estimated Time

| Step | Estimated Time |
| --- | --- |
| KubeOrg | 5 minutes |
| KubePool (EKS cluster) | 20–30 minutes |
| KubeProject | 10 minutes |
| KubeAppTemplate | 5 minutes |
| KubeApp | 10 minutes |

Start with [Prerequisites](prerequisites.md) before creating any resources.
