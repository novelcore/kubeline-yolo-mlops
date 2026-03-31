# Kubeline YOLO MLOps

An end-to-end machine learning pipeline for YOLO pose estimation, built to run on the [KAOS platform](https://novelcore.github.io/kubecore-operator/).

The pipeline takes a configuration file, loads your dataset from S3 or LakeFS, trains a YOLO pose model on a GPU node, and registers the result in MLflow — all fully automated through Argo Workflows.

## Pipeline Overview

```mermaid
graph LR
    A["⚙️ Config\nValidation"] --> B["📦 Dataset\nLoading"]
    B --> C["🧠 Model\nTraining"]
    C --> D["📋 Model\nRegistration"]

    style A fill:#4a148c,stroke:#7b1fa2,color:#fff
    style B fill:#4a148c,stroke:#7b1fa2,color:#fff
    style C fill:#6a1b9a,stroke:#9c27b0,color:#fff,stroke-width:2px
    style D fill:#4a148c,stroke:#7b1fa2,color:#fff
```

| Step | Compute | Typical Duration |
| --- | --- | --- |
| Config Validation | CPU | Seconds |
| Dataset Loading | CPU | Minutes |
| Model Training | GPU | Hours to days |
| Model Registration | CPU | Seconds |

## Which track are you on?

<div class="grid cards" markdown>

- :material-rocket-launch-outline: **Quick Start**

    ---

    You already have a KAOS environment running with Argo Workflows and MLflow available.

    Jump straight to configuring your first experiment.

    [:octicons-arrow-right-24: Prerequisites](quick-start/prerequisites.md)

- :material-wrench-outline: **Platform Setup**

    ---

    You need to deploy the pipeline template on KAOS from scratch.

    Start here to provision the full infrastructure stack.

    [:octicons-arrow-right-24: Platform Setup Overview](platform-setup/index.md)

</div>

---

> **Questions?** Open an issue on [GitHub](https://github.com/novelcore/kubeline-yolo-mlops/issues).
