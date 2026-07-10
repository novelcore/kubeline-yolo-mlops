# Platform Setup Prerequisites

Confirm you have all of the following before starting.

## KAOS Platform

- A KAOS platform licence is active for your organisation.
- You have admin access to the KAOS control-plane cluster.

## GCP Project

You need a GCP project with GCP IAM permissions for:

| Service | Why |
| --- | --- |
| GKE | Create and manage Kubernetes clusters |
| Compute Engine | Provision GPU and CPU nodes via GKE node pools + the cluster autoscaler |
| Artifact Registry | Store Docker images for pipeline steps |
| GCS | Store datasets, checkpoints, and MLflow artifacts |
| GCP IAM | Create service accounts (Workload Identity) and node service accounts |

## GitHub

- A GitHub organisation where the pipeline template repository will be stored.
- A GitHub personal access token or GitHub App with `repo` scope.

## CLI Tools

Install and configure the following on your local machine:

| Tool | Purpose | Install |
| --- | --- | --- |
| `kubectl` | Interact with Kubernetes clusters | [docs.kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| `gcloud` | GCP CLI for verifying resources | [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) |
| `gh` | GitHub CLI for repository verification | [cli.github.com](https://cli.github.com/) |

Verify each tool is configured:

```bash
kubectl version --client
gcloud auth list
gcloud config get-value project
gh auth status
```

All three commands should return without errors.

---

[:octicons-arrow-right-24: Infrastructure Setup](kaos-setup.md)
