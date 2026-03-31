# Platform Setup Prerequisites

Confirm you have all of the following before starting.

## KAOS Platform

- A KAOS platform licence is active for your organisation.
- You have admin access to the KAOS control-plane cluster.

## AWS Account

You need an AWS account with IAM permissions for:

| Service | Why |
| --- | --- |
| EKS | Create and manage Kubernetes clusters |
| EC2 | Provision GPU and CPU nodes via Karpenter |
| ECR | Store Docker images for pipeline steps |
| S3 | Store datasets, checkpoints, and MLflow artifacts |
| IAM | Create service accounts and node roles |

## GitHub

- A GitHub organisation where the pipeline template repository will be stored.
- A GitHub personal access token or GitHub App with `repo` scope.

## CLI Tools

Install and configure the following on your local machine:

| Tool | Purpose | Install |
| --- | --- | --- |
| `kubectl` | Interact with Kubernetes clusters | [docs.kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| `aws` | AWS CLI for verifying resources | [aws.amazon.com/cli](https://aws.amazon.com/cli/) |
| `gh` | GitHub CLI for repository verification | [cli.github.com](https://cli.github.com/) |

Verify each tool is configured:

```bash
kubectl version --client
aws sts get-caller-identity
gh auth status
```

All three commands should return without errors.

---

[:octicons-arrow-right-24: Infrastructure Setup](kaos-setup.md)
