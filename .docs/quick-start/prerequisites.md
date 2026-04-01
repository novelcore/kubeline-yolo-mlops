# Prerequisites

Before you can run a pipeline, confirm you have the following two things.

## 1. SSO Access to Argo Workflows

Your organisation's KAOS account grants access to Argo Workflows via Single Sign-On (SSO).

Ask your platform administrator for the Argo Workflows URL. It will look like:

```
https://argo.your-org.kaos.io
```

Log in with your organisation account. You should land on the Argo Workflows dashboard.

!!! tip "Can't log in?"
    Contact your platform administrator and confirm your account has been added to the organisation on the KAOS platform.

## 2. Your MLflow URL

MLflow is where your experiment results, metrics, and trained models are stored.

Ask your platform administrator for your project's MLflow URL:

```
https://mlflow.your-project.kaos.io
```

You will use this address to monitor runs and access your trained models after each pipeline execution.

---

Once you have Argo access and the MLflow URL, you are ready to run. Your WorkflowTemplate is already configured with sensible defaults for all parameters — you just override what you need at submission time.

[:octicons-arrow-right-24: Configure Your Experiment](../configure/index.md)
