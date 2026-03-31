# Prerequisites

Before you can run a pipeline, confirm you have the following three things.

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

## 3. Your `pipeline_config.yaml`

The pipeline is controlled by a single YAML configuration file. Before running a pipeline, you need to fill this file in with your experiment settings.

If you do not have a `pipeline_config.yaml` yet, start here:

[:octicons-arrow-right-24: Configure Your Experiment](../configure/index.md)

---

Once you have Argo access, the MLflow URL, and a filled-in `pipeline_config.yaml`, you are ready to run.

[:octicons-arrow-right-24: Run a Pipeline — Argo Workflows UI](../run/argo-ui.md)
