# Run via Argo Workflows UI

The Argo Workflows UI is the primary way to submit and monitor pipeline runs.

## Before You Start

- [ ] You have filled in `pipeline_config.yaml` — see [Configure Your Experiment](../configure/index.md)
- [ ] You can log in to the Argo Workflows UI (ask your administrator for the URL if you do not have it)

## Step 1: Log In

Open the Argo Workflows URL in your browser and log in using your organisation's SSO credentials.

You will land on the **Workflows** page, which lists all workflow runs in the selected namespace.

## Step 2: Select Your Namespace

In the top navigation bar, find the **Namespace** dropdown. Select the namespace for your project — it follows the pattern `ml-<project-name>`.

!!! tip "Can't find your namespace?"
    Contact your platform administrator. The namespace is created automatically when a KubeProject is provisioned.

## Step 3: Find the WorkflowTemplate

In the left sidebar, click **Workflow Templates**. You will see a list of available templates.

Find the template named after your pipeline (e.g., `kubeline-yolo-mlops-<project>-<app>`). This template was generated automatically by KAOS when your KubeApp was created.

## Step 4: Submit the Workflow

1. Click on your WorkflowTemplate to open it.
2. Click the **Submit** button (top right).
3. A **Submit Workflow** dialog appears. You need to provide your pipeline configuration as a parameter.

In the **pipeline-config** parameter field, paste the full contents of your `pipeline_config.yaml` file.

!!! info "Parameters"
    The WorkflowTemplate accepts your `pipeline_config.yaml` content as a parameter. The platform routes it to each step automatically.

4. Click **Submit**.

## Step 5: Watch the DAG

After submitting, you are taken to the **Workflow detail page**. This shows the pipeline as a directed acyclic graph (DAG):

```
[Config Validation] → [Dataset Loading] → [Model Training] → [Model Registration]
```

Each node changes colour as it progresses:

| Colour | Meaning |
| --- | --- |
| Grey | Pending — waiting to start |
| Yellow / pulsing | Running |
| Green | Succeeded |
| Red | Failed |

Click any node to see its logs in real time.

## Step 6: Read Step Logs

Click a running or completed step to open its detail panel. Switch to the **Logs** tab to see the step's standard output.

!!! tip "What to look for"
    - **Config Validation:** `✓ Config validated` — confirms your YAML is valid
    - **Dataset Loading:** shows download progress and number of images found
    - **Model Training:** shows per-epoch loss and mAP metrics
    - **Model Registration:** shows the registered model name and version number

## Step 7: Confirm Completion

When all four nodes are green, your pipeline run is complete. The trained model is now registered in MLflow.

Continue to check your results:

[:octicons-arrow-right-24: MLflow Dashboard](../monitor/mlflow.md)
