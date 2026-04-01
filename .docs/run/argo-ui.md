# Run via Argo Workflows UI

The Argo Workflows UI is the primary way to submit and monitor pipeline runs.

## Before You Start

- [ ] You can log in to the Argo Workflows UI (ask your administrator for the URL if you do not have it)
- [ ] You know which experiment parameters you want to override (see [Configure Your Experiment](../configure/index.md))

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
3. A **Submit Workflow** dialog appears with a **parameters form**. Every parameter has a default value already filled in.

**Platform parameters** (like `mlflow-endpoint`, `lakefs-endpoint`, `s3-artifacts-bucket`) are pre-filled by the platform. Do not change these.

**Experiment parameters** are what you override to configure your run. For example, to run a quick 50-epoch experiment on a smaller model with a data subset:

| Parameter | Override to |
| --- | --- |
| `epochs` | `50` |
| `model-config` | `yolov8s-pose.pt` |
| `dataset-version` | `v2` |
| `batch-size` | `32` |
| `dataset-sample-size` | `500` |

Leave everything else at its default. You only need to change the parameters relevant to your experiment.

4. Click **Submit**.

!!! tip "Most runs only need a few overrides"
    The defaults are tuned for a reasonable baseline. For a first run, you might only change `dataset-version` and `epochs`. See the [full parameter reference](../configure/pipeline-config.md) for what each parameter does.

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
    - **Config Validation:** `✓ Config validated` — confirms your parameters are valid
    - **Dataset Loading:** shows download progress and number of images found
    - **Model Training:** shows per-epoch loss and mAP metrics
    - **Model Registration:** shows the registered model name and version number

## Step 7: Confirm Completion

When all four nodes are green, your pipeline run is complete. The trained model is now registered in MLflow.

Continue to check your results:

[:octicons-arrow-right-24: MLflow Dashboard](../monitor/mlflow.md)
