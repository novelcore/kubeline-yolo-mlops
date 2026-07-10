# Configure Your Experiment

When the KAOS platform creates your pipeline, it scaffolds an Argo WorkflowTemplate with **sensible defaults for every parameter**. You configure each experiment at submission time by overriding only the parameters you care about.

## How Configuration Works

The WorkflowTemplate has roughly 60 parameters. They fall into two categories:

| Category | Who Sets It | When |
| --- | --- | --- |
| **Experiment parameters** | You | At workflow submission time, in the Argo UI |
| **Platform parameters** | KAOS (automatic) | Pre-filled when the WorkflowTemplate is created |

Platform parameters (MLflow endpoint, lakeFS endpoint, S3 buckets, etc.) are injected automatically. You never need to change them.

## What You Configure at Submission Time

When you click **Submit** on your WorkflowTemplate in the Argo UI, you see a parameters form. Every parameter has a default value — you only override what you want to change.

The parameters are grouped into these categories:

```
Experiment     ← Description for this run in MLflow
Dataset        ← Which data version to train on and how much
Model          ← Which YOLO variant to use
Training       ← All hyperparameters (epochs, batch size, learning rate, etc.)
Loss Gains     ← Pose-specific loss weights
Early Stopping ← When to stop if the model converges
Checkpointing  ← How often to save progress and how to resume
Augmentation   ← Image augmentation settings
Quantization   ← INT8 quantization mode (none/ptq/qat) and its settings
Registration   ← How to save and promote the trained model
```

## Most Users Only Need a Handful of Parameters

For a typical experiment, you might override just four or five values:

- `dataset-ref` — which lakeFS branch (dataset) to train on
- `epochs` — how long to train
- `model-variant` — which YOLO variant to use
- `batch-size` — how many images per gradient update

Everything else stays at its default. As you get more advanced, you can tune learning rates, augmentation, loss gains, and more.

Continue to the full parameter-by-parameter reference:

[:octicons-arrow-right-24: Parameter Reference](pipeline-config.md)
