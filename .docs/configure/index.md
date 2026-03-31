# Configure Your Experiment

Before submitting a pipeline run, you fill in a single YAML file: `pipeline_config.yaml`.

## Two Layers of Configuration

The pipeline has two configuration layers. As a data scientist, **you only need to touch one of them**.

| Layer | File | What It Controls | Who Sets It |
| --- | --- | --- | --- |
| **Pipeline config** | `pipeline_config.yaml` | Your experiment — dataset, model, hyperparameters, registration | You |
| **Runtime config** | Environment variables (`.env`) | Platform plumbing — server URLs, credentials, timeouts | Platform administrator |

The runtime configuration is handled by the KAOS platform automatically. The only file you need to edit is `pipeline_config.yaml`.

## Getting the Config File

The `pipeline_config.yaml` is stored in the pipeline's Git repository. To start a new experiment:

1. Copy the example config from the repository:

    ```bash
    cp pipeline_config.example.yaml pipeline_config.yaml
    ```

2. Edit `pipeline_config.yaml` with your experiment settings.

3. Submit the pipeline (see [Run a Pipeline](../run/argo-ui.md)).

## What's in pipeline_config.yaml

The file has seven sections:

```
experiment    ← Name and tags for this run in MLflow
dataset       ← Where to load data from and how much
model         ← Which YOLO variant to use
training      ← All hyperparameters (epochs, batch size, learning rate, etc.)
checkpointing ← How often to save progress and where
augmentation  ← Image augmentation settings
registration  ← How to save the trained model
```

Continue to the full field-by-field walkthrough:

[:octicons-arrow-right-24: pipeline_config.yaml Reference](pipeline-config.md)
