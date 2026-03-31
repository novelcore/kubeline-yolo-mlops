# Run via GitHub PR

You can trigger a pipeline run by opening a pull request that changes `pipeline_config.yaml`. This is useful when you want a Git audit trail of every experiment configuration.

## How It Works

1. The pipeline repository has a GitHub Actions workflow (`dag-update.yml`) that monitors changes to `pipeline_config.yaml`.
2. When you open a PR that modifies `pipeline_config.yaml`, the workflow re-renders the Argo `WorkflowTemplate` with the new parameters.
3. When the PR is merged to the default branch, the updated WorkflowTemplate is applied to the cluster — and an Argo workflow is submitted automatically.

## Step-by-Step

1. **Fork or branch** from the pipeline repository's default branch.

2. **Edit** `pipeline_config.yaml` with your new experiment settings.

3. **Commit** your changes:

    ```bash
    git add pipeline_config.yaml
    git commit -m "experiment: increase epochs to 200, adjust learning rate"
    ```

4. **Open a pull request** against the default branch.

    The PR triggers a dry-run that validates your configuration and shows what the updated WorkflowTemplate will look like.

5. **Merge the PR** once the validation checks pass.

    Within a minute, the updated WorkflowTemplate is applied to the cluster and a new workflow run is submitted automatically.

6. **Monitor** the run in the [Argo Workflows UI](argo-ui.md#step-5-watch-the-dag).

## When to Use This Approach

Use the GitHub PR approach when:

- You want every experiment configuration tracked in Git history
- You are collaborating with a team and want PR review before running experiments
- You want automated runs triggered by config changes (e.g., in a CI/CD pipeline)

Use the [Argo UI approach](argo-ui.md) for faster iteration when you are working alone.
