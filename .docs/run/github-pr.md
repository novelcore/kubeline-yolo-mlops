# Run via GitHub PR

You can use a pull request workflow to propose and review experiment configurations before submitting them. This is useful when you want a Git audit trail or team review of experiment changes.

## How It Works

1. Create a branch in the pipeline repository.
2. Update any configuration or pipeline code you want to change.
3. Open a PR for team review.
4. After the PR is merged and any pipeline code changes are applied, submit your experiment from the [Argo Workflows UI](argo-ui.md) with the parameters you want.

The actual experiment configuration (hyperparameters, dataset version, model variant, etc.) is set at workflow submission time in the Argo UI — not through files in the repository. The PR workflow is for reviewing pipeline code changes, not for configuring individual experiment runs.

## When to Use This Approach

Use the GitHub PR approach when:

- You are changing pipeline code, step scripts, or the WorkflowTemplate structure
- You are collaborating with a team and want PR review before deploying pipeline changes
- You want Git history for structural changes to the pipeline

Use the [Argo UI approach](argo-ui.md) for configuring and submitting individual experiment runs — that is where you set parameters like epochs, learning rate, dataset version, and model variant.
