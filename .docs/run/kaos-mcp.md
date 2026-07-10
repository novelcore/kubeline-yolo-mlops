# Run via KAOS MCP Agent

!!! note "Coming Soon"
    The KAOS MCP agent integration for pipeline submission is currently in development and not yet available.

## What It Will Do

The KAOS MCP agent will let you submit, monitor, and manage pipeline runs using natural language through any MCP-compatible client (including Claude Code).

Example interactions:

```
You: Submit a new YOLO training run with batch_size 32 and 200 epochs on the io-spacecraft dataset.

KAOS: Submitting workflow to ml-example-project namespace...
      Run ID: spacecraft-pose-v2-20260401-143022
      Status: RUNNING
      Monitor at: https://argo.your-org.kaos.io/...
```

```
You: What's the status of my last training run?

KAOS: Run spacecraft-pose-v2-20260401-143022 is currently on epoch 47/200.
      val/mAP50: 0.61 (↑ from 0.58 last epoch)
      Estimated completion: 4 hours 22 minutes.
```

## In the Meantime

Use the [Argo Workflows UI](argo-ui.md) to submit and monitor pipeline runs.

---

Watch the [repository](https://github.com/novelcore/kubeline-yolo-mlops) for updates on MCP agent availability.
