# Quantization Evaluation Plan (PRD-174)

How we decide whether the INT8 quantization is **correct**, whether the results
are **expected / good quality**, and whether **QAT is worth using** for the pose
model — versus PTQ, or versus not quantizing at all.

> **Key distinction.** "The pipeline is green" only proves the *mechanism* runs.
> It says nothing about model *quality*. This plan is about quality and the
> QAT-vs-PTQ decision, which are separate (and harder) questions.

---

## 0. Current state

| | Status |
|---|---|
| PTQ + QAT mechanism (export → INT8 tflite → register) | ✅ green e2e on cluster |
| Correctness of the *results* | ❌ not evaluated |
| Quality (accuracy vs FP32) | ❌ not evaluated |
| QAT-vs-PTQ (is QAT worth it) | ❌ not evaluated |

---

## 1. Prerequisites (must be true before any evaluation is meaningful)

- **P1 — A properly-trained FP32 model.** The current smoke model has **pose
  mAP = 0** (keypoints never trained). You cannot measure "quality preserved"
  when there is no quality, nor compare PTQ vs QAT starting from noise.
  → Needs one real training run (more data, 100–300 epochs) until pose mAP is
  materially non-zero. **This is the #1 blocker.**
- **P2 — Labelled val available to the quant step.** ✅ *Done* — the shim now
  downloads each frame's YOLO label, so `model.val` has ground truth.
- **P3 — Metrics wired.** ✅ *Done* — `qat_finetune_loss` (convergence) and the
  `fp32_/int8_/delta_` mAP-delta (accuracy) are logged to MLflow.

---

## 2. Level 1 — Correctness ("is QAT running *right*, not just running?")

| Check | How | Pass criterion |
|---|---|---|
| Fine-tune actually learns | `qat_finetune_loss` per epoch (MLflow) | loss trends **down** |
| Output is truly INT8 | inspect tflite tensor dtypes; file size | int8 tensors; **~4× smaller** than FP32 |
| Reproducible | `calibration_seed` fixed | same seed → same result |
| Produces sane output | run INT8 inference on sample frames | detections resemble FP32 (not noise) |

---

## 3. Level 2 — Quality ("are the results expected?")

The only real measure is the **task metric**, not tensor parity. Run `model.val`
on both models and compare:

- **FP32** mAP50 / mAP50-95, **box (B)** and **pose (P)**
- **INT8** mAP50 / mAP50-95, box and pose
- **Accuracy retention = INT8 ÷ FP32** per metric

*Expected for a healthy quantization:* box mAP within ~1–2% of FP32; pose within
an acceptable drop (keypoint regression is more quantization-sensitive).

→ Logged automatically by the mAP-delta feature (`fp32_*`, `int8_*`, `delta_*`,
`int8_map_retention_mAP50B`). The `map_delta_reference_map50b` flag self-documents
when the reference is ~0 (untrained/unlabelled → not meaningful).

---

## 4. Level 3 — The decisive comparison ("is QAT worth it *here*?")

QAT and PTQ produce the **same size (~4×)** and **same INT8 latency** — so the
**only** thing that differs is **accuracy**. The entire decision reduces to one
comparison, on the **same FP32 checkpoint**:

```
FP32 mAP   →   PTQ-INT8 mAP   →   QAT-INT8 mAP   (box + pose)
```

| Outcome | Verdict |
|---|---|
| PTQ-INT8 ≈ FP32 | **QAT NOT worth it** — PTQ is simpler/cheaper, no GPU fine-tune |
| PTQ-INT8 drops, QAT-INT8 recovers it | **QAT worth it** — quantify the recovery |
| Both drop a lot, QAT doesn't help | Neither INT8 is deployable — revisit scheme/calibration |

Pose specifically often benefits more from QAT (precise coordinate regression is
sensitive to quantization) — but that's a hypothesis to **measure**, not assume.

---

## 5. Level 4 — Deployment value ("why quantize at all?")

Measured **on the target edge hardware**, not in the pipeline:

- **Model size**: FP32 `.pt` vs INT8 `.tflite` (expect ~4×).
- **Latency / throughput**: INT8 inference speed on the device.
- **Memory footprint** on the device.

These justify quantization in general; §4 decides *which* quantization.

---

## 6. Decision criteria (the verdict template)

> QAT is worth it for the pose model iff **QAT-INT8 recovers ≥ X pose-mAP points
> over PTQ-INT8** (bringing it within Y% of FP32), and that gain justifies the
> extra **GPU fine-tune cost** (~Z GPU-hours per run).

Fill X / Y / Z from real numbers once P1 is met. Suggested starting bar:
QAT worth it if it recovers **≥ 2–3 pose-mAP points** over PTQ.

---

## 7. Execution steps (the runs to do, in order)

1. **Real training run** → a good FP32 checkpoint (pose mAP ≫ 0).  *(prereq P1)*
2. **PTQ run** on that checkpoint → PTQ-INT8 + its mAP-delta.
3. **QAT run** on the *same* checkpoint → QAT-INT8 + mAP-delta + loss curve.
4. **Compare in MLflow** (§3, §4) — same experiment, three runs to line up.
5. **Latency benchmark** of both tflites on the target device (§5).
6. **Write the verdict** (§6).

---

## 8. Automated vs manual

| Automated (in the pipeline / MLflow) | Manual / external |
|---|---|
| `qat_finetune_loss`, parity, mAP-delta, model sizes | Real training run (choose data/epochs) |
| PTQ & QAT runs, registration | Latency benchmark on target hardware |
| | The final QAT-vs-PTQ verdict |

---

## 9. Known caveats

- Parity `max_abs_error` (headless feature space) is a *sanity* signal, **not** a
  quality metric — the raw pre-NMS tensor mixes coord magnitudes and is dominated
  by background anchors. Trust the **mAP-delta**, not parity, for quality.
- INT8 calibration currently samples a bounded set of frames; for a real run,
  ensure calibration frames are representative (include the target objects).
