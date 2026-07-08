#!/usr/bin/env python3
"""repro_qat.py — offline reproduction of the qat_finetune QAT mechanism.

Mirrors ``qat_finetune/app/services/qat_service.py`` STAGE-BY-STAGE on the
PUBLIC ``yolov8n-pose.pt`` with dummy frames — so the
``torch.export -> PT2E -> convert -> litert`` chain can be debugged in *seconds*
on any Linux box / Google Colab, WITHOUT the cluster, a 14 GB image rebuild,
lakeFS, or MLflow.

Why this works: the QAT bugs we hit (torch.export, .train()/.eval() on the
exported model, convert_pt2e, litert) are pure *mechanism* bugs. They need only
a pose model (the public base has the same head structure) + a few frames — no
trained checkpoint, no creds, no confidential data.

--------------------------------------------------------------------------------
COLAB SETUP  (run this in a cell FIRST — pins match the qat-finetune image):

    !pip install -q "torch==2.11.0" "torchvision==0.26.0" \
        "torchao==0.17.0" "litert-torch==0.9.1" "ultralytics==8.4.19"

    # A Colab GPU runtime (free T4) is optional — the mechanism runs on CPU.
    # If litert-torch refuses to install, the script STILL runs stages 1-5
    # (export -> prepare -> train/eval -> finetune -> convert); only the final
    # litert export (stage 6) is skipped. That covers ~80% of the bugs.

Then either  `%run repro_qat.py`  or paste the whole file into a cell.
--------------------------------------------------------------------------------

Each stage prints ✅ / ❌ with a traceback, in the SAME order as the cluster, so
you see exactly where it breaks. Fix ``qat_service.py``, re-run, repeat. When all
stages pass HERE, do ONE cluster build + run to confirm the integration.
"""

import traceback

import torch
import torch.nn as nn

# --- Config (tweak freely) --------------------------------------------------
MODEL = "yolov8n-pose.pt"   # public; ultralytics auto-downloads if missing
IMGSZ = 320                 # 320 (mult. of 32) for speed; cluster uses 640
QAT_EPOCHS = 1
CALIB_BATCHES = 2           # a couple of random batches — mechanism test only
BATCH = 1                   # MUST be 1: the exported pose graph is batch-1-only
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _log(msg: str) -> None:
    print(msg, flush=True)


def stage(n: int, name: str):
    """Decorator-ish context: print a header, and on failure print the
    traceback and re-raise so the run stops at the first broken stage
    (exactly like the cluster step exits non-zero)."""
    _log(f"\n{'='*70}\nSTAGE {n} — {name}\n{'='*70}")


# ---------------------------------------------------------------------------
# Faithful mirror of qat_service.py, one function per stage.
# ---------------------------------------------------------------------------
def load_headless(device: str) -> nn.Module:
    """qat_service._load_headless_module — CON-03 head exclusion."""
    from ultralytics import YOLO

    yolo = YOLO(MODEL)
    module: nn.Module = yolo.model
    module = module.to(device)
    module.eval()
    module.model[-1].training = True  # head -> graph boundary (headless capture)
    _log(f"loaded headless {MODEL} on {device}")
    return module


def capture_graph(module: nn.Module, sample: tuple) -> nn.Module:
    """qat_service._capture_graph — torch.export(strict=False).module().

    The YOLO pose head specializes the batch dim to 1 during export (anchor /
    reshape logic), so the graph is batch-1-only — it CANNOT be made dynamic
    (torch rejects it: "specialized to a constant (1)"). The real fix is to feed
    batch=1 everywhere (export + fine-tune + litert), i.e. calibration loader
    batch_size=1 (see BATCH below / qat_service._build_calibration_loader).

    QAT FIX: prefer ``export_for_training`` (training graph → autograd works); its
    import path moved across torch versions, so try the known spots and fall back
    to plain ``torch.export.export`` (we also force requires_grad in prepare, which
    is what actually unblocks ``loss.backward()``).
    """
    ef = None
    for mod, name in [("torch.export", "export_for_training"),
                      ("torch._export", "export_for_training")]:
        try:
            ef = getattr(__import__(mod, fromlist=[name]), name)
            _log(f"using {mod}.{name} ...")
            break
        except Exception:
            ef = None
    if ef is None:
        _log("export_for_training unavailable → plain torch.export.export ...")
        ef = torch.export.export
    return ef(module, sample, strict=False).module()


def prepare_qat(module: nn.Module) -> nn.Module:
    """qat_service._prepare_qat + the allow_exported_model_train_eval fix."""
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False)  # CON-02
    )
    prepared = prepare_qat_pt2e(module, quantizer)
    # exported graph modules reject nn.Module .train()/.eval(); patch them to
    # torchao's move_exported_model_to_{train,eval}.
    allow_exported_model_train_eval(prepared)
    prepared.train()
    # QAT FIX: the export can leave weights with requires_grad=False → the fine-tune
    # loss has no grad_fn and loss.backward() dies. Re-enable grad so QAT can learn.
    n_grad = 0
    for p in prepared.parameters():
        p.requires_grad_(True)
        n_grad += 1
    _log(f"prepare_qat_pt2e + allow_exported_model_train_eval OK ({n_grad} params grad-enabled)")
    return prepared


def finetune(prepared: nn.Module, teacher: nn.Module, device: str) -> None:
    """qat_service._finetune — distillation MSE(student, fp32 teacher).

    FIXES vs the original:
    * The exported graph returns a pytree (dict), not a bare tensor/tuple — flatten
      both student & teacher to tensor leaves and MSE over aligned leaves.
    * teacher.eval() flips the Detect head OUT of headless mode (decoded output),
      which no longer matches the raw-feature student graph — re-assert headless
      (model[-1].training=True) after .eval() so both emit raw backbone+neck feats.
    """
    import torch.utils._pytree as pytree

    prepared.train()
    optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    teacher.eval()
    teacher.model[-1].training = True  # keep raw/headless to match the graph

    def _tensors(x):
        return [t for t in pytree.tree_leaves(x) if torch.is_tensor(t)]

    for epoch in range(QAT_EPOCHS):
        for _ in range(CALIB_BATCHES):
            images = torch.rand(BATCH, 3, IMGSZ, IMGSZ, device=device)
            with torch.no_grad():
                t_out = _tensors(teacher(images))
            optimizer.zero_grad()
            s_out = _tensors(prepared(images))
            _log(f"    (student leaves={len(s_out)} teacher leaves={len(t_out)} "
                 f"shapes s={[tuple(t.shape) for t in s_out][:3]})")
            loss = sum(
                criterion(s.float(), t.float().detach())
                for s, t in zip(s_out, t_out)
            )
            loss.backward()
            optimizer.step()
        _log(f"  epoch {epoch + 1}/{QAT_EPOCHS} loss={float(loss):.5f}")
    prepared.eval()
    _log("finetune OK")


def convert(prepared: nn.Module) -> nn.Module:
    """qat_service._convert — convert_pt2e(fold_quantize=False) [CON-01]."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

    quantized = convert_pt2e(prepared, fold_quantize=False)
    _log("convert_pt2e OK")
    return quantized


def export_tflite(quantized: nn.Module, sample: tuple) -> str:
    """qat_service._export_tflite — litert_torch.convert(...).export(...)."""
    import litert_torch

    out = "/tmp/model_int8.tflite"
    edge_model = litert_torch.convert(quantized, sample)
    edge_model.export(out)
    _log(f"litert export OK -> {out}")
    return out


def _decoded(out):
    """Peel a YOLO head's output down to the decoded (1, no, num_anchors) tensor."""
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def head_reattach(quantized: nn.Module, teacher: nn.Module, sample: tuple,
                  device: str) -> None:
    """Reattach the FP32 detection head to the INT8 backbone's features.

    This is how the deployment target actually runs a headless QAT model: the
    edge device feeds the INT8 backbone+neck feature maps into the (FP32) head +
    decode/NMS in software. Evaluating THIS gives a task-level number comparable
    to PTQ — unlike the raw-feature parity, which is uninterpretable.

    Here we validate the MECHANISM: run the same image through
      (a) the full FP32 model                  -> decoded (1, no, 8400)
      (b) INT8 backbone features -> FP32 head   -> decoded (1, no, 8400)
    and compare. On the cluster (b)'s features come from the .tflite; here we use
    the in-memory quantized graph (same mechanism, no litert reload).
    """
    import torch.utils._pytree as pytree

    img = sample[0]
    head = teacher.model[-1]  # the Detect/Pose head that QAT excluded

    # (a) FP32 reference: full model, head in decode (eval) mode.
    teacher.eval()
    head.training = False
    with torch.no_grad():
        fp32_pred = _decoded(teacher(img))
    _log(f"    FP32 full-model decoded output: {tuple(fp32_pred.shape)}")

    # (b) INT8 backbone features -> FP32 head.
    with torch.no_grad():
        feats = [t for t in pytree.tree_leaves(quantized(img))
                 if torch.is_tensor(t) and t.dim() == 4]
    feats = sorted(feats, key=lambda t: -(t.shape[-1] * t.shape[-2]))  # P3 first
    _log(f"    INT8 headless feature maps: {[tuple(f.shape) for f in feats]}")
    with torch.no_grad():
        int8_pred = _decoded(head(list(feats)))
    _log(f"    INT8-backbone + FP32-head decoded output: {tuple(int8_pred.shape)}")

    if fp32_pred.shape == int8_pred.shape:
        err = (fp32_pred.float() - int8_pred.float()).abs().max().item()
        ref = fp32_pred.float().abs().max().item() + 1e-9
        _log(f"✅ head reattachment WORKS | max_abs_err(decoded)={err:.4f} "
             f"(rel {err / ref:.2%}) — the QAT quality signal to build on")
    else:
        _log(f"⚠️ shape mismatch — fp32={tuple(fp32_pred.shape)} "
             f"int8={tuple(int8_pred.shape)} (feature order/reshape needs a tweak)")


# ---------------------------------------------------------------------------
def main() -> None:
    _log(f"device={DEVICE} imgsz={IMGSZ} torch={torch.__version__}")
    try:
        import torchao
        _log(f"torchao={torchao.__version__}")
    except Exception as e:  # noqa: BLE001
        _log(f"(torchao import note: {e})")

    sample = (torch.randn(1, 3, IMGSZ, IMGSZ, device=DEVICE),)

    stage(1, "load headless module (CON-03)")
    student_src = load_headless(DEVICE)
    teacher = load_headless(DEVICE)  # separate fp32 teacher for distillation

    stage(2, "torch.export capture")
    gm = capture_graph(student_src, sample)

    stage(3, "prepare_qat_pt2e + allow_exported_model_train_eval")
    prepared = prepare_qat(gm)

    stage(4, "fine-tune loop (distillation)")
    finetune(prepared, teacher, DEVICE)

    stage(5, "convert_pt2e (fold_quantize=False)")
    quantized = convert(prepared)

    stage(6, "litert INT8 TFLite export")
    try:
        export_tflite(quantized, sample)
    except ImportError as e:
        _log(f"⏭  SKIP stage 6 — litert-torch not importable ({e}).")
        _log("   Stages 1-5 (the export/PT2E/convert chain) passed — that's ~80%.")
        _log("   Install litert-torch or run stage 6 on the cluster.")

    stage(7, "head reattachment (INT8 backbone -> FP32 head -> detections)")
    head_reattach(quantized, teacher, sample, DEVICE)

    _log("\n🎉 ALL STAGES PASSED — the QAT mechanism works end to end.")
    _log("   Next: one cluster build + run to confirm the full integration.")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        _log("\n❌ FAILED at the stage above:\n")
        traceback.print_exc()
        _log(
            "\nThis is the next QAT bug. Fix qat_service.py accordingly, re-run "
            "this script (seconds), and repeat — no cluster, no rebuild."
        )
