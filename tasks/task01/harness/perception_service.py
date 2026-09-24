"""The perception models, served on their own socket. ROOT-ONLY, like the daemon.

A SERVICE, not a library, for two reasons. SAM3 and Contact-GraspNet take tens of seconds
to load and several GB of GPU memory, and the agent runs many short scripts -- an
in-process import would pay that on every one of them, and the L2 comparison would partly
measure model load time. And it keeps one story about what the agent's uid can open: a
service with a contract, not a tree of modules it can monkeypatch.

NOT METERED, and it cannot be: this holds no env, imports nothing from the metering
daemon, and answers only from the arrays in the request, so it cannot advance an episode.
The agent pays for it in wall clock.

Started by the image entrypoint at the levels that ship the library, and by nothing else.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
from typing import Any

import numpy as np

from . import perception_protocol as P

SOCKET = "/run/rlebench/perception.sock"

# Where the vendored models were staged in the image. Both carry their own weights: SAM3
# resolves `facebook/sam3` through the HuggingFace cache baked beside it (HF_HUB_OFFLINE
# is set, so nothing reaches the network at run time), and Contact-GraspNet's checkpoint
# ships inside its own repo.
VENDOR = os.environ.get("RLEBENCH_PERCEPTION_DIR", "/opt/perception")
CGN_CHECKPOINT = "contact_graspnet_pytorch/checkpoints/contact_graspnet"

_MODELS: dict[str, Any] = {}
_GPU = threading.Lock()      # one model at a time: two forwards race for GPU memory


def _device() -> str:
    import torch
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    return "cuda" if torch.cuda.is_available() else "cpu"


def load() -> dict:
    """Build both models once. Called before the socket is bound, so a client that
    connects at all is talking to something ready."""
    import torch

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = _device()
    model = build_sam3_image_model(device=device, eval_mode=True,
                                   enable_inst_interactivity=True)
    _MODELS["sam3"] = Sam3Processor(model, device=device)

    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.config_utils import load_config
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    ckpt_dir = os.path.join(VENDOR, CGN_CHECKPOINT)
    config = load_config(ckpt_dir, batch_size=1)
    estimator = GraspEstimator(config)
    CheckpointIO(checkpoint_dir=os.path.join(ckpt_dir, "checkpoints"),
                 model=estimator.model).load("model.pt")
    _MODELS["cgn"] = estimator

    # Inference only, and deterministically: these run inside a scored comparison, so a
    # frame must segment the same way twice (CLAUDE.md invariant #3).
    torch.set_grad_enabled(False)
    _MODELS["device"] = device
    return _MODELS


def _results(output: dict) -> list[dict]:
    """SAM3's tensors -> the list the agent's client documents. Highest score first."""
    masks, boxes, scores = (output.get("masks"), output.get("boxes"),
                            output.get("scores"))
    if masks is None or boxes is None or scores is None:
        return []
    masks = _np(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    boxes, scores = _np(boxes), _np(scores)
    out = [{"mask": masks[i] > 0, "box": [float(x) for x in boxes[i]],
            "score": float(scores[i])} for i in range(len(scores))]
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def _np(tensor) -> np.ndarray:
    """Tensor -> ndarray, bfloat16 included (numpy has no bfloat16)."""
    import torch

    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        return (t.float() if t.dtype == torch.bfloat16 else t).numpy()
    return np.asarray(tensor)


def _segment(rgb: np.ndarray, *, text: str | None = None, boxes=None, labels=None,
             threshold: float = 0.5) -> list[dict]:
    """One frame, one prompt. Text, or boxes -- the model has no point prompt."""
    import torch
    from PIL import Image

    proc = _MODELS["sam3"]
    frame = np.asarray(rgb, dtype=np.uint8)
    height, width = frame.shape[:2]
    image = Image.fromarray(frame)
    with _GPU, torch.autocast("cuda" if _MODELS["device"] == "cuda" else "cpu",
                              dtype=torch.bfloat16):
        proc.set_confidence_threshold(float(threshold))
        state = proc.set_image(image)
        if text is not None:
            output = proc.set_text_prompt(state=state, prompt=str(text))
        else:
            output = None
            for box, label in zip(boxes or [], labels or []):
                output = proc.add_geometric_prompt(
                    box=_normalised_box(box, width, height), label=bool(label),
                    state=state)
    return _results(output or {})


def _normalised_box(box, width: int, height: int) -> list:
    """Agent-facing pixel [x1, y1, x2, y2] -> the model's normalised [cx, cy, w, h].

    Converted here rather than in the client: the pixel corners are the format results
    come back in, so a box from one call can be fed straight into the next.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
            abs(x2 - x1) / width, abs(y2 - y1) / height]


def _plan_grasp(points: np.ndarray, segments: dict | None,
                max_grasps: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidates for a cloud, IN THE CLOUD'S OWN FRAME -- nothing here re-frames them."""
    cloud = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(cloud) < 32:
        return np.zeros((0, 4, 4)), np.zeros((0,))
    parts = {str(k): np.asarray(v, dtype=np.float32).reshape(-1, 3)
             for k, v in (segments or {}).items()}
    import torch

    # SAM3's tracking predictor enters a bf16 autocast at build time and never exits
    # it; CGN must not inherit it, or its float32 outputs come back as bfloat16.
    with _GPU, torch.autocast("cuda" if _MODELS.get("device") == "cuda" else "cpu",
                              enabled=False):
        grasps, scores, _, _ = _MODELS["cgn"].predict_scene_grasps(
            cloud, pc_segments=parts, local_regions=bool(parts),
            filter_grasps=bool(parts), forward_passes=1)
    # predict_scene_grasps returns dicts keyed by segment (or by -1 for the whole scene).
    flat_g = [g for group in grasps.values() for g in np.asarray(group).reshape(-1, 4, 4)]
    flat_s = [float(s) for group in scores.values() for s in np.asarray(group).reshape(-1)]
    if not flat_g:
        return np.zeros((0, 4, 4)), np.zeros((0,))
    order = np.argsort(flat_s)[::-1][:int(max_grasps)]
    return np.asarray(flat_g)[order], np.asarray(flat_s)[order]


def handle(msg: dict) -> dict:
    op = msg.get("op")
    if op == "info":
        return {"ok": True, "device": _MODELS.get("device"),
                "models": sorted(k for k in _MODELS if k != "device")}
    if op == "segment_text":
        return {"ok": True, "results": _segment(
            msg["rgb"], text=msg.get("text", ""),
            threshold=float(msg.get("threshold", 0.5)))}
    if op == "segment_boxes":
        return {"ok": True, "results": _segment(
            msg["rgb"], boxes=msg.get("boxes", []), labels=msg.get("labels", []),
            threshold=float(msg.get("threshold", 0.5)))}
    if op == "plan_grasp":
        grasps, scores = _plan_grasp(msg["points"], msg.get("segments"),
                                     int(msg.get("max_grasps", 50)))
        return {"ok": True, "grasps": grasps, "scores": scores}
    raise ValueError(f"unknown op {op!r}")


def serve(path: str = SOCKET, once: bool = False) -> None:
    """Bind and answer. One connection at a time -- these calls are GPU-bound anyway, and
    the lock above already serialises the part that matters."""
    if os.path.exists(path):
        os.unlink(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    # 0666 like the metering socket: reaching it spends the agent's own wall clock and
    # nothing else, so there is nothing to protect.
    os.chmod(path, 0o666)
    server.listen(8)
    while True:
        conn, _ = server.accept()
        conn.settimeout(30)
        try:
            msg = P.LineReader(conn).read()
            if msg is None:
                continue
            try:
                reply = handle(msg)
            except Exception as exc:  # noqa: BLE001
                # Never die on a bad request: the agent can send anything, and a
                # perception service that fell over would take the rest of the run with
                # it while looking like a harness fault.
                reply = {"ok": False, "error": "perception request failed"}
            P.send(conn, reply)
        except (ConnectionError, OSError, ValueError, P.ProtocolError):
            pass
        finally:
            conn.close()
        if once:
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", default=SOCKET)
    args = ap.parse_args(argv)
    load()
    print(f"[perception] ready on {args.socket} ({_MODELS['device']})", flush=True)
    serve(args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
