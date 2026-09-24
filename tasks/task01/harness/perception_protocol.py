"""Wire protocol between the agent's client and the metering daemon.

Newline-delimited JSON, with numpy arrays carried as base64 so observations
(camera images plus proprioception) survive the hop without pickle.

TRUST ASYMMETRY -- the reason this module exists rather than `pickle`:

    daemon -> agent   large payloads (observations). The agent may decode however
                      it likes; it is trusting the daemon, which is fine.
    agent  -> daemon  actions and control messages. This direction crosses INTO the
                      trusted side, so it must never be unpickled and never
                      eval'd. Everything is JSON, and arrays are validated for
                      dtype, shape and finiteness before reaching the simulator.

An action that fails validation is rejected outright rather than coerced: silently
clipping a malformed action would let a broken policy look better than it is, and
NaNs propagate into MuJoCo as hard-to-attribute failures.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # generous for image obs, bounded against OOM
_ARRAY_TAG = "__ndarray__"

# Only dtypes the simulator legitimately produces/consumes. Anything else (object
# arrays above all) is refused.
_ALLOWED_DTYPES = frozenset(
    {"uint8", "int32", "int64", "float32", "float64", "bool"}
)


class ProtocolError(RuntimeError):
    pass


def encode(obj: Any) -> Any:
    """Recursively make `obj` JSON-safe, tagging numpy arrays."""
    if isinstance(obj, np.ndarray):
        return {
            _ARRAY_TAG: True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def decode(obj: Any) -> Any:
    """Inverse of `encode`. Rejects unknown dtypes and size-inconsistent arrays."""
    if isinstance(obj, dict):
        if obj.get(_ARRAY_TAG):
            dtype = str(obj["dtype"])
            if dtype not in _ALLOWED_DTYPES:
                raise ProtocolError(f"refusing array dtype {dtype!r}")
            shape = tuple(int(v) for v in obj["shape"])
            if any(d < 0 for d in shape):
                raise ProtocolError(f"bad array shape {shape}")
            raw = base64.b64decode(obj["data"])
            arr = np.frombuffer(raw, dtype=np.dtype(dtype))
            expected = int(np.prod(shape)) if shape else 1
            if arr.size != expected:
                raise ProtocolError(
                    f"array payload has {arr.size} elements, shape {shape} "
                    f"needs {expected}"
                )
            return arr.reshape(shape).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


def send(sock, payload: dict) -> None:
    line = json.dumps(encode(payload), separators=(",", ":")).encode() + b"\n"
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message too large: {len(line)} bytes")
    sock.sendall(line)


class LineReader:
    """Buffered newline-delimited reader over a socket."""

    def __init__(self, sock, max_bytes: int = MAX_MESSAGE_BYTES):
        self._sock = sock
        self._buf = bytearray()
        self._max = max_bytes

    def read(self) -> dict | None:
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = bytes(self._buf[:nl])
                del self._buf[: nl + 1]
                if not line.strip():
                    continue
                return decode(json.loads(line))
            chunk = self._sock.recv(65536)
            if not chunk:
                return None  # peer closed
            self._buf.extend(chunk)
            if len(self._buf) > self._max:
                raise ProtocolError("message exceeded size limit before newline")


def validate_action(value: Any, expected_dim: int) -> np.ndarray:
    """Coerce an agent-supplied action, refusing anything the sim should not see.

    Refuses (rather than repairs) wrong length, non-finite values and bad dtypes.
    Values ARE clipped to [-1, 1], which is legitimate: every RoboCasa action
    component is defined on that interval, so clipping is the documented contract
    rather than hidden leniency.
    """
    arr = np.asarray(value, dtype=np.float64) if not isinstance(value, np.ndarray) \
        else value.astype(np.float64, copy=False)
    if arr.ndim != 1:
        raise ProtocolError(f"action must be 1-D, got shape {arr.shape}")
    if arr.shape[0] != expected_dim:
        raise ProtocolError(
            f"action must have {expected_dim} components, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ProtocolError("action contains NaN or infinity")
    return np.clip(arr, -1.0, 1.0)


def validate_actions(values: Any, expected_dim: int) -> list[np.ndarray]:
    """Validate a whole batch before any of it is applied.

    All or nothing: one bad action rejects the batch, so a caller never has to work out
    how much of a partially applied sequence reached the simulator.

    `actions` is always PLURAL on the wire, and a non-list is refused rather than guessed
    at: reading one action as a batch of twelve components would spend twelve steps for
    the one requested.
    """
    if not isinstance(values, (list, tuple)):
        raise ProtocolError(f"actions must be a list, got {type(values).__name__}")
    if not values:
        raise ProtocolError("actions must not be empty")
    return [validate_action(v, expected_dim) for v in values]
