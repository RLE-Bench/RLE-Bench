"""Bounded JSON messages. No executable serialization crosses either socket."""
import base64
import json
import math

MAX_REQUEST = 256 * 1024
MAX_REPLY = 32 * 1024 * 1024
MAX_BATCH = 200


class RequestError(ValueError):
    def __init__(self, message, kind="bad_request"):
        super().__init__(message)
        self.kind = kind


def encode(value):
    # Keep numpy out of the supervisor unless arrays actually reach this function.
    if type(value).__module__.startswith("numpy"):
        import numpy as np
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            if a.dtype.kind not in "biuf":
                raise ValueError("unsupported array dtype")
            return {"__array__": base64.b64encode(a.tobytes()).decode(),
                    "dtype": a.dtype.str, "shape": list(a.shape)}
        return value.item()
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if "__array__" in value:
            import numpy as np
            dtype = np.dtype(value["dtype"])
            shape = value["shape"]
            if dtype.kind not in "biuf" or not isinstance(shape, list) or any(
                    type(n) is not int or n < 0 for n in shape):
                raise ValueError("invalid array")
            data = base64.b64decode(value["__array__"], validate=True)
            if math.prod(shape) * dtype.itemsize != len(data):
                raise ValueError("invalid array size")
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def dumps(value):
    return json.dumps(encode(value), allow_nan=False, separators=(",", ":")).encode() + b"\n"


def loads(data):
    def reject(value):
        raise ValueError("nonfinite JSON number")
    value = json.loads(data, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def actions(value, dimension):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_BATCH:
        raise RequestError(f"provide 1–{MAX_BATCH} actions")
    result = []
    for action in value:
        if not isinstance(action, list) or len(action) != dimension or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in action):
            raise RequestError(f"each action needs {dimension} finite numbers")
        result.append([max(-1., min(1., v)) for v in action])
    return result
