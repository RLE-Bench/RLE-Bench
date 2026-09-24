"""Exclusive simulator control. Disconnecting never ends an episode."""
from dataclasses import asdict, dataclass
import os
import socket
import threading
import uuid

from . import protocol as P


@dataclass(frozen=True)
class ObsSpec:
    width: int | None = None
    height: int | None = None
    cameras: tuple[str, ...] | None = None
    depth: bool = False

    def as_wire(self):
        return asdict(self)


class RemoteError(RuntimeError):
    def __init__(self, message, kind="remote_error"):
        super().__init__(message)
        self.kind = kind


class SimClient:
    def __init__(self, socket_path="/run/rlebench/sim.sock"):
        self.socket_path = socket_path
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._socket = self._reader = None
        self._connect()

    def _connect(self):
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(None)
        try:
            self._socket.connect(self.socket_path)
            self._reader = self._socket.makefile("rb")
            self._read()  # Admission, including an immediate busy response.
        except BaseException:
            self.disconnect()
            raise

    def _read(self):
        raw = self._reader.readline(P.MAX_REPLY + 1)
        if not raw or len(raw) > P.MAX_REPLY or not raw.endswith(b"\n"):
            raise RemoteError("connection lost; reconnect and inspect status before acting", "disconnected")
        reply = P.decode(P.loads(raw))
        if not reply.pop("ok", False):
            raise RemoteError(reply.get("error", "request failed"), reply.get("kind"))
        return reply

    def _request(self, op, **fields):
        if os.getpid() != self._pid:
            raise RemoteError("create a separate client in each process", "forked_client")
        if self._socket is None:
            raise RemoteError("client is disconnected; create a new client", "disconnected")
        with self._lock:
            request = dict(op=op, request_id=uuid.uuid4().hex, **fields)
            payload = P.dumps(request)
            if len(payload) > P.MAX_REQUEST:
                raise ValueError("request too large")
            try:
                self._socket.sendall(payload)
                return self._read()
            except (OSError, RemoteError) as exc:
                if isinstance(exc, RemoteError) and exc.kind != "disconnected":
                    raise
                self.disconnect()
                raise RemoteError("connection lost; reconnect and inspect status before acting", "disconnected") from None

    def task_info(self):
        return self._request("task_info")

    def status(self):
        return self._request("status")

    def list_tasks(self):
        return self._request("list_tasks")["tasks"]

    def observe(self, spec=None):
        return self._request("observe", spec=asdict(spec or ObsSpec()))

    def step(self, actions, obs_spec=None):
        if hasattr(actions, "tolist"):
            actions = actions.tolist()
        if actions and not isinstance(actions[0], (list, tuple)):
            actions = [actions]
        return self._request("step", actions=actions, spec=asdict(obs_spec or ObsSpec()))

    def reset(self, task=None, seed=None):
        return self._request("reset", task=task, seed=seed)

    def end_development(self):
        return self._request("end_development")

    def finish_trial(self):
        return self._request("finish_trial")

    def next_trial(self):
        return self._request("next_trial")

    def disconnect(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
