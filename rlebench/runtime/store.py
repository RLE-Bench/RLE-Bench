"""One private SQLite writer, with an atomic state and event journal."""
import json
import os
from pathlib import Path
import sqlite3


class Store:
    def __init__(self, path, initial):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink():
            raise ValueError("ledger must not be a symlink")
        self.db = sqlite3.connect(path)
        os.chmod(path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, operation TEXT NOT NULL, done INTEGER NOT NULL);
        """)
        row = self.db.execute("SELECT value FROM state WHERE id=1").fetchone()
        self.state = json.loads(row[0]) if row else initial
        if row is None:
            self.save("start", {})

    def save(self, kind, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES(1,?)", (json.dumps(self.state, allow_nan=False),))
            self.db.execute("INSERT INTO events(kind,value) VALUES(?,?)", (kind, json.dumps(value, allow_nan=False)))

    def begin(self, request_id, operation):
        with self.db:
            row = self.db.execute("SELECT operation,done FROM requests WHERE id=?", (request_id,)).fetchone()
            if row:
                return False
            self.db.execute("INSERT INTO requests VALUES(?,?,0)", (request_id, operation))
        self.state["last_request"] = {"id": request_id, "operation": operation, "done": False}
        self.save("request", self.state["last_request"])
        return True

    def complete(self, request_id):
        with self.db:
            self.db.execute("UPDATE requests SET done=1 WHERE id=?", (request_id,))
            self.state["last_request"]["done"] = True
            self.save("request_done", {"id": request_id})

    def close(self):
        self.db.close()


def read_state(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or invalid ledger")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("invalid ledger")
        return json.loads(db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
