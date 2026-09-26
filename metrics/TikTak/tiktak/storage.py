"""Transactional evaluation cache, budgets, and restart state (single host).

SQLite must live on a local filesystem, not NFS/Lustre. Transactions are held
only for bookkeeping, never while solving a model. Each worker has its own
connection. A parent-only advisory lock prevents competing coordinators.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np


class BudgetExhausted(Exception):
    """The global evaluation or wall-clock budget was reached."""


class LocalBudgetExhausted(Exception):
    """The local search used its evaluation allocation."""


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def point_key(unit, screening=False):
    u = np.asarray(unit, dtype="<f8").copy()
    u[u == 0] = 0  # Canonicalize signed zero; never round distinct points.
    # Screening evaluations of a staged run come from a different (cheaper)
    # objective, so they are cached under their own key.
    return hashlib.sha256(u.tobytes() + (b"screen" if screening else b"")).hexdigest()


def _has_screening_column(connection):
    return any(row[1] == "screening" for row in connection.execute("PRAGMA table_info(evaluations)"))


@contextmanager
def coordinator_lock(directory):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "coordinator.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another coordinator is using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.connection = sqlite3.connect(self.directory / "history.sqlite3", timeout=60,
                                          isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute("PRAGMA synchronous=FULL")

    def close(self):
        self.connection.close()

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def initialize(self, specification, *, resume, max_evals, deadline):
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (
                key TEXT PRIMARY KEY, unit TEXT NOT NULL, parameters TEXT NOT NULL,
                status TEXT NOT NULL, value REAL, moments TEXT, residuals TEXT,
                error TEXT, seconds REAL, updated REAL NOT NULL,
                screening INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, key TEXT NOT NULL, task TEXT NOT NULL,
                started REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_task ON attempts(task);
            CREATE TABLE IF NOT EXISTS task_points (
                task TEXT NOT NULL, key TEXT NOT NULL, PRIMARY KEY(task, key));
            CREATE TABLE IF NOT EXISTS locals (
                id INTEGER PRIMARY KEY, start TEXT NOT NULL, seed TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT);
        """)
        if not _has_screening_column(self.connection):  # Runs created before staged screening.
            self.connection.execute("ALTER TABLE evaluations ADD COLUMN screening INTEGER NOT NULL DEFAULT 0")
        existing = self.get("specification")
        if existing is not None:
            if not resume:
                raise FileExistsError("run already exists; choose a new directory or resume=True")
            if existing != specification:
                raise ValueError("restart specification differs; use a new run and warm_start")
        elif resume:
            raise FileNotFoundError("there is no initialized run to resume")
        with self.transaction():
            self.put("specification", specification)
            self.put("max_evals", max_evals)
            self.put("deadline", deadline)
            # Keep charges for interrupted calls: work may have been performed.
            self.connection.execute("UPDATE evaluations SET status='abandoned' WHERE status='pending'")

    def get(self, key):
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key, value):
        self.connection.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, encode(value)))

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]

    def exhausted(self):
        deadline = self.get("deadline")
        return self.count() >= self.get("max_evals") or (deadline is not None and time.time() >= deadline)

    def claim(self, unit, parameters, task, local_limit=None, screening=False):
        """Return (key, cached row or None); reserve a hard budget slot atomically.

        ``screening=True`` marks an evaluation by the screening objective of a
        staged run, which has its own cache entry.
        """
        key = point_key(unit, screening)
        while True:
            with self.transaction():
                row = self.connection.execute("SELECT * FROM evaluations WHERE key=?", (key,)).fetchone()
                if row is not None and row["status"] in ("ok", "failed", "error"):
                    self.connection.execute("INSERT OR IGNORE INTO task_points VALUES (?, ?)", (task, key))
                    if row["status"] == "error":
                        raise RuntimeError("cached unexpected model error: " + row["error"])
                    return key, dict(row)
                if self.exhausted():
                    raise BudgetExhausted()
                if local_limit is not None:
                    count = self.connection.execute("SELECT COUNT(*) FROM attempts WHERE task=?", (task,)).fetchone()[0]
                    if count >= local_limit:
                        raise LocalBudgetExhausted()
                if row is None or row["status"] == "abandoned":
                    now = time.time()
                    self.connection.execute(
                        "INSERT OR REPLACE INTO evaluations(key,unit,parameters,status,updated,screening) "
                        "VALUES (?,?,?,?,?,?)",
                        (key, encode(unit.tolist()), encode(parameters.tolist()), "pending", now, int(screening)))
                    self.connection.execute("INSERT INTO attempts(key,task,started) VALUES (?,?,?)", (key, task, now))
                    self.connection.execute("INSERT OR IGNORE INTO task_points VALUES (?,?)", (task, key))
                    return key, None
            # Another worker is already evaluating this exact point. No second
            # model call and no second budget charge. Check deadlines while waiting.
            time.sleep(0.05)

    def finish(self, key, *, value=None, moments=None, residuals=None, error=None, seconds=0,
               unexpected=False):
        status = "error" if unexpected else ("ok" if value is not None else "failed")
        self.connection.execute(
            "UPDATE evaluations SET status=?,value=?,moments=?,residuals=?,error=?,seconds=?,updated=? WHERE key=?",
            (status, value, None if moments is None else encode(moments.tolist()),
             None if residuals is None else encode(residuals.tolist()), error, seconds, time.time(), key))

    def best(self, task=None):
        """Best finite full-accuracy evaluation; screening values of a staged run are not comparable."""
        if task is None:
            row = self.connection.execute(
                "SELECT * FROM evaluations WHERE status='ok' AND screening=0 ORDER BY value,key LIMIT 1").fetchone()
        else:
            row = self.connection.execute(
                "SELECT e.* FROM evaluations e JOIN task_points t ON e.key=t.key "
                "WHERE t.task=? AND e.status='ok' AND e.screening=0 ORDER BY e.value,e.key LIMIT 1",
                (task,)).fetchone()
        return None if row is None else dict(row)

    def local_rows(self):
        return [dict(row) for row in self.connection.execute("SELECT * FROM locals ORDER BY id")]

    def create_local(self, index, start, seed):
        self.connection.execute("INSERT INTO locals VALUES (?,?,?,'pending',NULL)",
                                (index, encode(start.tolist()), encode(seed.tolist())))

    def finish_local(self, index, result, complete):
        self.connection.execute("UPDATE locals SET status=?, result=? WHERE id=?",
                                ("done" if complete else "pending", encode(result), index))

    def export(self, result):
        # Only the coordinator exports the convenient human-readable snapshot.
        temporary = self.directory / "result.json.tmp"
        with temporary.open("w") as stream:
            stream.write(encode(result) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temporary.replace(self.directory / "result.json")


def load_estimates(directory, limit=20):
    """Read the best evaluated parameter vectors for a NEW search.

    Their objective values are deliberately not imported: a changed model,
    weighting matrix, or simulation design requires reevaluation. In a staged
    run, full-accuracy evaluations are ranked first and screening evaluations
    fill any remaining slots.
    """
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    path = Path(directory).resolve() / "history.sqlite3"
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        order = "screening,value,key" if _has_screening_column(connection) else "value,key"
        rows = connection.execute(
            f"SELECT parameters FROM evaluations WHERE status='ok' ORDER BY {order} LIMIT ?", (limit,)).fetchall()
    return np.asarray([json.loads(row[0]) for row in rows])
