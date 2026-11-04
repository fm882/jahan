"""Where a partition lives while its run is alive, and what happens when it finishes.

A tick arrives as one call and lands in one SQLite transaction or not at all (ADR 0033): the
runner hands over a whole tick, its events plus the `tick_closed` that marks it, and this
module writes them together. A re-sent tick is refused rather than duplicated, so a retry
after an ambiguous crash is safe — and an identical re-send is a no-op.

Layout under the root::

    registry.db
    runs/<run_id>/<world_id>/live.sqlite
    runs/<run_id>/<world_id>/events.parquet
    runs/<run_id>/<world_id>/beliefs.parquet
    runs/<run_id>/<world_id>/edges.parquet
    runs/<run_id>/<world_id>/partition.json

The world's own `state.db` may sit beside a partition for filesystem convenience; nothing
here reads it (ADR 0011).
"""

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from simcore.schemas import PartitionHeader, TraceEvent
from simcore.schemas.base import canonical_payload

from . import derive as _derive
from .errors import DuplicateSequenceError, FinalizedError, UnknownRunError, UnknownWorldError
from .registry import SqliteRunRegistry

_LIVE_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
    "CREATE TABLE IF NOT EXISTS events ("
    " world_id TEXT NOT NULL,"
    " seq INTEGER NOT NULL,"
    " event_id TEXT NOT NULL UNIQUE,"
    " tick INTEGER NOT NULL,"
    " persona_id TEXT,"
    " kind TEXT NOT NULL,"
    " event_json TEXT NOT NULL,"
    " PRIMARY KEY (world_id, seq))"
)


def _connect_live(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(_LIVE_SCHEMA)
    connection.commit()
    return connection


def _meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row is not None else None


def _same_header(stored: str, header: PartitionHeader) -> bool:
    """Whether a pinned header is the world this header describes.

    Compared as canonical payloads — set-valued fields sorted, output fields
    excluded — so a resume in another process agrees with the process that
    started the world. Raw JSON compares set iteration order, which follows
    the process hash seed: the same study re-pinned its world as foreign
    about as often as not. A stored header that no longer validates falls
    back to the raw comparison rather than refusing on a technicality.
    """
    try:
        return canonical_payload(PartitionHeader.model_validate(json.loads(stored))) == canonical_payload(header)
    except Exception:
        return json.loads(stored) == json.loads(header.model_dump_json())


class TraceStore:
    """The one writer and the backend behind every view. Satisfies the `TraceSink` port."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self.registry = SqliteRunRegistry(self._root / "registry.db")

    # -- layout -----------------------------------------------------------

    def _world_dir(self, run_id: str, world_id: str) -> Path:
        return self._root / "runs" / run_id / world_id

    def _find_world(self, world_id: str) -> tuple[str, Path]:
        """The `(run_id, dir)` holding this world, or an unknown-world refusal."""
        runs = self._root / "runs"
        if runs.is_dir():
            for run_dir in sorted(runs.iterdir()):
                candidate = run_dir / world_id
                if candidate.is_dir():
                    return run_dir.name, candidate
        raise UnknownWorldError(f"no record of world {world_id}")

    def create_world(self, run_id: str, header: PartitionHeader) -> Path:
        """Pin this world's header before its first tick. Repeating it is a no-op."""
        if not isinstance(header, PartitionHeader):
            raise TypeError(f"a world is created from a header, not a mapping ({type(header).__name__})")
        world_dir = self._world_dir(run_id, header.world_id)
        world_dir.mkdir(parents=True, exist_ok=True)
        with _connect_live(world_dir / "live.sqlite") as connection:
            stored = _meta(connection, "header")
            if stored is not None and not _same_header(stored, header):
                raise ValueError(f"world {header.world_id} already runs under a different header")
            connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("header", header.model_dump_json()))
            connection.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", ("contract_version", header.contract_version)
            )
            connection.commit()
        return world_dir

    # -- write ------------------------------------------------------------

    def write(self, events: Sequence[TraceEvent]) -> None:
        """One tick's events, in one transaction. A `(world_id, seq)` already held is refused;
        an identical re-send is a no-op. Takes payload models, never mappings."""
        if isinstance(events, Mapping) or isinstance(events, (str, bytes)):
            raise TypeError("write takes a sequence of events, not a mapping")
        events = tuple(events)
        if not events:
            return
        for event in events:
            if not isinstance(event, TraceEvent):
                raise TypeError(
                    "the write path takes payload models, not mappings;"
                    f" build the event first ({type(event).__name__})"
                )
        worlds = {event.world_id for event in events}
        if len(worlds) != 1:
            raise ValueError(f"one call writes one world's tick, got worlds {sorted(worlds)}")
        world_id = next(iter(worlds))
        run_id, world_dir = self._locate_world_for_write(world_id)
        with _connect_live(world_dir / "live.sqlite") as connection:
            if _meta(connection, "finalized") == "1":
                raise FinalizedError(f"world {world_id} is finalized and read-only")
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._insert_all(connection, events)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def _locate_world_for_write(self, world_id: str) -> tuple[str, Path]:
        try:
            return self._find_world(world_id)
        except UnknownWorldError:
            runs = self._root / "runs"
            if runs.is_dir():
                for run_dir in sorted(runs.iterdir()):
                    entry = self.registry.entry(run_dir.name)
                    if entry is not None and world_id in entry.world_ids:
                        world_dir = self._world_dir(run_dir.name, world_id)
                        world_dir.mkdir(parents=True, exist_ok=True)
                        return run_dir.name, world_dir
            raise UnknownRunError(f"no run on record holds world {world_id}")

    @staticmethod
    def _insert_all(connection: sqlite3.Connection, events: tuple[TraceEvent, ...]) -> None:
        seqs = [event.seq for event in events]
        rows = connection.execute(
            "SELECT seq, event_id, event_json FROM events WHERE world_id = ? AND seq IN (%s)" % ",".join("?" * len(seqs)),
            (events[0].world_id, *seqs),
        ).fetchall()
        if rows:
            held = {seq: (event_id, payload) for seq, event_id, payload in rows}
            identical = (
                len(held) == len(events)
                and all(event.seq in held for event in events)
                and all(
                    held[event.seq][0] == event.event_id
                    and json.loads(held[event.seq][1]) == json.loads(event.model_dump_json())
                    for event in events
                )
            )
            if identical:
                return
            raise DuplicateSequenceError(
                f"world {events[0].world_id} already holds sequence numbers"
                f" {sorted(held)}; a re-sent tick must be identical"
            )
        connection.executemany(
            "INSERT INTO events (world_id, seq, event_id, tick, persona_id, kind, event_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    event.world_id,
                    event.seq,
                    event.event_id,
                    event.tick,
                    event.persona_id,
                    event.payload.kind,
                    event.model_dump_json(),
                )
                for event in events
            ],
        )

    # -- live reads -------------------------------------------------------

    def read_live(self, run_id: str, world_id: str) -> tuple[TraceEvent, ...]:
        """Every event of one live world, strictly validated, in sequence order."""
        world_dir = self._world_dir(run_id, world_id)
        live = world_dir / "live.sqlite"
        if not live.exists():
            raise UnknownWorldError(f"no live record of world {world_id}")
        with _connect_live(live) as connection:
            rows = connection.execute(
                "SELECT event_json FROM events WHERE world_id = ? ORDER BY seq", (world_id,)
            ).fetchall()
        return tuple(TraceEvent.model_validate(json.loads(payload)) for (payload,) in rows)

    def world_ids(self, run_id: str) -> tuple[str, ...]:
        run_dir = self._root / "runs" / run_id
        if not run_dir.is_dir():
            return ()
        return tuple(sorted(child.name for child in run_dir.iterdir() if child.is_dir()))

    def is_finalized(self, run_id: str, world_id: str) -> bool:
        live = self._world_dir(run_id, world_id) / "live.sqlite"
        if not live.exists():
            return False
        with _connect_live(live) as connection:
            return _meta(connection, "finalized") == "1"

    # -- finalize ---------------------------------------------------------

    def finalize(self, world_id: str) -> Path:
        """Turn a finished world's live record into its lasting one. Idempotent: calling it
        twice changes nothing. A finalized world refuses further writes."""
        from .finalize import finalize_world

        run_id, world_dir = self._find_world(world_id)
        return finalize_world(self, run_id, world_id, world_dir)

    # -- views ------------------------------------------------------------

    def view(self, run_id: str, world_id: str | None = None):
        """The five questions over this run's record, live or finalized — the caller is never
        told which backend answered.

        `world_id` scopes the view to one world. A sweep's cells are separate studies sharing
        one budget, and a digest names one scenario, so they are read apart.
        """
        from .views import ParquetTraceView, SqliteTraceView

        entry = self.registry.entry(run_id)
        if entry is None:
            raise UnknownRunError(f"no entry for run {run_id}")
        worlds = self.world_ids(run_id)
        if world_id is not None:
            if world_id not in worlds:
                raise UnknownWorldError(f"{world_id} is not a world of run {run_id}")
            worlds = (world_id,)
        finalized_world = world_id is not None and self.is_finalized(run_id, world_id)
        if (entry.status.value == "completed" or finalized_world) and all(
            (self._world_dir(run_id, world) / "events.parquet").exists() for world in worlds
        ):
            return ParquetTraceView(self._root, run_id, worlds)
        return SqliteTraceView(self, run_id, worlds)


def _events_hash(events: Sequence[TraceEvent]) -> str:
    digest = hashlib.sha256()
    for event in sorted(events, key=lambda e: e.seq):
        digest.update(event.model_dump_json().encode("utf-8"))
    return digest.hexdigest()


# Module-level functions, for callers that hold a root rather than a store.

def create_world(root: str | Path, run_id: str, header: PartitionHeader) -> Path:
    return TraceStore(root).create_world(run_id, header)


def write(root: str | Path, events: Sequence[TraceEvent]) -> None:
    TraceStore(root).write(events)


def finalize(root: str | Path, world_id: str) -> Path:
    return TraceStore(root).finalize(world_id)


def view(root: str | Path, run_id: str, world_id: str | None = None):
    return TraceStore(root).view(run_id, world_id)


__all__ = ["TraceStore", "create_world", "finalize", "view", "write"]
