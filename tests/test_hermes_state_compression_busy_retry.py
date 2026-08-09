"""A live compression lock must delay a concurrent append, not destroy the turn.

``append_message`` refused immediately when another writer held the session's
compression lock. The conversation loop turns that into
``session_persistence_failed`` and tells the operator to check disk space and
permissions, when in fact the store is healthy and busy for a few seconds.

The write lock already gets a patience budget for exactly this reason (#74478).
A compression hold carries its own ``expires_at`` lease, so the writer waits on
that correctness boundary instead of a fixed five-second assumption.

The sibling condition, a compressor finding its own lease gone, is permanent
and must still fail fast rather than spin out the whole budget.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from hermes_state import (
    CompressionSessionBusyError,
    SessionCompressionInProgressError,
    SessionDB,
)


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def test_append_waits_out_a_live_compression_lock(db: SessionDB) -> None:
    """The classic race: a steer lands while compression owns the session."""
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    released = threading.Event()

    def _release_soon():
        time.sleep(0.3)
        db.release_compression_lock("sess1", "compressor")
        released.set()

    t = threading.Thread(target=_release_soon, daemon=True)
    t.start()
    try:
        started = time.monotonic()
        # No compression_lock_holder: this is an ordinary turn writer.
        db.append_message("sess1", role="user", content="steered mid-compression")
        elapsed = time.monotonic() - started
    finally:
        t.join(timeout=5)

    assert released.is_set(), "test bug: lock was never released"
    assert elapsed >= 0.25, "append returned before the lock could clear"
    rows = db.get_messages("sess1")
    assert any(r["content"] == "steered mid-compression" for r in rows), (
        "the message the user sent was lost"
    )


def test_append_still_gives_up_when_the_lock_never_clears(
    db: SessionDB, monkeypatch
) -> None:
    """The wait is bounded: a lock that never clears is still refused.

    The lease is a correctness boundary, so a genuinely long-running or wedged
    compression must not end with a stale turn landing in the parent.
    """
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.5)
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 0.5, raising=False
    )
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        db.append_message("sess1", role="user", content="never lands")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.4, "gave up before spending the patience budget"
    assert elapsed < 10, "did not give up within a bounded time"


def test_append_waits_past_fixed_budget_until_live_lease_clears(
    db: SessionDB, monkeypatch
) -> None:
    """A healthy multi-minute compression must not be cut off at five seconds.

    Small timings model the production ratio: the durable lease outlives both
    the old fixed compression wait and the transcript write patience.
    """
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 2.0, raising=False
    )
    monkeypatch.setattr(SessionDB, "_TRANSCRIPT_WRITE_PATIENCE_S", 0.2)
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=1.0
    ) is True

    released = threading.Event()

    def _release_after_old_budgets():
        time.sleep(0.6)
        db.release_compression_lock("sess1", "compressor")
        released.set()

    t = threading.Thread(target=_release_after_old_budgets, daemon=True)
    t.start()
    try:
        started = time.monotonic()
        db.append_message("sess1", role="user", content="waited on the lease")
        elapsed = time.monotonic() - started
    finally:
        t.join(timeout=5)

    assert released.is_set(), "test bug: lock was never released"
    assert elapsed >= 0.5, "append was still capped by a fixed patience budget"
    assert any(
        row["content"] == "waited on the lease" for row in db.get_messages("sess1")
    )


def test_append_atomically_invalidates_expired_lease(
    db: SessionDB, monkeypatch
) -> None:
    """Expiry and append are serialized so the old holder cannot revive."""
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 2.0, raising=False
    )
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=0.4
    ) is True

    started = time.monotonic()
    db.append_message("sess1", role="user", content="landed after invalidation")
    elapsed = time.monotonic() - started

    assert elapsed >= 0.3, "gave up before the live lease boundary"
    assert elapsed < 2.0, "wait crossed its bounded lease deadline"
    assert db.refresh_compression_lock("sess1", "compressor") is False
    assert any(
        row["content"] == "landed after invalidation"
        for row in db.get_messages("sess1")
    )


def test_fresh_retry_invalidates_expired_row_before_appending(
    db: SessionDB, monkeypatch
) -> None:
    """A later gateway retry must not bypass a revivable expired row."""
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 0.1, raising=False
    )
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=0.4
    ) is True

    with pytest.raises(CompressionSessionBusyError):
        db.append_message("sess1", role="user", content="first attempt")

    time.sleep(0.35)
    db.append_message("sess1", role="user", content="fresh retry")

    assert db.refresh_compression_lock("sess1", "compressor") is False
    with pytest.raises(CompressionSessionBusyError, match="lease lost"):
        db.append_message(
            "sess1",
            role="assistant",
            content="stale owner flush",
            compression_lock_holder="compressor",
        )
    with pytest.raises(CompressionSessionBusyError):
        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="stale-child",
            source="test",
            messages=[{"role": "assistant", "content": "stale snapshot"}],
            compression_lock_holder="compressor",
        )
    assert db.get_session("stale-child") is None
    assert all(
        row["content"] != "stale owner flush" for row in db.get_messages("sess1")
    )
    assert any(
        row["content"] == "fresh retry" for row in db.get_messages("sess1")
    )


def test_append_follows_same_holder_lease_refresh(
    db: SessionDB, monkeypatch
) -> None:
    """Progress heartbeats may extend one holder's lease within the hard cap."""
    monkeypatch.setattr(SessionDB, "_COMPRESSION_BUSY_WAIT_S", 0.1)
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 2.0, raising=False
    )
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=0.3
    ) is True

    released = threading.Event()

    def _refresh_then_release():
        time.sleep(0.15)
        assert db.refresh_compression_lock(
            "sess1", "compressor", ttl_seconds=0.5
        ) is True
        time.sleep(0.35)
        db.release_compression_lock("sess1", "compressor")
        released.set()

    t = threading.Thread(target=_refresh_then_release, daemon=True)
    t.start()
    try:
        started = time.monotonic()
        db.append_message("sess1", role="user", content="followed refresh")
        elapsed = time.monotonic() - started
    finally:
        t.join(timeout=5)

    assert released.is_set(), "test bug: refreshed lock was never released"
    assert elapsed >= 0.4, "writer ignored the refreshed lease"
    assert any(
        row["content"] == "followed refresh" for row in db.get_messages("sess1")
    )


def test_append_follows_same_holder_lease_contraction(
    db: SessionDB, monkeypatch
) -> None:
    """A shortened lease must not leave a later cached admission deadline."""
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 2.0, raising=False
    )
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 0.05)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", 0.05)
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=1.0
    ) is True

    shortened = threading.Event()

    def _shorten_lease():
        time.sleep(0.1)
        assert db.refresh_compression_lock(
            "sess1", "compressor", ttl_seconds=0.2
        ) is True
        shortened.set()

    t = threading.Thread(target=_shorten_lease, daemon=True)
    t.start()
    try:
        started = time.monotonic()
        db.append_message("sess1", role="user", content="followed contraction")
        elapsed = time.monotonic() - started
    finally:
        t.join(timeout=5)

    assert shortened.is_set(), "test bug: lease was never shortened"
    assert elapsed < 0.8, "writer retained the original, later lease deadline"
    assert db.refresh_compression_lock("sess1", "compressor") is False
    assert any(
        row["content"] == "followed contraction"
        for row in db.get_messages("sess1")
    )


def test_append_rechecks_release_at_lease_deadline(
    db: SessionDB, monkeypatch
) -> None:
    """A release during the final sleep must beat stale cached metadata."""
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 2.0, raising=False
    )
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 1.0)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", 1.0)
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=0.6
    ) is True

    released = threading.Event()

    def _release_before_deadline():
        time.sleep(0.4)
        db.release_compression_lock("sess1", "compressor")
        released.set()

    t = threading.Thread(target=_release_before_deadline, daemon=True)
    t.start()
    try:
        db.append_message("sess1", role="user", content="release won")
    finally:
        t.join(timeout=5)

    assert released.is_set(), "test bug: lock was never released"
    assert any(
        row["content"] == "release won" for row in db.get_messages("sess1")
    )


def test_busy_error_carries_authoritative_lease_metadata(
    db: SessionDB, monkeypatch
) -> None:
    before = time.time()
    assert db.try_acquire_compression_lock(
        "sess1", "compressor", ttl_seconds=30.0
    ) is True
    monkeypatch.setattr(
        SessionDB, "_COMPRESSION_BUSY_WAIT_MAX_S", 0.0, raising=False
    )

    with pytest.raises(SessionCompressionInProgressError) as caught:
        db.append_message("sess1", role="user", content="blocked")

    assert caught.value.session_id == "sess1"
    assert caught.value.holder == "compressor"
    assert caught.value.expires_at is not None
    assert caught.value.expires_at >= before + 29.0


def test_the_lock_owner_is_never_delayed_by_its_own_lock(db: SessionDB) -> None:
    assert db.try_acquire_compression_lock("sess1", "compressor") is True

    started = time.monotonic()
    db.append_message(
        "sess1",
        role="assistant",
        content="written by the compressor",
        compression_lock_holder="compressor",
    )
    assert time.monotonic() - started < 0.2


def test_transient_error_is_a_subclass_of_the_original(db: SessionDB) -> None:
    """Existing `except CompressionSessionBusyError` handlers must still catch."""
    assert issubclass(SessionCompressionInProgressError, CompressionSessionBusyError)
    assert SessionCompressionInProgressError().args == ()
    assert SessionCompressionInProgressError("one", "two").args == ("one", "two")


def test_no_lock_means_no_delay(db: SessionDB) -> None:
    started = time.monotonic()
    db.append_message("sess1", role="user", content="uncontended")
    assert time.monotonic() - started < 0.2


def test_a_lost_compression_lease_still_fails_fast(db: SessionDB) -> None:
    """The other CompressionSessionBusyError case must NOT be retried.

    ``publish_compression_child`` raises the same base class when the
    compressor discovers its own lease is gone. That is permanent, so
    retrying would burn the whole patience budget before failing anyway.
    Only the transient subclass raised by ``append_message`` is retried.
    """
    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=[{"role": "user", "content": "compacted"}],
            compression_lock_holder="not-the-holder",
            require_compression_lease=True,
        )
    assert time.monotonic() - started < 0.5, (
        "a lost lease is permanent and must not spend the retry budget"
    )
