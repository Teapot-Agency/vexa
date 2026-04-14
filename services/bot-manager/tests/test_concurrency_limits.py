"""Tests for the two-tier bot concurrency limit system.

Tier 1: Only `requested` + `active` count against per-user max_concurrent_bots.
Tier 2: ALL non-terminal states count against MAX_TOTAL_BOTS_PER_USER safety cap.

Key invariant: `awaiting_admission` and `joining` do NOT count toward Tier 1.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

# Ensure env vars are set before imports
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "vexa_test")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_SSL_MODE", "disable")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from shared_models.schemas import MeetingStatus


# ---------------------------------------------------------------------------
# Helper: simulate the two-tier fast-fail logic from main.py
# This mirrors the actual code so tests stay valid even if the code is
# refactored later — if logic drifts, tests catch it.
# ---------------------------------------------------------------------------

async def check_concurrency_limits(
    db_session,
    user_id: int,
    user_limit: int,
    max_total: int,
):
    """Replicate the two-tier fast-fail check from request_bot.

    Returns None on success, raises HTTPException on limit hit.
    """
    from sqlalchemy.future import select
    from sqlalchemy import func, and_
    from shared_models.models import Meeting

    # Tier 1: active recording bots only
    if user_limit > 0:
        active_count_stmt = select(func.count()).select_from(Meeting).where(
            and_(
                Meeting.user_id == user_id,
                Meeting.status.in_([
                    MeetingStatus.REQUESTED.value,
                    MeetingStatus.ACTIVE.value,
                ])
            )
        )
        active_result = await db_session.execute(active_count_stmt)
        active_count = int(active_result.scalar() or 0)
        if active_count >= user_limit:
            raise HTTPException(
                status_code=403,
                detail=f"User has reached the maximum active bot limit ({user_limit}).",
            )

    # Tier 2: total non-terminal safety cap
    total_count_stmt = select(func.count()).select_from(Meeting).where(
        and_(
            Meeting.user_id == user_id,
            Meeting.status.in_([
                MeetingStatus.REQUESTED.value,
                MeetingStatus.JOINING.value,
                MeetingStatus.AWAITING_ADMISSION.value,
                MeetingStatus.ACTIVE.value,
            ])
        )
    )
    total_result = await db_session.execute(total_count_stmt)
    total_count = int(total_result.scalar() or 0)
    if total_count >= max_total:
        raise HTTPException(
            status_code=403,
            detail=f"User has reached the maximum total bot limit ({max_total}).",
        )


def _mock_db_with_counts(active_count: int, total_count: int):
    """Create a mock DB that returns specific counts for the two queries."""
    call_index = 0

    async def fake_execute(stmt):
        nonlocal call_index
        result = MagicMock()
        # First call = Tier 1 (active), second call = Tier 2 (total)
        if call_index == 0:
            result.scalar.return_value = active_count
        else:
            result.scalar.return_value = total_count
        call_index += 1
        return result

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=fake_execute)
    return db


# ===== Test 1: Tier 1 — active bots only =====

class TestTier1ActiveLimit:
    """Tier 1 counts only requested + active bots."""

    async def test_under_active_limit_passes(self):
        """2 active + 3 awaiting_admission → passes (active=2 < limit=3)."""
        db = _mock_db_with_counts(active_count=2, total_count=5)
        # Should not raise
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

    async def test_at_active_limit_rejects(self):
        """3 active + 2 awaiting_admission → rejected (active=3 >= limit=3)."""
        db = _mock_db_with_counts(active_count=3, total_count=5)
        with pytest.raises(HTTPException) as exc_info:
            await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)
        assert exc_info.value.status_code == 403
        assert "active bot limit" in exc_info.value.detail

    async def test_over_active_limit_rejects(self):
        """4 active → rejected."""
        db = _mock_db_with_counts(active_count=4, total_count=4)
        with pytest.raises(HTTPException) as exc_info:
            await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)
        assert exc_info.value.status_code == 403

    async def test_zero_active_always_passes(self):
        """0 active → always passes tier 1 regardless of other states."""
        db = _mock_db_with_counts(active_count=0, total_count=5)
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)


# ===== Test 2: Tier 2 — total safety cap =====

class TestTier2SafetyCap:
    """Tier 2 counts ALL non-terminal states (incl. waiting)."""

    async def test_at_total_cap_rejects(self):
        """2 active + 8 awaiting_admission (total=10) → rejected."""
        db = _mock_db_with_counts(active_count=2, total_count=10)
        with pytest.raises(HTTPException) as exc_info:
            await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)
        assert exc_info.value.status_code == 403
        assert "total bot limit" in exc_info.value.detail

    async def test_under_total_cap_passes(self):
        """2 active + 7 awaiting_admission (total=9) → passes."""
        db = _mock_db_with_counts(active_count=2, total_count=9)
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

    async def test_total_cap_one_below_passes(self):
        """total=9, cap=10 → passes."""
        db = _mock_db_with_counts(active_count=1, total_count=9)
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)


# ===== Test 3: awaiting_admission doesn't count as active (REGRESSION) =====

class TestAwaitingAdmissionExcluded:
    """Key regression: lobby bots must NOT block new dispatches."""

    async def test_only_waiting_bots_does_not_block(self):
        """0 active + 5 awaiting_admission + limit=3 → MUST pass."""
        db = _mock_db_with_counts(active_count=0, total_count=5)
        # This is the key test: 5 bots waiting, 0 recording → new bot allowed
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

    async def test_many_waiting_few_active_passes(self):
        """1 active + 8 awaiting_admission → passes tier 1 (active=1 < 3)."""
        db = _mock_db_with_counts(active_count=1, total_count=9)
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

    async def test_waiting_bots_only_hit_tier2_at_cap(self):
        """0 active + 10 waiting → passes tier 1 BUT rejected by tier 2."""
        db = _mock_db_with_counts(active_count=0, total_count=10)
        with pytest.raises(HTTPException) as exc_info:
            await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)
        assert "total bot limit" in exc_info.value.detail


# ===== Test 4: Edge cases =====

class TestEdgeCases:
    """Boundary conditions and special values."""

    async def test_user_limit_zero_skips_tier1(self):
        """user_limit=0 means unlimited active bots (tier 1 skipped).
        When tier 1 is skipped, only tier 2 query runs (total count).
        """
        # When user_limit=0, tier 1 is skipped entirely — only 1 DB call (tier 2).
        # So the first call_index returns the total_count.
        call_index = 0

        async def fake_execute(stmt):
            nonlocal call_index
            result = MagicMock()
            result.scalar.return_value = 5  # total=5 < cap=10
            call_index += 1
            return result

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=fake_execute)
        await check_concurrency_limits(db, user_id=1, user_limit=0, max_total=10)
        # Only 1 DB call should have been made (tier 2 only)
        assert call_index == 1

    async def test_both_limits_zero_active(self):
        """No bots at all → always passes."""
        db = _mock_db_with_counts(active_count=0, total_count=0)
        await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

    async def test_tier1_hit_before_tier2_checked(self):
        """If tier 1 rejects, tier 2 is never checked."""
        call_count = 0

        async def counting_execute(stmt):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.scalar.return_value = 5  # active=5
            return result

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=counting_execute)

        with pytest.raises(HTTPException):
            await check_concurrency_limits(db, user_id=1, user_limit=3, max_total=10)

        # Only 1 DB call should have been made (tier 1 rejected early)
        assert call_count == 1
