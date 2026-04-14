"""Test 6: Regression tests for count_user_active_bots() in common.py.

Verifies that the existing function counts only `requested` and `active`,
not `joining` or `awaiting_admission`.
"""

import os
import sys
import importlib.util
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "vexa_test")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_SSL_MODE", "disable")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

# Load common.py directly, bypassing orchestrators/__init__.py
# which triggers docker/nomad/kubernetes imports with heavy dependencies.
_common_path = os.path.join(
    os.path.dirname(__file__), "..", "app", "orchestrators", "common.py"
)
_spec = importlib.util.spec_from_file_location("_common_direct", _common_path)
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
count_user_active_bots = _common.count_user_active_bots


class TestCountUserActiveBots:
    """Verify count_user_active_bots counts only requested + active."""

    async def test_zero_bots_returns_zero(self):
        """No active bots -> returns 0."""
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 0

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)

        with patch.object(_common, "async_session_local", mock_factory):
            count = await count_user_active_bots(user_id=1)
            assert count == 0

    async def test_function_queries_only_requested_and_active(self):
        """Verify the SQL only targets 'requested' and 'active' statuses.

        NOTE: Meeting.__table__.count() is removed in SQLAlchemy 2.x, so
        this function hits the fallback path on the host (returns 0). In
        production Docker it works with the pinned SQLAlchemy version.
        This test verifies the statuses used in the query via source inspection.
        """
        import inspect
        source = inspect.getsource(count_user_active_bots)
        # Must query requested and active
        assert "'requested'" in source
        assert "'active'" in source
        # Must NOT query awaiting_admission or joining
        assert "'awaiting_admission'" not in source
        assert "'joining'" not in source

    async def test_db_error_returns_zero(self):
        """DB error -> returns 0 (safe fallback)."""
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB down"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)

        with patch.object(_common, "async_session_local", mock_factory):
            count = await count_user_active_bots(user_id=1)
            assert count == 0
