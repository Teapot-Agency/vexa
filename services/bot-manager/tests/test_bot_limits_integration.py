"""Test 4: Integration test — POST /bots with mocked orchestrator.

Tests the full API endpoint code path (auth -> dedup -> limit check)
with mocked DB and orchestrator. No real containers are spawned.

NOTE: Bot-manager has heavy Docker-specific dependencies (requests_unixsocket,
aiodocker, etc.) that aren't available on the host. We mock them before import.
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "vexa_test")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_SSL_MODE", "disable")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

# Mock heavy dependencies that aren't installed on the host
# before importing app.main which triggers the orchestrator chain.
for mod_name in [
    "requests_unixsocket",
    "aiodocker",
    "celery",
    "boto3",
    "kubernetes",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Also mock sub-modules that may be accessed
sys.modules.setdefault("celery.schedules", MagicMock())
sys.modules.setdefault("kubernetes.client", MagicMock())
sys.modules.setdefault("kubernetes.config", MagicMock())

import httpx
from shared_models.models import User
from shared_models.schemas import MeetingStatus


def _make_test_user(max_concurrent_bots=3):
    user = MagicMock(spec=User)
    user.id = 1
    user.email = "test@example.com"
    user.name = "Test User"
    user.max_concurrent_bots = max_concurrent_bots
    user.data = {}
    return user


def _mock_db_session(existing_meeting=None, active_count=0, total_count=0):
    """Build a mock DB session that handles the request_bot query sequence.

    Query order in request_bot:
      1. SELECT existing meeting (dedup check)
      2. SELECT COUNT active bots (tier 1)
      3. SELECT COUNT total bots (tier 2)
      4+. downstream (commit, refresh, etc.)
    """
    call_index = 0

    async def fake_execute(stmt):
        nonlocal call_index
        idx = call_index
        call_index += 1

        if idx == 0:
            # Query 1: existing meeting check -> scalars().first()
            result = MagicMock()
            scalars = MagicMock()
            scalars.first.return_value = existing_meeting
            result.scalars.return_value = scalars
            return result
        elif idx == 1:
            # Query 2: tier 1 active count
            result = MagicMock()
            result.scalar.return_value = active_count
            return result
        elif idx == 2:
            # Query 3: tier 2 total count
            result = MagicMock()
            result.scalar.return_value = total_count
            return result
        else:
            # Subsequent calls
            result = MagicMock()
            result.scalar.return_value = 0
            return result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=fake_execute)
    session.commit = AsyncMock()
    session.refresh = AsyncMock(side_effect=lambda obj: setattr(obj, 'id', 100))
    session.add = MagicMock()
    session.close = AsyncMock()
    return session


class TestPostBotsLimitIntegration:
    """Test POST /bots endpoint with two-tier limit enforcement."""

    async def test_at_active_limit_returns_403(self):
        """3 active, limit=3 -> 403 Forbidden."""
        from app.main import app
        from shared_models.database import get_db
        from app.auth import get_user_and_token

        test_user = _make_test_user(max_concurrent_bots=3)
        mock_db = _mock_db_session(active_count=3, total_count=5)

        async def override_db():
            yield mock_db

        async def override_auth():
            return ("test-api-key", test_user)

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_user_and_token] = override_auth

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/bots",
                    json={
                        "platform": "google_meet",
                        "native_meeting_id": "abc-defg-hij",
                    },
                    headers={"X-API-Key": "test-api-key"},
                )
            assert resp.status_code == 403
            assert "active bot limit" in resp.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    async def test_waiting_bots_dont_block_new_dispatch(self):
        """0 active + 5 waiting, limit=3 -> NOT 403 (key regression test)."""
        from app.main import app
        from shared_models.database import get_db
        from app.auth import get_user_and_token

        test_user = _make_test_user(max_concurrent_bots=3)
        mock_db = _mock_db_session(active_count=0, total_count=5)

        async def override_db():
            yield mock_db

        async def override_auth():
            return ("test-api-key", test_user)

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_user_and_token] = override_auth

        try:
            with patch("app.main.start_bot_container", new_callable=AsyncMock) as mock_start:
                mock_start.return_value = {"container_id": "fake-456"}
                with patch("app.main.publish_meeting_status_change", new_callable=AsyncMock):
                    with patch("app.main.redis_client", new_callable=AsyncMock):
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=app),
                            base_url="http://test",
                        ) as client:
                            resp = await client.post(
                                "/bots",
                                json={
                                    "platform": "google_meet",
                                    "native_meeting_id": "xyz-mnop-qrs",
                                },
                                headers={"X-API-Key": "test-api-key"},
                            )
                        assert resp.status_code != 403, (
                            f"Lobby bots should NOT block dispatch! Got 403: {resp.text}"
                        )
        finally:
            app.dependency_overrides.clear()

    async def test_total_cap_rejects_at_limit(self):
        """2 active + 8 waiting (total=10), cap=10 -> 403."""
        from app.main import app
        from shared_models.database import get_db
        from app.auth import get_user_and_token

        test_user = _make_test_user(max_concurrent_bots=3)
        mock_db = _mock_db_session(active_count=2, total_count=10)

        async def override_db():
            yield mock_db

        async def override_auth():
            return ("test-api-key", test_user)

        app.dependency_overrides[get_db] = override_db
        app.dependency_overrides[get_user_and_token] = override_auth

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/bots",
                    json={
                        "platform": "google_meet",
                        "native_meeting_id": "tot-alca-ppp",
                    },
                    headers={"X-API-Key": "test-api-key"},
                )
            assert resp.status_code == 403
            assert "total bot limit" in resp.json()["detail"]
        finally:
            app.dependency_overrides.clear()
