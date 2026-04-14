"""Shared test fixtures for bot-manager tests.

Uses mock DB sessions and FastAPI dependency overrides to test
without touching real databases or spawning real bot containers.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set DB env vars BEFORE importing shared_models (it validates at import time)
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "vexa_test")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_SSL_MODE", "disable")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from shared_models.models import User, APIToken, Meeting, Base
from shared_models.schemas import MeetingStatus


@pytest.fixture
def test_user():
    """A test user with max_concurrent_bots=3."""
    user = MagicMock(spec=User)
    user.id = 1
    user.email = "test@example.com"
    user.name = "Test User"
    user.max_concurrent_bots = 3
    user.data = {}
    return user


@pytest.fixture
def mock_db():
    """A mock async DB session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.close = AsyncMock()
    return session


def make_scalar_result(value):
    """Create a mock DB result that returns a scalar value."""
    result = MagicMock()
    result.scalar.return_value = value
    return result


def make_scalars_result(items):
    """Create a mock DB result for scalars().first() queries."""
    result = MagicMock()
    scalars = MagicMock()
    scalars.first.return_value = items[0] if items else None
    scalars.all.return_value = items
    result.scalars.return_value = scalars
    return result
