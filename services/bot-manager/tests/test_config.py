"""Test 5: Config constants load correctly from env vars."""

import os
import pytest


class TestMaxTotalBotsPerUser:
    """MAX_TOTAL_BOTS_PER_USER env var and default."""

    def test_default_value(self):
        """Default is 10 when env var not set."""
        os.environ.pop("MAX_TOTAL_BOTS_PER_USER", None)
        # Re-import to pick up fresh value
        import importlib
        import app.config as config_module
        importlib.reload(config_module)
        assert config_module.MAX_TOTAL_BOTS_PER_USER == 10

    def test_env_var_override(self):
        """Env var overrides the default."""
        os.environ["MAX_TOTAL_BOTS_PER_USER"] = "25"
        try:
            import importlib
            import app.config as config_module
            importlib.reload(config_module)
            assert config_module.MAX_TOTAL_BOTS_PER_USER == 25
        finally:
            os.environ.pop("MAX_TOTAL_BOTS_PER_USER", None)

    def test_env_var_string_converted_to_int(self):
        """Env var string is converted to int."""
        os.environ["MAX_TOTAL_BOTS_PER_USER"] = "7"
        try:
            import importlib
            import app.config as config_module
            importlib.reload(config_module)
            assert isinstance(config_module.MAX_TOTAL_BOTS_PER_USER, int)
            assert config_module.MAX_TOTAL_BOTS_PER_USER == 7
        finally:
            os.environ.pop("MAX_TOTAL_BOTS_PER_USER", None)
