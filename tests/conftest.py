import pytest

from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_tests_from_local_dotenv(monkeypatch):
    """Keep real developer credentials out of normal deterministic tests."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
