
import pytest


@pytest.fixture(autouse=True)
def _legacy_tests_run_without_route_guard(request, monkeypatch):
    """Older tests call the API anonymously. Security tests opt back in with
    @pytest.mark.enforce_auth so the real RBAC guard is exercised."""
    if request.node.get_closest_marker("enforce_auth"):
        monkeypatch.setenv("EXAM_AGENT_AUTH_ENFORCE", "1")
    else:
        monkeypatch.setenv("EXAM_AGENT_AUTH_ENFORCE", "0")
