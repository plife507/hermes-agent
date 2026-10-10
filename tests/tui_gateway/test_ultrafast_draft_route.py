"""Draft tier validation uses the chosen route before an agent exists."""

from unittest.mock import patch

import pytest

from tui_gateway import server

CODEX = {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}
API = {"provider": "openai-api", "base_url": "https://api.openai.com/v1"}
PROXY = {"provider": "openai-codex", "base_url": "https://proxy.example/v1"}


@pytest.mark.parametrize("profile_route,override,accepted", [
    (API, CODEX, True), (CODEX, API, False), (CODEX, PROXY, False),
    (CODEX, {}, True), (API, {}, False),
    (API, {"provider": "openai-codex"}, True),
    (PROXY, {"provider": "openai-codex"}, False),
    (CODEX, {"provider": "openrouter"}, False),
    (CODEX, {"provider": "anthropic"}, False),
    (CODEX, {"provider": "xai"}, False),
])
def test_lazy_draft_validates_route_and_never_writes_global(profile_route, override, accepted):
    session = {"session_key": "draft-key", "agent": None,
               "model_override": {"model": "gpt-6.1-sol-900k", **override}}
    config = {"model": {"default": "gpt-6-astra", **profile_route}}
    with patch.dict(server._sessions, {"draft": session}, clear=True), \
            patch.object(server, "_load_cfg", return_value=config), \
            patch.object(server, "_write_config_key") as write, \
            patch.object(server, "_start_agent_build") as build:
        response = server._methods["config.set"]("r", {"session_id": "draft", "key": "fast", "value": "ultrafast"})
    if accepted:
        assert response["result"]["value"] == "ultrafast"
        assert session["create_service_tier_override"] == "ultrafast"
    else:
        assert response["error"]["code"] == 4002
        assert "create_service_tier_override" not in session
    write.assert_not_called()
    build.assert_not_called()


@pytest.mark.parametrize("route,accepted", [(CODEX, True), (API, False), (PROXY, False)])
def test_profile_default_validates_the_route_without_agent_build(route, accepted):
    session = {"session_key": "draft-key", "agent": None}
    with patch.dict(server._sessions, {"draft": session}, clear=True), \
            patch.object(server, "_load_cfg", return_value={"model": {"default": "gpt-6.1-sol", **route}}), \
            patch.object(server, "_write_config_key") as write:
        response = server._methods["config.set"]("r", {"session_id": "draft", "key": "fast", "value": "ultrafast"})
    assert ("result" in response) == accepted
    assert session.get("create_service_tier_override") == ("ultrafast" if accepted else None)
    write.assert_not_called()


def test_draft_reads_its_own_profile_config_not_the_launch_profile(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home_override

    launch, secondary = tmp_path / "launch", tmp_path / "secondary"
    launch.mkdir()
    secondary.mkdir()
    launch_cfg = "model:\n  default: gpt-6.1-sol\n  provider: openai-api\n  base_url: https://api.openai.com/v1\n"
    secondary_cfg = "model:\n  default: gpt-6.1-sol\n  provider: openai-codex\n  base_url: https://chatgpt.com/backend-api/codex\n"
    (launch / "config.yaml").write_text(launch_cfg)
    (secondary / "config.yaml").write_text(secondary_cfg)
    monkeypatch.setattr(server, "_hermes_home", launch)
    sessions = {"launch": {"session_key": "launch", "agent": None, "profile_home": launch},
                "secondary": {"session_key": "secondary", "agent": None, "profile_home": secondary}}
    previous_home = get_hermes_home_override()
    with patch.dict(server._sessions, sessions, clear=True), patch.object(server, "_write_config_key") as write:
        for sid, accepted in (("secondary", True), ("launch", False)):
            response = server._methods["config.set"]("r", {"session_id": sid, "key": "fast", "value": "ultrafast"})
            assert ("result" in response) == accepted
    assert sessions["secondary"]["create_service_tier_override"] == "ultrafast"
    assert "create_service_tier_override" not in sessions["launch"]
    assert get_hermes_home_override() == previous_home
    assert (launch / "config.yaml").read_text() == launch_cfg
    assert (secondary / "config.yaml").read_text() == secondary_cfg
    write.assert_not_called()
