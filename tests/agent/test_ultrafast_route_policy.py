"""Ultrafast is a Codex subscription opt-in, never an API/proxy request."""

from types import SimpleNamespace

import pytest

from agent.fast_mode import effective_request_overrides
from hermes_cli.models import resolve_fast_mode_overrides

CODEX_URL = "https://chatgpt.com/backend-api/codex"
REJECTED_ROUTES = [
    (None, None), (None, CODEX_URL),
    ("openai", None), ("openai-api", "https://api.openai.com/v1"),
    ("anthropic", "https://api.anthropic.com"), ("xai", "https://api.x.ai/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"), ("custom", "https://proxy.example/v1"),
    ("openai-codex", "https://proxy.example/v1"),
    ("openai-codex", "https://chatgpt.com.proxy.example/v1"),
    ("openai-codex", "https://api.openai.com/v1"),
    ("openai-codex", "proxy.example/v1"),
]


@pytest.mark.parametrize("provider,base_url", REJECTED_ROUTES)
def test_request_gate_refuses_non_subscription_routes(provider, base_url):
    assert resolve_fast_mode_overrides("gpt-6.1-sol-900k", provider=provider,
                                       base_url=base_url, tier="ultrafast") is None


@pytest.mark.parametrize("provider,base_url", REJECTED_ROUTES)
@pytest.mark.parametrize("service_tier", ["ultrafast", None])
def test_stale_or_manual_ultrafast_is_removed_before_request(provider, base_url, service_tier):
    stored = {"service_tier": "ultrafast", "temperature": 0.2}
    agent = SimpleNamespace(model="gpt-6.1-sol", provider=provider, base_url=base_url,
                            api_mode="codex_responses", service_tier=service_tier, request_overrides=stored)
    assert effective_request_overrides(agent) == {"temperature": 0.2}
    assert stored["service_tier"] == "ultrafast"  # per-request filtering must not mutate saved state


def test_switching_route_or_model_rechecks_ultrafast_without_changing_reasoning():
    agent = SimpleNamespace(model="gpt-6.1-sol", provider="openai-codex", base_url=CODEX_URL,
                            api_mode="codex_responses", service_tier="ultrafast",
                            request_overrides={"service_tier": "ultrafast", "reasoning_effort": "high"})
    assert effective_request_overrides(agent) == agent.request_overrides
    agent.provider, agent.base_url = "openai-api", "https://api.openai.com/v1"
    assert effective_request_overrides(agent) == {"reasoning_effort": "high"}
    agent.provider, agent.base_url, agent.model = "openai-codex", CODEX_URL, "gpt-6-sol"
    assert effective_request_overrides(agent) == {"reasoning_effort": "high"}
    agent.request_overrides["service_tier"] = "priority"
    assert effective_request_overrides(agent) == agent.request_overrides


def test_manual_extra_body_ultrafast_cannot_bypass_the_route_gate():
    stored = {"extra_body": {"service_tier": "ultrafast", "keep": 1}, "reasoning_effort": "high"}
    agent = SimpleNamespace(model="gpt-6.1-sol", provider="openai-codex", base_url=CODEX_URL,
                            api_mode="codex_responses", service_tier=None, request_overrides=stored)
    assert effective_request_overrides(agent) == stored
    agent.provider, agent.base_url = "openai-api", "https://api.openai.com/v1"
    assert effective_request_overrides(agent) == {"extra_body": {"keep": 1}, "reasoning_effort": "high"}
    assert stored["extra_body"]["service_tier"] == "ultrafast"
