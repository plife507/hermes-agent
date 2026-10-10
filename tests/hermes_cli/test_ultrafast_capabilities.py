"""The picker advertises exact request-tier eligibility, not generic Fast support."""

import pytest

from hermes_cli.inventory import _apply_capabilities
from hermes_cli.models import resolve_fast_mode_overrides


@pytest.mark.parametrize("provider,base_url", [
    ("openai-codex", "https://chatgpt.com/backend-api/codex"),
    ("openai", "https://api.openai.com/v1"),
    ("openai-api", "https://api.openai.com/v1"),
    ("openai-codex", "https://proxy.example/v1"),
    ("custom", "https://proxy.example/v1"),
])
def test_catalog_ultrafast_matches_the_exact_route_gate(provider, base_url):
    row = {"slug": provider, "api_url": base_url, "models": ["gpt-6.1-sol-900k", "gpt-6-astra", "gpt-6-sol"]}
    _apply_capabilities([row], metadata_config={})
    for model, caps in row["capabilities"].items():
        eligible = provider == "openai-codex" and "chatgpt.com/" in base_url and model != "gpt-6-sol"
        assert bool(caps.get("ultrafast")) == eligible
        assert caps["fast"] == (resolve_fast_mode_overrides(model, provider=provider, base_url=base_url) is not None)
