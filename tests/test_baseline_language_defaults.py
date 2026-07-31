"""Language defaults for bundled baseline agents."""

from agent.baselines import auto_seed, react_160k_compact_30k


class _FakeToolClient:
    def __init__(self, *args, **kwargs):
        pass


def test_baseline_agents_default_to_english(monkeypatch):
    monkeypatch.setattr(auto_seed, "MerchantBenchToolClient", _FakeToolClient)
    monkeypatch.setattr(react_160k_compact_30k, "MerchantBenchToolClient", _FakeToolClient)

    auto = auto_seed.AutoSeedAgent.__new__(auto_seed.AutoSeedAgent)
    compact = react_160k_compact_30k.ReActAgent.__new__(react_160k_compact_30k.ReActAgent)

    auto_seed.AutoSeedAgent.__init__(auto, "http://example.test", "run_id", "agent_0")
    react_160k_compact_30k.ReActAgent.__init__(
        compact,
        "http://example.test",
        "run_id",
        "agent_0",
        openai_client=object(),
        model="fake-model",
    )

    assert auto.language == "en"
    assert compact.language == "en"


def test_baseline_brief_language_fallback_is_english():
    auto = auto_seed.AutoSeedAgent.__new__(auto_seed.AutoSeedAgent)
    compact = react_160k_compact_30k.ReActAgent.__new__(react_160k_compact_30k.ReActAgent)

    auto.system_prompt = None
    compact.system_prompt = None

    auto_seed.AutoSeedAgent._update_brief_from_obs(auto, {"brief": {"system_prompt": "rules"}})
    react_160k_compact_30k.ReActAgent._update_brief_from_obs(compact, {"brief": {"system_prompt": "rules"}})

    assert auto.language == "en"
    assert compact.language == "en"
    assert auto.system_prompt == "rules"
    assert compact.system_prompt == "rules"
