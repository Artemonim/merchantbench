"""Dashboard rendering affordances for cached-token usage."""

from pathlib import Path


def test_dashboard_shows_cache_read_write_and_reasoning_kpis():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "Cache read" in html
    assert "Cache write" in html
    assert "Reasoning" in html
    assert "cacheReadTokens(totals).toLocaleString()" in html
    assert "(totals.cache_write || 0).toLocaleString()" in html
    assert "(totals.reasoning || 0).toLocaleString()" in html


def test_dashboard_turn_label_shows_cache_read_write_and_reasoning_tokens():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "const tokLabelParts = []" in html
    assert "`${cacheRead} cache read`" in html
    assert "`${tu.cache_write} cache write`" in html
    assert "`${tu.reasoning} reasoning`" in html


def test_dashboard_cache_read_falls_back_to_legacy_cached_tokens():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "const cacheReadTokens = (tu) => tu?.cache_read ?? tu?.cached ?? 0;" in html
    assert "tokCached += cacheReadTokens(tu);" in html
    assert "const cacheRead = cacheReadTokens(tu);" in html


def test_dashboard_shows_context_and_compaction_labels():
    html = Path("env/web/templates/dashboard.html").read_text(encoding="utf-8")

    assert "function fmtContextTokens" in html
    assert "contextByStep" in html
    assert "`ctx ${fmtContextTokens(stepContext.tokens)}`" in html
    assert "`compact x${stepContext.compactions}`" in html
    assert "`ctx ${fmtContextTokens(stepContext.tokens)} tok`" in html
    assert "if (turnContext.compacted === true) metaParts.push('compacted');" in html
