from pathlib import Path


def test_scenario_yaml_editor_has_no_horizontal_background_lines():
    theme_html = Path("env/web/templates/_pixel_theme.html").read_text(encoding="utf-8")

    assert "background-size: 100% 24px" not in theme_html
    assert "linear-gradient(var(--pixel-grid) 1px, transparent 1px),\n      var(--panel)" not in theme_html
