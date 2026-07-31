from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "demo" / "build_trainer.py"
GENERATED = (ROOT / "index.html", ROOT / "demo" / "trainer_demo.html")


class _IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])


def test_mobile_player_loop_contract_is_generated_from_source():
    source = SOURCE.read_text()
    required = (
        "const SESSION_SIZE=10",
        "class=\"session-hud\"",
        "w.className=\"tv duel\"",
        ".acts.n-3{grid-template-columns:repeat(3",
        "class=\"fb-actions\"",
        "function showSessionEnd()",
        "trainer-progress",
        "function normalizeStats(x)",
        "aria-pressed",
        "aria-current",
        "role=\"dialog\"",
        "setView(\"train\")",
    )
    for marker in required:
        assert marker in source


def test_generated_trainers_have_unique_control_ids():
    for path in GENERATED:
        parser = _IdCollector()
        parser.feed(path.read_text())
        duplicates = {item for item in parser.ids if parser.ids.count(item) > 1}
        assert not duplicates, f"{path}: duplicate ids {sorted(duplicates)}"


def test_generated_trainers_include_mobile_session_and_feedback_controls():
    required_ids = {
        "play-card",
        "session-step",
        "session-total",
        "session-end",
        "retry-leaks",
        "new-session",
        "learn",
        "coach-open",
        "progress-note",
        "terms-learned",
        "learn-summary",
    }
    for path in GENERATED:
        parser = _IdCollector()
        parser.feed(path.read_text())
        assert required_ids <= set(parser.ids)


def test_generated_trainers_expose_accessible_state_and_modal_semantics():
    for path in GENERATED:
        html = path.read_text()
        assert 'role="progressbar"' in html
        assert 'role="dialog" aria-modal="true" aria-labelledby="verdict"' in html
        assert 'aria-label="Primary"' in html
        assert 'setAttribute("aria-pressed",String(on))' in html
        assert 'setAttribute("aria-current","page")' in html
        assert 'id="play-card" tabindex="-1"' in html
        assert 'document.getElementById("hd-back").focus' in html
        assert 'document.getElementById("play-card").focus' in html
        assert 'e.setAttribute("role","img")' in html
        assert '" of "+(SUIT_NAME[s]||"spades")' in html


def test_mobile_ux_copy_and_compact_layout_contract():
    source = SOURCE.read_text()
    assert ".hint,.act .k{display:none}" in source
    assert ".sit .demo{display:none}" in source
    assert '"Why "+actionPrimary(pref)+" is stronger"' in source
    assert '"How the choices compare"' in source
    assert 'oppAct="bet "+(q.bet_pct||66)+"%"' in source
    assert '"You are "+q.acting_player+" · "+cap1(q.street)' in source


def test_settings_language_change_does_not_open_feedback_over_settings():
    source = SOURCE.read_text()
    assert 'if(!document.getElementById("v-train").classList.contains("on"))closeSheet(false);' in source


def test_settings_copy_matches_later_street_raise_support():
    source = SOURCE.read_text()
    assert "with Fold/Call/Raise available on facing-a-bet nodes" in source
    assert "facing-a-bet there is Fold/Call" not in source
