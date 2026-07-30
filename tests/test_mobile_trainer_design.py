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
    }
    for path in GENERATED:
        parser = _IdCollector()
        parser.feed(path.read_text())
        assert required_ids <= set(parser.ids)
