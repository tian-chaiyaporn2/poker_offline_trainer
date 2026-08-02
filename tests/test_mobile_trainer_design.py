import json
import re
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


def _embedded_json(html, name):
    match = re.search(rf"const {name} = (\[[^\n]+\]);", html)
    assert match, f"missing embedded {name}"
    return json.loads(match.group(1))


def test_mobile_player_loop_contract_is_generated_from_source():
    source = SOURCE.read_text()
    required = (
        "const SESSION_SIZE=10",
        "class=\"session-hud\"",
        "w.className=\"stage duel\"",
        ".acts.n-3{grid-template-columns:repeat(3",
        "class=\"fb-actions\"",
        "function showSessionEnd()",
        "trainer-progress",
        "function normalizeStats(x)",
        "aria-pressed",
        "aria-current",
        "role=\"dialog\"",
        "setView(\"train\")",
        "function skipBonus()",
        "html.sheet-open,html.sheet-open body{overflow:hidden}",
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
        "session-hud",
        "session-step",
        "session-total",
        "session-progress",
        "session-end",
        "retry-leaks",
        "new-session",
        "skip-bonus",
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
        assert 'document.getElementById("session-hud").hidden=true' in html


def test_generated_question_data_is_internally_consistent():
    street_cards = {"flop": 3, "turn": 4, "river": 5}
    for path in GENERATED:
        questions = _embedded_json(path.read_text(), "Q")
        signatures = set()
        for q in questions:
            actions = q["actions"]
            assert len(actions) >= 2
            assert len(actions) == len(set(actions))
            target = q["answer"] if q.get("preflop") else q["preferred"]
            assert target in actions
            cards = q["hand"] if q.get("preflop") else q["board"] + q["hero"]
            assert len(cards) == len(set(cards))
            if not q.get("preflop"):
                assert len(q["board"]) == street_cards[q["street"]]
                assert set(actions) <= set(q["ev"])
                assert set(actions) <= set(q["freq"])
                assert set(actions) <= set(q["grades"])
                assert sum(q["freq"][action] for action in actions) == 100
                assert 0 < q["bet_pct"] <= 500
                assert "villain" in q and "is_oop" in q
            signature = (
                "preflop" if q.get("preflop") else q["street"],
                q.get("node") or q.get("ctx"),
                q.get("pos") or q.get("acting_player"),
                q.get("villain"),
                q.get("is_oop"),
                tuple(cards),
                tuple(actions),
            )
            assert signature not in signatures
            signatures.add(signature)


def test_mobile_ux_copy_and_compact_layout_contract():
    source = SOURCE.read_text()
    assert ".hint,.act .k{display:none}" in source
    assert ".sit .demo{display:none}" in source
    assert '"Why "+actionPrimary(pref)+" is stronger"' in source
    assert '"How the choices compare"' in source
    assert 'oppAct="Bets "+(q.bet_pct||66)+"%"' in source
    assert 'document.getElementById("sitcontext").textContent=cap1(q.street)' in source
    assert "with Fold/Call/Raise available on facing-a-bet nodes" in source


def test_session_history_and_comparison_practice_are_accounted_correctly():
    source = SOURCE.read_text()
    assert "function shownStep()" in source
    assert "step:pos,bonus:false" in source
    assert "step:shownStep(),bonus:true" in source
    assert 'document.getElementById("session-counter").hidden=bonus' in source
    assert "const inSession=!(hist[hidx]&&hist[hidx].bonus)" in source
    assert "if(inSession)recordGrade(stats,tier,hit)" in source
    assert "if(inSession&&!hit)sessionMisses.push(cur)" in source
    assert "queued&&queued.bonus&&queued.q===c.q" in source
    assert "if(hist[hidx]&&hist[hidx].bonus&&!answered){skipBonus();return;}" in source


def test_persisted_state_and_modal_escape_are_hardened():
    source = SOURCE.read_text()
    assert "const classified=out.solid+out.ok+out.leak;out.n=classified" in source
    assert "const VALID_TERMS=new Set" in source
    assert "raw.filter(t=>VALID_TERMS.has(t))" in source
    assert 'typeof c!=="object"||Array.isArray(c)' in source
    assert 'if(e.key==="Escape"){e.preventDefault();closeSheet();return;}' in source
    assert 'if(!document.getElementById("session-end").hidden)return;' in source
    assert 'e.target.closest("button,summary,a[href],input,select,textarea"))return;' in source
    assert 'document.querySelectorAll("#cats button").forEach(b=>{b.disabled=true;});' in source


def test_modal_and_view_transitions_restore_or_contain_focus():
    source = SOURCE.read_text()
    assert 'id="session-end" tabindex="-1"' in source
    assert "function trapModalTab(root,e)" in source
    assert 'if(e.key==="Tab")trapModalTab(document.getElementById("handdetail"),e)' in source
    assert 'if(e.key==="Tab")trapModalTab(document.getElementById("fb"),e)' in source
    assert 'document.getElementById("session-end").focus' in source
    assert 'document.getElementById("next").focus({preventScroll:true});}  // review' in source
    assert 'fb.addEventListener("touchcancel"' in source
    assert "closeSheet(false);" in source


def test_correct_preflop_answers_unlock_position_language():
    source = SOURCE.read_text()
    assert "const gained=hit?tryUnlockPreflop():[]" in source
    assert "function tryUnlockPreflop()" in source
    assert "preflop?pfActLabel(a):actionPrimary(a)" in source
    assert '"Why "+pfActLabel(a)+" works here"' in source
    assert '"Why "+pfActLabel(q.answer)+" is stronger"' in source


def test_settings_language_change_does_not_open_feedback_over_settings():
    source = SOURCE.read_text()
    assert 'if(!document.getElementById("v-train").classList.contains("on"))closeSheet(false);' in source
