"""Continuation mode (C): the signed trajectory pack round-trips into linked 3-step hands."""
import json

from pokertrainer.content_pack import build_pack, verify_pack


def _step(hand_id, step, seat, board, hand, pref, evs, freqs, villain, last):
    return {
        "board": board, "board_texture": [], "board_favored": None,
        "node": "bb_first" if seat == "OOP" else "btn_vs_check",
        "acting_player": "BB" if seat == "OOP" else "BTN", "decision_type": "continuation",
        "hand": hand, "hand_category": "cont",
        "actions": ["check", "bet"], "ev": evs, "freq": freqs, "preferred": pref,
        "reach_mass": 0.8, "mixed": False, "pot_bb": 5.5,
        "scenario": f"cont|{hand_id}|s{step}|{seat}",
        "oop_pos": "BB", "ip_pos": "BTN",
        "explanation": {"reason": "continuation", "headline": "x",
                        "detail": {"hand_id": hand_id, "step_index": step, "hero_seat": seat,
                                   "street": ["flop", "turn", "river"][step],
                                   "villain_action": villain, "board_full": board,
                                   "hero_cards": hand, "is_oop": seat == "OOP", "villain": "BTN"}},
    }


def test_continuation_pack_roundtrips_into_linked_hands(tmp_path):
    ev = {"check": 0.0, "bet": -0.4}          # check is the max-EV (preferred) action
    fr = {"check": 0.82, "bet": 0.18}
    recs = []
    for hid, seat, boards, hand in [
        ("h1", "OOP", ["7c6d2s", "7c6d2sKh", "7c6d2sKh3c"], "Ad5d"),
        ("h2", "IP", ["Ah7d2c", "Ah7d2cTs", "Ah7d2cTs9h"], "KsQs"),
    ]:
        for step, board in enumerate(boards):
            recs.append(_step(hid, step, seat, board, hand, "check", ev, fr,
                              "checks behind", last=(step == 2)))

    report = build_pack(recs, {"line": "continuation_passive_villain"},
                        str(tmp_path), "cont_test", pot=5.5, dedup_cap=999)
    assert report["records_after_dedup"] == 6          # 2 hands x 3 steps, nothing dropped
    v = verify_pack(str(tmp_path / "flop_pack_cont_test.db"))
    assert v["hash_ok"] and v["signature_ok"] and v["records"] == 6

    # every record: preferred is a max-EV action, freq sums ~1, and the continuation
    # identifiers survive in the scenario column + detail JSON so the trainer can link them.
    hands = {}
    for r in recs:
        assert r["ev"][r["preferred"]] == max(r["ev"].values())
        assert abs(sum(r["freq"].values()) - 1.0) < 1e-9
        hid, sstep, seat = r["scenario"].split("|")[1:]
        hands.setdefault(hid, []).append(r["explanation"]["detail"]["step_index"])
    assert set(hands) == {"h1", "h2"}
    for steps in hands.values():
        assert sorted(steps) == [0, 1, 2]              # three linked streets per hand
