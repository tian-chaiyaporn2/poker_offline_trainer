"""Card model and canonical notation (MIT).

Cards are represented as integers 0..51 for speed:

    card_int = rank_index * 4 + suit_index

    rank_index: 0..12  ->  2 3 4 5 6 7 8 9 T J Q K A
    suit_index: 0..3   ->  c d h s   (see SUITS)

Canonical string notation (see docs/scenario_format.md):
    rank in "23456789TJQKA", suit in "shdc" (lowercase).

Suit ordering for isomorphism/normalisation is s > h > d > c. We store suits
with that ordering baked into the index so that a larger suit_index means a
"higher" suit in canonical order.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

RANKS = "23456789TJQKA"          # index 0..12
# Canonical priority order: s > h > d > c. suit_index 3=s,2=h,1=d,0=c.
SUITS = "cdhs"                    # index 0..3 (so 's' is highest index)

_RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}
_SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}


def make_card(rank_idx: int, suit_idx: int) -> int:
    return rank_idx * 4 + suit_idx


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def parse_card(text: str) -> int:
    """Parse a 2-char card like 'Ah' or 'Td' into an int 0..51."""
    if len(text) != 2:
        raise ValueError(f"card must be 2 chars, got {text!r}")
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in _RANK_TO_IDX:
        raise ValueError(f"bad rank in {text!r}")
    if suit not in _SUIT_TO_IDX:
        raise ValueError(f"bad suit in {text!r}")
    return make_card(_RANK_TO_IDX[rank], _SUIT_TO_IDX[suit])


def card_str(card: int) -> str:
    return RANKS[card_rank(card)] + SUITS[card_suit(card)]


def parse_cards(text: str | Iterable[str]) -> List[int]:
    """Parse 'AhKd' or ['Ah','Kd'] or 'Ah Kd' into a list of card ints."""
    if isinstance(text, str):
        cleaned = text.replace(" ", "")
        if len(cleaned) % 2 != 0:
            raise ValueError(f"odd-length card string {text!r}")
        tokens = [cleaned[i : i + 2] for i in range(0, len(cleaned), 2)]
    else:
        tokens = list(text)
    return [parse_card(t) for t in tokens]


def cards_str(cards: Iterable[int]) -> str:
    return "".join(card_str(c) for c in cards)


def parse_hand(text: str) -> Tuple[int, int]:
    """Parse a two-card hand, returned high-card-first (by rank then suit)."""
    cards = parse_cards(text)
    if len(cards) != 2:
        raise ValueError(f"hand must be 2 cards, got {text!r}")
    a, b = cards
    if a == b:
        raise ValueError(f"duplicate card in hand {text!r}")
    return (a, b) if a > b else (b, a)


def hand_str(hand: Tuple[int, int]) -> str:
    a, b = hand
    hi, lo = (a, b) if a > b else (b, a)
    return card_str(hi) + card_str(lo)


FULL_DECK: Tuple[int, ...] = tuple(range(52))
