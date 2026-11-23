"""Score a converted game against the book's expected game.

The question that matters is not "how many moves are right" but "is every wrong move flagged": a flagged mistake
costs the reader a click, a silent one corrupts the PGN.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher

NEAR = 1  # a flag this many plies away still points the reader at the mistake


@dataclass
class Score:
    expected: int
    correct: int
    wrong: int  # expected moves missing or different in the output
    extra: int  # output moves past or beside the book's game
    unflagged_wrong: int
    silent_plies: list[int] = field(default_factory=list)  # output plies (1-based) of the silent errors
    exact: bool = False


def score(expected: list[str], output: list[str], flagged: list[int],
          accept: dict[int, list[str]] | None = None) -> Score:
    """`expected` and `output` are SAN main lines; `flagged` are the output's flagged plies (1-based);
    `accept` maps an expected ply (1-based) to the SAN moves accepted there."""
    output = list(output)
    for ply, alternatives in (accept or {}).items():
        if ply <= len(output) and ply <= len(expected) and output[ply - 1] in alternatives:
            output[ply - 1] = expected[ply - 1]

    def near_flag(ply: int) -> bool:
        return any(abs(f - ply) <= NEAR for f in flagged)

    correct = wrong = extra = 0
    silent: list[int] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=expected, b=output, autojunk=False).get_opcodes():
        if tag == "equal":
            correct += i2 - i1
            continue
        wrong += i2 - i1
        if j2 > j1:  # output moves that aren't the book's: each needs a flag
            extra += max(0, (j2 - j1) - (i2 - i1))
            silent += [j + 1 for j in range(j1, j2) if not near_flag(j + 1)]
        elif j1 < len(output) and not (near_flag(j1) or near_flag(j1 + 1)):  # book moves dropped mid-game
            silent.append(j1 + 1)
    return Score(expected=len(expected), correct=correct, wrong=wrong, extra=extra, unflagged_wrong=len(silent),
                 silent_plies=silent, exact=(output == expected))
