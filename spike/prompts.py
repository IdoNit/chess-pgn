"""Prompts shared by both pipelines."""

OCR_PROMPT = "Text Recognition:"

_RULES = """\
Rules:
- Output ONLY the PGN: the header tags, a blank line, then the movetext. No explanations, no code fences.
- Headers: Event, Site, Date, Round, White, Black, Result, and Opening if the book names it.
  Use what the page says. Unknown parts are written PGN-style: Date "1946.??.??" if only the year is known, Round "?".
- Moves in standard English algebraic notation (K Q R B N; pawns have no letter).
  Convert figurines (♔♕♖♗♘) and descriptive notation (P-K4, N-KB3, QxP...) to algebraic.
  Books often abbreviate pawn captures as two files ("ed", "dc", "gf"): write the full move, e.g. exd5.
  Castling is O-O / O-O-O. "++" in a book means double check; write a single "+".
- The main line is the game itself. Side lines in brackets [ ] or ( ) are alternatives to the game moves:
  write them as PGN variations in ( ) placed right after the game move they replace.
  A ";" inside brackets starts ANOTHER alternative, which may branch from a different move number
  (for example "[8...Ne5! 9 Bb3? c5; 9 Be2 d5]": "9 Be2" is an alternative to "9 Bb3" inside that side line).
  Nest variations accordingly.
- Keep the book's move symbols (! ? !! ?? !? ?!) attached to moves. An "=" evaluation becomes $10.
- Short remarks inside side lines (e.g. "double check", "with the idea ...") go in {curly braces}.
  Long prose commentary is dropped.
- Result: 1-0, 0-1, 1/2-1/2. "1:0" means 1-0. "and won" / "resigned" tell you who won.
- Every move must be legal. If a side line in the book contains an impossible move, end that variation
  before it and put the rest of the side line text in a {comment}.
"""

NORMALIZE_FROM_TEXT = f"""\
Below is OCR text of a page from a chess book. It contains one game.
Convert that game into PGN.

{_RULES}
OCR text:
<<<
@@OCR_TEXT@@
>>>
"""

DIRECT_FROM_IMAGE = f"""\
The image(s) show a page from a chess book containing one game (if there are several images, they are
consecutive pages of the same game, in order). Convert that game into PGN.

{_RULES}"""

REPAIR = """\
Your PGN has a move that is illegal on the board:

{problem}

Re-read the source and return the complete corrected PGN (same rules as before, PGN only).
If the source itself gives an impossible move in a side line, end that variation before it and put
the rest of the side line in a {{comment}}.

Your previous PGN:
{pgn}
"""

REPAIR_TEXT_SOURCE = "\n\nSource OCR text:\n<<<\n{ocr_text}\n>>>\n"
