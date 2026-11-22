import sys, time
from pathlib import Path
import chess.pgn, io
import decoder

ocr = Path(sys.argv[1]).read_text(encoding="utf-8")
expected = chess.pgn.read_game(io.StringIO(Path(sys.argv[2]).read_text(encoding="utf-8")))
exp = list(expected.mainline_moves())
t = time.perf_counter()
d = decoder.decode(ocr)
print(f"convention={d.convention} cost={d.cost:.1f} plies={len(d.moves)}/{len(exp)} time={time.perf_counter()-t:.1f}s result={d.result}")
match = next((i for i,(a,b) in enumerate(zip(d.moves, exp)) if a!=b), min(len(d.moves),len(exp)))
print("exact" if d.moves==exp else f"first difference at ply {match+1}")
for r in d.repairs: print("  repair:", r)
print("runner-up:", d.runner_up_cost, "differs at ply", d.runner_up_differs_at)
print(decoder.to_pgn_movetext(d))
