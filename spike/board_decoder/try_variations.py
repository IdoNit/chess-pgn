import sys
from pathlib import Path
import variations
EX = Path("/mnt/user-data/uploads/chess_pgn_project/examples")
cases = [("testpages/korchnoi_razov.txt", "1. Korchnoi-Razov - Leningrad 1946.pgn"),
         ("testpages/zikov_korchnoi.txt", "2. Zikov-Korchnoi - Leningrad 1946.pgn")]
for page, expected_name in cases:
    game, decoded = variations.build_game(Path(page).read_text(encoding="utf-8"), "english")
    expected = variations.read_pgn((EX / expected_name).read_text(encoding="utf-8"))
    main_ok = list(game.mainline_moves()) == list(expected.mainline_moves())
    got, exp = variations.all_variation_moves(game), variations.all_variation_moves(expected)
    print(f"== {page}: main line {'exact' if main_ok else 'DIFFERENT'}; side-line moves: {len(got & exp)}/{len(exp)} expected found, {len(got - exp)} extra")
    exporter = __import__("chess.pgn").pgn.StringExporter(headers=False)
    print(game.accept(exporter)); print()
