from __future__ import annotations

import os
import sys


def _configure_utf8_output() -> None:
    """
    Make Windows console output resilient to player names with non-ASCII chars.
    """
    os.environ["PYTHONUTF8"] = "1"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_output()

from scripts.predict_ast import main as predict_ast_main
from scripts.predict_fg3 import main as predict_fg3_main
from scripts.predict_pts import main as predict_pts_main
from scripts.predict_reb import main as predict_reb_main


def main() -> None:
    print("[STEP] Predict AST...")
    predict_ast_main()

    print("[STEP] Predict FG3...")
    predict_fg3_main()

    print("[STEP] Predict PTS...")
    predict_pts_main()

    print("[STEP] Predict REB...")
    predict_reb_main()

    print("[DONE] All predictions complete.")


if __name__ == "__main__":
    main()