from __future__ import annotations

from model_training.common.slate import latest_results_dir
from ticket.build_ticket import build_all_tickets


def main() -> None:
    slate_date, results_dir = latest_results_dir()

    print(f"[TICKETS] Using slate_date = {slate_date}")

    out = build_all_tickets(results_dir=results_dir)

    print("\n===== SUMMARY =====")
    print(out["ticket_summary"].to_string(index=False))


if __name__ == "__main__":
    main()