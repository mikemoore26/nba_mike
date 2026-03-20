from __future__ import annotations

from scripts.predict_all import main as predict_all
from scripts.build_tickets import main as build_tickets
from scripts.export_projection_report import main as export_projection_report
from scripts.export_ticket_report import main as export_ticket_report


def main() -> None:
    print("=" * 60)
    print("STEP 1: PREDICT ALL")
    print("=" * 60)
    predict_all()

    print()
    print("=" * 60)
    print("STEP 2: BUILD TICKETS")
    print("=" * 60)
    build_tickets()

    print()
    print("=" * 60)
    print("STEP 3: EXPORT PROJECTION REPORT")
    print("=" * 60)
    export_projection_report()

    print()
    print("=" * 60)
    print("STEP 4: EXPORT TICKET REPORT")
    print("=" * 60)
    export_ticket_report()

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Outputs generated in the latest results folder:")
    print("  pred_ast.csv")
    print("  pred_reb.csv")
    print("  pred_fg3.csv")
    print("  pred_pts.csv")
    print("  ticket_safe.csv")
    print("  ticket_balanced.csv")
    print("  ticket_lotto.csv")
    print("  ticket_summary.csv")
    print("  projection_report.pdf")
    print("  ticket_report.pdf")


if __name__ == "__main__":
    main()