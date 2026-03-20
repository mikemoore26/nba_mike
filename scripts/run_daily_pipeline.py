from __future__ import annotations

from scripts.predict_all import main as predict_all_main
from scripts.build_tickets import main as build_tickets_main
from scripts.export_ticket_pdf import TicketPDFExporter


def main() -> None:
    print("=" * 60)
    print("STEP 1/3: RUNNING PREDICTIONS")
    print("=" * 60)
    predict_all_main()

    print("\n" + "=" * 60)
    print("STEP 2/3: BUILDING TICKETS")
    print("=" * 60)
    build_tickets_main()

    print("\n" + "=" * 60)
    print("STEP 3/3: EXPORTING PDF")
    print("=" * 60)
    exporter = TicketPDFExporter()
    exporter.export_pdf()

    print("\n" + "=" * 60)
    print("DAILY PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
