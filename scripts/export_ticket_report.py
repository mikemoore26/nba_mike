from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model_training.common.slate import latest_results_dir


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_num(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _make_summary_table(df: pd.DataFrame) -> Table:
    data = [["Ticket", "Legs", "Avg Pred", "Avg Minutes"]]

    if df.empty:
        data.append(["No summary found", "", "", ""])
    else:
        for _, row in df.iterrows():
            data.append(
                [
                    str(row.get("ticket_name", "")),
                    str(row.get("n_legs", "")),
                    _fmt_num(row.get("avg_pred_mean", ""), 1),
                    _fmt_num(row.get("avg_minutes_proj", ""), 1),
                ]
            )

    table = Table(
        data,
        colWidths=[1.6 * inch, 0.8 * inch, 1.0 * inch, 1.2 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _make_ticket_table(df: pd.DataFrame) -> Table:
    headers = ["Player", "Team", "Opp", "Stat", "Pred", "Min"]
    data = [headers]

    if df.empty:
        data.append(["No ticket generated", "", "", "", "", ""])
    else:
        for _, row in df.iterrows():
            data.append(
                [
                    str(row.get("player", "")),
                    str(row.get("team", "")),
                    str(row.get("opp", "")),
                    str(row.get("stat", "")),
                    _fmt_num(row.get("pred_mean", ""), 1),
                    _fmt_num(row.get("minutes_proj", ""), 1),
                ]
            )

    table = Table(
        data,
        colWidths=[
            2.2 * inch,
            0.7 * inch,
            0.7 * inch,
            0.8 * inch,
            0.8 * inch,
            0.8 * inch,
        ],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _build_pdf(
    *,
    out_path: Path,
    game_date: str,
    summary_df: pd.DataFrame,
    safe_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    lotto_df: pd.DataFrame,
) -> None:
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1D4ED8"),
        spaceAfter=8,
        spaceBefore=8,
    )

    story = []

    story.append(Paragraph("NBA Projection Tickets", title_style))
    story.append(
        Paragraph(
            f"Slate date: {game_date} &nbsp;&nbsp;&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            subtitle_style,
        )
    )

    story.append(Paragraph("Summary", section_style))
    story.append(_make_summary_table(summary_df))
    story.append(Spacer(1, 0.22 * inch))

    story.append(Paragraph("Safe Ticket", section_style))
    story.append(_make_ticket_table(safe_df))
    story.append(Spacer(1, 0.22 * inch))

    story.append(Paragraph("Balanced Ticket", section_style))
    story.append(_make_ticket_table(balanced_df))
    story.append(Spacer(1, 0.22 * inch))

    story.append(Paragraph("Lotto Ticket", section_style))
    story.append(_make_ticket_table(lotto_df))

    doc.build(story)


def main() -> None:
    slate_date, results_dir = latest_results_dir()

    print(f"[REPORT] Using slate_date = {slate_date}")

    summary_df = _load_csv(results_dir / "ticket_summary.csv")
    safe_df = _load_csv(results_dir / "ticket_safe.csv")
    balanced_df = _load_csv(results_dir / "ticket_balanced.csv")
    lotto_df = _load_csv(results_dir / "ticket_lotto.csv")

    out_path = results_dir / "ticket_report.pdf"
    _build_pdf(
        out_path=out_path,
        game_date=slate_date,
        summary_df=summary_df,
        safe_df=safe_df,
        balanced_df=balanced_df,
        lotto_df=lotto_df,
    )

    print(f"[REPORT] Saved -> {out_path}")


if __name__ == "__main__":
    main()