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


# 🔥 COLOR HELPER (new)
def _p_hit_color(p):
    try:
        p = float(p)
        if p >= 0.72:
            return colors.HexColor("#16A34A")  # green
        elif p >= 0.65:
            return colors.HexColor("#2563EB")  # blue
        else:
            return colors.HexColor("#B45309")  # orange
    except:
        return colors.black


def _make_ticket_table(df: pd.DataFrame) -> Table:
    headers = ["Player", "Tm", "Opp", "Stat", "Line", "Pred", "p_hit"]
    data = [headers]

    if df.empty:
        data.append(["No ticket", "", "", "", "", "", ""])
    else:
        for _, row in df.iterrows():
            data.append(
                [
                    str(row.get("player", "")),
                    str(row.get("team", "")),
                    str(row.get("opp", "")),
                    str(row.get("stat", "")),
                    _fmt_num(row.get("line", ""), 1),
                    _fmt_num(row.get("pred_mean", ""), 1),
                    _fmt_num(row.get("p_hit", ""), 2),
                ]
            )

    table = Table(
        data,
        colWidths=[1.9 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch],
        repeatRows=1,
    )

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ])

    # 🎯 Color p_hit column dynamically
    for i, row in enumerate(data[1:], start=1):
        style.add(
            "TEXTCOLOR",
            (6, i),
            (6, i),
            _p_hit_color(row[6])
        )

    table.setStyle(style)
    return table


def _build_pdf(
    *,
    out_path: Path,
    game_date: str,
    safe_df: pd.DataFrame,
    balanced_df: pd.DataFrame,
    lotto_df: pd.DataFrame,
) -> None:

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#111827"),
        spaceAfter=10,
    )

    subtitle = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=12,
    )

    section = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=8,
        spaceAfter=6,
    )

    story = []

    # 🔥 HEADER
    story.append(Paragraph("NBA TICKET CARD", title))
    story.append(
        Paragraph(
            f"{game_date} • Generated {datetime.now().strftime('%H:%M')}",
            subtitle,
        )
    )

    # 🔐 SAFE
    story.append(Paragraph("SAFE (High Stability)", section))
    story.append(_make_ticket_table(safe_df))
    story.append(Spacer(1, 0.2 * inch))

    # ⚖️ BALANCED
    story.append(Paragraph("BALANCED (Core Mix)", section))
    story.append(_make_ticket_table(balanced_df))
    story.append(Spacer(1, 0.2 * inch))

    # 🎯 LOTTO
    story.append(Paragraph("LOTTO (Ceiling Play)", section))
    story.append(_make_ticket_table(lotto_df))

    doc.build(story)


def main() -> None:
    slate_date, results_dir = latest_results_dir()

    tickets_dir = results_dir / "tickets"

    safe_df = _load_csv(tickets_dir / "ticket_safe.csv")
    balanced_df = _load_csv(tickets_dir / "ticket_balanced.csv")
    lotto_df = _load_csv(tickets_dir / "ticket_lotto.csv")

    # 🔥 NEW NAME
    out_path = results_dir / f"nba_ticket_card_{slate_date}.pdf"

    _build_pdf(
        out_path=out_path,
        game_date=slate_date,
        safe_df=safe_df,
        balanced_df=balanced_df,
        lotto_df=lotto_df,
    )

    print(f"[CARD] Saved -> {out_path}")


if __name__ == "__main__":
    main()  