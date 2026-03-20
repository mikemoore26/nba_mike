# scripts/export_projection_report.py

from __future__ import annotations

from pathlib import Path
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from datetime import datetime


# ---------- CONFIG ----------
N_TOP = 15


# ---------- HELPERS ----------
def load_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # Keep only eligible
    if "is_eligible" in df.columns:
        df = df[df["is_eligible"] == 1]

    return df


def build_table(df: pd.DataFrame, title: str, cols: list[str]):
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    if df.empty:
        elements.append(Paragraph("No data available", styles["Normal"]))
        elements.append(Spacer(1, 20))
        return elements

    df = df[cols].copy()

    # Round numbers
    for c in df.columns:
        if df[c].dtype != "object":
            df[c] = df[c].round(2)

    data = [cols] + df.values.tolist()

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 20))
    return elements


# ---------- MAIN ----------
def main():
    base = Path("results")


    today_str = datetime.today().strftime("%Y-%m-%d")
    today_dir = base / today_str

    if not today_dir.exists():
        raise ValueError(f"No results found for today: {today_str}")

    latest_dir = today_dir

    pts = clean_df(load_if_exists(latest_dir / "pred_pts.csv"))
    reb = clean_df(load_if_exists(latest_dir / "pred_reb.csv"))
    ast = clean_df(load_if_exists(latest_dir / "pred_ast.csv"))
    fg3 = clean_df(load_if_exists(latest_dir / "pred_fg3.csv"))

    # Sort
    pts_top = pts.sort_values("pred_mean", ascending=False).head(N_TOP)
    reb_top = reb.sort_values("pred_mean", ascending=False).head(N_TOP)
    ast_top = ast.sort_values("pred_mean", ascending=False).head(N_TOP)
    fg3_top = fg3.sort_values("pred_mean", ascending=False).head(N_TOP)

    # Delta leaders (combine all)
    combined = pd.concat([pts, reb, ast, fg3], ignore_index=True)
    delta_top = combined.sort_values("delta_mean", ascending=False).head(N_TOP)

    # ---------- PDF ----------
    pdf_path = latest_dir / "projection_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>Projection Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Sections
    elements += build_table(
        pts_top,
        "Top Points Projections",
        ["player", "team", "opp", "pred_mean", "minutes_proj"],
    )

    elements += build_table(
        reb_top,
        "Top Rebounds Projections",
        ["player", "team", "opp", "pred_mean", "minutes_proj"],
    )

    elements += build_table(
        ast_top,
        "Top Assists Projections",
        ["player", "team", "opp", "pred_mean", "minutes_proj"],
    )

    elements += build_table(
        fg3_top,
        "Top 3PT Made Projections",
        ["player", "team", "opp", "pred_mean", "minutes_proj"],
    )

    elements += build_table(
        delta_top,
        "Top Value (Delta vs Baseline)",
        ["player", "stat", "team", "opp", "pred_mean", "delta_mean"],
    )

    doc.build(elements)

    print(f"[REPORT] Saved -> {pdf_path}")


if __name__ == "__main__":
    main()