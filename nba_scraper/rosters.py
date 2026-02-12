# nba_scraper/rosters.py
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment
from pandas.errors import EmptyDataError

from nba_scraper.browser import fetch_html, log
from nba_scraper.config import BASE_URL, DATA_DIR, TEAMS_CSV, PLAYERS_CSV, PROGRESS_DIR
from nba_scraper.teams import get_teams

# ---------------------------------------------------------------------------
# Paths / progress
# ---------------------------------------------------------------------------

ROSTER_PROGRESS_FILE: Path = PROGRESS_DIR / "progress_rosters.json"

PLAYERS_SCHEMA = ["name", "year", "pos", "team", "number", "birth_date", "href"]

def _repair_players_csv(path: Path) -> None:
    """
    One-time repair for PLAYERS_CSV if columns drifted or bad rows exist.
    Keeps only known columns, coerces year to int, drops invalid rows, de-dupes.
    """
    if not path.exists():
        return

    df = pd.read_csv(path)

    # If header is missing or wrong, bail loudly (don’t guess)
    missing = [c for c in PLAYERS_SCHEMA if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path} missing columns {missing}. "
            "CSV header/schema is corrupted; fix manually or restore from backup."
        )

    df = df[PLAYERS_SCHEMA].copy()

    # Coerce year to numeric; anything non-numeric becomes NaN
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop broken rows (your bug is here: href ended up in year)
    bad = df["year"].isna()
    if bad.any():
        log(f"Repair: dropping {int(bad.sum())} rows with non-numeric year in {path}.", "WARN")
        df = df.loc[~bad].copy()

    df["year"] = df["year"].astype(int)

    # Drop rows with empty href
    df = df[df["href"].astype(str).str.startswith("http")].copy()

    # De-dupe by (href, year)
    df = df.drop_duplicates(subset=["href", "year"], keep="last")

    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log(f"Repair complete: rewrote {path} with {len(df)} rows.", "INFO")



def load_roster_progress() -> set[tuple[int, str]]:
    """
    Backward compatible:
      - old format: [[year, team], ...]
      - new format: {"year|team": "YYYY-MM-DD"} (ignored value; keys treated as done)
    """
    if not ROSTER_PROGRESS_FILE.exists():
        return set()

    with ROSTER_PROGRESS_FILE.open("r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return {(int(year), str(team)) for year, team in data}

    if isinstance(data, dict):
        out: set[tuple[int, str]] = set()
        for k in data.keys():
            y, t = str(k).split("|", 1)
            out.add((int(y), t))
        return out

    return set()


def save_roster_progress(done_set: set[tuple[int, str]]) -> None:
    """
    Atomic write (crash-safe).
    Keep simple list format to stay compatible with older runs.
    """
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ROSTER_PROGRESS_FILE.with_suffix(".json.tmp")
    data = [[year, team] for (year, team) in sorted(done_set)]
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(ROSTER_PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Block / throttle detection
# ---------------------------------------------------------------------------

def is_blocked_html(html: str) -> bool:
    if not html:
        return False   # ← critical change
    h = html.lower()
    signals = [
        "captcha",
        "verify you are human",
        "unusual traffic",
        "access denied",
        "temporarily blocked",
        "request blocked",
        "cloudflare",
        "incapsula",
        "automated queries",
    ]
    return any(s in h for s in signals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cell_text(row, data_stat: str, header: bool = False) -> str:
    tag = "th" if header else "td"
    cell = row.find(tag, {"data-stat": data_stat})
    return cell.get_text(strip=True) if cell else ""


def _normalize_team_token(team_val: str) -> str:
    t = str(team_val).strip()

    if t.startswith("http"):
        t = t.split("/teams/")[-1]

    if "/teams/" in t:
        parts = t.strip("/").split("/")
        if len(parts) >= 2:
            return parts[1]

    return t.strip("/")


# ---------------------------------------------------------------------------
# Parse roster page
# ---------------------------------------------------------------------------

def parse_roster_table(html: str, team: str, year: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    roster_table = soup.find("table", {"id": "roster"})
    if not roster_table:
        # Basketball-Reference sometimes hides tables inside HTML comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if 'id="roster"' in c:
                roster_table = BeautifulSoup(c, "html.parser").find("table", id="roster")
                break

    if not roster_table:
        log(f"No roster table for team {team} {year}. Skipping.", "WARN")
        return []

    tbody = roster_table.find("tbody")
    if not tbody:
        log(f"No tbody for roster table {team} {year}. Skipping.", "WARN")
        return []

    rows = tbody.find_all("tr")

    players: list[dict] = []
    for row in rows:
        player_cell = row.find("td", {"data-stat": "player"})
        if not player_cell:
            continue

        name = player_cell.get_text(strip=True)
        href_tag = player_cell.find("a")

        players.append(
            {
                "name": name,
                "year": year,
                "pos": _get_cell_text(row, "pos"),
                "team": team,
                "number": _get_cell_text(row, "number", header=True),
                "birth_date": _get_cell_text(row, "birth_date"),
                "href": BASE_URL + href_tag["href"] if href_tag else "",
            }
        )

    return players


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def initialize_players(
    start_year: int = 2024,
    end_year: int = 2025,
    delay: float = 3.0,
) -> list[dict]:
    """
    Goals:
      - idempotent reruns (dedupe by (href, year))
      - progress is success-only (don't mark done on blocked/empty HTML)
      - driver reuse (performance + stability)
      - chunked key-load from PLAYERS_CSV (no full read into RAM)
      - deterministic iteration order
    """
    DATA_DIR.mkdir(exist_ok=True)
    PROGRESS_DIR.mkdir(exist_ok=True)

    if TEAMS_CSV.exists():
        teams_df = pd.read_csv(TEAMS_CSV)
    else:
        teams_df = get_teams()
        teams_df.to_csv(TEAMS_CSV, index=False)

    # expecting teams_df["href"] to be either team abbrev or a path-ish token
    teams_raw = teams_df["href"].tolist()
    teams = sorted(_normalize_team_token(t) for t in teams_raw)

    done = load_roster_progress()
    log(f"Loaded {len(done)} completed (year, team) combos from roster progress.")

    # Returned rows from THIS run (keeps memory bounded)
    players: list[dict] = []

    seen_keys: set[tuple[str, int]] = set()
    file_initialized = False

    if PLAYERS_CSV.exists():
        _repair_players_csv(PLAYERS_CSV)

    try:
        for chunk in pd.read_csv(PLAYERS_CSV, usecols=["href", "year"], chunksize=200_000):
            # year now guaranteed numeric after repair, but keep safe coercion anyway
            chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
            chunk = chunk.dropna(subset=["href", "year"])
            for href_val, year_val in chunk.itertuples(index=False, name=None):
                seen_keys.add((str(href_val), int(year_val)))

        file_initialized = PLAYERS_CSV.stat().st_size > 0
        log(f"Loaded {len(seen_keys)} existing (href,year) keys from {PLAYERS_CSV}.")
    except EmptyDataError:
        log(f"{PLAYERS_CSV} exists but is empty. Starting fresh.", "WARN")

    driver = None
    try:
        for year in range(start_year, end_year + 1):
            log(f"Season {year} starting...")

            for i, team in enumerate(teams, start=1):
                if (year, team) in done:
                    log(f"{team} {year} already processed", "SKIP")
                    continue

                log(f"({i}/{len(teams)}) Scraping roster for {team} {year}")
                url = f"{BASE_URL}/teams/{team}/{year}.html"

                html, driver = fetch_html(
                    driver,
                    url,
                    retries=3,
                    retry_delay=max(3.0, float(delay)),
                    wait_for_css_any=["#roster", "body"],
                    timeout=15.0,
                )

                # ✅ DO NOT mark done on blocked/empty. That causes missed rosters until next run.
                if is_blocked_html(html):
                    log(f"Blocked/throttled HTML for {team} {year}. Backing off + skipping.", "WARN")
                    time.sleep(30)
                    continue

                if not html:
                    log(f"Empty HTML for {team} {year}. Skipping (not marking done).", "WARN")
                    continue

                new_players = parse_roster_table(html, team, year)

                if new_players:
                    unique_new: list[dict] = []
                    for p in new_players:
                        key = (p["href"], p["year"])
                        if key not in seen_keys:
                            seen_keys.add(key)
                            unique_new.append(p)

                    if unique_new:
                        df = pd.DataFrame(unique_new, columns=PLAYERS_SCHEMA)
                        mode = "a" if file_initialized else "w"
                        df.to_csv(PLAYERS_CSV, mode=mode, index=False, header=not file_initialized)
                        file_initialized = True
                        players.extend(unique_new)
                        
                # ✅ success = page loaded + parsed (even if roster empty)
                done.add((year, team))
                save_roster_progress(done)

                time.sleep(delay)

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass

    log(f"Finished roster scraping. Total NEW players appended this run: {len(players)}")
    return players
