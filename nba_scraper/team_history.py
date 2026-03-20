from __future__ import annotations


# Canonical team -> season-aware Basketball Reference request code
# `year` here is the Basketball Reference season year, e.g.
# 2026 means the 2025-26 season page /teams/XXX/2026.html
TEAM_HISTORY: dict[str, list[tuple[int, int, str]]] = {
    # Nets
    "BKN": [
        (2013, 9999, "BKN"),
        (1978, 2012, "NJN"),
    ],

    # Hornets / Bobcats / old Hornets lineage
    # For modern Charlotte franchise pages:
    "CHA": [
        (2015, 9999, "CHO"),   # BBR often uses CHO for recent Charlotte pages
        (2005, 2014, "CHA"),   # Bobcats era on BBR
        (1989, 2002, "CHH"),   # original Hornets
    ],

    # Pelicans / Hornets history
    "NOP": [
        (2014, 9999, "NOP"),
        (2003, 2013, "NOH"),
        (2006, 2007, "NOK"),   # special Katrina seasons if ever needed
    ],

    # Suns
    "PHX": [
        (1969, 9999, "PHO"),
    ],

    # Warriors
    "GSW": [
        (1963, 9999, "GSW"),
    ],

    # Spurs
    "SAS": [
        (1977, 9999, "SAS"),
    ],
}


def team_code_for_bbr(team: str, year: int) -> str:
    """
    Convert canonical team code into the Basketball Reference code
    needed to request that season's roster page.
    """
    team = str(team).strip().upper()

    if team not in TEAM_HISTORY:
        return team

    for start_year, end_year, bbr_code in TEAM_HISTORY[team]:
        if start_year <= year <= end_year:
            return bbr_code

    return team