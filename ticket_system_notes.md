NBA PROJECTION + TICKET SYSTEM NOTES

CORE PHILOSOPHY
- Predict end-game stat outcomes first
- Convert projections into candidate legs
- Score legs by projection strength, hit probability, role quality, and edge
- Build tickets as a portfolio, not just a sorted top-N list

MODEL LAYER
- Primary target is player end-game stat prediction
- Stats:
  - pts
  - reb
  - ast
  - fg3
- Outputs used by ticket layer:
  - pred_mean
  - line
  - p_hit
  - delta
  - minutes_proj
  - projection_rank_score
  - role_score
  - stability_score
  - usage_score
  - fragility_score

SCORING LOGIC
- safe_score:
  - prioritize p_hit
  - prioritize role stability
  - cap delta influence so volatility does not dominate
- balanced_score:
  - blend p_hit + delta + role + projection rank
- lotto_score:
  - prioritize ceiling via delta
  - allow more volatility
  - but do not reward negative-edge plays

SAFE TICKET PHILOSOPHY
- best small-core plays
- highest confidence
- avoid same-game stacking
- avoid fragile minute profiles
- should be built from stable role players first

BALANCED TICKET PHILOSOPHY
- blend of confidence and upside
- can allow some same-game exposure
- still should not overly cluster one game environment
- should avoid obvious trap/minutes-risk legs

LOTTO TICKET PHILOSOPHY
- ceiling-focused
- can tolerate lower p_hit than safe
- must still require real minutes and positive edge
- should be diversified enough that one single game does not kill the whole card

TICKET CONSTRUCTION RULES
1. SAFE:
   - max 1 leg from same game
   - max 2 legs from same team
   - prefer different stat types
   - prefer stronger role_score / lower fragility_score

2. BALANCED:
   - max 2 legs from same game
   - max 2 legs from same team
   - mixed stat exposure
   - penalize games already used in safe

3. LOTTO:
   - max 2 legs from same game
   - max 2 legs from same team
   - allow more delta-driven plays
   - penalize games already used in safe and balanced
   - reject low-minute fake ceiling plays

CORRELATION RULES
- Same-game legs are correlated even when stat types differ
- Same-player legs are forbidden
- Same-team legs should be limited
- Same-game exposure should be penalized across tickets
- If two legs are close in score, choose the one from a different game

ANTI-TRAP RULES
- no 0-minute or near-0-minute players
- no lotto legs with negative delta unless explicitly allowed later
- no weak fg3 spam on 0.5 lines unless probability is truly strong
- avoid overly fragile players in safe
- downgrade thin-role bench pieces in balanced unless edge is strong

FG3 RULES
- fg3 should not be suppressed just because raw delta is smaller
- fg3 deserves inclusion when:
  - minutes are real
  - p_hit is acceptable
  - role/usage support the attempt volume
- fg3 is especially useful in balanced and lotto, but should still respect game-correlation caps

EXPOSURE RULES
- player exposure penalty should reduce repeated use across tickets
- game exposure penalty should reduce repeated stacking of the same matchup across tickets
- exposure penalty should soften repeated usage, not completely zero it out

CURRENT CONSTRUCTION TAKEAWAYS
- good picks can still lose if ticket structure is over-correlated
- portfolio quality matters as much as individual leg quality
- recent misses were mostly close misses, which suggests projection quality is reasonable
- next improvement is portfolio construction, not major model overhaul

NEXT SYSTEM LAYER
- market layer
- compare projection line vs sportsbook line
- compute true edge:
  - edge = pred_mean - market_line
- then rebuild tickets using market-aware edge rather than only internal pseudo-lines