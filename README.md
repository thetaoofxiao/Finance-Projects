# Attention Radar

A weekly engine that measures abnormal attention on S&P 500 names (plus a watchlist), checks it against price, and sorts each name into a quadrant. It runs on GitHub Actions, using free data only. A separate Claude scheduled task reads `attention-radar/data/latest.json` each Monday, adds commentary, and refreshes the dashboard.

## What it measures

| Window | Attention days | Price days (trading) |
|---|---|---|
| 3d | last 3 calendar days | 3 |
| 2w | last 14 calendar days | 10 |
| 1m | last 30 calendar days | 21 |

**Attention score:** how far this window's attention sits from the name's own normal level. Normal is the 90 days that end where the 1-month window starts, so a spike never sits in its own baseline. Each source is scored against its own history.

**Blend:** Wikipedia 50%, news 30%, Google Trends 20%. If a source is missing, the other weights are rescaled. Reddit is reported separately as a crowding flag.

**Price check:** return over each window versus the S&P 500, distance from the 52-week closing high, recent volume versus normal, and 12-1 momentum.

## Quadrants (checked in this order)

| Quadrant | Rule | Read |
|---|---|---|
| Peak Hype | 2-week attention >= 2.0 and 1-month return beat the S&P 500 by 10 points or more | Crowded. Defer new buys. |
| Under Fire | 2-week attention >= 1.5 and 1-month return trailed the S&P 500 by 10 points or more | Bad-news attention. Check for a one-off event. |
| Confirmed | 2-week attention >= 1.0, beat the S&P 500 over 1 month, within 10% of 52-week high, no heavy insider selling without buys | Supports momentum. |
| Early | 3-day attention >= 1.5 and rising, 2-week price within 5 points of the S&P 500 | Watch list. |
| Fading | 1-month attention >= 1.0, 3-day attention falling, lagging over 2 weeks | Story rolling over. |
| Quiet | everything else | |

All thresholds sit in `config.json`. They are starting assumptions, not fitted values.

## Files

* `attention-radar/radar.py` the engine. Writes `data/latest.json`, `data/history/`, `data/reddit/`.
* `attention-radar/edition.py` turns `latest.json` into the scored part of a dashboard edition.
* `attention-radar/config.json` thresholds, weights and watchlist.
* `attention-radar/requirements.txt` Python packages.
* `.github/workflows/attention-radar.yml` (repo root) runs the engine Mondays 10:15 UTC and saves results.

## Known limitations

* Wikipedia attention includes non-investor interest. It measures attention, not sentiment.
* GDELT name matching is noisy for common-word names. Use `news_query_overrides`.
* Google Trends and Yahoo Finance access is unofficial and can break or rate limit. A failed source is reported, never fatal.
* OpenInsider sales include pre-planned 10b5-1 sales. Buys are the better signal.
