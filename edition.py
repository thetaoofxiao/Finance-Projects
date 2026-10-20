#!/usr/bin/env python3
"""
Turn the engine's data/latest.json into the scored half of a dashboard edition.
Claude's weekly run then fills summary, macro, windows, leaks, ceos, government,
investors and insiders from published sources, and writes the full edition to
the dashboard database.

usage: python edition.py latest.json out.json
"""
import argparse
import json

KEYS = {"Peak Hype": "peak_hype", "Under Fire": "under_fire", "Confirmed": "confirmed",
        "Early": "early", "Fading": "fading"}
ACTION = {"Peak Hype": "Defer: wait for 2 weeks out of Peak Hype or skip to next name",
          "Under Fire": "Run single-event veto check before buying",
          "Confirmed": "Supportive; buy per screen",
          "Early": "Supportive; attention building",
          "Fading": "Caution; confirm momentum is not rolling over",
          "Quiet": "No attention flag; buy per screen",
          "No data": "No attention data; buy per screen"}


def f(v):
    return None if v is None else round(float(v), 3)


def reason(r):
    bits = [f"attn 3d/2w/1m {r.get('a3d') or 0:+.1f}/{r.get('a2w') or 0:+.1f}/{r.get('a1m') or 0:+.1f}"]
    if r.get("ex_ret_1m") is not None:
        bits.append(f"1m vs SPY {r['ex_ret_1m'] * 100:+.0f}%")
    if r.get("dist_52wh") is not None:
        bits.append(f"{r['dist_52wh'] * 100:+.0f}% vs 52w high")
    if r.get("reddit_rank"):
        bits.append(f"Reddit #{int(r['reddit_rank'])}")
    if (r.get("ins_buy_usd") or 0) > 0:
        bits.append(f"insider buys ${r['ins_buy_usd'] / 1e6:.1f}M")
    if (r.get("ins_sell_usd") or 0) > 1e6:
        bits.append(f"insider sales ${r['ins_sell_usd'] / 1e6:.1f}M")
    return "; ".join(bits)


def build(latest, holdings):
    rows = latest["rows"]
    by = {r["ticker"]: r for r in rows}
    quads = {k: [] for k in KEYS.values()}
    for r in sorted(rows, key=lambda r: -(abs(r.get("a2w") or 0))):
        k = KEYS.get(r.get("quadrant"))
        if k and len(quads[k]) < 10:
            quads[k].append({"ticker": r["ticker"], "name": r["name"], "reason": reason(r),
                             "a2w": f(r.get("a2w")), "x1m": f(r.get("ex_ret_1m")), "a3d": f(r.get("a3d"))})
    scatter = [{"t": r["ticker"], "a2w": f(r["a2w"]), "x1m": f(r["ex_ret_1m"]), "q": r["quadrant"]}
               for r in rows if r.get("a2w") is not None and r.get("ex_ret_1m") is not None]
    wl = sorted([r for r in rows if r.get("wl_screen_rank")], key=lambda r: r["wl_screen_rank"])[:10]
    wl_out = [{"rank": int(r["wl_screen_rank"]), "ticker": r["ticker"], "mom_12_1": f(r.get("mom_12_1")),
               "dist_52wh": f(r.get("dist_52wh")), "quadrant": r["quadrant"],
               "action": ACTION.get(r["quadrant"], "")} for r in wl]
    hold = []
    for t in holdings:
        r = by.get(t)
        if r:
            hold.append({"ticker": t, "quadrant": r["quadrant"], "note": reason(r)})
        else:
            hold.append({"ticker": t, "quadrant": "Not scored", "note": "Not in engine universe (ETF or unlisted)."})
    reddit = [{"ticker": r.get("ticker"), "mentions": r.get("mentions"), "prev": r.get("mentions_24h_ago")}
              for r in latest.get("reddit_top", [])[:15]]
    return {
        "engine": {"status": "ok", "as_of": latest["as_of"], "sources": latest["sources"],
                   "quadrant_counts": latest.get("quadrant_counts"), "params": latest.get("params")},
        "quadrants": quads, "scatter": scatter,
        "overlay": {"holdings": hold, "wl_screen": wl_out}, "reddit": reddit,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("latest")
    ap.add_argument("out")
    ap.add_argument("--holdings", default="")
    a = ap.parse_args()
    latest = json.load(open(a.latest))
    out = build(latest, [h.strip() for h in a.holdings.split(",") if h.strip()])
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print(json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()}))
