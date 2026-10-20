#!/usr/bin/env python3
"""
Attention Radar engine (weekly, free sources only).

Measures abnormal attention for S&P 500 names plus a watchlist over 3-day,
2-week and 1-month windows, checks it against price, and sorts each name into
Peak Hype / Under Fire / Confirmed / Early / Fading / Quiet.

Attention z-score = (mean of log(1+daily attention) over the window
                     minus the baseline median) / baseline robust SD.
Baseline = the N days that end where the 1-month window starts, so a
recent spike never contaminates its own baseline.

Sources (all free, each optional; a failed source is reported, never fatal):
  wiki    Wikipedia pageviews (user agent, daily)        full universe
  news    GDELT DOC 2.0 volume intensity (daily)         top candidates
  trends  Google Trends via pytrends ("<TICKER> stock")  top candidates
  reddit  ApeWisdom mention counts (24h snapshot)        all stocks
  insider OpenInsider open-market buys and sales, 30d    all stocks
  price   Yahoo Finance via yfinance                     full universe
Outputs: data/latest.json, data/history/<date>.json, data/history/scores.csv,
         data/reddit/<date>.csv
"""
import datetime as dt
import io
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data"
UA = os.environ.get("RADAR_UA") or "AttentionRadar/1.0 (personal research; github.com)"
HTTP = requests.Session()
HTTP.headers["User-Agent"] = UA
log = logging.getLogger("radar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SOURCES = {}  # name -> {"status": ok|partial|failed|skipped, "n": int, "note": str}


def status(name, st, n=0, note=""):
    SOURCES[name] = {"status": st, "n": int(n), "note": note[:200]}


def get(url, params=None, tries=3, sleep=2.0, timeout=30):
    for i in range(tries):
        try:
            r = HTTP.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(sleep * (i + 2) * 3)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if i == tries - 1:
                raise
            time.sleep(sleep * (i + 1))
    raise RuntimeError(f"rate limited: {url}")


# ---------------------------------------------------------------- universe
def load_universe():
    """S&P 500 from Wikipedia's constituents table (keeps article titles), plus watchlist."""
    from bs4 import BeautifulSoup

    rows = []
    try:
        html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", {"id": "constituents"})
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            ticker = tds[0].get_text(strip=True)
            a = tds[1].find("a")
            title = a["href"].split("/wiki/")[-1] if a and a.get("href", "").startswith("/wiki/") else None
            rows.append({"ticker": ticker, "name": tds[1].get_text(strip=True),
                         "sector": tds[2].get_text(strip=True), "wiki": title, "sp500": True})
        status("universe", "ok", len(rows))
    except Exception as e:  # fallback: GitHub dataset, no article titles
        log.warning("Wikipedia list failed (%s); using GitHub constituents", e)
        df = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
        for _, r in df.iterrows():
            rows.append({"ticker": r["Symbol"], "name": r["Security"], "sector": r.get("GICS Sector", ""),
                         "wiki": str(r["Security"]).replace(" ", "_"), "sp500": True})
        status("universe", "partial", len(rows), "fallback list, wiki titles guessed")
    have = {r["ticker"] for r in rows}
    for w in CFG["watchlist"]:
        if w["ticker"] not in have:
            rows.append({**w, "sp500": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring core
def window_z(series, end, baseline_days, windows=None, min_base=None):
    """series: pd.Series of daily values indexed by date. Returns {window: z} or NaNs."""
    windows = windows or CFG["attention_windows_days"]
    out = {k: np.nan for k in windows}
    if series is None or len(series) == 0:
        return out
    s = np.log1p(series.astype(float).clip(lower=0))
    longest = max(windows.values())
    base_end = end - dt.timedelta(days=longest)
    base = s[(s.index > base_end - dt.timedelta(days=baseline_days)) & (s.index <= base_end)]
    min_base = min_base or int(baseline_days * 0.66)
    if len(base) < min_base:
        return out
    med = base.median()
    sd = (base - med).abs().median() * 1.4826
    if not sd or sd < 0.05:
        sd = max(base.std(), 0.05)
    for k, d in windows.items():
        w = s[(s.index > end - dt.timedelta(days=d)) & (s.index <= end)]
        if len(w) >= max(1, int(d * 0.66)):
            out[k] = float((w.mean() - med) / sd)
    return out


def classify(r, t=CFG["thresholds"]):
    a3, a14, a30 = r.get("a3d"), r.get("a2w"), r.get("a1m")
    x1m, x2w, d52 = r.get("ex_ret_1m"), r.get("ex_ret_2w"), r.get("dist_52wh")
    nan = lambda v: v is None or (isinstance(v, float) and math.isnan(v))
    if nan(a14) or nan(x1m):
        return "No data"
    if a14 >= t["peak_attn_2w"] and x1m >= t["peak_exret_1m"]:
        return "Peak Hype"
    if a14 >= t["underfire_attn_2w"] and x1m <= t["underfire_exret_1m"]:
        return "Under Fire"
    if (a14 >= t["confirmed_attn_2w_min"] and x1m > t["confirmed_exret_1m_min"]
            and not nan(d52) and d52 >= t["confirmed_dist52_min"]
            and not (r.get("ins_sell_usd", 0) > t["insider_sell_flag_usd"] and r.get("ins_buy_usd", 0) == 0)):
        return "Confirmed"
    if (not nan(a3) and not nan(a30) and a3 >= t["early_attn_3d"] and a3 - a30 >= t["early_accel"]
            and not nan(x2w) and abs(x2w) <= t["early_abs_exret_2w_max"]):
        return "Early"
    if (not nan(a30) and not nan(a3) and a30 >= t["fading_attn_1m"] and a3 <= a30 - t["fading_drop"]
            and not nan(x2w) and x2w < 0):
        return "Fading"
    return "Quiet"


def blend(row, w=CFG["source_weights"]):
    """Weighted mean of available source z-scores, per window."""
    out = {}
    for k in CFG["attention_windows_days"]:
        num = den = 0.0
        for src, wt in w.items():
            v = row.get(f"{src}_z_{k}")
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                num += wt * v
                den += wt
        out[f"a{k}"] = num / den if den else np.nan
    return out


# ---------------------------------------------------------------- sources
def fetch_wiki(uni, end):
    start = end - dt.timedelta(days=CFG["baseline_days"] + 40)
    res, ok = {}, 0
    base = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"
    for _, r in uni.iterrows():
        title = r["wiki"]
        if not title:
            continue
        url = f"{base}/{quote(title, safe='')}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"
        try:
            j = HTTP.get(url, timeout=30)
            if j.status_code == 404:  # try resolving a redirect
                q = get("https://en.wikipedia.org/w/api.php", {"action": "query", "titles": title.replace("_", " "),
                                                                "redirects": 1, "format": "json"}).json()
                pages = q.get("query", {}).get("pages", {})
                t2 = next(iter(pages.values())).get("title", "").replace(" ", "_")
                if t2 and t2 != title:
                    j = HTTP.get(f"{base}/{quote(t2, safe='')}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00", timeout=30)
            j.raise_for_status()
            items = j.json().get("items", [])
            s = pd.Series({dt.datetime.strptime(i["timestamp"][:8], "%Y%m%d").date(): i["views"] for i in items})
            res[r["ticker"]] = s
            ok += 1
        except Exception as e:
            log.debug("wiki %s: %s", r["ticker"], e)
        time.sleep(0.05)
    status("wiki", "ok" if ok > 0.9 * len(uni) else ("partial" if ok else "failed"), ok)
    return res


def fetch_prices(uni):
    import yfinance as yf

    tick = [t.replace(".", "-") for t in uni["ticker"]] + ["SPY"]
    try:
        df = yf.download(tick, period="15mo", interval="1d", auto_adjust=True, group_by="ticker",
                         threads=True, progress=False)
    except Exception as e:
        status("price", "failed", 0, str(e))
        return {}, None
    out = {}
    for t in tick:
        try:
            sub = df[t][["Close", "Volume"]].dropna()
            if len(sub) > 30:
                out[t.replace("-", ".")] = sub
        except Exception:
            pass
    status("price", "ok" if len(out) > 0.9 * len(tick) else "partial", len(out))
    return out, out.get("SPY")


def price_metrics(px, spy):
    m = {}
    if px is None or len(px) < 30:
        return m
    c, v = px["Close"], px["Volume"]
    for k, n in CFG["price_windows_trading_days"].items():
        if len(c) > n:
            r = c.iloc[-1] / c.iloc[-1 - n] - 1
            m[f"ret_{k}"] = float(r)
            if spy is not None and len(spy) > n:
                m[f"ex_ret_{k}"] = float(r - (spy["Close"].iloc[-1] / spy["Close"].iloc[-1 - n] - 1))
    m["dist_52wh"] = float(c.iloc[-1] / c.iloc[-252:].max() - 1)
    if len(v) > 70 and v.iloc[-65:-5].mean() > 0:
        m["vol_ratio"] = float(v.iloc[-5:].mean() / v.iloc[-65:-5].mean())
    if len(c) >= 253:
        m["mom_12_1"] = float(c.iloc[-22] / c.iloc[-253] - 1)
    m["close"] = float(c.iloc[-1])
    m["px_date"] = str(c.index[-1].date())
    return m


def fetch_reddit():
    rows = []
    try:
        for p in range(1, CFG["reddit_pages"] + 1):
            j = get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{p}").json()
            rows += j.get("results", [])
            if p >= int(j.get("pages", 1)):
                break
            time.sleep(1)
        df = pd.DataFrame(rows)
        for c in ["rank", "mentions", "mentions_24h_ago", "upvotes", "rank_24h_ago"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        status("reddit", "ok", len(df))
        return df
    except Exception as e:
        status("reddit", "failed", len(rows), str(e))
        return pd.DataFrame(rows)


def reddit_history_z(today_df, today):
    """Once 8+ weekly snapshots exist, z-score this week's mentions vs prior snapshots."""
    files = sorted((DATA / "reddit").glob("*.csv"))
    hist = [pd.read_csv(f).assign(d=f.stem) for f in files if f.stem < str(today)]
    if len(hist) < 8 or today_df.empty:
        return {}
    h = pd.concat(hist)
    out = {}
    for t, g in h.groupby("ticker"):
        base = np.log1p(g["mentions"].astype(float))
        if len(base) < 6:
            continue
        cur = today_df.loc[today_df["ticker"] == t, "mentions"]
        if cur.empty:
            continue
        sd = max(base.std(), 0.25)
        out[t] = float((np.log1p(float(cur.iloc[0])) - base.median()) / sd)
    return out


def clean_name(n):
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[,.]?\s+(Inc|Incorporated|Corporation|Corp|Company|Co|plc|PLC|Ltd|Limited|N\.V|S\.A|Holdings?|Group|Class [A-C])\.?$", "", n.strip())
    n = re.sub(r"[,.]?\s+(Inc|Corp|Co|plc|Ltd)\.?$", "", n.strip())
    return n.strip(" ,.")


def fetch_news(cands, uni, end):
    res, ok = {}, 0
    start = end - dt.timedelta(days=89)
    ov = CFG.get("news_query_overrides", {})
    names = uni.set_index("ticker")["name"].to_dict()
    for t in cands:
        q = ov.get(t) or f'"{clean_name(names.get(t, t))}"'
        params = {"query": f"{q} sourcelang:english", "mode": "timelinevol", "format": "json",
                  "startdatetime": f"{start:%Y%m%d}000000", "enddatetime": f"{end:%Y%m%d}235959"}
        try:
            j = get("https://api.gdeltproject.org/api/v2/doc/doc", params, tries=2, sleep=6).json()
            data = j["timeline"][0]["data"]
            s = pd.Series({dt.datetime.strptime(d["date"][:8], "%Y%m%d").date(): d["value"] * 1000 for d in data})
            res[t] = s
            ok += 1
        except Exception as e:
            log.debug("gdelt %s: %s", t, e)
        time.sleep(5.5)  # GDELT asks for one request per 5 seconds
    status("news", "ok" if ok >= 0.8 * len(cands) else ("partial" if ok else "failed"), ok)
    return res


def fetch_trends(cands):
    res, fails = {}, 0
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=300, timeout=(10, 30))
    except Exception as e:
        status("trends", "failed", 0, str(e))
        return res
    for t in cands:
        if fails >= 3:
            break
        try:
            pt.build_payload([f"{t} stock"], timeframe="today 3-m", geo="US")
            df = pt.interest_over_time()
            if not df.empty:
                res[t] = pd.Series(df.iloc[:, 0].values, index=[d.date() for d in df.index])
            time.sleep(8)
        except Exception as e:
            fails += 1
            log.info("trends %s failed: %s", t, e)
            time.sleep(30)
    status("trends", "ok" if len(res) >= 0.8 * len(cands) else ("partial" if res else "failed"), len(res),
           "Google Trends is unofficial and rate limited; best effort")
    return res


def fetch_insiders():
    url = ("http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago="
           "&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh="
           "&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=5000&page=1")
    try:
        html = get(url, timeout=60).text
        tables = pd.read_html(io.StringIO(html))
        df = next(t for t in tables if "Ticker" in t.columns and "Value" in t.columns)
        df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
        df["usd"] = pd.to_numeric(df["Value"].astype(str).str.replace(r"[^\d\-]", "", regex=True), errors="coerce")
        df["kind"] = np.where(df["Trade Type"].astype(str).str.startswith("P"), "buy", "sell")
        g = df.groupby(["Ticker", "kind"]).agg(usd=("usd", lambda x: x.abs().sum()), n=("Insider Name", "nunique"))
        out = {}
        for (t, k), r in g.iterrows():
            o = out.setdefault(t, {"ins_buy_usd": 0.0, "ins_sell_usd": 0.0, "ins_buyers": 0, "ins_sellers": 0})
            if k == "buy":
                o["ins_buy_usd"], o["ins_buyers"] = float(r["usd"]), int(r["n"])
            else:
                o["ins_sell_usd"], o["ins_sellers"] = float(r["usd"]), int(r["n"])
        status("insider", "ok", len(df), "OpenInsider 30d; sales include 10b5-1 plan sales")
        return out
    except Exception as e:
        status("insider", "failed", 0, str(e))
        return {}


# ---------------------------------------------------------------- main
def r4(v):
    return None if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else round(float(v), 4)


def run():
    today = dt.date.today()
    end = today - dt.timedelta(days=1)  # pageviews and GDELT lag about a day
    uni = load_universe()
    log.info("universe %d", len(uni))

    wiki = fetch_wiki(uni, end)
    prices, spy = fetch_prices(uni)
    reddit = fetch_reddit()
    insiders = fetch_insiders()

    rows = []
    for _, u in uni.iterrows():
        t = u["ticker"]
        r = {"ticker": t, "name": u["name"], "sector": u["sector"], "sp500": bool(u["sp500"])}
        for k, v in window_z(wiki.get(t), end, CFG["baseline_days"]).items():
            r[f"wiki_z_{k}"] = v
        r.update(price_metrics(prices.get(t), spy))
        r.update(insiders.get(t, {"ins_buy_usd": 0.0, "ins_sell_usd": 0.0, "ins_buyers": 0, "ins_sellers": 0}))
        rows.append(r)
    df = pd.DataFrame(rows)

    # candidates for the slower sources: biggest wiki movers + Reddit leaders + watchlist
    df["_wmax"] = df[["wiki_z_3d", "wiki_z_2w"]].max(axis=1)
    rtop = set(reddit.head(40)["ticker"]) if not reddit.empty else set()
    cands = list(dict.fromkeys(
        list(df.sort_values("_wmax", ascending=False).head(CFG["news_top_n"])["ticker"])
        + [t for t in df["ticker"] if t in rtop]
        + [w["ticker"] for w in CFG["watchlist"]]))
    news = fetch_news(cands, uni, end)
    trends = fetch_trends(cands[: CFG["trends_top_n"]])
    for i, r in df.iterrows():
        for src, res, bd in (("news", news, CFG["news_baseline_days"]), ("trends", trends, CFG["news_baseline_days"])):
            for k, v in window_z(res.get(r["ticker"]), end, bd, min_base=int(bd * 0.5)).items():
                df.at[i, f"{src}_z_{k}"] = v

    # Reddit
    if not reddit.empty:
        rd = reddit.set_index("ticker")
        df["reddit_rank"] = df["ticker"].map(rd["rank"])
        df["reddit_mentions"] = df["ticker"].map(rd["mentions"])
        df["reddit_mentions_prev"] = df["ticker"].map(rd["mentions_24h_ago"])
        hz = reddit_history_z(reddit, today)
        df["reddit_z_week"] = df["ticker"].map(hz)
        (DATA / "reddit").mkdir(parents=True, exist_ok=True)
        reddit.to_csv(DATA / "reddit" / f"{today}.csv", index=False)

    # blend and classify
    bl = df.apply(lambda r: pd.Series(blend(r)), axis=1)
    df = pd.concat([df.drop(columns=["_wmax"]), bl], axis=1)
    df["accel"] = df["a3d"] - df["a1m"]
    df["quadrant"] = df.apply(lambda r: classify(r.to_dict()), axis=1)
    df["n_sources"] = df[[c for c in df.columns if c.endswith("_z_2w")]].notna().sum(axis=1)

    # Winners' Ladder proxy screen: top decile 12-1 momentum, then closest to 52w high
    sp = df[df["sp500"] & df["mom_12_1"].notna()]
    cut = sp["mom_12_1"].quantile(1 - CFG["thresholds"]["wl_momentum_top_pct"])
    wl = sp[sp["mom_12_1"] >= cut].sort_values("dist_52wh", ascending=False)
    df["wl_screen_rank"] = df["ticker"].map({t: i + 1 for i, t in enumerate(wl["ticker"])})

    # write
    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    recs = []
    for r in df.to_dict("records"):
        recs.append({k: (r4(v) if k in num_cols else v) for k, v in r.items()})
    payload = {
        "as_of": str(end), "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "engine_version": CFG["engine_version"], "params": {k: CFG[k] for k in
            ["baseline_days", "news_baseline_days", "attention_windows_days", "price_windows_trading_days",
             "source_weights", "thresholds"]},
        "sources": SOURCES,
        "quadrant_counts": df["quadrant"].value_counts().to_dict(),
        "rows": recs,
        "reddit_top": reddit.head(40)[[c for c in ["ticker", "name", "rank", "mentions", "mentions_24h_ago",
                                                   "rank_24h_ago"] if c in reddit]].to_dict("records")
        if not reddit.empty else [],
    }
    (DATA / "history").mkdir(parents=True, exist_ok=True)
    txt = json.dumps(payload, default=str, separators=(",", ":"))
    (DATA / "latest.json").write_text(txt)
    (DATA / "history" / f"{today}.json").write_text(txt)
    keep = ["ticker", "a3d", "a2w", "a1m", "wiki_z_2w", "news_z_2w", "trends_z_2w", "reddit_mentions",
            "ret_1m", "ex_ret_1m", "dist_52wh", "mom_12_1", "quadrant", "wl_screen_rank"]
    hist = df[[c for c in keep if c in df]].assign(date=str(today))
    f = DATA / "history" / "scores.csv"
    hist.to_csv(f, mode="a", header=not f.exists(), index=False)
    log.info("done: %s", payload["quadrant_counts"])
    log.info("sources: %s", json.dumps(SOURCES))
    return payload


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
