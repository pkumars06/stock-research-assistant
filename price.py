"""The latest share price, fetched online while the user waits. This is the one figure that is not from a filing.

BSE first, because the filings come from BSE; Yahoo Finance's NSE price if BSE fails. Both are unofficial, free
endpoints with no promise behind them, so a failure returns an error and the reader sees "unavailable", never an
old number. The two exchanges really differ for these thinly traded shares (Speciality at Friday 30 Oct's close:
BSE 138.85, NSE 135.16, measured 2026-10-31), so the exchange and the time always travel with the price.

python price.py     prints both companies
"""
import json
import sys
import time
import urllib.request

BSE_CODES = {"Stylam": "526951", "Speciality": "534425"}
NSE_SYMBOLS = {"Stylam": "STYLAMIND.NS", "Speciality": "SPECIALITY.NS"}
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "application/json"}
TIMEOUT = 6            # seconds per source, so a slow site cannot hold up the page for long
KEEP_FOR = 300         # seconds; a page reload should not call the exchange each time
IST = 5.5 * 3600
_kept = {}


def _get_json(url, headers=BROWSER):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _bse(stock):
    d = _get_json("https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w?Debtflag=&seriesid="
                  f"&scripcode={BSE_CODES[stock]}", {**BROWSER, "Referer": "https://www.bseindia.com/"})
    h = d["Header"]
    return {"price_rupees": float(h["LTP"].replace(",", "")), "exchange": "BSE",
            "as_of": h["Ason"].replace(" | ", ", ") + " IST", "source": "bseindia.com"}


def _nse(stock):
    d = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{NSE_SYMBOLS[stock]}?range=1d&interval=1d")
    m = d["chart"]["result"][0]["meta"]
    return {"price_rupees": float(m["regularMarketPrice"]), "exchange": "NSE",
            "as_of": time.strftime("%d %b %y, %H:%M IST", time.gmtime(m["regularMarketTime"] + IST)),
            "source": "Yahoo Finance"}


def share_price(stock):
    """{"stock", "price_rupees", "exchange", "as_of", "source"}, or {"error"} when no source gives a usable price."""
    if stock not in BSE_CODES:
        return {"error": f"no share price for {stock!r}: the app covers Stylam and Speciality"}
    kept = _kept.get(stock)
    if kept and time.time() - kept[0] < KEEP_FOR:
        return kept[1]
    for source in (_bse, _nse):
        try:
            r = {"stock": stock, **source(stock)}
        except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
            continue       # a site that is down, blocks us, or changed its format: try the next
        if r["price_rupees"] > 0:
            _kept[stock] = (time.time(), r)
            return r
    return {"error": f"the latest share price of {stock} could not be fetched just now"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name in BSE_CODES:
        print(share_price(name))
