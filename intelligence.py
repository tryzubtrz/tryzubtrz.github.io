"""Market intelligence — RSS news sentiment (feedparser) with keyword scoring."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

log = logging.getLogger("tryzub.intel")

FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss.xml",
]

BULLISH = {"surge", "rally", "etf", "approval", "adoption", "bull", "record", "inflow", "partnership"}
BEARISH = {"hack", "ban", "lawsuit", "crash", "bear", "outflow", "fraud", "sec charge", "collapse", "liquidation"}


class NewsIntelligence:
    def __init__(self) -> None:
        self.last: Dict[str, Any] = {"score": 0.0, "items": [], "updated_at": None}

    def fetch(self, limit_per_feed: int = 8) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        score = 0.0
        try:
            import feedparser
        except ImportError:
            log.warning("feedparser not installed — news intelligence disabled")
            return {"score": 0.0, "items": [], "updated_at": datetime.now(timezone.utc).isoformat()}

        for url in FEEDS:
            try:
                parsed = feedparser.parse(url)
                for e in (parsed.entries or [])[:limit_per_feed]:
                    title = (e.get("title") or "").strip()
                    summary = (e.get("summary") or e.get("description") or "")[:280]
                    text = f"{title} {summary}".lower()
                    s = 0.0
                    for w in BULLISH:
                        if w in text:
                            s += 0.15
                    for w in BEARISH:
                        if w in text:
                            s -= 0.2
                    s = max(-1.0, min(1.0, s))
                    score += s
                    items.append({"title": title, "score": s, "link": e.get("link", ""), "source": url})
                log.info("News feed ok: %s (+%s items)", url, min(limit_per_feed, len(parsed.entries or [])))
            except Exception as exc:
                log.warning("News feed failed %s: %s", url, exc)

        n = max(len(items), 1)
        avg = max(-1.0, min(1.0, score / n))
        self.last = {
            "score": float(avg),
            "items": items[:40],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info("News intelligence score=%.3f items=%s", avg, len(items))
        return self.last

    def bias_signal(self, symbol: str) -> Dict[str, Any]:
        """Map news score to a soft trading bias."""
        try:
            if not self.last.get("updated_at"):
                self.fetch()
            sc = float(self.last.get("score") or 0)
            if sc >= 0.15:
                return {"symbol": symbol, "direction": "long", "confidence": min(0.7, 0.45 + sc), "strategy": "news"}
            if sc <= -0.15:
                return {"symbol": symbol, "direction": "short", "confidence": min(0.7, 0.45 + abs(sc)), "strategy": "news"}
            return {"symbol": symbol, "direction": "flat", "confidence": 0.0, "strategy": "news", "score": sc}
        except Exception as exc:
            log.error("News bias error: %s", exc)
            return {"symbol": symbol, "direction": "flat", "confidence": 0.0, "strategy": "news", "error": str(exc)}
