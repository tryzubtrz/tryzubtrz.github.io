"""Tournament Mode — run strategies in parallel and pick the current winner."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("tryzub.tournament")


class TournamentMode:
    def __init__(self, state_path: Path, enabled: bool = False) -> None:
        self.enabled = enabled
        self.state_path = state_path
        self.scores: Dict[str, float] = {}
        self.trades: Dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.scores = {k: float(v) for k, v in (raw.get("scores") or {}).items()}
                self.trades = {k: int(v) for k, v in (raw.get("trades") or {}).items()}
            except Exception as exc:
                log.warning("Tournament load failed: %s", exc)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"scores": self.scores, "trades": self.trades}, indent=2),
            encoding="utf-8",
        )

    def record(self, strategy: str, pnl_pct: float) -> None:
        if not strategy:
            return
        self.scores[strategy] = self.scores.get(strategy, 0.0) + float(pnl_pct)
        self.trades[strategy] = self.trades.get(strategy, 0) + 1
        self.save()
        log.info("Tournament record %s pnl_pct=%.4f total=%.4f", strategy, pnl_pct, self.scores[strategy])

    def winner(self) -> Optional[str]:
        if not self.scores:
            return None
        return max(self.scores, key=self.scores.get)

    def pick(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        candidates: list of signal dicts with direction/confidence/strategy.
        If tournament disabled → highest confidence.
        If enabled → prefer current winner strategy when it has a non-flat signal.
        """
        try:
            valid = [c for c in candidates if c.get("direction") in {"long", "short"} and float(c.get("confidence") or 0) > 0]
            if not valid:
                return {"direction": "flat", "confidence": 0.0, "strategy": "tournament", "candidates": candidates}
            if not self.enabled:
                best = max(valid, key=lambda x: float(x.get("confidence") or 0))
                return {**best, "tournament_mode": False}
            win = self.winner()
            if win:
                for c in valid:
                    if c.get("strategy") == win:
                        log.info("Tournament picks winner strategy=%s", win)
                        return {**c, "tournament_mode": True, "winner": win}
            best = max(valid, key=lambda x: float(x.get("confidence") or 0))
            return {**best, "tournament_mode": True, "winner": win}
        except Exception as exc:
            log.error("Tournament pick error: %s", exc)
            return {"direction": "flat", "confidence": 0.0, "strategy": "tournament", "error": str(exc)}

    def snapshot(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "scores": self.scores, "trades": self.trades, "winner": self.winner()}
