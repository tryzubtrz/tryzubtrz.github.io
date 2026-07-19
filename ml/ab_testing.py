"""A/B testing of strategy parameter sets."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from ml.genetic import _apply_params, _backtest_fitness
from strategies.ensemble import EnsembleStrategy

logger = logging.getLogger("ai.ab")


class ABTester:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def compare(
        self,
        df: pd.DataFrame,
        variants: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Any]:
        """
        variants: name -> nested strategy params dict
        """
        scored: List[Tuple[str, float]] = []
        details = []
        for name, params in variants.items():
            ens = EnsembleStrategy()
            if params:
                _apply_params(ens, params)
            fitness = _backtest_fitness(df, ens)
            scored.append((name, fitness))
            details.append({"name": name, "fitness": fitness, "params": params})
            logger.info("A/B %s fitness=%.5f", name, fitness)
        scored.sort(key=lambda x: x[1], reverse=True)
        winner = scored[0][0] if scored else "control"
        self.results = details
        return {
            "winner": winner,
            "scores": {n: s for n, s in scored},
            "details": details,
        }
