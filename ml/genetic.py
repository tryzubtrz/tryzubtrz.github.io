"""Genetic algorithm to evolve strategy parameters."""
from __future__ import annotations

import logging
import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from indicators.technical import add_all_indicators
from strategies.ensemble import EnsembleStrategy
from strategies.base import StrategySignal

logger = logging.getLogger("ai.genetic")


PARAM_SPACE = {
    "trend_following": {
        "adx_min": (10.0, 40.0),
        "rsi_long_max": (55.0, 80.0),
        "rsi_short_min": (20.0, 45.0),
    },
    "mean_reversion": {
        "rsi_oversold": (20.0, 40.0),
        "rsi_overbought": (60.0, 80.0),
    },
    "breakout": {
        "lookback": (10, 40),
        "volume_mult": (1.1, 3.0),
    },
    "scalping": {
        "stoch_low": (15.0, 35.0),
        "stoch_high": (65.0, 85.0),
    },
}


def _random_individual() -> Dict[str, Dict[str, float]]:
    ind: Dict[str, Dict[str, float]] = {}
    for strat, space in PARAM_SPACE.items():
        ind[strat] = {}
        for k, (lo, hi) in space.items():
            if isinstance(lo, int) and isinstance(hi, int):
                ind[strat][k] = random.randint(lo, hi)
            else:
                ind[strat][k] = random.uniform(float(lo), float(hi))
    return ind


def _mutate(ind: Dict[str, Dict[str, float]], rate: float = 0.3) -> Dict[str, Dict[str, float]]:
    out = deepcopy(ind)
    for strat, space in PARAM_SPACE.items():
        for k, (lo, hi) in space.items():
            if random.random() < rate:
                if isinstance(lo, int) and isinstance(hi, int):
                    out[strat][k] = random.randint(lo, hi)
                else:
                    jitter = (float(hi) - float(lo)) * 0.15
                    val = float(out[strat][k]) + random.uniform(-jitter, jitter)
                    out[strat][k] = max(float(lo), min(float(hi), val))
    return out


def _crossover(a: Dict, b: Dict) -> Dict:
    child = deepcopy(a)
    for strat in PARAM_SPACE:
        for k in PARAM_SPACE[strat]:
            if random.random() < 0.5:
                child[strat][k] = b[strat][k]
    return child


def _apply_params(ensemble: EnsembleStrategy, ind: Dict) -> None:
    for strat in ensemble.strategies:
        if strat.name in ind:
            strat.set_params(**ind[strat.name])


def _backtest_fitness(df: pd.DataFrame, ensemble: EnsembleStrategy) -> float:
    """Simple vectorized-ish walk-forward PnL fitness with drawdown penalty."""
    if len(df) < 80:
        return -999.0
    data = add_all_indicators(df)
    equity = 0.0
    peak = 0.0
    dd = 0.0
    position = 0  # -1 short, 1 long
    entry = 0.0
    for i in range(60, len(data) - 1):
        window = df.iloc[: i + 1].tail(120)
        sig = ensemble.generate("SYM", window)
        price = float(data["close"].iloc[i])
        nxt = float(data["close"].iloc[i + 1])
        if position == 0:
            if sig.direction == "long" and sig.confidence >= 0.55:
                position, entry = 1, price
            elif sig.direction == "short" and sig.confidence >= 0.55:
                position, entry = -1, price
        else:
            # Exit on opposite or flat
            ret = (nxt - entry) / entry * position
            # Force exit after move or opposite signal
            if sig.direction == "flat" or (position == 1 and sig.direction == "short") or (
                position == -1 and sig.direction == "long"
            ) or abs(ret) > 0.01:
                equity += ret
                position = 0
            peak = max(peak, equity)
            dd = max(dd, peak - equity)
    return float(equity - 0.5 * dd)


class GeneticOptimizer:
    def __init__(self, population: int = 16, generations: int = 8) -> None:
        self.population = population
        self.generations = generations
        self.best: Dict[str, Any] = {}

    def evolve(self, df: pd.DataFrame) -> Dict[str, Any]:
        pop = [_random_individual() for _ in range(self.population)]
        best_score = -1e9
        best_ind = pop[0]
        for gen in range(self.generations):
            scored: List[Tuple[float, Dict]] = []
            for ind in pop:
                ens = EnsembleStrategy()
                _apply_params(ens, ind)
                score = _backtest_fitness(df, ens)
                scored.append((score, ind))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best_ind = scored[0]
            logger.info("GA gen=%s best=%.5f", gen + 1, scored[0][0])
            elites = [ind for _, ind in scored[: max(2, self.population // 4)]]
            new_pop = elites[:]
            while len(new_pop) < self.population:
                a, b = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
                child = _mutate(_crossover(a, b))
                new_pop.append(child)
            pop = new_pop
        self.best = {"params": best_ind, "fitness": best_score}
        return self.best

    def apply_to_ensemble(self, ensemble: EnsembleStrategy) -> None:
        if self.best.get("params"):
            _apply_params(ensemble, self.best["params"])
