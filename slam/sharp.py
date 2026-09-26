"""Sharp analog IR: ADC value -> distance, from a measured calibration table.

A Sharp sensor's output falls roughly like 1/distance, so a formula fitted
from a datasheet is often off by several cm on a real unit. Instead we store
the (adc, metres) pairs measured with `python -m tools.calibrate_sharp` and
interpolate linearly between them. Readings outside the table -> None.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

# Rough GP2Y0A21 curve for a 10-bit / 3.3 V ADC. Only a placeholder so the
# robot can run before calibration; every real sensor must be calibrated.
PLACEHOLDER_TABLE: Tuple[Tuple[int, float], ...] = (
    (713, 0.10),
    (512, 0.15),
    (403, 0.20),
    (285, 0.30),
    (232, 0.40),
    (186, 0.50),
    (155, 0.60),
    (124, 0.80),
)


class SharpCalibration:
    def __init__(self, pairs: Sequence[Tuple[float, float]], source: str = "") -> None:
        """``pairs`` = (adc, metres). The ADC must fall as distance grows."""
        table = sorted((float(a), float(d)) for a, d in pairs)
        if len(table) < 2:
            raise ValueError("need at least two calibration points")
        adcs = [a for a, _ in table]
        dists = [d for _, d in table]
        if len(set(adcs)) != len(adcs) or any(d2 >= d1 for d1, d2 in zip(dists, dists[1:])):
            raise ValueError("distance must strictly decrease as ADC increases")
        self.adcs: List[float] = adcs
        self.dists: List[float] = dists
        self.source = source

    @property
    def min_m(self) -> float:
        return self.dists[-1]

    @property
    def max_m(self) -> float:
        return self.dists[0]

    def distance(self, adc: Optional[float]) -> Optional[float]:
        if adc is None or adc < self.adcs[0] or adc > self.adcs[-1]:
            return None
        for (a1, d1), (a2, d2) in zip(zip(self.adcs, self.dists), zip(self.adcs[1:], self.dists[1:])):
            if a1 <= adc <= a2:
                return d1 + (d2 - d1) * (adc - a1) / (a2 - a1)
        return None

    # ---- files -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "SharpCalibration":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([(p["adc"], p["distance_m"]) for p in data["points"]], source=path)

    @classmethod
    def load_or_placeholder(cls, path: str) -> "SharpCalibration":
        if os.path.exists(path):
            return cls.load(path)
        return cls(PLACEHOLDER_TABLE, source="placeholder")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        points = [{"adc": a, "distance_m": d} for a, d in zip(self.adcs, self.dists)]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"points": points}, f, indent=2)
