"""
Layer 1 — Data Foundation

Loads the frozen Criteo snapshot and partitions it into three pseudo-campaigns.
The label column (ground truth) is kept internally; agents never see it directly.

Campaign layout (by row index):
  camp_feed    rows 0–3499    CVR ~8%   high converter
  camp_reels   rows 3500–6999 CVR ~4%   medium converter
  camp_stories rows 7000–9999 CVR ~2%   low converter
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Absolute path to the frozen snapshot — works regardless of cwd
_DATA_DIR = Path(__file__).parent.parent / "data"
SNAPSHOT_PATH = _DATA_DIR / "ad_logs_sampled.csv"

# Campaign index bands (start inclusive, end exclusive)
CAMPAIGN_BANDS: Dict[str, Tuple[int, int]] = {
    "camp_feed":    (0,    3500),
    "camp_reels":   (3500, 7000),
    "camp_stories": (7000, 10000),
}

CAMPAIGN_NAMES = list(CAMPAIGN_BANDS.keys())

# Feature columns (what the agent can potentially use via its feature mask)
INTEGER_FEATURES = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_FEATURES = [f"C{i}" for i in range(1, 27)]
ALL_FEATURES = INTEGER_FEATURES + CATEGORICAL_FEATURES


class CriteoSnapshot:
    """
    Immutable wrapper around the frozen Criteo CSV.

    Provides:
      - get_batch(start_row, n_rows)  → raw DataFrame slice
      - get_labels(start_row, n_rows) → numpy array of 0/1 labels
      - campaign_slice(campaign, start_row, n_rows) → labels for one campaign
      - total_rows property
    """

    def __init__(self, path: str | Path = SNAPSHOT_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Criteo snapshot not found at {path}. "
                "Run data/generate_snapshot.py first."
            )
        self._df = pd.read_csv(path)
        self._labels = self._df["label"].to_numpy(dtype=np.int32)
        # Expose only ML features — label and campaign_id are internal
        self._features = self._df.drop(columns=["label", "campaign_id"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_rows(self) -> int:
        return len(self._df)

    def get_labels(self, start_row: int, n_rows: int) -> np.ndarray:
        """Return ground-truth labels for rows [start_row, start_row+n_rows)."""
        end = min(start_row + n_rows, self.total_rows)
        return self._labels[start_row:end].copy()

    def get_batch(self, start_row: int, n_rows: int) -> pd.DataFrame:
        """Return feature rows (no label, no campaign_id) for the given slice."""
        end = min(start_row + n_rows, self.total_rows)
        return self._features.iloc[start_row:end].copy()

    def campaign_labels(
        self,
        campaign: str,
        start_row: int,
        n_rows: int,
    ) -> np.ndarray:
        """
        Return labels for rows that belong to a specific campaign band,
        intersected with the slice [start_row, start_row+n_rows).

        Campaign bands are fixed index ranges; rows outside the band return
        an empty array.
        """
        if campaign not in CAMPAIGN_BANDS:
            raise ValueError(f"Unknown campaign '{campaign}'. "
                             f"Choose from {CAMPAIGN_NAMES}")
        band_start, band_end = CAMPAIGN_BANDS[campaign]
        # Intersection of the requested slice with the campaign band
        effective_start = max(start_row, band_start)
        effective_end = min(start_row + n_rows, band_end)
        if effective_start >= effective_end:
            return np.array([], dtype=np.int32)
        return self._labels[effective_start:effective_end].copy()

    def true_cvr(self, campaign: str) -> float:
        """Oracle CVR for a campaign — used only by graders, never the agent."""
        band_start, band_end = CAMPAIGN_BANDS[campaign]
        labels = self._labels[band_start:band_end]
        return float(labels.mean()) if len(labels) > 0 else 0.0

    def campaign_row_count(self, campaign: str) -> int:
        band_start, band_end = CAMPAIGN_BANDS[campaign]
        return band_end - band_start


# Module-level singleton — loaded once, reused everywhere
_snapshot: CriteoSnapshot | None = None


def get_snapshot() -> CriteoSnapshot:
    """Return the module-level singleton, loading it on first call."""
    global _snapshot
    if _snapshot is None:
        _snapshot = CriteoSnapshot()
    return _snapshot


# ------------------------------------------------------------------
# Quick sanity check when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    snap = get_snapshot()
    print(f"Snapshot loaded: {snap.total_rows} rows, {len(ALL_FEATURES)} features")
    print()
    for camp in CAMPAIGN_NAMES:
        cvr = snap.true_cvr(camp)
        n = snap.campaign_row_count(camp)
        print(f"  {camp:<15} rows={n}  CVR={cvr:.3f}")
    print()
    # Test a batch pull
    batch = snap.get_batch(0, 5)
    print(f"Sample batch (5 rows, {len(batch.columns)} feature cols):")
    print(batch[["I1", "I2", "C1", "C2"]].to_string())
