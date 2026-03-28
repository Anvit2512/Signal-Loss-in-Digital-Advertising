"""
Generate a synthetic Criteo-schema snapshot for the Meta-Signal environment.

Criteo schema: label (0/1), campaign_id, I1-I13 (integer features), C1-C26 (categorical features)
Campaign partitioning by row index bands -- deterministic, reproducible with seed=42.

CVR targets:
  camp_feed    rows 0-3499      target ~8%
  camp_reels   rows 3500-6999   target ~4%
  camp_stories rows 7000-9999   target ~2%
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_ROWS = 10_000

CAMPAIGN_BANDS = {
    "camp_feed":    (0,    3500, 0.08),
    "camp_reels":   (3500, 7000, 0.04),
    "camp_stories": (7000, 10000, 0.02),
}


def _make_integer_features(n: int) -> pd.DataFrame:
    cols = {}
    for i in range(1, 14):
        # Sparse count-like features -- mostly low values, long tail
        vals = RNG.negative_binomial(1, 0.3, size=n).astype(float)
        # ~15% missing, matching real Criteo sparsity
        mask = RNG.random(n) < 0.15
        vals[mask] = np.nan
        cols[f"I{i}"] = vals
    return pd.DataFrame(cols)


def _make_categorical_features(n: int) -> pd.DataFrame:
    cols = {}
    vocab_sizes = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145,
                   5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4,
                   7046547, 18, 15, 286181, 105, 142572]
    for i, vsize in enumerate(vocab_sizes, start=1):
        hashed = RNG.integers(0, min(vsize, 100_000), size=n).astype(str)
        # ~10% missing
        mask = RNG.random(n) < 0.10
        cols[f"C{i}"] = [None if m else v for m, v in zip(mask, hashed)]
    return pd.DataFrame(cols)


def generate() -> pd.DataFrame:
    labels = np.zeros(N_ROWS, dtype=int)
    campaign_ids = np.empty(N_ROWS, dtype=object)

    for camp, (start, end, cvr) in CAMPAIGN_BANDS.items():
        n = end - start
        labels[start:end] = RNG.binomial(1, cvr, size=n)
        campaign_ids[start:end] = camp

    int_feats = _make_integer_features(N_ROWS)
    cat_feats = _make_categorical_features(N_ROWS)

    df = pd.concat(
        [
            pd.DataFrame({"label": labels, "campaign_id": campaign_ids}),
            int_feats,
            cat_feats,
        ],
        axis=1,
    )

    # Shuffle within each campaign band so rows aren't trivially ordered
    # campaign_id travels with each row so bands remain correct after shuffle
    indices = np.concatenate([
        RNG.permutation(np.arange(start, end))
        for _, (start, end, _) in CAMPAIGN_BANDS.items()
    ])
    df = df.iloc[indices].reset_index(drop=True)

    return df


def verify(df: pd.DataFrame) -> bool:
    """
    Assert data integrity. Callable from env.py at startup so any corruption
    is caught before an episode begins, not mid-episode.
    """
    assert len(df) == N_ROWS, f"Expected {N_ROWS} rows, got {len(df)}"
    assert "label" in df.columns, "Missing 'label' column"
    assert "campaign_id" in df.columns, "Missing 'campaign_id' column"

    actual_camps = set(df["campaign_id"].unique())
    expected_camps = set(CAMPAIGN_BANDS.keys())
    assert actual_camps == expected_camps, (
        f"Campaign mismatch: got {actual_camps}, expected {expected_camps}"
    )

    for camp, (start, end, target_cvr) in CAMPAIGN_BANDS.items():
        band = df.iloc[start:end]
        actual_cvr = band["label"].mean()
        assert abs(actual_cvr - target_cvr) < 0.02, (
            f"{camp} CVR {actual_cvr:.3f} too far from target {target_cvr:.3f}"
        )
        assert (band["campaign_id"] == camp).all(), (
            f"campaign_id mismatch inside band for {camp}"
        )

    return True


if __name__ == "__main__":
    import os
    out_path = os.path.join(os.path.dirname(__file__), "ad_logs_sampled.csv")
    df = generate()
    verify(df)
    df.to_csv(out_path, index=False)

    feed = df.iloc[0:3500]
    reels = df.iloc[3500:7000]
    stories = df.iloc[7000:10000]

    print(f"Generated {len(df)} rows -> {out_path}")
    print(f"Columns: {list(df.columns[:6])} ... ({len(df.columns)} total)")
    print(f"\nCampaign CVR check:")
    print(f"  camp_feed    CVR = {feed['label'].mean():.3f}  (target 0.080)")
    print(f"  camp_reels   CVR = {reels['label'].mean():.3f}  (target 0.040)")
    print(f"  camp_stories CVR = {stories['label'].mean():.3f}  (target 0.020)")
    print(f"\nMissing values sample: I1={df['I1'].isna().sum()}, C1={df['C1'].isna().sum()}")
    print(f"\nverify() passed: True")
