from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import build_features
from src.utils import load_train_cached_sample, train_val_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick local smoke test for the feature pipeline."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.01,
        help="Fraction of srch_id groups to cache/load for the smoke test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sample creation and train/validation split.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached sample before running the smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_train_cached_sample(
        sample_frac=args.sample_frac,
        random_state=args.random_state,
        refresh=args.refresh_cache,
    )
    print(
        f"Loaded cached sample: {len(df):,} rows across "
        f"{df['srch_id'].nunique():,} searches"
    )

    raw_cols = set(df.columns)
    df = build_features(df)
    feature_cols = sorted(set(df.columns) - raw_cols)
    print(f"Added {len(feature_cols)} engineered feature columns")

    feature_object_cols = [
        col for col in feature_cols if str(df[col].dtype) in {"object", "string"}
    ]
    if feature_object_cols:
        raise ValueError(
            f"Unexpected object/string dtypes in engineered features: {feature_object_cols}"
        )

    train_df, val_df = train_val_split(df, random_state=args.random_state)
    train_searches = set(train_df["srch_id"].unique())
    val_searches = set(val_df["srch_id"].unique())
    overlap = train_searches & val_searches
    if overlap:
        raise ValueError(f"Train/val search leakage detected for {len(overlap)} srch_id values")

    feature_nan_counts = (
        df[feature_cols].isna().sum().sort_values(ascending=False).head(10).to_dict()
    )
    feature_dtype_counts = df[feature_cols].dtypes.astype(str).value_counts().to_dict()

    print(f"Train rows: {len(train_df):,}  Val rows: {len(val_df):,}")
    print(f"Top NaN counts in engineered features: {feature_nan_counts}")
    print(f"Engineered feature dtypes: {feature_dtype_counts}")


if __name__ == "__main__":
    main()
