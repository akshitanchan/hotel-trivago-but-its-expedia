from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DTYPES_TRAIN = {
    "srch_id": "int32",
    "site_id": "int8",
    "visitor_location_country_id": "int16",
    "visitor_hist_starrating": "float32",
    "visitor_hist_adr_usd": "float32",
    "prop_country_id": "int16",
    "prop_id": "int32",
    "prop_starrating": "int8",
    "prop_review_score": "float32",
    "prop_brand_bool": "int8",
    "prop_location_score1": "float32",
    "prop_location_score2": "float32",
    "prop_log_historical_price": "float32",
    "position": "int8",
    "price_usd": "float32",
    "promotion_flag": "int8",
    "srch_destination_id": "int32",
    "srch_length_of_stay": "int8",
    "srch_booking_window": "int16",
    "srch_adults_count": "int8",
    "srch_children_count": "int8",
    "srch_room_count": "int8",
    "srch_saturday_night_bool": "int8",
    "srch_query_affinity_score": "float32",
    "orig_destination_distance": "float32",
    "random_bool": "int8",
    "comp1_rate": "float32",
    "comp1_inv": "float32",
    "comp1_rate_percent_diff": "float32",
    "comp2_rate": "float32",
    "comp2_inv": "float32",
    "comp2_rate_percent_diff": "float32",
    "comp3_rate": "float32",
    "comp3_inv": "float32",
    "comp3_rate_percent_diff": "float32",
    "comp4_rate": "float32",
    "comp4_inv": "float32",
    "comp4_rate_percent_diff": "float32",
    "comp5_rate": "float32",
    "comp5_inv": "float32",
    "comp5_rate_percent_diff": "float32",
    "comp6_rate": "float32",
    "comp6_inv": "float32",
    "comp6_rate_percent_diff": "float32",
    "comp7_rate": "float32",
    "comp7_inv": "float32",
    "comp7_rate_percent_diff": "float32",
    "comp8_rate": "float32",
    "comp8_inv": "float32",
    "comp8_rate_percent_diff": "float32",
    "click_bool": "int8",
    "gross_bookings_usd": "float32",
    "booking_bool": "int8",
}

DTYPES_TEST = {
    k: v
    for k, v in DTYPES_TRAIN.items()
    if k not in ("position", "click_bool", "gross_bookings_usd", "booking_bool")
}


def _validate_sample_frac(sample_frac: float) -> None:
    if not 0 < sample_frac <= 1:
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")


def _subsample_by_search_id(
    df: pd.DataFrame, sample_frac: float, random_state: int = 42
) -> pd.DataFrame:
    _validate_sample_frac(sample_frac)
    srch_ids = df["srch_id"].unique()
    rng = np.random.RandomState(random_state)
    keep = rng.choice(srch_ids, size=max(1, int(len(srch_ids) * sample_frac)), replace=False)
    return df[df["srch_id"].isin(set(keep))].reset_index(drop=True)


def get_train_sample_cache_path(sample_frac: float, random_state: int = 42) -> Path:
    """Return the local cache path for a sampled training dataframe."""
    _validate_sample_frac(sample_frac)
    pct = int(round(sample_frac * 100))
    return DATA_DIR / f"training_sample_{pct:02d}pct_seed{random_state}.pkl"


def load_train(sample_frac: float | None = None, random_state: int = 42) -> pd.DataFrame:
    """Load training set. sample_frac subsamples by srch_id after the CSV is read."""
    df = pd.read_csv(
        DATA_DIR / "training_set_VU_DM.csv",
        dtype=DTYPES_TRAIN,
        na_values=["NULL"],
    )
    if sample_frac is not None:
        df = _subsample_by_search_id(df, sample_frac=sample_frac, random_state=random_state)
    return df


def load_test() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "test_set_VU_DM.csv",
        dtype=DTYPES_TEST,
        na_values=["NULL"],
    )


def build_train_sample_cache(
    sample_frac: float = 0.15,
    random_state: int = 42,
    output_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """
    Cache a sampled training dataframe locally for faster reruns.

    The first call still reads the full CSV once. Later calls reuse the cached
    subset, which is stored as a pandas pickle.
    """
    cache_path = Path(output_path) if output_path is not None else get_train_sample_cache_path(
        sample_frac=sample_frac,
        random_state=random_state,
    )
    if cache_path.exists() and not force:
        return cache_path

    df = load_train(sample_frac=sample_frac, random_state=random_state)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    return cache_path


def load_train_cached_sample(
    sample_frac: float = 0.15,
    random_state: int = 42,
    refresh: bool = False,
    sample_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a cached srch_id-level sample, creating it on first use."""
    cache_path = build_train_sample_cache(
        sample_frac=sample_frac,
        random_state=random_state,
        output_path=sample_path,
        force=refresh,
    )
    return pd.read_pickle(cache_path)


def train_val_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by srch_id groups. Returns (train_df, val_df)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["srch_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def dcg_at_k(relevances: np.ndarray, k: int = 5) -> float:
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int = 5) -> float:
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def compute_ndcg(
    df: pd.DataFrame,
    score_col: str,
    k: int = 5,
    relevance_col: str | None = None,
) -> float:
    """
    Compute mean NDCG@k across all searches (loop version).

    df must contain srch_id, booking_bool and click_bool, unless relevance_col
    is given. score_col names the column of predicted scores (higher is better).
    """
    if relevance_col is None:
        rel = df["booking_bool"] * 5 + (df["click_bool"] - df["booking_bool"]).clip(lower=0)
    else:
        rel = df[relevance_col]

    scores = []
    for _, group in df.groupby("srch_id"):
        group_sorted = group.sort_values(score_col, ascending=False)
        scores.append(ndcg_at_k(rel.loc[group_sorted.index].values, k))

    return float(np.mean(scores))


def compute_ndcg_fast(
    df: pd.DataFrame,
    score_col: str,
    k: int = 5,
) -> float:
    """Vectorised NDCG@k for large datasets."""
    df = df.copy()
    df["_rel"] = df["booking_bool"] * 5 + (df["click_bool"] - df["booking_bool"]).clip(lower=0)
    df["_neg_score"] = -df[score_col]
    df["_rank"] = df.groupby("srch_id")["_neg_score"].rank(method="first").astype(int)

    top_k = df[df["_rank"] <= k].copy()
    top_k["_dcg"] = top_k["_rel"] / np.log2(top_k["_rank"] + 1)
    dcg_per_query = top_k.groupby("srch_id")["_dcg"].sum()

    df_sorted = df.sort_values(["srch_id", "_rel"], ascending=[True, False])
    df_sorted["_ideal_rank"] = df_sorted.groupby("srch_id").cumcount() + 1
    ideal_top_k = df_sorted[df_sorted["_ideal_rank"] <= k].copy()
    ideal_top_k["_idcg"] = ideal_top_k["_rel"] / np.log2(ideal_top_k["_ideal_rank"] + 1)
    idcg_per_query = ideal_top_k.groupby("srch_id")["_idcg"].sum()

    ndcg = dcg_per_query / idcg_per_query
    ndcg = ndcg.fillna(0.0)
    return float(ndcg.mean())


def make_submission(
    df: pd.DataFrame,
    score_col: str,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate Kaggle submission ranked by score descending within each search.
    """
    df = df.copy()
    df["_neg_score"] = -df[score_col]
    df = df.sort_values(["srch_id", "_neg_score"])
    sub = df[["srch_id", "prop_id"]]
    if output_path is not None:
        sub.to_csv(output_path, index=False)
    return sub
