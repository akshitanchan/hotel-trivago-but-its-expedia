from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

from src.features import (
    add_competitor_features,
    add_missing_indicators,
    add_price_features,
    add_relative_features,
    add_temporal_features,
    impute_score2,
)
from src.position_features import build_position_features
from src.svd_features import build_svd_features
from src.utils import compute_ndcg_fast


DROP_COLS = [
    "srch_id",
    "date_time",
    "position",
    "click_bool",
    "booking_bool",
    "gross_bookings_usd",
    "random_bool",
    "relevance",
]

COMP_RAW = [
    f"comp{i}_{suffix}"
    for i in range(1, 9)
    for suffix in ("rate", "inv", "rate_percent_diff")
]

# Confirmed low/zero-gain drop from the boost experiments.
T3B_DROP = [
    "visitor_hist_starrating_missing",
    "prop_review_score_missing",
    "has_visitor_history",
]

TARGET_RANK_COLS = [
    "hotel_ctr_unbiased",
    "hotel_book_rate_unbiased",
    "dest_hotel_affinity",
    "svd_affinity_score",
    "hotel_book_rate_by_dest",
    "hotel_book_rate_by_country",
    "hotel_book_rate_by_site",
    "dest_star_book_rate_unbiased",
    "dest_star_ctr_unbiased",
    "dest_star_click_only_rate_unbiased",
    "hotel_ctr_by_window_unbiased",
    "hotel_book_rate_by_window_unbiased",
    "hotel_click_only_rate_by_window_unbiased",
    "prop_id_book_rate_oof",
    "prop_id_click_rate_oof",
    "prop_id_click_only_rate_oof",
]

TARGET_PRIOR_COLS = [
    "dest_star_book_rate_unbiased",
    "dest_star_ctr_unbiased",
    "dest_star_click_only_rate_unbiased",
    "dest_star_count_unbiased",
    "hotel_ctr_by_window_unbiased",
    "hotel_book_rate_by_window_unbiased",
    "hotel_click_only_rate_by_window_unbiased",
]

PROP_ID_PRIOR_COLS = [
    "prop_id_book_rate_oof",
    "prop_id_click_rate_oof",
    "prop_id_click_only_rate_oof",
]


def make_relevance(frame: pd.DataFrame) -> np.ndarray:
    """Kaggle relevance: booking=5, click-only=1, no action=0."""
    return (
        frame["booking_bool"] * 5
        + (frame["click_bool"] - frame["booking_bool"]).clip(lower=0)
    ).astype("int8").to_numpy()


def get_feature_cols(
    frame: pd.DataFrame,
    extra_drop: Iterable[str] | None = None,
    keep_svd_factors: bool = True,
) -> list[str]:
    """Return model feature columns with targets, IDs, raw comp columns removed."""
    drop = set(DROP_COLS + COMP_RAW + T3B_DROP)
    if extra_drop is not None:
        drop.update(extra_drop)
    if not keep_svd_factors:
        drop.update(c for c in frame.columns if c.startswith(("svd_dest_f", "svd_hotel_f")))
    return [c for c in frame.columns if c not in drop]


def sort_for_group_model(
    frame: pd.DataFrame,
    feature_cols: list[str],
    label: np.ndarray | pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Sort rows by srch_id before constructing grouped LightGBM/CatBoost data."""
    sorted_frame = frame.sort_values(["srch_id", "prop_id"], kind="mergesort")
    if label is None:
        y = make_relevance(sorted_frame) if {"booking_bool", "click_bool"}.issubset(sorted_frame.columns) else None
    else:
        y = pd.Series(label, index=frame.index).loc[sorted_frame.index].to_numpy()
    sorted_frame = sorted_frame.reset_index(drop=True)
    X = sorted_frame[feature_cols]

    group = sorted_frame.groupby("srch_id", sort=False).size().to_numpy()
    return sorted_frame, X, y, group


def add_reference_count_features(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Popularity counts fitted from an explicit reference frame."""
    prop_counts = reference_frame["prop_id"].value_counts()
    dest_counts = reference_frame["srch_destination_id"].value_counts()
    frame["prop_id_count"] = frame["prop_id"].map(prop_counts).fillna(0).astype("int32")
    frame["dest_id_count"] = (
        frame["srch_destination_id"].map(dest_counts).fillna(0).astype("int32")
    )
    return frame


def add_score2_rank_in_dest(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Rank each hotel's score2 among hotels seen for the same destination."""
    dest_score = (
        reference_frame[["srch_destination_id", "prop_id", "prop_location_score2"]]
        .dropna(subset=["prop_location_score2"])
        .groupby(["srch_destination_id", "prop_id"], as_index=False)["prop_location_score2"]
        .mean()
    )
    dest_score["score2_rank_in_dest"] = (
        dest_score.groupby("srch_destination_id")["prop_location_score2"].rank(pct=True)
    ).astype("float32")
    frame = frame.merge(
        dest_score[["srch_destination_id", "prop_id", "score2_rank_in_dest"]],
        on=["srch_destination_id", "prop_id"],
        how="left",
        validate="many_to_one",
    )
    frame["score2_rank_in_dest"] = frame["score2_rank_in_dest"].fillna(0.5).astype("float32")
    return frame


def add_imputed_score2_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add imputed score2 relative features without replacing raw score2 features."""
    col = "prop_location_score2"
    g = frame.groupby("srch_id")[col]
    frame[f"{col}_imp_rank"] = g.rank(pct=True).astype("float32")
    frame[f"{col}_imp_diff_mean"] = (frame[col] - g.transform("mean")).astype("float32")
    frame[f"{col}_imp_diff_median"] = (frame[col] - g.transform("median")).astype("float32")
    return frame


def add_liu_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Liu-style interaction features without frame-local count maps."""
    frame["score2ma"] = (
        frame["prop_location_score2"].fillna(0) * frame["prop_location_score1"]
    ).astype("float32")
    frame["score1d2"] = (
        frame["prop_location_score1"] / (frame["prop_location_score2"].fillna(0) + 1)
    ).astype("float32")
    frame["score1d2_liu"] = (
        (frame["prop_location_score2"].fillna(0) + 0.0001)
        / (frame["prop_location_score1"] + 0.0001)
    ).astype("float32")
    frame["score2ma_liu"] = (
        frame["prop_location_score2"].fillna(0)
        * frame["srch_query_affinity_score"].fillna(0)
    ).astype("float32")
    return frame


def _booking_window_bucket(window: pd.Series) -> pd.Series:
    """Shared booking-window bucket used by direct and target-derived features."""
    return pd.cut(
        window,
        bins=[-1, 0, 1, 7, 14, 30, 90, 365, 9999],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    ).astype("float32")


def _multiindex_map(
    frame: pd.DataFrame,
    key_cols: list[str],
    mapping: pd.Series,
) -> pd.Series:
    keys = pd.MultiIndex.from_frame(frame[key_cols])
    return pd.Series(keys.map(mapping), index=frame.index)


def add_reference_price_tier_features(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-search price normalization by destination and hotel star tier."""
    key_cols = ["srch_destination_id", "prop_starrating"]
    ref_price = reference_frame["price_usd"].replace([np.inf, -np.inf], np.nan)
    global_mean = float(ref_price.mean())
    global_median = float(ref_price.median())
    global_std = float(ref_price.std())
    if not np.isfinite(global_std) or global_std <= 1e-6:
        global_std = 1.0

    tier_stats = (
        reference_frame.assign(_price_usd=ref_price)
        .groupby(key_cols)["_price_usd"]
        .agg(["mean", "median", "std", "count"])
    )
    tier_stats["std"] = tier_stats["std"].replace(0, np.nan)

    tier_mean = _multiindex_map(frame, key_cols, tier_stats["mean"]).fillna(global_mean)
    tier_median = _multiindex_map(frame, key_cols, tier_stats["median"]).fillna(global_median)
    tier_std = _multiindex_map(frame, key_cols, tier_stats["std"]).fillna(global_std)
    tier_count = _multiindex_map(frame, key_cols, tier_stats["count"]).fillna(0)
    tier_std = tier_std.replace(0, global_std)

    frame["price_tier_mean"] = tier_mean.astype("float32")
    frame["price_tier_median"] = tier_median.astype("float32")
    frame["price_tier_std"] = tier_std.astype("float32")
    frame["price_tier_count"] = tier_count.astype("int32")
    frame["price_z_in_tier"] = (
        (frame["price_usd"] - frame["price_tier_mean"]) / frame["price_tier_std"]
    ).astype("float32")
    frame["price_vs_tier_median"] = (
        frame["price_usd"] / (frame["price_tier_median"].clip(lower=1.0))
    ).astype("float32")
    return frame


def add_search_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Search-level context and dispersion features."""
    price_g = frame.groupby("srch_id")["price_usd"]
    price_mean = price_g.transform("mean")
    price_std = price_g.transform("std").fillna(0)
    frame["search_price_mean"] = price_mean.astype("float32")
    frame["search_price_std"] = price_std.astype("float32")
    frame["search_price_cv"] = (price_std / (price_mean.abs() + 1.0)).astype("float32")
    frame["price_z_in_search"] = (
        (frame["price_usd"] - price_mean) / price_std.replace(0, 1)
    ).astype("float32")

    frame["search_brand_frac"] = (
        frame.groupby("srch_id")["prop_brand_bool"].transform("mean").astype("float32")
    )
    star_g = frame.groupby("srch_id")["prop_starrating"]
    search_mean_star = star_g.transform("mean")
    frame["search_mean_star"] = search_mean_star.astype("float32")
    frame["star_diff_from_search_mean"] = (
        frame["prop_starrating"] - search_mean_star
    ).astype("float32")

    frame["search_promo_frac"] = (
        frame.groupby("srch_id")["promotion_flag"].transform("mean").astype("float32")
    )
    return frame


def add_final_direct_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Non-target direct features from the Shockwave plan."""
    guests = (frame["srch_adults_count"] + frame["srch_children_count"]).clip(lower=1)
    nights = frame["srch_length_of_stay"].clip(lower=1)
    price = frame["price_usd"].clip(lower=1.0)

    frame["is_domestic"] = (
        frame["visitor_location_country_id"] == frame["prop_country_id"]
    ).astype("int8")
    frame["is_family"] = (frame["srch_children_count"] > 0).astype("int8")
    frame["total_guests"] = guests.astype("int8")
    frame["children_ratio"] = (frame["srch_children_count"] / guests).astype("float32")
    frame["price_per_night"] = (frame["price_usd"] / nights).astype("float32")
    frame["log_distance"] = np.log1p(frame["orig_destination_distance"]).astype("float32")
    frame["distance_price_interaction"] = (
        frame["orig_destination_distance"] * frame["price_usd"]
    ).astype("float32")
    frame["booking_window_x_saturday"] = (
        frame["srch_booking_window"] * frame["srch_saturday_night_bool"]
    ).astype("float32")
    frame["starrating_price_ratio"] = (frame["prop_starrating"] / price).astype("float32")
    frame["starrating_x_brand"] = (
        frame["prop_starrating"] * frame["prop_brand_bool"]
    ).astype("int8")
    frame["visitor_hist_price_diff"] = (
        frame["price_usd"] - frame["visitor_hist_adr_usd"]
    ).astype("float32")
    frame["prop_quality_score"] = (
        frame["prop_starrating"] * 2
        + frame["prop_review_score"].fillna(0)
        + frame["prop_location_score1"]
        + frame["prop_location_score2"]
    ).astype("float32")

    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
    comp_data_count = frame[comp_rate_cols].notna().sum(axis=1).replace(0, np.nan)
    weighted = frame[comp_rate_cols].fillna(0).to_numpy() * frame[comp_diff_cols].fillna(0).to_numpy()
    frame["comp_rate_weighted_diff"] = (
        pd.Series(weighted.sum(axis=1), index=frame.index) / comp_data_count
    ).fillna(0).astype("float32")
    frame["comp_advantage_score"] = (
        (frame["comp_cheaper_count"] - frame["comp_pricier_count"])
        / frame["comp_data_count"].replace(0, np.nan)
    ).fillna(0).astype("float32")
    frame["comp_no_inventory_frac"] = (
        frame["comp_no_inventory_count"] / frame["comp_data_count"].replace(0, np.nan)
    ).fillna(0).astype("float32")
    frame["any_comp_no_inventory"] = (frame["comp_no_inventory_count"] > 0).astype("int8")

    frame["price_rank_per_star"] = (
        frame.groupby(["srch_id", "prop_starrating"])["price_usd"].rank(pct=True)
    ).astype("float32")
    frame["visitor_hist_price_diff_rank"] = (
        frame.groupby("srch_id")["visitor_hist_price_diff"].rank(pct=True)
    ).astype("float32")
    frame["prop_quality_score_rank"] = (
        frame.groupby("srch_id")["prop_quality_score"].rank(pct=True)
    ).astype("float32")
    frame["srch_query_affinity_score_rank"] = (
        frame.groupby("srch_id")["srch_query_affinity_score"].rank(pct=True)
    ).astype("float32")
    frame["comp_advantage_rank"] = (
        frame.groupby("srch_id")["comp_advantage_score"].rank(pct=True)
    ).astype("float32")

    frame["is_last_minute"] = (frame["srch_booking_window"] <= 1).astype("int8")
    frame["is_advance_purchase"] = (frame["srch_booking_window"] >= 30).astype("int8")
    frame["is_leisure_search"] = (
        (frame["srch_saturday_night_bool"] == 1) | (frame["srch_length_of_stay"] >= 3)
    ).astype("int8")
    frame["last_minute_x_price_rank"] = (
        frame["is_last_minute"] * frame["price_usd_rank"]
    ).astype("float32")
    frame["window_x_deal"] = (
        frame["srch_booking_window"] * frame["is_deal"]
    ).astype("float32")

    return frame


def add_destination_star_target_features(
    train_fold_df: pd.DataFrame,
    frame: pd.DataFrame,
    m: int = 50,
) -> pd.DataFrame:
    """Bayesian-smoothed destination/star priors from unbiased training rows."""
    random_train = train_fold_df[train_fold_df["random_bool"] == 1]
    if random_train.empty:
        frame["dest_star_book_rate_unbiased"] = np.float32(0.0)
        frame["dest_star_ctr_unbiased"] = np.float32(0.0)
        frame["dest_star_click_only_rate_unbiased"] = np.float32(0.0)
        frame["dest_star_count_unbiased"] = np.int32(0)
        return frame

    global_book = float(random_train["booking_bool"].mean())
    global_ctr = float(random_train["click_bool"].mean())
    click_only = (random_train["click_bool"] - random_train["booking_bool"]).clip(lower=0)
    global_click_only = float(click_only.mean())
    key_cols = ["srch_destination_id", "prop_starrating"]

    agg = (
        random_train.assign(_click_only=click_only)
        .groupby(key_cols)
        .agg(
            book_sum=("booking_bool", "sum"),
            click_sum=("click_bool", "sum"),
            click_only_sum=("_click_only", "sum"),
            count=("booking_bool", "size"),
        )
    )
    agg["dest_star_book_rate_unbiased"] = (
        (agg["book_sum"] + m * global_book) / (agg["count"] + m)
    ).astype("float32")
    agg["dest_star_ctr_unbiased"] = (
        (agg["click_sum"] + m * global_ctr) / (agg["count"] + m)
    ).astype("float32")
    agg["dest_star_click_only_rate_unbiased"] = (
        (agg["click_only_sum"] + m * global_click_only) / (agg["count"] + m)
    ).astype("float32")

    frame["dest_star_book_rate_unbiased"] = _multiindex_map(
        frame, key_cols, agg["dest_star_book_rate_unbiased"]
    ).fillna(global_book).astype("float32")
    frame["dest_star_ctr_unbiased"] = _multiindex_map(
        frame, key_cols, agg["dest_star_ctr_unbiased"]
    ).fillna(global_ctr).astype("float32")
    frame["dest_star_click_only_rate_unbiased"] = _multiindex_map(
        frame, key_cols, agg["dest_star_click_only_rate_unbiased"]
    ).fillna(global_click_only).astype("float32")
    frame["dest_star_count_unbiased"] = _multiindex_map(
        frame, key_cols, agg["count"]
    ).fillna(0).astype("int32")
    return frame


def add_hotel_window_target_features(
    train_fold_df: pd.DataFrame,
    frame: pd.DataFrame,
    m: int = 50,
) -> pd.DataFrame:
    """Hotel conversion priors split by booking-window bucket."""
    random_train = train_fold_df[train_fold_df["random_bool"] == 1].copy()
    if random_train.empty:
        frame["hotel_ctr_by_window_unbiased"] = np.float32(0.0)
        frame["hotel_book_rate_by_window_unbiased"] = np.float32(0.0)
        frame["hotel_click_only_rate_by_window_unbiased"] = np.float32(0.0)
        return frame

    if "booking_window_bucket" not in random_train.columns:
        random_train["booking_window_bucket"] = _booking_window_bucket(
            random_train["srch_booking_window"]
        )
    if "booking_window_bucket" not in frame.columns:
        frame["booking_window_bucket"] = _booking_window_bucket(frame["srch_booking_window"])

    click_only = (random_train["click_bool"] - random_train["booking_bool"]).clip(lower=0)
    global_ctr = float(random_train["click_bool"].mean())
    global_book = float(random_train["booking_bool"].mean())
    global_click_only = float(click_only.mean())
    key_cols = ["prop_id", "booking_window_bucket"]

    agg = (
        random_train.assign(_click_only=click_only)
        .groupby(key_cols)
        .agg(
            click_sum=("click_bool", "sum"),
            book_sum=("booking_bool", "sum"),
            click_only_sum=("_click_only", "sum"),
            count=("booking_bool", "size"),
        )
    )
    agg["hotel_ctr_by_window_unbiased"] = (
        (agg["click_sum"] + m * global_ctr) / (agg["count"] + m)
    ).astype("float32")
    agg["hotel_book_rate_by_window_unbiased"] = (
        (agg["book_sum"] + m * global_book) / (agg["count"] + m)
    ).astype("float32")
    agg["hotel_click_only_rate_by_window_unbiased"] = (
        (agg["click_only_sum"] + m * global_click_only) / (agg["count"] + m)
    ).astype("float32")

    frame["hotel_ctr_by_window_unbiased"] = _multiindex_map(
        frame, key_cols, agg["hotel_ctr_by_window_unbiased"]
    ).fillna(global_ctr).astype("float32")
    frame["hotel_book_rate_by_window_unbiased"] = _multiindex_map(
        frame, key_cols, agg["hotel_book_rate_by_window_unbiased"]
    ).fillna(global_book).astype("float32")
    frame["hotel_click_only_rate_by_window_unbiased"] = _multiindex_map(
        frame, key_cols, agg["hotel_click_only_rate_by_window_unbiased"]
    ).fillna(global_click_only).astype("float32")
    return frame


def add_target_prior_features(
    train_fold_df: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply all target-derived prior maps from train_fold_df onto frame."""
    frame = add_destination_star_target_features(train_fold_df, frame)
    frame = add_hotel_window_target_features(train_fold_df, frame)
    return frame


def add_oof_target_prior_features(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Add target priors with out-of-fold values for training rows.

    Holdout/test rows are mapped from the full training frame. This avoids
    training-row self-label target encoding while keeping validation/test maps
    train-only.
    """
    n_groups = train_df["srch_id"].nunique()
    splits = min(n_splits, n_groups)
    if splits < 2:
        train_df = add_target_prior_features(train_df, train_df)
    else:
        for col in TARGET_PRIOR_COLS:
            train_df[col] = np.nan
        splitter = GroupKFold(n_splits=splits)
        for fit_idx, map_idx in splitter.split(train_df, groups=train_df["srch_id"]):
            fit_df = train_df.iloc[fit_idx]
            fold_df = train_df.iloc[map_idx].copy()
            fold_df = add_target_prior_features(fit_df, fold_df)
            train_df.loc[train_df.index[map_idx], TARGET_PRIOR_COLS] = fold_df[
                TARGET_PRIOR_COLS
            ].to_numpy()
        for col in TARGET_PRIOR_COLS:
            if train_df[col].isna().any():
                train_df[col] = train_df[col].fillna(train_df[col].mean())

    if holdout_df is not None:
        holdout_df = add_target_prior_features(train_df, holdout_df)
    for target_frame in [train_df, holdout_df]:
        if target_frame is None:
            continue
        for col in TARGET_PRIOR_COLS:
            if col not in target_frame.columns:
                continue
            if col == "dest_star_count_unbiased":
                target_frame[col] = target_frame[col].fillna(0).astype("int32")
            else:
                target_frame[col] = target_frame[col].astype("float32")
    return train_df, holdout_df


def add_prop_id_target_prior_features(
    train_fold_df: pd.DataFrame,
    frame: pd.DataFrame,
    m: int = 50,
) -> pd.DataFrame:
    """Bayesian-smoothed prop_id priors fitted from an explicit training frame."""
    if train_fold_df.empty:
        for col in PROP_ID_PRIOR_COLS:
            frame[col] = np.float32(0.0)
        return frame

    click_only = (train_fold_df["click_bool"] - train_fold_df["booking_bool"]).clip(lower=0)
    global_book = float(train_fold_df["booking_bool"].mean())
    global_click = float(train_fold_df["click_bool"].mean())
    global_click_only = float(click_only.mean())

    agg = (
        train_fold_df.assign(_click_only=click_only)
        .groupby("prop_id")
        .agg(
            book_sum=("booking_bool", "sum"),
            click_sum=("click_bool", "sum"),
            click_only_sum=("_click_only", "sum"),
            count=("booking_bool", "size"),
        )
    )
    agg["prop_id_book_rate_oof"] = (
        (agg["book_sum"] + m * global_book) / (agg["count"] + m)
    ).astype("float32")
    agg["prop_id_click_rate_oof"] = (
        (agg["click_sum"] + m * global_click) / (agg["count"] + m)
    ).astype("float32")
    agg["prop_id_click_only_rate_oof"] = (
        (agg["click_only_sum"] + m * global_click_only) / (agg["count"] + m)
    ).astype("float32")

    frame["prop_id_book_rate_oof"] = (
        frame["prop_id"].map(agg["prop_id_book_rate_oof"]).fillna(global_book).astype("float32")
    )
    frame["prop_id_click_rate_oof"] = (
        frame["prop_id"].map(agg["prop_id_click_rate_oof"]).fillna(global_click).astype("float32")
    )
    frame["prop_id_click_only_rate_oof"] = (
        frame["prop_id"]
        .map(agg["prop_id_click_only_rate_oof"])
        .fillna(global_click_only)
        .astype("float32")
    )
    return frame


def add_full_train_oof_prop_id_priors(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Add prop_id target priors with OOF values for training rows.

    Unlike the random-order priors, these use all training impressions. Validation
    and test rows are still mapped from training rows only.
    """
    n_groups = train_df["srch_id"].nunique()
    splits = min(n_splits, n_groups)
    if splits < 2:
        train_df = add_prop_id_target_prior_features(train_df, train_df)
    else:
        for col in PROP_ID_PRIOR_COLS:
            train_df[col] = np.nan
        splitter = GroupKFold(n_splits=splits)
        for fit_idx, map_idx in splitter.split(train_df, groups=train_df["srch_id"]):
            fit_df = train_df.iloc[fit_idx]
            fold_df = train_df.iloc[map_idx].copy()
            fold_df = add_prop_id_target_prior_features(fit_df, fold_df)
            train_df.loc[train_df.index[map_idx], PROP_ID_PRIOR_COLS] = fold_df[
                PROP_ID_PRIOR_COLS
            ].to_numpy()
        for col in PROP_ID_PRIOR_COLS:
            if train_df[col].isna().any():
                train_df[col] = train_df[col].fillna(train_df[col].mean())

    if holdout_df is not None:
        unseen_rate = 1.0 - holdout_df["prop_id"].isin(train_df["prop_id"]).mean()
        print(f"prop_id prior unseen fill rate: {unseen_rate:.4%}")
        holdout_df = add_prop_id_target_prior_features(train_df, holdout_df)

    for target_frame in [train_df, holdout_df]:
        if target_frame is None:
            continue
        for col in PROP_ID_PRIOR_COLS:
            if col in target_frame.columns:
                target_frame[col] = target_frame[col].astype("float32")
    return train_df, holdout_df


def add_prop_id_numerical_aggregates(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Train-reference prop_id price aggregates and current-row deviation."""
    price = reference_frame["price_usd"].replace([np.inf, -np.inf], np.nan)
    global_mean = float(price.mean())
    agg = (
        reference_frame.assign(_price_usd=price)
        .groupby("prop_id")["_price_usd"]
        .agg(["mean", "std"])
    )
    frame["prop_id_price_mean"] = (
        frame["prop_id"].map(agg["mean"]).fillna(global_mean).astype("float32")
    )
    frame["prop_id_price_std"] = (
        frame["prop_id"].map(agg["std"]).fillna(0.0).astype("float32")
    )
    frame["price_dev_from_prop_mean"] = (
        frame["price_usd"] - frame["prop_id_price_mean"]
    ).astype("float32")
    return frame


def add_oof_prop_id_numerical_aggregates(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Prop_id price aggregates with OOF values for training rows."""
    cols = ["prop_id_price_mean", "prop_id_price_std", "price_dev_from_prop_mean"]
    n_groups = train_df["srch_id"].nunique()
    splits = min(n_splits, n_groups)
    if splits < 2:
        train_df = add_prop_id_numerical_aggregates(train_df, train_df)
    else:
        for col in cols:
            train_df[col] = np.nan
        splitter = GroupKFold(n_splits=splits)
        for fit_idx, map_idx in splitter.split(train_df, groups=train_df["srch_id"]):
            fit_df = train_df.iloc[fit_idx]
            fold_df = train_df.iloc[map_idx].copy()
            fold_df = add_prop_id_numerical_aggregates(fold_df, fit_df)
            train_df.loc[train_df.index[map_idx], cols] = fold_df[cols].to_numpy()
        for col in cols:
            if train_df[col].isna().any():
                train_df[col] = train_df[col].fillna(train_df[col].mean())
            train_df[col] = train_df[col].astype("float32")

    if holdout_df is not None:
        holdout_df = add_prop_id_numerical_aggregates(holdout_df, train_df)
    return train_df, holdout_df


def add_choice_set_dominance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compact within-search top dominance features for quality/value axes."""
    higher_better = [
        "prop_location_score2",
        "prop_quality_score",
        "hotel_ctr_unbiased",
        "svd_affinity_score",
    ]
    for col in higher_better:
        if col not in frame.columns:
            continue
        g = frame.groupby("srch_id")[col]
        frame[f"{col}_gap_best"] = (g.transform("max") - frame[col]).astype("float32")
        frame[f"{col}_is_top1"] = (
            g.rank(ascending=False, method="first").eq(1).astype("int8")
        )

    if "price_per_night" in frame.columns:
        g = frame.groupby("srch_id")["price_per_night"]
        frame["price_per_night_gap_best"] = (
            frame["price_per_night"] - g.transform("min")
        ).astype("float32")
        frame["price_per_night_is_top1"] = (
            g.rank(ascending=True, method="first").eq(1).astype("int8")
        )
    return frame


def add_target_feature_ranks(
    frame: pd.DataFrame,
    cols: Iterable[str] = TARGET_RANK_COLS,
) -> pd.DataFrame:
    """Within-search ranks and deviations for fitted target-derived signals."""
    for col in cols:
        if col not in frame.columns:
            continue
        g = frame.groupby("srch_id")[col]
        frame[f"{col}_rank"] = g.rank(pct=True).astype("float32")
        frame[f"{col}_diff_mean"] = (frame[col] - g.transform("mean")).astype("float32")
    return frame


def build_base_feature_frame(
    frame: pd.DataFrame,
    impute_fit_frame: pd.DataFrame,
    count_reference_frame: pd.DataFrame,
    dest_rank_reference_frame: pd.DataFrame | None = None,
    price_tier_reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build non-target features while preserving raw and imputed score2 variants."""
    frame = frame.copy()
    if price_tier_reference_frame is None:
        price_tier_reference_frame = count_reference_frame
    frame = add_missing_indicators(frame)
    frame = add_relative_features(frame)
    frame = add_price_features(frame)
    frame = add_competitor_features(frame)
    frame = add_temporal_features(frame)
    frame = add_liu_interaction_features(frame)
    frame = impute_score2(frame, impute_fit_frame)
    frame = add_imputed_score2_relative_features(frame)
    frame = add_reference_count_features(frame, count_reference_frame)
    frame = add_final_direct_features(frame)
    frame = add_reference_price_tier_features(frame, price_tier_reference_frame)
    frame = add_search_context_features(frame)
    if dest_rank_reference_frame is None:
        dest_rank_reference_frame = count_reference_frame
    if not {"srch_destination_id", "prop_id", "prop_location_score2"}.issubset(
        dest_rank_reference_frame.columns
    ):
        dest_rank_reference_frame = impute_fit_frame
    rank_reference = dest_rank_reference_frame
    if {"prop_country_id", "prop_location_score2"}.issubset(rank_reference.columns):
        rank_reference = impute_score2(rank_reference.copy(), impute_fit_frame)
    frame = add_score2_rank_in_dest(frame, rank_reference)
    return frame


def build_validation_feature_frames(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    count_reference_frame: pd.DataFrame,
    svd_components: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leakage-safe validation features: target-derived maps fitted from train only."""
    train_df = build_base_feature_frame(
        train_raw,
        impute_fit_frame=train_raw,
        count_reference_frame=count_reference_frame,
        dest_rank_reference_frame=count_reference_frame,
        price_tier_reference_frame=train_raw,
    )
    val_df = build_base_feature_frame(
        val_raw,
        impute_fit_frame=train_raw,
        count_reference_frame=count_reference_frame,
        dest_rank_reference_frame=count_reference_frame,
        price_tier_reference_frame=train_raw,
    )

    train_df = build_svd_features(train_df, train_df, n_components=svd_components)
    val_df = build_svd_features(train_df, val_df, n_components=svd_components)
    train_df = build_position_features(train_df, train_df)
    val_df = build_position_features(train_df, val_df)
    train_df, val_df = add_oof_target_prior_features(train_df, val_df)
    train_df, val_df = add_full_train_oof_prop_id_priors(train_df, val_df)
    train_df = add_prop_id_numerical_aggregates(train_df, train_df)
    val_df = add_prop_id_numerical_aggregates(val_df, train_df)
    train_df = add_choice_set_dominance_features(train_df)
    val_df = add_choice_set_dominance_features(val_df)
    train_df = add_target_feature_ranks(train_df)
    val_df = add_target_feature_ranks(val_df)
    return train_df, val_df


def build_test_feature_frame(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    count_reference_frame: pd.DataFrame,
    svd_components: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full-train and test features for final submission."""
    train_df = build_base_feature_frame(
        train_raw,
        impute_fit_frame=train_raw,
        count_reference_frame=count_reference_frame,
        dest_rank_reference_frame=count_reference_frame,
        price_tier_reference_frame=train_raw,
    )
    test_df = build_base_feature_frame(
        test_raw,
        impute_fit_frame=train_raw,
        count_reference_frame=count_reference_frame,
        dest_rank_reference_frame=count_reference_frame,
        price_tier_reference_frame=train_raw,
    )
    train_df = build_svd_features(train_df, train_df, n_components=svd_components)
    test_df = build_svd_features(train_df, test_df, n_components=svd_components)
    train_df = build_position_features(train_df, train_df)
    test_df = build_position_features(train_df, test_df)
    train_df, test_df = add_oof_target_prior_features(train_df, test_df)
    train_df, test_df = add_full_train_oof_prop_id_priors(train_df, test_df)
    train_df = add_prop_id_numerical_aggregates(train_df, train_df)
    test_df = add_prop_id_numerical_aggregates(test_df, train_df)
    train_df = add_choice_set_dominance_features(train_df)
    test_df = add_choice_set_dominance_features(test_df)
    train_df = add_target_feature_ranks(train_df)
    test_df = add_target_feature_ranks(test_df)
    return train_df, test_df


def minmax_normalize_per_search(frame: pd.DataFrame, col: str) -> pd.Series:
    g = frame.groupby("srch_id")[col]
    min_v = g.transform("min")
    max_v = g.transform("max")
    denom = (max_v - min_v).replace(0, 1)
    return ((frame[col] - min_v) / denom).astype("float32")


def z_normalize_per_search(frame: pd.DataFrame, col: str) -> pd.Series:
    g = frame.groupby("srch_id")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, 1).fillna(1)
    return ((frame[col] - mean) / std).astype("float32")


def rrf_score(frame: pd.DataFrame, score_cols: list[str], k: int = 60) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype="float64")
    for col in score_cols:
        ranks = frame.groupby("srch_id")[col].rank(ascending=False, method="first")
        score += 1.0 / (k + ranks)
    return score.astype("float32")


def optimize_blend_weights(
    frame: pd.DataFrame,
    normalized_score_cols: list[str],
    maxiter: int = 500,
) -> tuple[np.ndarray, float]:
    """Optimize non-negative blend weights against validation NDCG@5."""
    preds = [frame[col].to_numpy(dtype="float64") for col in normalized_score_cols]
    tmp = frame[["srch_id", "booking_bool", "click_bool"]].copy()

    def objective(raw_weights: np.ndarray) -> float:
        weights = np.abs(raw_weights)
        denom = weights.sum()
        if denom < 1e-12:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / denom
        blend = np.zeros(len(frame), dtype="float64")
        for weight, pred in zip(weights, preds):
            blend += weight * pred
        tmp["_blend"] = blend
        return -compute_ndcg_fast(tmp, "_blend")

    x0 = np.ones(len(normalized_score_cols), dtype="float64") / len(normalized_score_cols)
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-5},
    )
    weights = np.abs(result.x)
    denom = weights.sum()
    weights = np.ones_like(weights) / len(weights) if denom < 1e-12 else weights / denom
    return weights, -float(result.fun)


def optimize_signed_blend_weights(
    frame: pd.DataFrame,
    normalized_score_cols: list[str],
    maxiter: int = 500,
    l2: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Optimize a signed L1-normalized blend so auxiliary models can penalize."""
    preds = [frame[col].to_numpy(dtype="float64") for col in normalized_score_cols]
    tmp = frame[["srch_id", "booking_bool", "click_bool"]].copy()

    def objective(raw_weights: np.ndarray) -> float:
        weights = raw_weights.astype("float64")
        denom = np.abs(weights).sum()
        if denom < 1e-12:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / denom
        blend = np.zeros(len(frame), dtype="float64")
        for weight, pred in zip(weights, preds):
            blend += weight * pred
        tmp["_blend"] = blend
        penalty = l2 * float(np.square(weights).sum())
        return -compute_ndcg_fast(tmp, "_blend") + penalty

    n_cols = len(normalized_score_cols)
    starts = [np.ones(n_cols, dtype="float64") / n_cols]
    click_only_idx = [i for i, col in enumerate(normalized_score_cols) if "click_only" in col]
    click_idx = [
        i
        for i, col in enumerate(normalized_score_cols)
        if "click_score" in col and "click_only" not in col
    ]
    if click_only_idx:
        x0 = starts[0].copy()
        x0[click_only_idx] = -0.25
        starts.append(x0)
    if click_idx or click_only_idx:
        x0 = starts[0].copy()
        x0[click_idx + click_only_idx] = -0.20
        starts.append(x0)

    per_start_maxiter = max(50, maxiter // len(starts))
    best_result = None
    best_objective = np.inf
    for x0 in starts:
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": per_start_maxiter, "xatol": 1e-4, "fatol": 1e-5},
        )
        value = float(result.fun)
        if value < best_objective:
            best_objective = value
            best_result = result

    if best_result is None:
        weights = starts[0]
    else:
        weights = best_result.x.astype("float64")
    denom = np.abs(weights).sum()
    weights = np.ones_like(weights) / len(weights) if denom < 1e-12 else weights / denom

    blend = np.zeros(len(frame), dtype="float64")
    for weight, pred in zip(weights, preds):
        blend += weight * pred
    tmp["_blend"] = blend
    return weights, compute_ndcg_fast(tmp, "_blend")


def save_run_json(path: str | Path, payload: dict) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
