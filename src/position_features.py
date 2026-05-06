import numpy as np
import pandas as pd


def build_position_features(
    train_fold_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute position-debiased hotel quality features from unbiased
    training-fold rows and merge them onto full_df.

    Leakage-safe: only train_fold_df rows with random_bool=1 are used,
    so validation targets never leak into features.

    Returns full_df with 6 new columns:
      - hotel_ctr_unbiased
      - hotel_book_rate_unbiased
      - hotel_appearance_count
      - dest_book_rate
      - dest_hotel_affinity
      - country_pair_affinity
    """
    full_df = full_df.copy()

    # Only use randomly-displayed rows from the training fold
    random_train = train_fold_df[train_fold_df["random_bool"] == 1]

    # Global fallback rates (from unbiased training data)
    global_ctr = random_train["click_bool"].mean()
    global_book_rate = random_train["booking_bool"].mean()

    # --- Per-hotel click and booking rates ---
    hotel_stats = (
        random_train.groupby("prop_id")
        .agg(
            hotel_ctr_unbiased=("click_bool", "mean"),
            hotel_book_rate_unbiased=("booking_bool", "mean"),
            hotel_appearance_count=("booking_bool", "size"),
        )
    )

    full_df = full_df.merge(hotel_stats, on="prop_id", how="left")
    full_df["hotel_ctr_unbiased"] = (
        full_df["hotel_ctr_unbiased"].fillna(global_ctr).astype("float32")
    )
    full_df["hotel_book_rate_unbiased"] = (
        full_df["hotel_book_rate_unbiased"].fillna(global_book_rate).astype("float32")
    )
    full_df["hotel_appearance_count"] = (
        full_df["hotel_appearance_count"].fillna(0).astype("int32")
    )

    # --- Per-destination booking rate ---
    dest_bookr = (
        random_train.groupby("srch_destination_id")["booking_bool"]
        .mean()
        .rename("dest_book_rate")
    )
    full_df = full_df.merge(dest_bookr, on="srch_destination_id", how="left")
    full_df["dest_book_rate"] = (
        full_df["dest_book_rate"].fillna(global_book_rate).astype("float32")
    )

    # --- Per-(destination, hotel) affinity ---
    dest_hotel = (
        random_train.groupby(["srch_destination_id", "prop_id"])["booking_bool"]
        .mean()
        .rename("dest_hotel_affinity")
    )
    full_df = full_df.merge(
        dest_hotel, on=["srch_destination_id", "prop_id"], how="left"
    )
    full_df["dest_hotel_affinity"] = (
        full_df["dest_hotel_affinity"].fillna(global_book_rate).astype("float32")
    )

    # --- Per-(visitor_country, prop_country) affinity ---
    country_affinity = (
        random_train.groupby(
            ["visitor_location_country_id", "prop_country_id"]
        )["booking_bool"]
        .mean()
        .rename("country_pair_affinity")
    )
    full_df = full_df.merge(
        country_affinity,
        on=["visitor_location_country_id", "prop_country_id"],
        how="left",
    )
    full_df["country_pair_affinity"] = (
        full_df["country_pair_affinity"].fillna(global_book_rate).astype("float32")
    )

    return full_df