import numpy as np
import pandas as pd


def build_position_features(
    train_fold_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute position-debiased hotel quality features from unbiased
    training-fold rows and merge them onto full_df.

    Only train_fold_df rows with random_bool=1 are used, so validation targets
    never reach the features.

    Returns full_df with these new columns, one per line below.
      - hotel_ctr_unbiased
      - hotel_book_rate_unbiased
      - hotel_appearance_count
      - dest_book_rate
      - dest_hotel_affinity
      - country_pair_affinity
      - country_pair_click_rate
      All target-derived rates use Bayesian smoothing (m=30) to damp noise
      from low-count groups.
    """
    full_df = full_df.copy()

    # Use only randomly ordered rows from the training fold
    random_train = train_fold_df[train_fold_df["random_bool"] == 1]

    # Global fallback rates from the unbiased training rows
    global_ctr = random_train["click_bool"].mean()
    global_book_rate = random_train["booking_bool"].mean()
    m = 30  # smoothing strength

    # Per-hotel click and booking rates
    hotel_agg = (
        random_train.groupby("prop_id")
        .agg(
            click_sum=("click_bool", "sum"),
            book_sum=("booking_bool", "sum"),
            hotel_appearance_count=("booking_bool", "size"),
        )
    )
    hotel_agg["hotel_ctr_unbiased"] = (
        (hotel_agg["click_sum"] + m * global_ctr) / (hotel_agg["hotel_appearance_count"] + m)
    ).astype("float32")
    hotel_agg["hotel_book_rate_unbiased"] = (
        (hotel_agg["book_sum"] + m * global_book_rate) / (hotel_agg["hotel_appearance_count"] + m)
    ).astype("float32")
    hotel_agg["hotel_appearance_count"] = hotel_agg["hotel_appearance_count"].astype("int32")

    # Keep only the output columns for the merge
    hotel_stats = hotel_agg[["hotel_ctr_unbiased", "hotel_book_rate_unbiased", "hotel_appearance_count"]]

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

    # Per-destination booking rate
    dest_agg = random_train.groupby("srch_destination_id")["booking_bool"].agg(["sum", "count"])
    dest_agg["dest_book_rate"] = (
        (dest_agg["sum"] + m * global_book_rate) / (dest_agg["count"] + m)
    ).astype("float32")
    dest_bookr = dest_agg["dest_book_rate"]
    full_df = full_df.merge(dest_bookr, on="srch_destination_id", how="left")
    full_df["dest_book_rate"] = (
        full_df["dest_book_rate"].fillna(global_book_rate).astype("float32")
    )

    # Per (destination, hotel) affinity
    dh_agg = random_train.groupby(["srch_destination_id", "prop_id"])["booking_bool"].agg(["sum", "count"]).reset_index()
    dh_agg["dest_hotel_affinity"] = (
        (dh_agg["sum"] + m * global_book_rate) / (dh_agg["count"] + m)
    ).astype("float32")
    dest_hotel = dh_agg.set_index(["srch_destination_id", "prop_id"])["dest_hotel_affinity"]
    full_df = full_df.merge(
        dest_hotel, on=["srch_destination_id", "prop_id"], how="left"
    )
    full_df["dest_hotel_affinity"] = (
        full_df["dest_hotel_affinity"].fillna(global_book_rate).astype("float32")
    )

    # Per (visitor country, hotel country) affinity
    cp_agg = random_train.groupby(
        ["visitor_location_country_id", "prop_country_id"]
    )["booking_bool"].agg(["sum", "count"]).reset_index()
    cp_agg["country_pair_affinity"] = (
        (cp_agg["sum"] + m * global_book_rate) / (cp_agg["count"] + m)
    ).astype("float32")
    country_affinity = cp_agg.set_index(
        ["visitor_location_country_id", "prop_country_id"]
    )["country_pair_affinity"]
    full_df = full_df.merge(
        country_affinity,
        on=["visitor_location_country_id", "prop_country_id"],
        how="left",
    )
    full_df["country_pair_affinity"] = (
        full_df["country_pair_affinity"].fillna(global_book_rate).astype("float32")
    )

    # Per (visitor country, hotel country) click rate
    cpc_agg = random_train.groupby(
        ["visitor_location_country_id", "prop_country_id"]
    )["click_bool"].agg(["sum", "count"]).reset_index()
    cpc_agg["country_pair_click_rate"] = (
        (cpc_agg["sum"] + m * global_ctr) / (cpc_agg["count"] + m)
    ).astype("float32")
    country_click = cpc_agg.set_index(
        ["visitor_location_country_id", "prop_country_id"]
    )["country_pair_click_rate"]
    full_df = full_df.merge(
        country_click, on=["visitor_location_country_id", "prop_country_id"], how="left"
    )
    full_df["country_pair_click_rate"] = (
        full_df["country_pair_click_rate"].fillna(global_ctr).astype("float32")
    )

    return full_df