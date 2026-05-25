import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def build_svd_features(
    train_fold_df: pd.DataFrame,
    full_df: pd.DataFrame,
    n_components: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute SVD latent factors from unbiased training-fold rows and merge
    them as features onto full_df.

    Only train_fold_df rows with random_bool=1 are used to build the
    interaction matrix, so validation targets never reach the features.

    Returns full_df with 44 new columns, listed below.
      - 20 destination factors (svd_dest_f0..f19)
      - 20 hotel factors (svd_hotel_f0..f19)
      - 1 affinity score (svd_affinity_score)
      - 3 item-based CF signals (hotel_book_rate_by_site, hotel_book_rate_by_country,
        hotel_book_rate_by_dest)
    """
    full_df = full_df.copy()

    # Keep only unbiased (randomly ordered) training rows
    random_train = train_fold_df[train_fold_df["random_bool"] == 1]

    # Destination x hotel interaction matrix
    interactions = (
        random_train.groupby(["srch_destination_id", "prop_id"])
        .agg(
            bookings=("booking_bool", "sum"),
            clicks=("click_bool", "sum"),
        )
        .reset_index()
    )
    # Every booking is also a click. Score follows the Kaggle relevance
    # (booking 5, click only 1, no action 0).
    interactions["score"] = 5 * interactions["bookings"] + (
        interactions["clicks"] - interactions["bookings"]
    )

    dest_cat = pd.Categorical(interactions["srch_destination_id"])
    prop_cat = pd.Categorical(interactions["prop_id"])

    sparse_matrix = csr_matrix(
        (interactions["score"].values, (dest_cat.codes, prop_cat.codes)),
        shape=(len(dest_cat.categories), len(prop_cat.categories)),
    )

    # Factorize the matrix
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    dest_factors = svd.fit_transform(sparse_matrix)
    hotel_factors = svd.components_.T

    # Lookup tables of factors by destination and hotel
    dest_cols = [f"svd_dest_f{i}" for i in range(n_components)]
    hotel_cols = [f"svd_hotel_f{i}" for i in range(n_components)]

    dest_factors_df = pd.DataFrame(
        dest_factors, index=dest_cat.categories, columns=dest_cols
    )
    hotel_factors_df = pd.DataFrame(
        hotel_factors, index=prop_cat.categories, columns=hotel_cols
    )

    # Mean factors used for unseen destinations and hotels
    default_dest = dest_factors.mean(axis=0)
    default_hotel = hotel_factors.mean(axis=0)

    # Merge the factors onto full_df
    full_df = full_df.merge(
        dest_factors_df, left_on="srch_destination_id", right_index=True, how="left"
    )
    full_df = full_df.merge(
        hotel_factors_df, left_on="prop_id", right_index=True, how="left"
    )

    # Fill unseen keys with the mean factors
    for i, col in enumerate(dest_cols):
        full_df[col] = full_df[col].fillna(default_dest[i])
    for i, col in enumerate(hotel_cols):
        full_df[col] = full_df[col].fillna(default_hotel[i])

    # Affinity score as the dot product of the two factor vectors
    dest_vals = full_df[dest_cols].values
    hotel_vals = full_df[hotel_cols].values
    full_df["svd_affinity_score"] = (dest_vals * hotel_vals).sum(axis=1)

    # Smoothed booking rates by hotel and context
    global_book_rate = random_train["booking_bool"].mean()
    m = 30  # smoothing strength

    # Per (hotel, site) booking rate
    hs_agg = random_train.groupby(["prop_id", "site_id"])["booking_bool"].agg(["sum", "count"]).reset_index()
    hs_agg["hotel_book_rate_by_site"] = (
        (hs_agg["sum"] + m * global_book_rate) / (hs_agg["count"] + m)
    ).astype("float32")
    hs_map = hs_agg.set_index(["prop_id", "site_id"])["hotel_book_rate_by_site"]
    full_df = full_df.merge(
        hs_map, left_on=["prop_id", "site_id"], right_index=True, how="left"
    )
    full_df["hotel_book_rate_by_site"] = full_df["hotel_book_rate_by_site"].fillna(global_book_rate)

    # Per (hotel, visitor country) booking rate
    hc_agg = random_train.groupby(["prop_id", "visitor_location_country_id"])["booking_bool"].agg(["sum", "count"]).reset_index()
    hc_agg["hotel_book_rate_by_country"] = (
        (hc_agg["sum"] + m * global_book_rate) / (hc_agg["count"] + m)
    ).astype("float32")
    hc_map = hc_agg.set_index(["prop_id", "visitor_location_country_id"])["hotel_book_rate_by_country"]
    full_df = full_df.merge(
        hc_map, left_on=["prop_id", "visitor_location_country_id"], right_index=True, how="left"
    )
    full_df["hotel_book_rate_by_country"] = full_df["hotel_book_rate_by_country"].fillna(global_book_rate)

    # Per (hotel, destination) booking rate
    hd_agg = random_train.groupby(["prop_id", "srch_destination_id"])["booking_bool"].agg(["sum", "count"]).reset_index()
    hd_agg["hotel_book_rate_by_dest"] = (
        (hd_agg["sum"] + m * global_book_rate) / (hd_agg["count"] + m)
    ).astype("float32")
    hd_map = hd_agg.set_index(["prop_id", "srch_destination_id"])["hotel_book_rate_by_dest"]
    full_df = full_df.merge(
        hd_map, left_on=["prop_id", "srch_destination_id"], right_index=True, how="left"
    )
    full_df["hotel_book_rate_by_dest"] = full_df["hotel_book_rate_by_dest"].fillna(global_book_rate)

    # Cast the new columns to float32
    svd_columns = dest_cols + hotel_cols + [
        "svd_affinity_score",
        "hotel_book_rate_by_site",
        "hotel_book_rate_by_country",
        "hotel_book_rate_by_dest",
    ]
    for col in svd_columns:
        full_df[col] = full_df[col].astype("float32")

    return full_df
