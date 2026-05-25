import numpy as np
import pandas as pd


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """How each hotel compares to others in the same search."""
    cols = [
        "price_usd",
        "prop_starrating",
        "prop_review_score",
        "prop_location_score1",
        "prop_location_score2",
    ]
    for col in cols:
        g = df.groupby("srch_id")[col]
        df[f"{col}_rank"] = g.rank(pct=True).astype("float32")
        df[f"{col}_diff_mean"] = (df[col] - g.transform("mean")).astype("float32")
        df[f"{col}_diff_median"] = (df[col] - g.transform("median")).astype("float32")
        df[f"{col}_min"] = g.transform("min").astype("float32")
        df[f"{col}_max"] = g.transform("max").astype("float32")
        df[f"{col}_std"] = g.transform("std").astype("float32")

    df["search_hotel_count"] = (
        df.groupby("srch_id")["prop_id"].transform("count").astype("int16")
    )
    df["price_is_cheapest"] = (
        (df["price_usd"] == df["price_usd_min"]).astype("int8")
    )
    df["price_is_most_expensive"] = (
        (df["price_usd"] == df["price_usd_max"]).astype("int8")
    )

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price competitiveness and value signals."""
    hist_price = np.exp(df["prop_log_historical_price"])
    df["price_vs_historical"] = (df["price_usd"] / (hist_price + 1)).astype("float32")

    df["price_vs_visitor_hist"] = (
        df["price_usd"] / (df["visitor_hist_adr_usd"] + 1)
    ).astype("float32")

    df["star_diff_visitor"] = (
        df["prop_starrating"] - df["visitor_hist_starrating"]
    ).astype("float32")

    total_guests = df["srch_adults_count"] + df["srch_children_count"]
    df["price_per_person"] = (
        df["price_usd"] / (total_guests * df["srch_length_of_stay"] + 1)
    ).astype("float32")

    df["is_deal"] = (df["price_vs_historical"] < 1).astype("int8")

    df["ump"] = (hist_price - df["price_usd"]).astype("float32")

    denom = (df["srch_adults_count"] + df["srch_children_count"]).clip(lower=1)
    df["per_fee"] = (df["price_usd"] * df["srch_room_count"] / denom).astype("float32")

    return df

def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the eight competitors' rate, inventory and price difference columns into summary features."""
    comp_rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    comp_inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    comp_diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
 
    rates = df[comp_rate_cols]
    inv = df[comp_inv_cols]
    diffs = df[comp_diff_cols]
 
    df["comp_cheaper_count"] = (rates == 1).sum(axis=1).astype("int8")
    df["comp_pricier_count"] = (rates == -1).sum(axis=1).astype("int8")
    df["comp_same_count"] = (rates == 0).sum(axis=1).astype("int8")
    df["comp_no_inventory_count"] = (inv == 1).sum(axis=1).astype("int8")
    df["comp_data_count"] = rates.notna().sum(axis=1).astype("int8")
    df["comp_avg_diff"] = diffs.mean(axis=1).astype("float32")
    df["comp_max_diff"] = diffs.max(axis=1).astype("float32")
 
    # Fraction of competitors Expedia undercuts (0 when there is no data)
    df["comp_cheaper_frac"] = (
        df["comp_cheaper_count"] / df["comp_data_count"].replace(0, np.nan)
    ).fillna(0).astype("float32")
 
    return df
 
 
def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Flag informative null patterns."""
    null_cols = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "srch_query_affinity_score",
        "orig_destination_distance",
        "prop_review_score",
        "prop_location_score2",
    ]
    for col in null_cols:
        df[f"{col}_missing"] = df[col].isnull().astype("int8")
 
    df["has_visitor_history"] = df["visitor_hist_starrating"].notna().astype("int8")
 
    return df
 
 
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based features from search timestamp."""
    dt = pd.to_datetime(df["date_time"])
    df["month"] = dt.dt.month.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["is_weekend_search"] = (dt.dt.dayofweek >= 5).astype("int8")
 
    # Booking window buckets, ordinal-encoded
    df["booking_window_bucket"] = pd.cut(
        df["srch_booking_window"],
        bins=[-1, 0, 1, 7, 14, 30, 90, 365, 9999],
        labels=[0, 1, 2, 3, 4, 5, 6, 7],
    ).astype("float32")
 
    return df
 
 
def add_liu_count_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count and interaction features from Liu et al. (2013).
 
    None of these use target columns, so they can be computed on the full frame.
    """
    # Product of the two location scores (score2 filled with 0)
    df["score2ma"] = (
        df["prop_location_score2"].fillna(0) * df["prop_location_score1"]
    ).astype("float32")
 
    # Location score ratio
    df["score1d2"] = (
        df["prop_location_score1"] / (df["prop_location_score2"].fillna(0) + 1)
    ).astype("float32")

    # Liu et al. formulas, kept alongside the variants above
    df["score1d2_liu"] = (
        (df["prop_location_score2"].fillna(0) + 0.0001)
        / (df["prop_location_score1"] + 0.0001)
    ).astype("float32")

    df["score2ma_liu"] = (
        df["prop_location_score2"].fillna(0) * df["srch_query_affinity_score"].fillna(0)
    ).astype("float32")

    # Hotel appearance count as a popularity proxy
    prop_counts = df["prop_id"].map(df["prop_id"].value_counts())
    df["prop_id_count"] = prop_counts.astype("int32")
 
    # Destination appearance count
    dest_counts = df["srch_destination_id"].map(df["srch_destination_id"].value_counts())
    df["dest_id_count"] = dest_counts.astype("int32")
 
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply relative and price feature engineering."""
    df = add_relative_features(df)
    df = add_price_features(df)
    df = add_competitor_features(df)
    df = add_missing_indicators(df)
    df = add_temporal_features(df)
    df = add_liu_count_features(df)
    return df


def impute_score2(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing prop_location_score2 with the first quartile by prop_country_id.

    Fill values come from train_df only, so no validation or test rows leak in.
    The prop_location_score2_missing indicator is kept so the trees can still
    split on it.
    """
    col = "prop_location_score2"
    fill_map = train_df.groupby("prop_country_id")[col].quantile(0.25)
    global_fill = train_df[col].quantile(0.25)
    df[col] = df[col].fillna(df["prop_country_id"].map(fill_map))
    df[col] = df[col].fillna(global_fill)
    return df
