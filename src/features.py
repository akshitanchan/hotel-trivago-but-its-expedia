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

    df["ump"] = (
        df["price_usd"] * df["srch_room_count"] / (hist_price + 1)
    ).astype("float32")

    df["per_fee"] = (
        df["price_usd"]
        / (df["srch_room_count"] * df["srch_length_of_stay"] + 1)
    ).astype("float32")

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply relative and price feature engineering."""
    df = add_relative_features(df)
    df = add_price_features(df)
    return df
