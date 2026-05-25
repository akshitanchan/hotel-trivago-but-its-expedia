"""Shared fixtures. All data here is synthetic, no real Expedia files are needed."""

import numpy as np
import pandas as pd
import pytest

from src.utils import DTYPES_TRAIN

N_DESTS = 6
HOTELS_PER_DEST = 8
COUNTRIES = [10, 20, 30]


def make_search_log(n_searches: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a small search log with the same columns as the training CSV."""
    rng = np.random.default_rng(seed)

    # Each destination has a fixed country and its own pool of hotels.
    dest_ids = [1000 + d for d in range(N_DESTS)]
    dest_country = {d: COUNTRIES[i % len(COUNTRIES)] for i, d in enumerate(dest_ids)}
    n_hotels = N_DESTS * HOTELS_PER_DEST
    hotel_star = rng.integers(1, 6, n_hotels)
    hotel_score1 = rng.uniform(0, 7, n_hotels)
    hotel_brand = rng.integers(0, 2, n_hotels)
    hotel_hist_price = rng.uniform(3.5, 6.0, n_hotels)

    rows = []
    for s in range(n_searches):
        dest_idx = s % N_DESTS
        dest = dest_ids[dest_idx]
        is_random = int((s // N_DESTS) % 2 == 0)
        n_rows = int(rng.integers(6, HOTELS_PER_DEST + 1))
        hotel_idx = dest_idx * HOTELS_PER_DEST + rng.choice(
            HOTELS_PER_DEST, size=n_rows, replace=False
        )

        booked_row = int(rng.integers(0, n_rows)) if rng.random() < 0.6 else -1
        search = {
            "srch_id": 5000 + s,
            "date_time": f"2013-0{1 + s % 9}-{10 + s % 15} {s % 24:02d}:30:00",
            "site_id": int(rng.choice([5, 14, 15])),
            "visitor_location_country_id": int(rng.choice(COUNTRIES)),
            "visitor_hist_starrating": rng.choice([np.nan, 3.5, 4.0]),
            "visitor_hist_adr_usd": rng.choice([np.nan, 90.0, 150.0]),
            "srch_destination_id": dest,
            "srch_length_of_stay": int(rng.integers(1, 6)),
            "srch_booking_window": int(rng.choice([0, 1, 5, 20, 60, 200])),
            "srch_adults_count": int(rng.integers(1, 4)),
            "srch_children_count": int(rng.integers(0, 3)),
            "srch_room_count": 1,
            "srch_saturday_night_bool": int(rng.integers(0, 2)),
            "srch_query_affinity_score": rng.choice([np.nan, -12.0, -8.5]),
            "orig_destination_distance": rng.choice([np.nan, 120.0, 900.0]),
            "random_bool": is_random,
        }
        for pos, h in enumerate(hotel_idx, start=1):
            booked = int(pos - 1 == booked_row)
            clicked = int(booked or rng.random() < 0.15)
            row = dict(search)
            row.update(
                {
                    "prop_country_id": dest_country[dest],
                    "prop_id": 100 + int(h),
                    "prop_starrating": int(hotel_star[h]),
                    "prop_review_score": rng.choice([np.nan, 3.0, 4.0, 4.5]),
                    "prop_brand_bool": int(hotel_brand[h]),
                    "prop_location_score1": float(hotel_score1[h]),
                    "prop_location_score2": rng.choice([np.nan, 0.05, 0.2, 0.6]),
                    "prop_log_historical_price": float(hotel_hist_price[h]),
                    "position": pos,
                    "price_usd": float(rng.uniform(50, 400)),
                    "promotion_flag": int(rng.integers(0, 2)),
                    "click_bool": clicked,
                    "gross_bookings_usd": float(rng.uniform(50, 400)) if booked else np.nan,
                    "booking_bool": booked,
                }
            )
            for i in range(1, 9):
                row[f"comp{i}_rate"] = rng.choice([np.nan, -1.0, 0.0, 1.0])
                row[f"comp{i}_inv"] = rng.choice([np.nan, 0.0, 1.0])
                row[f"comp{i}_rate_percent_diff"] = rng.choice([np.nan, 5.0, 12.0, 30.0])
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.astype({k: v for k, v in DTYPES_TRAIN.items() if k in df.columns})


@pytest.fixture
def make_log():
    """Factory so a test can ask for a different size or seed."""
    return make_search_log


@pytest.fixture
def search_log():
    """A fresh synthetic training log for each test, safe to mutate."""
    return make_search_log()


def _flip_labels(df: pd.DataFrame, mask) -> pd.DataFrame:
    """Return a copy where click and booking labels are flipped on masked rows."""
    out = df.copy()
    mask = np.asarray(mask)
    click = out["click_bool"].to_numpy().copy()
    book = out["booking_bool"].to_numpy().copy()
    new_click = np.where(mask, 1 - click, click)
    new_book = np.where(mask, (1 - book) * new_click, book)
    out["click_bool"] = new_click.astype(out["click_bool"].dtype)
    out["booking_bool"] = new_book.astype(out["booking_bool"].dtype)
    return out


@pytest.fixture
def flip_labels():
    """Helper that flips click and booking labels on the rows picked by a mask."""
    return _flip_labels
