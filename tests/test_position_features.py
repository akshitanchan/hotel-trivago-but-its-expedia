import numpy as np
import pandas as pd
import pytest

from src.position_features import build_position_features

NEW_COLS = [
    "hotel_ctr_unbiased",
    "hotel_book_rate_unbiased",
    "hotel_appearance_count",
    "dest_book_rate",
    "dest_hotel_affinity",
    "country_pair_affinity",
    "country_pair_click_rate",
]


def test_labels_on_non_random_rows_do_not_change_the_features(search_log, flip_labels):
    non_random = (search_log["random_bool"] == 0).to_numpy()
    flipped = flip_labels(search_log, non_random)
    base = build_position_features(search_log, search_log)
    after = build_position_features(flipped, search_log)
    pd.testing.assert_frame_equal(base[NEW_COLS], after[NEW_COLS])


def test_labels_on_random_rows_do_change_the_features(search_log, flip_labels):
    random_rows = (search_log["random_bool"] == 1).to_numpy()
    flipped = flip_labels(search_log, random_rows)
    base = build_position_features(search_log, search_log)
    after = build_position_features(flipped, search_log)
    assert not np.allclose(base["hotel_ctr_unbiased"], after["hotel_ctr_unbiased"])


def test_appearance_count_only_counts_random_rows(search_log):
    out = build_position_features(search_log, search_log)
    random_counts = search_log[search_log["random_bool"] == 1]["prop_id"].value_counts()
    expected = out["prop_id"].map(random_counts).fillna(0).astype(int)
    assert (out["hotel_appearance_count"] == expected).all()


def test_hotel_ctr_matches_the_smoothed_formula_on_a_tiny_frame():
    train = pd.DataFrame(
        {
            "prop_id": [1, 1, 1, 2, 2],
            "srch_destination_id": [7] * 5,
            "visitor_location_country_id": [1] * 5,
            "prop_country_id": [2] * 5,
            "random_bool": [1, 1, 0, 1, 1],
            "click_bool": [1, 0, 1, 0, 0],
            "booking_bool": [0, 0, 0, 0, 0],
        }
    )
    out = build_position_features(train, train)
    # Random rows are 1 click out of 4, hotel 1 has 1 click in 2 random rows.
    global_ctr = 1 / 4
    expected = (1 + 30 * global_ctr) / (2 + 30)
    assert out.loc[out["prop_id"] == 1, "hotel_ctr_unbiased"].iloc[0] == pytest.approx(
        expected, rel=1e-5
    )


def test_unseen_hotel_falls_back_to_the_global_rate(search_log):
    train = search_log[search_log["srch_destination_id"] != 1005]
    out = build_position_features(train, search_log)
    unseen = out[out["srch_destination_id"] == 1005]
    random_train = train[train["random_bool"] == 1]
    assert len(unseen) > 0
    assert (unseen["hotel_appearance_count"] == 0).all()
    np.testing.assert_allclose(
        unseen["hotel_ctr_unbiased"], random_train["click_bool"].mean(), rtol=1e-5
    )
