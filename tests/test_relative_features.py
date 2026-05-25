import numpy as np
import pandas as pd
import pytest

from src.features import add_relative_features, impute_score2

RELATIVE_INPUTS = [
    "price_usd",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
]


@pytest.fixture
def two_searches():
    """Two searches on very different price scales."""
    frame = pd.DataFrame(
        {
            "srch_id": [1, 1, 1, 2, 2, 2],
            "prop_id": [10, 11, 12, 10, 11, 12],
            "price_usd": [100.0, 200.0, 300.0, 3000.0, 1000.0, 2000.0],
        }
    )
    for col in RELATIVE_INPUTS[1:]:
        frame[col] = 3.0
    return frame


def test_price_rank_is_computed_inside_each_search(two_searches):
    out = add_relative_features(two_searches)
    # A global rank would give the first search the low ranks only.
    np.testing.assert_allclose(
        out["price_usd_rank"], [1 / 3, 2 / 3, 1.0, 1.0, 1 / 3, 2 / 3], rtol=1e-6
    )


def test_diff_from_mean_and_extremes_are_per_search(two_searches):
    out = add_relative_features(two_searches)
    np.testing.assert_allclose(
        out["price_usd_diff_mean"], [-100, 0, 100, 1000, -1000, 0], rtol=1e-6
    )
    assert out["price_usd_min"].tolist() == [100.0] * 3 + [1000.0] * 3
    assert out["price_usd_max"].tolist() == [300.0] * 3 + [3000.0] * 3


def test_cheapest_and_priciest_flags_are_one_per_search(two_searches):
    out = add_relative_features(two_searches)
    assert out["price_is_cheapest"].tolist() == [1, 0, 0, 0, 1, 0]
    assert out["price_is_most_expensive"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["search_hotel_count"].tolist() == [3] * 6


def test_relative_features_do_not_depend_on_row_order(search_log):
    shuffled = search_log.sample(frac=1.0, random_state=4)
    base = add_relative_features(search_log.copy())
    after = add_relative_features(shuffled.copy()).loc[base.index]
    cols = [f"{c}_rank" for c in RELATIVE_INPUTS] + ["price_usd_diff_mean"]
    np.testing.assert_allclose(
        base[cols].to_numpy(), after[cols].to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True
    )


def test_impute_score2_fills_from_train_quantiles_only():
    train = pd.DataFrame(
        {
            "prop_country_id": [1, 1, 1, 1, 2],
            "prop_location_score2": [0.1, 0.2, 0.3, 0.4, 0.9],
        }
    )
    target = pd.DataFrame(
        {
            "prop_country_id": [1, 2, 3],
            "prop_location_score2": [np.nan, np.nan, np.nan],
        }
    )
    out = impute_score2(target, train)
    # Country 1 uses its own Q1, country 3 is unseen so it uses the global Q1.
    assert out.loc[0, "prop_location_score2"] == pytest.approx(0.175)
    assert out.loc[1, "prop_location_score2"] == pytest.approx(0.9)
    assert out.loc[2, "prop_location_score2"] == pytest.approx(train["prop_location_score2"].quantile(0.25))
