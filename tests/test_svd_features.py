import numpy as np
import pytest

from src.svd_features import build_svd_features

N_COMPONENTS = 3
UNSEEN_DEST = 1005


def _svd_cols(n=N_COMPONENTS):
    return (
        [f"svd_dest_f{i}" for i in range(n)]
        + [f"svd_hotel_f{i}" for i in range(n)]
        + [
            "svd_affinity_score",
            "hotel_book_rate_by_site",
            "hotel_book_rate_by_country",
            "hotel_book_rate_by_dest",
        ]
    )


def test_output_has_expected_shape_column_names_and_finite_float32_values(search_log):
    out = build_svd_features(search_log, search_log, n_components=N_COMPONENTS)
    assert len(out) == len(search_log)
    assert out.shape[1] == search_log.shape[1] + 2 * N_COMPONENTS + 4
    values = out[_svd_cols()]
    assert (values.dtypes == np.float32).all()
    assert np.isfinite(values.to_numpy()).all()


def test_unseen_destination_and_hotels_get_the_mean_of_the_fitted_factors(search_log):
    train = search_log[search_log["srch_destination_id"] != UNSEEN_DEST]
    out = build_svd_features(train, search_log, n_components=N_COMPONENTS)
    unseen = out[out["srch_destination_id"] == UNSEEN_DEST]
    seen = out[out["srch_destination_id"] != UNSEEN_DEST]
    dest_cols = [f"svd_dest_f{i}" for i in range(N_COMPONENTS)]
    hotel_cols = [f"svd_hotel_f{i}" for i in range(N_COMPONENTS)]

    assert len(unseen) > 0
    assert np.isfinite(unseen[_svd_cols()].to_numpy()).all()
    fitted_dest = seen.drop_duplicates("srch_destination_id")[dest_cols]
    np.testing.assert_allclose(
        unseen[dest_cols].iloc[0], fitted_dest.mean(), rtol=1e-4, atol=1e-6
    )
    # Every unseen hotel shares one default hotel vector.
    assert (unseen[hotel_cols].nunique() == 1).all()


def test_unseen_hotel_rates_fall_back_to_the_global_booking_rate(search_log):
    train = search_log[search_log["srch_destination_id"] != UNSEEN_DEST]
    out = build_svd_features(train, search_log, n_components=N_COMPONENTS)
    unseen = out[out["srch_destination_id"] == UNSEEN_DEST]
    global_rate = train.loc[train["random_bool"] == 1, "booking_bool"].mean()
    for col in ["hotel_book_rate_by_site", "hotel_book_rate_by_country", "hotel_book_rate_by_dest"]:
        np.testing.assert_allclose(unseen[col], global_rate, rtol=1e-5)


def test_a_destination_shown_only_in_non_random_order_is_treated_as_unseen(search_log):
    train = search_log.copy()
    train.loc[train["srch_destination_id"] == UNSEEN_DEST, "random_bool"] = 0
    out = build_svd_features(train, search_log, n_components=N_COMPONENTS)
    unseen = out[out["srch_destination_id"] == UNSEEN_DEST]
    seen = out[out["srch_destination_id"] != UNSEEN_DEST]
    dest_cols = [f"svd_dest_f{i}" for i in range(N_COMPONENTS)]
    fitted_dest = seen.drop_duplicates("srch_destination_id")[dest_cols]
    np.testing.assert_allclose(
        unseen[dest_cols].iloc[0], fitted_dest.mean(), rtol=1e-4, atol=1e-6
    )


def test_labels_on_non_random_rows_do_not_change_the_features(search_log, flip_labels):
    non_random = (search_log["random_bool"] == 0).to_numpy()
    flipped = flip_labels(search_log, non_random)
    base = build_svd_features(search_log, search_log, n_components=N_COMPONENTS)
    after = build_svd_features(flipped, search_log, n_components=N_COMPONENTS)
    np.testing.assert_allclose(
        base[_svd_cols()].to_numpy(), after[_svd_cols()].to_numpy(), rtol=1e-5, atol=1e-6
    )
