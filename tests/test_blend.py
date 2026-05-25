import numpy as np
import pandas as pd
import pytest

from src.final_pipeline import (
    optimize_signed_blend_weights,
    z_normalize_per_search,
)
from src.utils import compute_ndcg_fast


@pytest.fixture
def scored(search_log):
    """The search log with one informative model and one noisy model."""
    rng = np.random.default_rng(3)
    rel = search_log["booking_bool"] * 5 + search_log["click_bool"]
    search_log["good"] = rel + rng.normal(scale=0.5, size=len(search_log))
    search_log["noise"] = rng.normal(size=len(search_log))
    return search_log


def test_z_norm_ignores_a_constant_added_to_each_search(scored):
    shifts = scored["srch_id"].map(lambda s: 37.0 * (s % 7) + 11.0)
    shifted = scored.assign(good=scored["good"] + shifts)
    np.testing.assert_allclose(
        z_normalize_per_search(shifted, "good"),
        z_normalize_per_search(scored, "good"),
        atol=1e-4,
    )


def test_z_norm_keeps_ndcg_unchanged(scored):
    scored["good_z"] = z_normalize_per_search(scored, "good")
    assert compute_ndcg_fast(scored, "good_z") == pytest.approx(
        compute_ndcg_fast(scored, "good"), abs=1e-9
    )


def test_z_norm_handles_constant_and_single_row_searches():
    frame = pd.DataFrame(
        {
            "srch_id": [1, 1, 1, 2, 3, 3],
            "score": [4.0, 4.0, 4.0, 9.0, 1.0, 3.0],
        }
    )
    z = z_normalize_per_search(frame, "score")
    assert np.isfinite(z).all()
    assert (z.iloc[:4] == 0).all()
    assert z.iloc[4] < 0 < z.iloc[5]


def test_signed_blend_with_a_single_model_reproduces_that_model(scored):
    scored["good_z"] = z_normalize_per_search(scored, "good")
    weights, ndcg = optimize_signed_blend_weights(scored, ["good_z"], maxiter=60)
    np.testing.assert_allclose(weights, [1.0])
    assert ndcg == pytest.approx(compute_ndcg_fast(scored, "good_z"))


def test_signed_blend_weights_are_l1_normalised(scored):
    for col in ["good", "noise"]:
        scored[f"{col}_z"] = z_normalize_per_search(scored, col)
    weights, _ = optimize_signed_blend_weights(scored, ["good_z", "noise_z"], maxiter=100)
    assert np.abs(weights).sum() == pytest.approx(1.0)
