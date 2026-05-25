import numpy as np
import pandas as pd
import pytest

from src.utils import compute_ndcg, compute_ndcg_fast


def _search(srch_id, scores, bookings, clicks):
    """One search as a frame. Clicks include the booked row, like the real data."""
    return pd.DataFrame(
        {
            "srch_id": srch_id,
            "score": scores,
            "booking_bool": bookings,
            "click_bool": clicks,
        }
    )


def test_perfect_ranking_scores_one():
    # Booking has the top score and the click-only row is second.
    df = _search(1, [8, 9, 7, 6, 5, 4], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0])
    assert compute_ndcg_fast(df, "score") == pytest.approx(1.0)


def test_worst_ordering_scores_zero_when_positives_fall_below_the_cutoff():
    # The booking sits at rank 8 so nothing relevant is inside the top 5.
    df = _search(
        1,
        scores=[8, 7, 6, 5, 4, 3, 2, 1],
        bookings=[0, 0, 0, 0, 0, 0, 0, 1],
        clicks=[0, 0, 0, 0, 0, 0, 0, 1],
    )
    assert compute_ndcg_fast(df, "score") == 0.0


def test_reversed_ranking_scores_below_perfect_ranking():
    perfect = _search(1, [6, 5, 4, 3, 2, 1], [1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0])
    worst = perfect.assign(score=-perfect["score"])
    assert compute_ndcg_fast(worst, "score") < compute_ndcg_fast(perfect, "score")


def test_hand_computed_case_uses_linear_gain_and_cutoff_five():
    # Ranked by score the relevances are [0, 1, 5, 0, 0, 5].
    # The last booking is at rank 6 so it is outside the cutoff.
    df = _search(
        1,
        scores=[6, 5, 4, 3, 2, 1],
        bookings=[0, 0, 1, 0, 0, 1],
        clicks=[0, 1, 1, 0, 0, 1],
    )
    dcg = 1 / np.log2(3) + 5 / np.log2(4)
    ideal = 5 / np.log2(2) + 5 / np.log2(3) + 1 / np.log2(4)
    assert compute_ndcg_fast(df, "score") == pytest.approx(dcg / ideal)


def test_booked_row_counts_as_five_not_six():
    # A booked row also has click_bool 1 and must not get relevance 1 + 5.
    df = _search(1, [2, 1], [1, 0], [1, 0])
    assert compute_ndcg_fast(df, "score") == pytest.approx(1.0)
    swapped = df.assign(score=[1, 2])
    assert compute_ndcg_fast(swapped, "score") == pytest.approx(1 / np.log2(3))


def test_search_without_positive_labels_scores_zero_and_stays_in_the_mean():
    good = _search(1, [3, 2, 1], [1, 0, 0], [1, 0, 0])
    empty = _search(2, [3, 2, 1], [0, 0, 0], [0, 0, 0])
    df = pd.concat([good, empty], ignore_index=True)
    assert compute_ndcg_fast(empty, "score") == 0.0
    assert compute_ndcg_fast(df, "score") == pytest.approx(0.5)


def test_fast_and_loop_versions_agree(search_log):
    rng = np.random.default_rng(1)
    search_log["score"] = rng.normal(size=len(search_log))
    assert compute_ndcg_fast(search_log, "score") == pytest.approx(
        compute_ndcg(search_log, "score")
    )
