import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from src.final_pipeline import (
    PROP_ID_PRIOR_COLS,
    TARGET_PRIOR_COLS,
    add_full_train_oof_prop_id_priors,
    add_oof_target_prior_features,
)

N_SPLITS = 5


def _fold_masks(df, n_splits=N_SPLITS):
    """Row masks for each held-out fold, split the same way the pipeline does."""
    masks = []
    for _, map_idx in GroupKFold(n_splits=n_splits).split(df, groups=df["srch_id"]):
        mask = np.zeros(len(df), dtype=bool)
        mask[map_idx] = True
        masks.append(mask)
    return masks


def _changed(before, after, cols, mask):
    return any(
        not np.allclose(before.loc[mask, c], after.loc[mask, c]) for c in cols
    )


def test_no_search_appears_in_more_than_one_fold(search_log):
    masks = _fold_masks(search_log)
    fold_of_search = {}
    for k, mask in enumerate(masks):
        for srch_id in search_log.loc[mask, "srch_id"].unique():
            assert fold_of_search.setdefault(srch_id, k) == k


@pytest.mark.parametrize(
    "encode, cols",
    [
        (add_full_train_oof_prop_id_priors, PROP_ID_PRIOR_COLS),
        (add_oof_target_prior_features, TARGET_PRIOR_COLS),
    ],
    ids=["prop_id_priors", "dest_star_and_window_priors"],
)
def test_flipping_labels_in_one_fold_leaves_that_folds_encodings_unchanged(
    search_log, flip_labels, encode, cols
):
    base, _ = encode(search_log.copy(), None, n_splits=N_SPLITS)
    masks = _fold_masks(search_log)
    held_out = masks[0]
    flipped_log = flip_labels(search_log, held_out)
    flipped, _ = encode(flipped_log, None, n_splits=N_SPLITS)

    assert not _changed(base, flipped, cols, held_out), "own-fold encodings moved"
    assert _changed(base, flipped, cols, ~held_out), "other folds should see the flip"


def test_holdout_encodings_use_train_labels_only(search_log, flip_labels):
    train = search_log[search_log["srch_id"] < 5040].copy()
    holdout = search_log[search_log["srch_id"] >= 5040].copy()
    all_rows = np.ones(len(holdout), dtype=bool)

    _, base = add_full_train_oof_prop_id_priors(train.copy(), holdout.copy())
    _, flipped = add_full_train_oof_prop_id_priors(
        train.copy(), flip_labels(holdout, all_rows)
    )
    np.testing.assert_array_equal(
        base[PROP_ID_PRIOR_COLS].to_numpy(), flipped[PROP_ID_PRIOR_COLS].to_numpy()
    )
