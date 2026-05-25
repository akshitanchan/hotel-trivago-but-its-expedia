"""
Re-run the bias analysis on the Supernova validation predictions and draw two figures.

Run from the project root with `python scripts/report_artifacts.py`.

Writes docs/figures/booking_rate_by_star.png and docs/figures/feature_importance.png,
and prints the bias numbers to stdout.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gc
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import (
    compute_ndcg_fast,
    load_train,
    load_train_cached_sample,
    train_val_split,
)

FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Helpers

def recall_at_k(df, score_col, k=5):
    """Fraction of positive-relevance items placed in top-k."""
    df = df.copy()
    df["_rel"] = df["booking_bool"] * 5 + (df["click_bool"] - df["booking_bool"]).clip(lower=0)
    df["_rank"] = df.groupby("srch_id")[score_col].rank(ascending=False, method="first")
    positives = df[df["_rel"] > 0]
    if len(positives) == 0:
        return 0.0
    return float((positives["_rank"] <= k).mean())


def segment_ndcg_and_recall(df, score_col, segment_col):
    """Compute NDCG@5 and Recall@5 per segment value."""
    search_seg = df.groupby("srch_id")[segment_col].first()
    results = {}
    for value in sorted(search_seg.unique()):
        sids = search_seg.index[search_seg == value]
        seg = df[df["srch_id"].isin(sids)]
        results[value] = {
            "ndcg5": compute_ndcg_fast(seg, score_col),
            "recall5": recall_at_k(seg, score_col, k=5),
            "n_searches": len(sids),
        }
    return results


# Part 1: bias re-run on the Supernova validation predictions

print("=" * 70)
print("PART 1: Bias analysis on Supernova validation predictions")
print("=" * 70)

with open(ROOT / "outputs" / "supernova_validation_run.json") as f:
    run = json.load(f)

score_cols = run["score_cols"]
signed_weights = np.array([run["signed_z_weights"][c] for c in score_cols], dtype="float64")

seg_diag = run["segment_diagnostics"]["signed_z_blend"]
print("\nSupernova signed-z-blend segment NDCG@5 (from validation run):")
for k, v in sorted(seg_diag.items()):
    print(f"  {k}: {v:.6f}")

val_parquet = ROOT / "outputs" / "supernova_validation_predictions.parquet"
if val_parquet.exists():
    print(f"\nLoading {val_parquet}...")
    pred_df = pd.read_parquet(val_parquet)
    HAS_PREDICTIONS = True
else:
    print(f"\n{val_parquet} not found.")
    print("Will use JSON diagnostics for detection and explain mitigation analytically.")
    HAS_PREDICTIONS = False

if HAS_PREDICTIONS:
    if "is_domestic" not in pred_df.columns:
        print("Loading raw training data for segment columns...")
        raw = load_train(sample_frac=None, random_state=RANDOM_STATE)
        _, val_raw = train_val_split(raw, test_size=0.2, random_state=RANDOM_STATE)
        pred_df = pred_df.merge(
            val_raw[["srch_id", "prop_id", "visitor_location_country_id", "prop_country_id",
                      "srch_children_count"]],
            on=["srch_id", "prop_id"],
            how="left",
        )
        pred_df["is_domestic"] = (
            pred_df["visitor_location_country_id"] == pred_df["prop_country_id"]
        ).astype("int8")
        pred_df["is_family"] = (pred_df["srch_children_count"] > 0).astype("int8")
        del raw, val_raw
        gc.collect()

    if "signed_z_blend" not in pred_df.columns:
        from src.final_pipeline import z_normalize_per_search
        for col in score_cols:
            if f"{col}_z" not in pred_df.columns:
                pred_df[f"{col}_z"] = z_normalize_per_search(pred_df, col)
        pred_df["signed_z_blend"] = sum(
            w * pred_df[f"{col}_z"] for w, col in zip(signed_weights, score_cols)
        )

    SCORE_COL = "signed_z_blend"

    print("\n--- Detection (Supernova signed-z-blend) ---")
    for seg_col in ["is_domestic", "is_family"]:
        results = segment_ndcg_and_recall(pred_df, SCORE_COL, seg_col)
        print(f"\n  {seg_col}:")
        for val, metrics in sorted(results.items()):
            label = f"{seg_col}={val}"
            print(f"    {label}: NDCG@5={metrics['ndcg5']:.6f}, Recall@5={metrics['recall5']:.4f}, n_searches={metrics['n_searches']}")

    print("\n--- Mitigation: +0.02 boost to international searches ---")
    pred_df["boosted_score"] = pred_df[SCORE_COL].copy()
    international_mask = pred_df["is_domestic"] == 0
    pred_df.loc[international_mask, "boosted_score"] += 0.02

    overall_before = compute_ndcg_fast(pred_df, SCORE_COL)
    overall_after = compute_ndcg_fast(pred_df, "boosted_score")
    print(f"  Overall NDCG@5 before: {overall_before:.6f}")
    print(f"  Overall NDCG@5 after:  {overall_after:.6f}")

    for seg_col in ["is_domestic"]:
        results_before = segment_ndcg_and_recall(pred_df, SCORE_COL, seg_col)
        results_after = segment_ndcg_and_recall(pred_df, "boosted_score", seg_col)
        print(f"\n  {seg_col} before/after mitigation:")
        for val in sorted(results_before.keys()):
            label = f"{seg_col}={val}"
            b = results_before[val]
            a = results_after[val]
            print(f"    {label}: NDCG@5 {b['ndcg5']:.6f} -> {a['ndcg5']:.6f}, Recall@5 {b['recall5']:.4f} -> {a['recall5']:.4f}")

    del pred_df
    gc.collect()
else:
    print("\nDetection numbers from the Supernova validation JSON:")
    print(f"  domestic NDCG@5:      {seg_diag.get('is_domestic=1', 'N/A'):.4f}")
    print(f"  international NDCG@5: {seg_diag.get('is_domestic=0', 'N/A'):.4f}")
    print(f"  family NDCG@5:        {seg_diag.get('is_family=1', 'N/A'):.4f}")
    print(f"  non-family NDCG@5:    {seg_diag.get('is_family=0', 'N/A'):.4f}")
    print("\nA constant +0.02 boost to all international hotels in a search does not")
    print("change the within-search ranking when every candidate in that search is")
    print("international. Only mixed-country searches are affected, and those are rare,")
    print("so the boost cannot close the gap.")
    print("\nExact mitigation numbers need the validation predictions parquet.")


# Part 2: booking rate by star rating plot

print("\n" + "=" * 70)
print("PART 2: Booking rate by star rating plot")
print("=" * 70)

sample_path = ROOT / "data" / "training_sample_15pct_seed42.pkl"
if sample_path.exists():
    print(f"Loading cached sample from {sample_path}...")
    train_df = pd.read_pickle(sample_path)
else:
    print("Loading full training data (no cache found)...")
    train_df = load_train(sample_frac=0.15, random_state=RANDOM_STATE)

star_stats = train_df.groupby("prop_starrating").agg(
    booking_rate=("booking_bool", "mean"),
    click_rate=("click_bool", "mean"),
    count=("booking_bool", "size"),
).reset_index()

print("\nBooking and click rates by star rating:")
print(star_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 3.5))
x = star_stats["prop_starrating"].to_numpy()
width = 0.35
ax.bar(x - width/2, star_stats["booking_rate"] * 100, width, label="Booking rate", color="#2c7bb6")
ax.bar(x + width/2, star_stats["click_rate"] * 100, width, label="Click rate", color="#fdae61")
ax.set_xlabel("Hotel star rating")
ax.set_ylabel("Rate (%)")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}" if s > 0 else "0\n(unknown)" for s in x], fontsize=8)
ax.legend(fontsize=8)
ax.set_title("Booking and click rates by star rating", fontsize=10)
fig.tight_layout()
out_path = FIG_DIR / "booking_rate_by_star.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved {out_path}")


# Part 3: feature importance plot

print("\n" + "=" * 70)
print("PART 3: Feature importance (top 20) from quick RankXENDCG")
print("=" * 70)

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Skipping feature importance plot.")
    sys.exit(0)

from src.final_pipeline import (
    build_validation_feature_frames,
    get_feature_cols,
    make_relevance,
    sort_for_group_model,
)

print("Building features on 15% sample...")
train_sub, val_sub = train_val_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

count_ref = train_sub[["prop_id", "srch_destination_id"]]
train_feat, _ = build_validation_feature_frames(train_sub, val_sub, count_reference_frame=count_ref, svd_components=20)
del val_sub, count_ref
gc.collect()

feature_cols = get_feature_cols(train_feat)
print(f"Feature count: {len(feature_cols)}")

sorted_df, X, y, groups = sort_for_group_model(train_feat, feature_cols, make_relevance(train_feat))

params = {
    "objective": "rank_xendcg",
    "metric": "ndcg",
    "ndcg_eval_at": [5],
    "num_leaves": 255,
    "learning_rate": 0.05,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "label_gain": [0, 1, 2, 3, 4, 5],
    "verbose": -1,
    "seed": RANDOM_STATE,
    "feature_pre_filter": False,
    "num_threads": -1,
}

try:
    params_gpu = {**params, "device": "gpu", "gpu_use_dp": False}
    train_set = lgb.Dataset(X, label=y, group=groups)
    model = lgb.train(params_gpu, train_set, num_boost_round=500)
    print("Trained on GPU, 500 rounds")
except Exception:
    train_set = lgb.Dataset(X, label=y, group=groups)
    model = lgb.train(params, train_set, num_boost_round=500)
    print("Trained on CPU, 500 rounds")

importance = model.feature_importance(importance_type="gain")
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance,
}).sort_values("importance", ascending=False).head(20)

print("\nTop 20 features by gain:")
print(feat_imp.to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
feat_imp_plot = feat_imp.sort_values("importance", ascending=True)
ax.barh(
    range(len(feat_imp_plot)),
    feat_imp_plot["importance"].to_numpy(),
    color="#2c7bb6",
)
ax.set_yticks(range(len(feat_imp_plot)))
names = feat_imp_plot["feature"].tolist()
names = [n.replace("_unbiased", "").replace("_oof", "") for n in names]
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel("Importance (gain)", fontsize=9)
ax.set_title("Top 20 features by LightGBM gain", fontsize=10)
fig.tight_layout()
out_path = FIG_DIR / "feature_importance.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Saved {out_path}")

del model, train_set, X, y, groups, sorted_df, train_feat, train_df
gc.collect()

print("\n" + "=" * 70)
print("Done. Figures written to docs/figures/.")
print("=" * 70)
