#!/usr/bin/env python3
"""Recompute bias metrics for the Supernova validation predictions.

Run this where the final validation predictions parquet exists. The script
does not train anything. It loads the predictions, rebuilds the signed z-score
blend if it is missing, and merges validation labels and segment columns if
they are missing. It then reports NDCG@5 and Recall@5 for domestic versus
international searches and for family versus non-family searches, plus the
effect of a constant score boost for international hotels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLS = [
    "srch_id",
    "prop_id",
    "visitor_location_country_id",
    "prop_country_id",
    "srch_children_count",
    "click_bool",
    "booking_bool",
]


def z_normalize_per_search(frame: pd.DataFrame, col: str) -> pd.Series:
    group = frame.groupby("srch_id")[col]
    mean = group.transform("mean")
    std = group.transform("std").replace(0, 1).fillna(1)
    return ((frame[col] - mean) / std).astype("float32")


def ensure_signed_z_blend(pred: pd.DataFrame, run: dict, score_col: str) -> pd.DataFrame:
    if score_col in pred.columns:
        return pred

    score_cols = run["score_cols"]
    weights = run.get("signed_z_weights")
    if not weights:
        raise ValueError("run JSON does not contain signed_z_weights")

    missing = [col for col in score_cols if col not in pred.columns]
    if missing:
        raise ValueError(
            "Prediction parquet is missing base score columns needed to rebuild "
            f"{score_col}: {missing}"
        )

    pred = pred.copy()
    pred[score_col] = 0.0
    for col in score_cols:
        z_col = f"{col}_z"
        if z_col not in pred.columns:
            pred[z_col] = z_normalize_per_search(pred, col)
        pred[score_col] += float(weights[col]) * pred[z_col]
    return pred


def attach_labels_and_segments(
    pred: pd.DataFrame,
    train_csv: Path,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    needed = {
        "click_bool",
        "booking_bool",
        "visitor_location_country_id",
        "prop_country_id",
        "srch_children_count",
    }
    if needed.issubset(pred.columns):
        out = pred.copy()
    else:
        from sklearn.model_selection import GroupShuffleSplit

        raw = pd.read_csv(train_csv, usecols=RAW_COLS, na_values=["NULL"])
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        _, val_idx = next(splitter.split(raw, groups=raw["srch_id"]))
        val_raw = raw.iloc[val_idx].reset_index(drop=True)
        merge_cols = ["srch_id", "prop_id"]
        out = pred.merge(val_raw, on=merge_cols, how="left", suffixes=("", "_raw"))
        for col in needed:
            raw_col = f"{col}_raw"
            if raw_col in out.columns:
                out[col] = out[col].fillna(out[raw_col]) if col in out.columns else out[raw_col]
                out = out.drop(columns=[raw_col])

    out["is_domestic"] = (
        out["visitor_location_country_id"] == out["prop_country_id"]
    ).astype("int8")
    out["is_family"] = (out["srch_children_count"] > 0).astype("int8")
    return out


def relevance(frame: pd.DataFrame) -> pd.Series:
    return frame["booking_bool"] * 5 + (
        frame["click_bool"] - frame["booking_bool"]
    ).clip(lower=0)


def compute_ndcg_fast(frame: pd.DataFrame, score_col: str, k: int = 5) -> float:
    df = frame.copy()
    df["_rel"] = relevance(df)
    df["_neg_score"] = -df[score_col]
    df["_rank"] = df.groupby("srch_id")["_neg_score"].rank(method="first").astype(int)

    top_k = df[df["_rank"] <= k].copy()
    top_k["_dcg"] = top_k["_rel"] / np.log2(top_k["_rank"] + 1)
    dcg_per_query = top_k.groupby("srch_id")["_dcg"].sum()

    ideal = df.sort_values(["srch_id", "_rel"], ascending=[True, False]).copy()
    ideal["_ideal_rank"] = ideal.groupby("srch_id").cumcount() + 1
    ideal_top_k = ideal[ideal["_ideal_rank"] <= k].copy()
    ideal_top_k["_idcg"] = ideal_top_k["_rel"] / np.log2(ideal_top_k["_ideal_rank"] + 1)
    idcg_per_query = ideal_top_k.groupby("srch_id")["_idcg"].sum()

    return float((dcg_per_query / idcg_per_query).fillna(0.0).mean())


def recall_at_k(frame: pd.DataFrame, score_col: str, k: int = 5) -> float:
    df = frame.copy()
    df["_rel"] = relevance(df)
    df["_rank"] = df.groupby("srch_id")[score_col].rank(ascending=False, method="first")
    positives = df[df["_rel"] > 0]
    if positives.empty:
        return 0.0
    return float((positives["_rank"] <= k).mean())


def segment_metrics(frame: pd.DataFrame, score_col: str, segment_col: str) -> dict:
    search_seg = frame.groupby("srch_id")[segment_col].first()
    results = {}
    for value in sorted(search_seg.unique()):
        search_ids = search_seg.index[search_seg == value]
        segment = frame[frame["srch_id"].isin(search_ids)]
        results[f"{segment_col}={value}"] = {
            "ndcg5": compute_ndcg_fast(segment, score_col),
            "recall5": recall_at_k(segment, score_col),
            "n_searches": int(len(search_ids)),
            "n_rows": int(len(segment)),
        }
    return results


def epr(results: dict, low_key: str, high_key: str) -> float:
    low = results[low_key]["ndcg5"]
    high = results[high_key]["ndcg5"]
    return float(min(low, high) / max(low, high))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="outputs/supernova_validation_predictions.parquet",
        help="Validation predictions parquet from the final run.",
    )
    parser.add_argument(
        "--run-json",
        default="outputs/supernova_validation_run.json",
        help="Final validation run JSON containing score columns and blend weights.",
    )
    parser.add_argument(
        "--train-csv",
        default="data/training_set_VU_DM.csv",
        help="Raw training CSV, used only if labels or segment columns are missing.",
    )
    parser.add_argument("--score-col", default="signed_z_blend")
    parser.add_argument("--boost", type=float, default=0.02)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", default="outputs/bias_metrics_check.json", help="JSON output path.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    run_json_path = Path(args.run_json)
    train_csv_path = Path(args.train_csv)

    with run_json_path.open() as f:
        run = json.load(f)

    pred = pd.read_parquet(predictions_path)
    pred = ensure_signed_z_blend(pred, run, args.score_col)
    pred = attach_labels_and_segments(
        pred,
        train_csv=train_csv_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    detection = {
        "overall": {
            "ndcg5": compute_ndcg_fast(pred, args.score_col),
            "recall5": recall_at_k(pred, args.score_col),
            "n_searches": int(pred["srch_id"].nunique()),
            "n_rows": int(len(pred)),
        },
        "is_domestic": segment_metrics(pred, args.score_col, "is_domestic"),
        "is_family": segment_metrics(pred, args.score_col, "is_family"),
    }
    detection["is_domestic_epr"] = epr(
        detection["is_domestic"], "is_domestic=0", "is_domestic=1"
    )
    detection["is_family_epr"] = epr(
        detection["is_family"], "is_family=0", "is_family=1"
    )

    boosted_col = f"{args.score_col}_intl_boost_{str(args.boost).replace('.', 'p')}"
    pred[boosted_col] = pred[args.score_col]
    pred.loc[pred["is_domestic"] == 0, boosted_col] += args.boost

    domestic_nunique = pred.groupby("srch_id")["is_domestic"].nunique()
    mixed_queries = int((domestic_nunique > 1).sum())
    post_processing = {
        "boost": args.boost,
        "mixed_domestic_queries": {
            "count": mixed_queries,
            "fraction": float(mixed_queries / pred["srch_id"].nunique()),
        },
        "before": {
            "overall_ndcg5": detection["overall"]["ndcg5"],
            "overall_recall5": detection["overall"]["recall5"],
            "is_domestic": detection["is_domestic"],
        },
        "after": {
            "overall_ndcg5": compute_ndcg_fast(pred, boosted_col),
            "overall_recall5": recall_at_k(pred, boosted_col),
            "is_domestic": segment_metrics(pred, boosted_col, "is_domestic"),
        },
    }

    result = {
        "score_col": args.score_col,
        "predictions": str(predictions_path),
        "detection": detection,
        "post_processing": post_processing,
    }

    print(json.dumps(result, indent=2))
    print("\nRounded values:")
    dom = detection["is_domestic"]
    fam = detection["is_family"]
    print(
        "Domestic:      "
        f"NDCG@5={dom['is_domestic=1']['ndcg5']:.4f}, "
        f"Recall@5={dom['is_domestic=1']['recall5']:.3f}"
    )
    print(
        "International: "
        f"NDCG@5={dom['is_domestic=0']['ndcg5']:.4f}, "
        f"Recall@5={dom['is_domestic=0']['recall5']:.3f}"
    )
    print(
        "Family:        "
        f"NDCG@5={fam['is_family=1']['ndcg5']:.4f}, "
        f"Recall@5={fam['is_family=1']['recall5']:.3f}"
    )
    print(
        "Non-family:    "
        f"NDCG@5={fam['is_family=0']['ndcg5']:.4f}, "
        f"Recall@5={fam['is_family=0']['recall5']:.3f}"
    )
    print(
        "Post-process boost overall: "
        f"{post_processing['before']['overall_ndcg5']:.6f} -> "
        f"{post_processing['after']['overall_ndcg5']:.6f}"
    )
    print(
        "Mixed domestic/international validation searches: "
        f"{mixed_queries:,} / {pred['srch_id'].nunique():,} "
        f"({post_processing['mixed_domestic_queries']['fraction']:.4%})"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
