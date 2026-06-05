# scripts/train/compare.py
"""
Compare all win model variants side by side.
Run this to decide which algorithm to use before training win.py.

Usage:
    PYTHONPATH=. python3 -m scripts.train.compare
"""

import pandas as pd
from features.training import build_features
from models import FEATURE_COLS, FEATURE_COLS_LEAN, fill_features, get_models, train_model


def compare():
    df = build_features()
    y = df["target"]
    test_mask = df["season"] == 20252026
    y_train, y_test = y[~test_mask], y[test_mask]

    feature_sets = {
        f"full ({len(FEATURE_COLS)})": FEATURE_COLS,
        f"lean ({len(FEATURE_COLS_LEAN)})": FEATURE_COLS_LEAN,
    }

    baseline = y_test.mean()
    print(f"\nTrain: {int((~test_mask).sum())} | Test: {int(test_mask.sum())}")
    print(f"Baseline (always pick home): {baseline:.3f}")

    results = []
    for feature_set_name, cols in feature_sets.items():
        X = fill_features(df[cols])
        X_train, X_test = X[~test_mask], X[test_mask]

        for name, model_cfg in get_models().items():
            print(f"\nTraining {feature_set_name} | {name}...")
            result = train_model(name, model_cfg, X_train, y_train, X_test, y_test)
            result["feature_set"] = feature_set_name
            result["feature_cols"] = cols
            results.append(result)

    if not results:
        print("\nNo model results to display.")
        return

    feature_col_width = max(len("Feature Set"), *[len(r["feature_set"]) for r in results])
    model_col_width = max(len("Model"), *[len(r["name"]) for r in results])
    accuracy_col_width = 10
    baseline_col_width = 12
    column_separator_width = 3  # three single-space separators between the table columns
    table_width = (
        feature_col_width
        + model_col_width
        + accuracy_col_width
        + baseline_col_width
        + column_separator_width
    )
    print("\n" + "=" * table_width)
    print(
        f"{'Feature Set':<{feature_col_width}} {'Model':<{model_col_width}} "
        f"{'Accuracy':>{accuracy_col_width}} {'vs Baseline':>{baseline_col_width}}"
    )
    print("=" * table_width)
    for r in results:
        print(
            f"{r['feature_set']:<{feature_col_width}} {r['name']:<{model_col_width}} "
            f"{r['accuracy']:.3f} {r['accuracy'] - baseline:+.3f}"
        )
    print("=" * table_width)

    for r in results:
        print(f"\n--- {r['feature_set']} | {r['name']} ---")
        print(r["report"])

    for r in results:
        model = r["model"]
        if hasattr(model, "feature_importances_"):
            print(f"\nFeature importances ({r['feature_set']} | {r['name']}):")
            for col, imp in sorted(zip(r["feature_cols"], model.feature_importances_), key=lambda x: x[1], reverse=True):
                print(f"  {col:<35} {imp:.4f}")


if __name__ == "__main__":
    compare()
