# scripts/train/compare.py
"""
Compare all win model variants side by side.
Run this to decide which algorithm to use before training win.py.

Usage:
    PYTHONPATH=. python3 -m scripts.train.compare
"""

import pandas as pd
from features.training import build_features
from models import FEATURE_COLS, get_models, train_model


def compare():
    df = build_features()

    X = df[FEATURE_COLS].fillna(0.5)
    y = df["target"]

    test_mask = df["season"] == 20242025
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    baseline = y_test.mean()
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    print(f"Baseline (always pick home): {baseline:.3f}")

    results = []
    for name, model_cfg in get_models().items():
        print(f"\nTraining {name}...")
        result = train_model(name, model_cfg, X_train, y_train, X_test, y_test)
        results.append(result)

    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'Accuracy':>10} {'vs Baseline':>12}")
    print("=" * 55)
    for r in results:
        print(f" {r['name']:<23} {r['accuracy']:.3f} {r['accuracy'] - baseline:+.3f}")
    print("=" * 55)

    for r in results:
        print(f"\n--- {r['name']} ---")
        print(r["report"])

    for r in results:
        model = r["model"]
        if hasattr(model, "feature_importances_"):
            print(f"\nFeature importances ({r['name']}):")
            for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True):
                print(f"  {col:<35} {imp:.4f}")


if __name__ == "__main__":
    compare()
