# models/playoff.py

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from models.game import FEATURE_COLS

PLAYOFF_EXTRA_COLS = [
    "series_game_number",
    "home_series_wins",
    "away_series_wins",
    "seed_diff",
]

# Full feature list: 41 base (same as win_model) + 4 playoff-specific
PLAYOFF_FEATURE_COLS = FEATURE_COLS + PLAYOFF_EXTRA_COLS


def get_playoff_model():
    """
    Gradient Boosting tuned for a smaller playoff dataset.
    subsample < 1 reduces overfitting on limited data.
    """
    return GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def train_playoff_model(model, X_train, y_train, X_test, y_test):
    """Train the playoff model and return results."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return {"model": model, "accuracy": acc, "report": report}