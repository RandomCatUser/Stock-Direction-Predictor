from __future__ import annotations("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,  # stochastic GB, reduces overfitting
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


@dataclass
class NaiveBaselines:
    """Two simple baselines every real model must beat to be worth using."""

    majority_class: int
    persistence_acc_train: float

    @staticmethod
    def compute(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        majority = int(train_df["target"].mode()[0])
        majority_acc = (test_df["target"] == majority).mean()
        # "persistence": predict today's direction repeats tomorrow
        if "return_1d" in test_df.columns:
            naive_pred = (test_df["return_1d"] > 0).astype(int)
            persistence_acc = (naive_pred == test_df["target"]).mean()
        else:
            persistence_acc = np.nan
        return {"majority_class_accuracy": majority_acc, "persistence_accuracy": persistence_acc}


def walk_forward_splits(n_rows: int, n_folds: int = 5, min_train_size: float = 0.4):
    """Yield (train_idx, test_idx) index arrays for expanding-window
    walk-forward validation. Each fold trains on everything up to a point
    and tests on the next equally-sized chunk."""
    start = int(n_rows * min_train_size)
    remaining = n_rows - start
    fold_size = remaining // n_folds
    for i in range(n_folds):
        train_end = start + i * fold_size
        test_end = train_end + fold_size if i < n_folds - 1 else n_rows
        yield np.arange(0, train_end), np.arange(train_end, test_end)


def train_model(pipeline: Pipeline, train_df: pd.DataFrame, feature_cols: list[str]):
    X_train, y_train = train_df[feature_cols], train_df["target"]
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, path: str = "models/model.joblib"):
    joblib.dump(pipeline, path)


def load_model(path: str = "models/model.joblib") -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    from data_loader import fetch_data
    from features import build_feature_table

    raw = fetch_data("DEMO", start="2015-01-01", end="2024-01-01", source="synthetic")
    table, feature_cols = build_feature_table(raw)
    train_df, test_df = chronological_split(table, test_size=0.2)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print("Baselines:", NaiveBaselines.compute(train_df, test_df))

    for name, pipe in get_model_zoo().items():
        trained = train_model(pipe, train_df, feature_cols)
        acc = trained.score(test_df[feature_cols], test_df["target"])
        print(f"{name}: test accuracy = {acc:.4f}")
