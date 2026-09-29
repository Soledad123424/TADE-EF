import numpy as np

from tade_ef.dataset import LabeledDataset
from tade_ef.training import train_oof


class _CentroidModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_CentroidModel":
        self.center = float(np.mean(X[y == 1, 0]))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probability = 1 / (1 + np.exp(-(X[:, 0] - self.center)))
        return np.column_stack((1 - probability, probability))


def test_three_fold_oof_fills_each_sample_once(tmp_path) -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    dataset = LabeledDataset(
        X=np.tile(np.arange(6, dtype=float)[:, None], (1, 36)),
        y=np.asarray([0, 1, 0, 1, 0, 1]),
        groups=groups,
        sample_ids=np.asarray([f"s{i}" for i in range(6)]),
        parent_ids=np.asarray([f"p{i}" for i in range(6)]),
        start_us=np.arange(6),
        end_us=np.arange(6) + 1,
    )
    folds = [
        {"name": "f1", "test": ["a"]},
        {"name": "f2", "test": ["b"]},
        {"name": "f3", "test": ["c"]},
    ]
    probabilities = train_oof(
        dataset,
        folds,
        output_dir=tmp_path,
        model_factory=_CentroidModel,
    )
    assert probabilities.shape == (6,)
    assert np.isfinite(probabilities).all()

