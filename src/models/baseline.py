"""Feature-based baseline classifier (scikit-learn).

This is one of the baselines FragGen must beat: a classifier over *flat* features
(byte histogram + statistical/entropy descriptors) with no sequence modelling. It
deliberately has access to the byte marginal, so its cross-corpus gap is the
reference the neural model should shrink. Also provides predict_proba, so the
reject option and gap metric run without torch.
"""
from __future__ import annotations
import numpy as np
from ..preprocessing.entropy import byte_histograms, aux_features


def featurize(X: np.ndarray) -> np.ndarray:
    """(N, L) uint8 -> (N, 256+8) float feature matrix."""
    return np.concatenate([byte_histograms(X), aux_features(X)], axis=1)


class FeatureBaseline:
    def __init__(self, kind="logreg", seed=0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        if kind == "logreg":
            clf = LogisticRegression(max_iter=1000, C=2.0)
        elif kind == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60,
                                random_state=seed)
        else:
            raise ValueError(kind)
        self.pipe = make_pipeline(StandardScaler(with_mean=False), clf)
        self.classes = None

    def fit(self, fs):
        self.classes = fs.classes
        self.pipe.fit(featurize(fs.X), fs.y)
        return self

    def predict_proba(self, fs):
        return self.pipe.predict_proba(featurize(fs.X))

    def predict(self, fs):
        return self.predict_proba(fs).argmax(1)
