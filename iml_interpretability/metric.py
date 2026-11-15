"""Interpretability metrics.

Implement your own metric by subclassing :class:`InterpretabilityMetric`. The metric must be
applicable to every model of the chosen model class (e.g. decision trees, random forests and
gradient-boosted trees), so that scores are comparable across models and datasets.
:class:`RelativeNumberOfLeaves` is a minimal example for a single decision tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from iml_interpretability.data import Dataset


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """
    
class AveragePathLength(InterpretabilityMetric):
    """
    Average Path Length (APL).

    Measures the average number of decision rules
    evaluated per sample.

    For ensembles, paths across all trees are summed.

    Lower APL indicates fewer decisions to follow.
    """

    name = "average_path_length"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:

        # Step 1: Identify the trees in the model
        if isinstance(model, DecisionTreeClassifier):
            trees = [model]

        elif isinstance(model, (
            RandomForestClassifier,
            GradientBoostingClassifier
        )):
            trees = np.asarray(
                model.estimators_,
                dtype=object
            ).ravel()

        else:
            raise TypeError(
                f"Unsupported model: {type(model).__name__}"
            )

        # Step 2: Get the dataset features
        X = np.asarray(dataset.X, dtype=np.float32)

        if len(X) == 0:
            raise ValueError("Cannot calculate APL on an empty dataset")

        total_apl = 0.0

        # Step 3: Calculate the APL for every tree
        for tree in trees:

            paths = tree.tree_.decision_path(X)

            # Number of visited nodes minus the leaf
            path_lengths = np.diff(paths.indptr) - 1

            # Average over all samples
            total_apl += np.mean(path_lengths)

        # Step 4: Return the final score
        return float(total_apl)
