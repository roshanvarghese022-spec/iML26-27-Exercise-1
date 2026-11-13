"""Inherently interpretable model candidates.

Extend or replace :func:`get_models` with the models you want to compare, e.g. trees of different
depths to probe the threshold at which a model stops being interpretable, or random forests and
gradient-boosted trees to cover a whole model class.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    return {
    # Single decision trees with different depths
    "tree_depth_1": DecisionTreeClassifier(
        max_depth=1, random_state=seed
    ),
    "tree_depth_2": DecisionTreeClassifier(
        max_depth=2, random_state=seed
    ),
    "tree_depth_3": DecisionTreeClassifier(
        max_depth=3, random_state=seed
    ),
    "tree_depth_5": DecisionTreeClassifier(
        max_depth=5, random_state=seed
    ),
    "tree_depth_8": DecisionTreeClassifier(
        max_depth=8, random_state=seed
    ),

    # Random forests
    "forest_10": RandomForestClassifier(
        n_estimators=10, max_depth=3, random_state=seed
    ),
    "forest_50": RandomForestClassifier(
        n_estimators=50, max_depth=3, random_state=seed
    ),

    # Gradient-boosted trees
    "boosting_10": GradientBoostingClassifier(
        n_estimators=10, max_depth=2, random_state=seed
    ),
    "boosting_50": GradientBoostingClassifier(
        n_estimators=50, max_depth=2, random_state=seed
    ),
}
