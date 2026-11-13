import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from iml_interpretability.data import Dataset
from iml_interpretability.metric import AveragePathLength


def test_apl_depth_one():

    # Four samples with two classes
    X = pd.DataFrame({
        "feature": [0, 1, 2, 3]
    })

    y = pd.Series([0, 0, 1, 1])

    dataset = Dataset(
        name="test_dataset",
        X=X,
        y=y
    )

    # Train a tree that can make only one decision
    model = DecisionTreeClassifier(
        max_depth=1,
        random_state=0
    )

    model.fit(X, y)

    # Calculate APL
    metric = AveragePathLength()
    result = metric.score(model, dataset)

    # All samples should traverse exactly one split
    assert result == pytest.approx(1.0)
def test_apl_unbalanced_tree():

    X = pd.DataFrame({
        "feature": [0, 1, 2, 3]
    })

    y = pd.Series([0, 1, 1, 0])

    dataset = Dataset(
        name="unbalanced_tree",
        X=X,
        y=y
    )

    model = DecisionTreeClassifier(
        max_depth=2,
        random_state=0
    )

    model.fit(X, y)

    metric = AveragePathLength()
    result = metric.score(model, dataset)

    assert model.get_depth() == 2

    assert result == pytest.approx(1.75)   
def test_apl_random_forest():
    X = pd.DataFrame({
        "feature": [0, 1, 2, 3, 4, 5]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])

    dataset = Dataset(
        name="forest_test",
        X=X,
        y=y
    )

    model = RandomForestClassifier(
        n_estimators=3,
        max_depth=1,
        random_state=0
    )
    model.fit(X, y)

    result = AveragePathLength().score(model, dataset)

    assert result == pytest.approx(3.0)


def test_apl_gradient_boosting():
    X = pd.DataFrame({
        "feature": [0, 1, 2, 3, 4, 5]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])

    dataset = Dataset(
        name="boosting_test",
        X=X,
        y=y
    )

    model = GradientBoostingClassifier(
        n_estimators=3,
        max_depth=1,
        random_state=0
    )
    model.fit(X, y)

    result = AveragePathLength().score(model, dataset)

    assert result == pytest.approx(3.0)
