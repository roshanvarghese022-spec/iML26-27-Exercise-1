import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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


def test_apl_does_not_exceed_tree_depth():

    X = pd.DataFrame({
        "feature": [0, 1, 2, 3, 4, 5, 6, 7]
    })

    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])

    dataset = Dataset(
        name="depth_validation",
        X=X,
        y=y
    )

    metric = AveragePathLength()

    for depth in [1, 2, 3, 5]:

        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=0
        )

        model.fit(X, y)

        apl = metric.score(model, dataset)

        assert 0 <= apl <= model.get_depth()

def test_apl_train_test_robustness():

    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=0
    )

    model.fit(X_train, y_train)

    train_dataset = Dataset(
        name="iris_train",
        X=X_train,
        y=y_train
    )

    test_dataset = Dataset(
        name="iris_test",
        X=X_test,
        y=y_test
    )

    metric = AveragePathLength()

    train_apl = metric.score(model, train_dataset)
    test_apl = metric.score(model, test_dataset)

    print(f"\nTraining APL: {train_apl:.4f}")
    print(f"Testing APL:  {test_apl:.4f}")
    print(f"Difference:   {abs(train_apl - test_apl):.4f}")

    assert 0 <= train_apl <= model.get_depth()
    assert 0 <= test_apl <= model.get_depth()