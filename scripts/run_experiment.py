"""Run APL interpretability experiments across multiple datasets and seeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
)

from iml_interpretability import (
    evaluate,
    get_models,
    load_titanic,
)

from iml_interpretability.data import Dataset
from iml_interpretability.metric import AveragePathLength


# --------------------------------------------------
# STEP 1: Settings
# --------------------------------------------------

OUTPUT = Path("results/scores.csv")

SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------
# STEP 2: Load datasets
# --------------------------------------------------

def get_datasets():

    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()

    return [
        load_titanic(),

        Dataset(
            name="iris",
            X=pd.DataFrame(
                iris.data,
                columns=iris.feature_names
            ),
            y=pd.Series(iris.target)
        ),

        Dataset(
            name="wine",
            X=pd.DataFrame(
                wine.data,
                columns=wine.feature_names
            ),
            y=pd.Series(wine.target)
        ),

        Dataset(
            name="breast_cancer",
            X=pd.DataFrame(
                cancer.data,
                columns=cancer.feature_names
            ),
            y=pd.Series(cancer.target)
        ),
    ]


# --------------------------------------------------
# STEP 3: Run experiments
# --------------------------------------------------

def main() -> None:

    datasets = get_datasets()

    all_results = []

    for seed in SEEDS:

        print(f"Running experiment with seed {seed}")

        results = evaluate(
            datasets=datasets,
            models=get_models(seed=seed),
            metrics=[AveragePathLength()],
            seed=seed,
        )

        results["seed"] = seed

        all_results.append(results)

    # --------------------------------------------------
    # STEP 4: Combine results
    # --------------------------------------------------

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    combined.to_csv(
        "results/apl_all_seeds.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 5: Calculate summary statistics
    # --------------------------------------------------

    summary = combined.groupby(
        ["dataset", "model"]
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        apl_mean=("average_path_length", "mean"),
        apl_std=("average_path_length", "std"),
    ).reset_index()

    # --------------------------------------------------
    # STEP 6: Save summary
    # --------------------------------------------------

    summary.to_csv(
        "results/apl_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # STEP 7: Display results
    # --------------------------------------------------

    print("\n=== APL EXPERIMENT SUMMARY ===")

    print(
        summary.round(3).to_string(index=False)
    )

    print("\nExperiment completed successfully.")
    print("Saved: results/apl_all_seeds.csv")
    print("Saved: results/apl_summary.csv")


if __name__ == "__main__":
    main()