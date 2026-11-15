
# Interpretable Machine Learning — Average Path Length (APL)

## 1. Project Overview

This project develops and evaluates **Average Path Length (APL)** as an interpretability metric for tree-based machine learning classifiers.

The work is part of the Interpretable Machine Learning (iML) Exercise 1.

The central research question is:

**Can the average number of decision rules evaluated during prediction serve as a useful proxy for model interpretability?**

The project implements APL within the provided evaluation framework and compares decision trees, random forests, and gradient-boosted trees across multiple classification datasets.

### Project Objectives

- Implement a custom interpretability metric using the provided `InterpretabilityMetric` interface.
- Measure prediction-path complexity for tree-based classifiers.
- Compare interpretability scores across different model configurations.
- Investigate the relationship between model complexity and predictive accuracy.
- Evaluate the metric across multiple datasets and random seeds.
- Validate mathematical correctness and examine metric robustness.
- Discuss the strengths and limitations of APL as an interpretability metric.

---

## 2. Mathematical Definition

Average Path Length measures the average number of decision nodes traversed by samples in a decision tree.

For a single decision tree:

\[
APL(M,D)=\frac{1}{N}\sum_{i=1}^{N}d_M(x_i)
\]

Where:

- \(M\) is the trained decision tree.
- \(D\) is the evaluation dataset.
- \(N\) is the number of samples.
- \(x_i\) is the \(i\)-th sample.
- \(d_M(x_i)\) is the number of decision nodes traversed by sample \(x_i\).

The terminal leaf is not counted as a decision.

### Extension to Ensembles

For random forests and gradient-boosted trees, the metric sums the path lengths across all constituent trees:

\[
APL(M,D)=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}d_t(x_i)
\]

Where \(T\) represents the number of constituent trees. For multiclass gradient boosting, all constituent regression trees are counted.

### Interpretation

**Lower APL indicates fewer decision steps per prediction and therefore lower decision-path complexity.**

Higher APL indicates that more decision rules are evaluated on average.

This is a structural proxy for interpretability, not a direct measurement of human understanding.

---

## 3. Project Structure

```text
iML26-27-Exercise-1/
│
├── iml_interpretability/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── metric.py
│   └── models.py
│
├── scripts/
│   └── run_experiment.py
│
├── tests/
│   ├── test_smoke.py
│   └── test_apl.py
│
├── data/
├── results/
├── presentation/
│
├── apl_study.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Main Components

| File | Description |
|---|---|
| `metric.py` | Defines the interpretability metric interface and the APL implementation |
| `models.py` | Defines the tree-based classifiers |
| `data.py` | Provides dataset loading and preprocessing |
| `evaluate.py` | Trains models and computes accuracy and interpretability scores |
| `scripts/run_experiment.py` | Runs experiments across datasets and random seeds |
| `tests/test_apl.py` | Contains correctness and robustness tests |
| `apl_study.py` | Standalone experimental implementation used during metric development |
| `results/` | Stores experimental results |

---

## 4. Models

The experiments evaluate three model families.

### Decision Trees

`DecisionTreeClassifier` with maximum depths:

- 1
- 2
- 3
- 5
- 8

These configurations investigate how decision-path complexity changes as the permitted tree depth increases.

### Random Forests

`RandomForestClassifier` with:

- 10 estimators, maximum depth 3
- 50 estimators, maximum depth 3

These configurations investigate the effect of ensemble size on total APL.

### Gradient-Boosted Trees

`GradientBoostingClassifier` with:

- 10 boosting stages, maximum depth 2
- 50 boosting stages, maximum depth 2

The metric sums the decision-path lengths of all constituent trees.

---

## 5. Datasets

Four classification datasets are used.

| Dataset | Classification task |
|---|---|
| Titanic | Passenger survival prediction |
| Iris | Flower species classification |
| Wine | Wine class classification |
| Breast Cancer | Breast cancer diagnostic classification |

Datasets are represented using the project's `Dataset` class, containing a feature matrix `X` and target vector `y`.

Dataset preprocessing is handled by the corresponding dataset-loading implementation.

---

## 6. Implementation

The APL metric is implemented in:

`iml_interpretability/metric.py`

The custom class inherits from:

```python
class InterpretabilityMetric(ABC):
```

The metric implementation defines:

```python
class AveragePathLength(InterpretabilityMetric):
    name = "average_path_length"
```

Its `score(model, dataset)` method:

1. Identifies the constituent trees of the fitted classifier.
2. Obtains the feature matrix from the dataset.
3. Determines which nodes each sample traverses.
4. Subtracts one from each visited-node count to exclude the leaf.
5. Calculates the mean number of decisions per tree.
6. Sums the mean path lengths across the constituent trees.
7. Returns the final score as a floating-point value.

The implementation uses scikit-learn's decision-path functionality and NumPy.

### Metric Direction

Lower score = fewer evaluated decision rules.

The metric is not normalized to a fixed interval. Scores depend on the fitted trees and the samples on which they are evaluated.

---

## 7. Experimental Methodology

The experiments use:

- 4 datasets
- 9 model configurations
- 5 random seeds: `0, 1, 2, 3, 4`
- A stratified training/test split with 25% test data

For each dataset, model configuration, and seed:

1. Split the dataset into training and testing subsets.
2. Train the selected classifier on the training subset.
3. Calculate predictive accuracy on the held-out test subset.
4. Calculate APL using the training subset.
5. Record the results.

The experiment contains:

\[
4\times9\times5=180
\]

individual model evaluations.

The results are aggregated to calculate the mean and sample standard deviation of both accuracy and APL across five random seeds.

### Evaluation Protocol

APL and test accuracy measure different aspects of the fitted model.

- **APL:** Mean decision-path complexity on training samples.
- **Accuracy:** Predictive performance on held-out samples.

This distinction is important when interpreting the results.

---

## 8. Running the Project

### Requirements

- Python compatible with the dependencies in `pyproject.toml`
- `uv`
- scikit-learn
- NumPy
- Pandas
- pytest for testing

The validated development environment used Python 3.14.3 on Windows.

### Run the Experiments

From the project root:

```bash
uv run python scripts/run_experiment.py
```

This runs the dataset–model comparisons across five random seeds.

### Run APL Tests

```bash
uv run --extra dev python -m pytest tests/test_apl.py -v -s
```

### Output Files

The experiment produces:

```text
results/apl_all_seeds.csv
results/apl_summary.csv
```

`apl_all_seeds.csv` contains the individual evaluations.

`apl_summary.csv` contains the mean accuracy, accuracy standard deviation, mean APL, and APL standard deviation for each dataset–model combination.

---

## 9. Experimental Results

The following results were obtained from five random seeds.

### Best Mean Accuracy by Dataset

| Dataset | Model | Mean accuracy | Mean APL |
|---|---|---:|---:|
| Titanic | Decision tree depth 3 | 79.5% | 3.000 |
| Iris | Random forest (10 trees) | 96.8% | 23.304 |
| Wine | Random forest (10 trees) | 98.2% | 27.165 |
| Breast Cancer | Random forest (10 trees) | 95.7% | 29.402 |

For Wine, the 50-tree random forest also achieved 98.2% mean accuracy at the reported precision, with a mean APL of 135.316.

### Key Observations

**1. Deeper trees generally produce longer decision paths.**

For Titanic, mean APL increased from 1.000 at maximum depth 1 to 7.226 at maximum depth 8.

**2. Higher APL does not guarantee higher accuracy.**

For Titanic, the depth-3 decision tree achieved the highest mean accuracy among the evaluated configurations, despite having substantially lower APL than the deeper tree and ensemble models.

**3. Ensemble size strongly affects total APL.**

For Wine, increasing the random forest from 10 to 50 trees increased mean APL from 27.165 to 135.316, while both models achieved approximately 98.2% mean accuracy.

**4. Additional permitted tree depth does not always increase APL.**

For Wine, depth-5 and depth-8 decision trees both obtained a mean APL of 2.899.

These findings indicate that APL captures an important aspect of structural prediction complexity and can help compare models with different decision-path requirements.

The experiments do not establish that lower APL always corresponds to greater human interpretability.

---

## 10. Metric Validation

Six validation tests were implemented.

| Test | Purpose | Result |
|---|---|---|
| Depth-1 tree | Verify a known path length | Passed |
| Unbalanced tree | Verify averaging over unequal paths | Passed |
| Random forest | Verify ensemble path aggregation | Passed |
| Gradient boosting | Verify boosted-tree path aggregation | Passed |
| Tree-depth bound | Verify mathematical bounds | Passed |
| Training/test robustness | Compare APL on different evaluation samples | Passed |

### Mathematical Validation

For a single tree:

\[
0\leq APL\leq d_{\max}
\]

The implementation passed a test checking this bound over several tree-depth configurations.

### Robustness Validation

An Iris decision tree with maximum depth 3 produced:

| Evaluation sample | APL |
|---|---:|
| Training subset | 2.3393 |
| Testing subset | 2.3158 |
| Absolute difference | 0.0235 |

This small difference provides initial evidence of stability for this fitted model and data split.

However, the test does not establish robustness across all datasets, models, and sampling distributions.

### Repeated Experiments

Five random seeds were used to measure variations in APL and predictive accuracy.

For example, the Iris depth-3 decision tree produced:

- Mean APL: 2.279
- APL standard deviation: 0.136

These results provide evidence about variability across the five tested train/test splits.

---

## 11. Interpretability Justification

The motivation for APL is that understanding a decision-tree prediction involves following a sequence of conditions from the root node to a prediction leaf.

A shorter decision path generally requires examining fewer rules.

Therefore, APL approximates the **amount of sequential decision information required to follow predictions**.

The metric has several useful properties:

- It has a clear mathematical definition.
- It can be calculated automatically from a fitted model.
- It accounts for the paths actually followed by evaluation samples.
- It distinguishes trees with different average path depths.
- Its summed extension measures total decision-path complexity across ensembles.
- It does not require human participants to compute the score.

APL is best interpreted as a measure of path-based structural complexity rather than a complete interpretability measure.

---

## 12. Limitations

### 12.1 Feature Complexity

APL does not consider how understandable individual features or decision thresholds are.

Two decision paths of equal length can differ substantially in human comprehensibility.

### 12.2 Ensemble Size

Summing path lengths makes APL strongly dependent on the number of constituent trees.

This is appropriate for measuring total split-decision count, but it does not directly capture the cognitive difficulty of understanding an ensemble.

### 12.3 Data Dependence

APL depends on the evaluation samples.

Different data distributions may produce different path-length averages for the same trained model.

### 12.4 No Human Evaluation

No human-subject study was performed to verify that shorter decision paths actually improve explanation comprehension.

Consequently, the experiments provide structural and empirical validation rather than direct human interpretability validation.

### 12.5 Dataset and Model Scope

The main experiments cover four classification datasets and three scikit-learn model families.

The results should not be generalized to all machine learning models or application domains.

### 12.6 Comparison Across Datasets

Absolute APL values may be influenced by dataset characteristics, feature distributions, and model structure.

Cross-dataset comparisons should therefore be interpreted cautiously.

---

## 13. Conclusion

This project implemented and evaluated Average Path Length as an interpretability metric for tree-based classification models.

The metric was integrated into the provided iML evaluation framework and tested on decision trees, random forests, and gradient-boosted trees.

Six tests passed, and 180 evaluations were completed across four datasets and five random seeds.

The experiments show that model complexity, measured through total decision-path length, can increase without corresponding improvements in predictive accuracy.

APL provides an interpretable and reproducible measure of **decision-path complexity**.

However, it should not be considered a complete measure of human interpretability. Its most defensible use is as a structural proxy that can complement predictive performance and other interpretability assessments.

---

## 14. Future Work

Potential extensions include:

- Evaluating APL consistently on held-out data.
- Comparing APL with tree depth, node count, and leaf count.
- Investigating normalized or per-tree ensemble variants.
- Evaluating additional datasets and model configurations.
- Conducting human-subject studies to examine whether APL correlates with explanation comprehension.
- Studying feature complexity alongside path length.

---

## 15. Reproducibility

The experiment uses fixed random seeds and records individual and aggregated results.

To reproduce the main evaluation:

```bash
uv run python scripts/run_experiment.py
```

To run the available APL validation tests:

```bash
uv run --extra dev python -m pytest tests/test_apl.py -v -s
```

The experiment configuration, model definitions, metric implementation, and evaluation procedure are contained in the repository source files.

**Project:** Interpretable Machine Learning — Exercise 1

**Selected Metric:** Average Path Length (APL)

**Model Class:** Tree-based classifiers

**Primary Implementation:** `iml_interpretability/metric.py`