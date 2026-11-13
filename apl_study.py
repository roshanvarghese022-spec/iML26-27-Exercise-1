"""
Average Path Length (APL) interpretability study
Metric : APL   = average number of yes/no questions per prediction
                 (summed over all trees for forests / boosting)
Score  : min(1, LIMIT / APL)
Models : decision trees, random forests, gradient-boosted trees
Run    : pip install scikit-learn numpy pandas matplotlib
         python apl_study.py
"""
# ------------------------------------------------------------------ STEP 1: imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # remove this line to see plot windows
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)

# ------------------------------------------------------------------ STEP 2: settings
LIMIT = 7                  # working-memory limit (score = 1 when APL <= 7)
LIMITS_TO_TEST = [5, 7, 10]  # sensitivity check for the threshold
SEEDS = [0, 1, 2, 3, 4]    # repeat on 5 different train/test splits

# ------------------------------------------------------------------ STEP 3: datasets
DATASETS = {
    "iris":          (load_iris, "clf"),
    "wine":          (load_wine, "clf"),
    "breast_cancer": (load_breast_cancer, "clf"),
    "diabetes":      (load_diabetes, "reg"),
}

# ------------------------------------------------------------------ STEP 4: models
def make_models(task, seed):
    clf = task == "clf"
    DT = DecisionTreeClassifier if clf else DecisionTreeRegressor
    RF = RandomForestClassifier if clf else RandomForestRegressor
    GB = GradientBoostingClassifier if clf else GradientBoostingRegressor
    models = {}
    for d in [1, 2, 3, 4, 5, 6, 8, 10, None]:
        models[f"tree d={d}"] = DT(max_depth=d, random_state=seed)
    for n in [3, 10, 50]:
        models[f"forest n={n} d=3"] = RF(n_estimators=n, max_depth=3, random_state=seed)
    for n in [3, 10, 50]:
        models[f"gbt n={n} d=2"] = GB(n_estimators=n, max_depth=2, random_state=seed)
    return models

# ------------------------------------------------------------------ STEP 5: the metric
def get_trees(model):
    if hasattr(model, "estimators_"):                    # forest or boosting
        return [e.tree_ for e in np.array(model.estimators_).ravel()]
    return [model.tree_]                                 # single tree

def avg_path_length(model, X):
    """APL = mean over samples of (questions asked), summed over all trees."""
    total = 0.0
    for t in get_trees(model):
        nodes_visited = t.decision_path(X.astype(np.float32)).sum(axis=1).A1
        total += (nodes_visited - 1).mean()              # minus 1: leaf is an answer
    return total

def n_leaves(model):                                     # reported for context only
    return sum(int((t.children_left == -1).sum()) for t in get_trees(model))

def score(apl, limit=LIMIT):
    return min(1.0, limit / apl)

# ------------------------------------------------------------------ STEP 6: sanity check
def sanity_check():
    X, y = load_iris(return_X_y=True)
    m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    print("=== SANITY CHECK: depth-2 tree on iris ===")
    print(export_text(m, feature_names=load_iris().feature_names))
    print("APL from code :", round(avg_path_length(m, X), 4))
    print("APL by hand   : (50*1 + 100*2) / 150 =", round((50 * 1 + 100 * 2) / 150, 4), "\n")

# ------------------------------------------------------------------ STEP 7: experiments
def run_experiments():
    rows = []
    for dname, (loader, task) in DATASETS.items():
        X, y = loader(return_X_y=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
            for mname, model in make_models(task, seed).items():
                model.fit(Xtr, ytr)
                rows.append(dict(dataset=dname, model=mname, family=mname.split()[0],
                                 apl=avg_path_length(model, Xtr),      # measured on training data
                                 leaves=n_leaves(model),
                                 perf=model.score(Xte, yte)))          # accuracy or R^2 on test data
    df = pd.DataFrame(rows)
    df = df.groupby(["dataset", "model", "family"], sort=False).mean(numeric_only=True).reset_index()
    df["score"] = df.apl.apply(score)
    return df

# ------------------------------------------------------------------ STEP 8: threshold analysis
def threshold_analysis(df):
    print("=== THRESHOLD ANALYSIS ===")
    for d, s in df.groupby("dataset", sort=False):
        print(f"\n[{d}]")
        for fam, f in s.groupby("family", sort=False):
            out = []
            for lim in LIMITS_TO_TEST:
                over = f[f.apl > lim]
                out.append(f"APL>{lim}: {over.model.iloc[0] if len(over) else 'never'}")
            print(f"  {fam:7s} first model exceeding ->  " + " | ".join(out))
        t = s[s.family == "tree"]
        ok = t[t.perf >= t.perf.max() - 0.02].sort_values("apl").iloc[0]
        print(f"  accuracy knee (simplest tree within 0.02 of best tree): {ok.model}, "
              f"APL={ok.apl:.1f}, leaves={ok.leaves:.0f}, perf={ok.perf:.3f} (best {t.perf.max():.3f})")

# ------------------------------------------------------------------ STEP 9: plots
def make_plots(df):
    colors = {"tree": "tab:blue", "forest": "tab:red", "gbt": "tab:green"}
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, (d, s) in zip(ax, df.groupby("dataset", sort=False)):
        for fam, f in s.groupby("family", sort=False):
            a.scatter(f.apl, f.perf, c=colors[fam], label=fam, s=45)
        a.axvline(LIMIT, ls="--", c="gray")
        a.set_xscale("log"); a.set_title(d); a.set_xlabel("average path length (log scale)")
    ax[0].set_ylabel("test accuracy / R²"); ax[0].legend()
    plt.tight_layout(); plt.savefig("fig_apl_tradeoff.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, (d, s) in zip(ax, df.groupby("dataset", sort=False)):
        t = s[s.family == "tree"]; x = list(range(len(t)))
        a.plot(x, t.apl, "o-", c="tab:blue")
        a.axhline(LIMIT, ls="--", c="gray"); a.set_title(d)
        a.set_xticks(x); a.set_xticklabels([m.split("=")[1] for m in t.model])
        a.set_xlabel("tree max depth")
    ax[0].set_ylabel("average path length")
    plt.tight_layout(); plt.savefig("fig_apl_by_depth.png", dpi=130); plt.close()

# ------------------------------------------------------------------ STEP 10: main
if __name__ == "__main__":
    sanity_check()
    df = run_experiments()
    df.to_csv("apl_results.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(df[["dataset", "model", "apl", "leaves", "score", "perf"]].round(2).to_string(index=False))
    threshold_analysis(df)
    make_plots(df)
    print("\nSaved: apl_results.csv, fig_apl_tradeoff.png, fig_apl_by_depth.png")
