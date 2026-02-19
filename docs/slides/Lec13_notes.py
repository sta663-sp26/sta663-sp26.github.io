## Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sklearn.set_config(display="text")


## Digits

digits = load_digits(as_frame=True)

## Classification Tree

X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, shuffle=True, random_state=1234
)

digits_tree = GridSearchCV(
  DecisionTreeClassifier(),
  param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": range(2,16)
  },
  cv = KFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(
  X_train, y_train
)

digits_tree.best_estimator_
digits_tree.best_score_

accuracy_score(y_test, digits_tree.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, digits_tree.best_estimator_.predict(X_test)
)


## Trees vs Forests

pipe = Pipeline([("clf", DecisionTreeClassifier())])

param_grid = [
  {
    "clf": [DecisionTreeClassifier()],
    "clf__criterion": ["gini", "entropy"],
    "clf__max_depth": range(2, 16)
  },
  {
    "clf": [RandomForestClassifier()],
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 5, 10]
  }
]

trees_vs_forests = GridSearchCV(
  pipe,
  param_grid,
  cv = StratifiedKFold(5, shuffle=True, random_state=12345),
  n_jobs = 4
).fit(X_train, y_train)

trees_vs_forests.best_estimator_
trees_vs_forests.best_params_
trees_vs_forests.best_score_

accuracy_score(y_test, trees_vs_forests.best_estimator_.predict(X_test))
confusion_matrix(
  y_test, trees_vs_forests.best_estimator_.predict(X_test)
)
