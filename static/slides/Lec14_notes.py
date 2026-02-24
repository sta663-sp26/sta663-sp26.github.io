import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class interact_features(TransformerMixin, BaseEstimator):
  
  def __init__(self, interaction_only = False, include_intercept = False):
    self.interaction_only = interaction_only
    self.include_intercept = include_intercept

  def fit(self, X, y=None):
    # validate_data checks shape/type, sets n_features_in_, and sets feature_names_in_ (if X has named columns)
    validate_data(self, X=X, reset=True, ensure_min_features=2)

    return self

  def transform(self, X, y=None):
    # Ensures fit() has been called before transform()
    check_is_fitted(self, "n_features_in_")

    # Validates shape and feature names match what was seen during fit (reset=False)
    validate_data(self, X=X, reset=False)

    X = np.array(X)

    # Generate all unique pairwise products (i < j avoids duplicates and self-interactions)
    new_cols = []
    for i in range(self.n_features_in_-1):
      for j in range(i+1, self.n_features_in_):
        new_cols.append( X[:,i] * X[:,j] )

    new_X = np.column_stack(new_cols)

    # Prepend original features unless only interaction terms are requested
    if not self.interaction_only:
      new_X = np.column_stack([X, new_X])

    # Prepend a column of ones as an intercept term
    if self.include_intercept:
      new_X = np.column_stack([np.ones((new_X.shape[0],1)), new_X])

    return new_X

  def get_feature_names_out(self):
    check_is_fitted(self, "n_features_in_")

    # Fall back to generic x0, x1, ... names when fitted on a plain numpy array
    if not hasattr(self, "feature_names_in_"):
      feat_names = ["x"+str(i) for i in range(self.n_features_in_)]
    else:
      feat_names = self.feature_names_in_

    # Mirror the same i < j loop used in transform()
    new_feat_names = []
    for i in range(self.n_features_in_-1):
      for j in range(i+1, self.n_features_in_):
        new_feat_names.append( feat_names[i] + " * " + feat_names[j] )

    if not self.interaction_only:
      new_feat_names = np.concatenate((feat_names, new_feat_names), axis=0)

    if self.include_intercept:
      new_feat_names = ["1"] + list(new_feat_names)

    return new_feat_names


# DataFrame fit: feature_names_in_ will be set from column names
X = pd.DataFrame({"x1": range(1,6), "x2": range(5, 0, -1)})
Y = pd.DataFrame({"x1": range(1,6)})
Z = np.array(X)

itf = interact_features().fit(X)
itf.n_features_in_
itf.feature_names_in_
itf.transform(X)
itf.get_feature_names_out()

# numpy array fit: feature_names_in_ is NOT set, generic names are used instead
itf2 = interact_features().fit(Z)
itf2.n_features_in_
hasattr(itf2, "feature_names_in_")  # False - no named columns in numpy array
itf2.transform(Z)
itf2.get_feature_names_out()


from sklearn.utils.estimator_checks import check_estimator
check_estimator(interact_features())
