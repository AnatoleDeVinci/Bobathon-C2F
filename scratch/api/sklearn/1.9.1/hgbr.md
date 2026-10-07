# HistGradientBoostingRegressor

Source: inspect: sklearn.ensemble.HistGradientBoostingRegressor @ 1.9.1
Probed: 2026-10-07

## Signature

```
(loss='squared_error', *, quantile=None, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
```

## help() (first 1200 chars)

```
Python Library Documentation: class HistGradientBoostingRegressor in module sklearn.ensemble._hist_gradient_boosting.gradient_boosting

class HistGradientBoostingRegressor(sklearn.base.RegressorMixin, BaseHistGradientBoosting)
 |  HistGradientBoostingRegressor(loss='squared_error', *, quantile=None, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, max_bins=255, categorical_features='from_dtype', monotonic_cst=None, interaction_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
 |
 |  Histogram-based Gradient Boosting Regression Tree.
 |
 |  This estimator is much faster than
 |  :class:`GradientBoostingRegressor<sklearn.ensemble.GradientBoostingRegressor>`
 |  for big datasets (n_samples >= 10 000).
 |
 |  This estimator has native support for missing values (NaNs). During
 |  training, the tree grower learns at each split point whether samples
 |  with missing values should go to the left or right child, based on the
 |  potential gain. When predicting, samples with missing values are
 |  assigned to
```

## Usage

- **Call:** `HistGradientBoostingRegressor(random_state=0)` — handles NaN natively, no imputation needed
- **Don't call:** with `SimpleImputer` before it — that hides the missingness signal
- **Trap:** string columns must be converted to `category` dtype before passing; pass numeric columns directly
- **Returns:** fitted estimator with `.predict(X)` → array of float
