# sklearn linear pipeline symbols

Source: inspect: sklearn @ 1.9.1
Probed: 2026-10-07

## Ridge

### Signature

```
(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
```

### help() (truncated to 800 chars)

```
Python Library Documentation: class Ridge in module sklearn.linear_model._ridge

class Ridge(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, _BaseRidge)
 |  Ridge(alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None)
 |
 |  Linear least squares with l2 regularization.
 |
 |  Minimizes the objective function::
 |
 |  ||y - Xw||^2_2 + alpha * ||w||^2_2
 |
 |  This model solves a regression model where the loss function is
 |  the linear least squares function and regularization is given by
 |  the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
 |  This estimator has built-in support for multi-variate regression
 |  (i.e., when y is a 2d-array of shape (n_samples, n_targets)).
 |
 |  Read mo
```

## SimpleImputer

### Signature

```
(*, missing_values=nan, strategy='mean', fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)
```

### help() (truncated to 800 chars)

```
Python Library Documentation: class SimpleImputer in module sklearn.impute._base

class SimpleImputer(_BaseImputer)
 |  SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)
 |
 |  Univariate imputer for completing missing values with simple strategies.
 |
 |  Replace missing values using a descriptive statistic (e.g. mean, median, or
 |  most frequent) along each column, or using a constant value.
 |
 |  Read more in the :ref:`User Guide <impute>`.
 |
 |  .. versionadded:: 0.20
 |     `SimpleImputer` replaces the previous `sklearn.preprocessing.Imputer`
 |     estimator which is now removed.
 |
 |  Parameters
 |  ----------
 |  missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
 |      The
```

## make_pipeline

### Signature

```
(*steps, memory=None, transform_input=None, verbose=False)
```

### help() (truncated to 800 chars)

```
Python Library Documentation: function make_pipeline in module sklearn.pipeline

make_pipeline(*steps, memory=None, transform_input=None, verbose=False)
    Construct a :class:`Pipeline` from the given estimators.

    This is a shorthand for the :class:`Pipeline` constructor; it does not
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list of Estimator objects
        List of the scikit-learn estimators that are chained together.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
  
```

