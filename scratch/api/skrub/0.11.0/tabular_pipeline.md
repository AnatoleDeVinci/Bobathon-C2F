# tabular_pipeline

Source: inspect: skrub.tabular_pipeline @ 0.11.0
Probed: 2026-10-07

## Signature

```
(estimator, *, n_jobs=None)
```

## help()

```
Python Library Documentation: function tabular_pipeline in module skrub._tabular_pipeline

tabular_pipeline(estimator, *, n_jobs=None)
    Get a simple machine-learning pipeline for tabular data.

    Given either a scikit-learn compatible estimator or one of the special-cased strings
    ``'regressor'``, ``'regression'``, ``'classifier'``, ``'classification'``, this
    function creates a scikit-learn pipeline that extracts numeric features, imputes
    missing values and scales the data if necessary, then applies the estimator.

    .. note::
       The heuristics used by the ``tabular_pipeline``
       to define an appropriate preprocessing based on the ``estimator`` may change
       in future releases.

    .. versionchanged:: 0.6.0
        The high cardinality encoder has been changed from
        :class:`~skrub.MinHashEncoder` to :class:`~skrub.StringEncoder`.

    .. versionchanged:: 0.7.0
        The :class:`~skrub.SquashingScaler` with `max_absolute_value=5` is now used instead of
        :class:`~sklearn.preprocessing.StandardScaler` for centering and scaling
        numerical features when using linear models.

    Parameters
    ----------
    estimator : {"regressor", "regression", "classifier", "classification"} or scikit-learn
        compatible estimator or scikit-learn pipeline

        The estimator to use as the final step in the pipeline. Based on the type of
        estimator, the previous preprocessing steps and their respective parameters are
        chosen. The possible values are:

        - ``'regressor'`` or ``'regression'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingRegressor` is used as the final
          step;
        - ``'classifier'`` or ``'classification'``: a
          :obj:`~sklearn.ensemble.HistGradientBoostingClassifier` is used as the final
          step;
        - a scikit-learn estimator: the provided estimator is used as the final step.
        - a scikit-learn pipeline : if given a pipeline the steps are ext
```

## Usage

- **Call:** `skrub.tabular_pipeline('regressor')` returns a sklearn-compatible Pipeline
- **Don't call:** `skrub.tabular_learner` — renamed to `tabular_pipeline` in 0.7+
- **Trap:** pass a DataFrame (not a numpy array) — TableVectorizer needs column names
- **Returns:** sklearn Pipeline with TableVectorizer + HistGradientBoostingRegressor
