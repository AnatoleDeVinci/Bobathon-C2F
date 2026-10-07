# skore project.put and CrossValidationReport

Source: inspect: skore @ 0.26.0
Probed: 2026-10-07

## Project.put

### Signature

```
(self, key: 'str', report: 'EstimatorReport | CrossValidationReport')
```

### help() (first 600 chars)

```
Python Library Documentation: function put in module skore._project.project

put(self, key: 'str', report: 'EstimatorReport | CrossValidationReport')
    Put a key-report pair to the project.

    If the key already exists, its last report is modified to point to this new
    report, while keeping track of the report history.

    Parameters
    ----------
    key : str
        The key to associate with ``report`` in the project.
        Name of the run for mode:mlflow
    report : EstimatorReport | CrossValidationReport
        The report to associate with ``key`` in the project.

    Returns
```

## CrossValidationReport

### Signature

```
(estimator: 'EstimatorLike', X: 'ArrayLike | None' = None, y: 'ArrayLike | None' = None, data: 'dict | None' = None, pos_label: 'PositiveLabel | None' = None, splitter: 'int | SKLearnCrossValidator | Generator | None' = None, n_jobs: 'int | None' = None) -> 'None'
```

### help() (first 600 chars)

```
Python Library Documentation: class CrossValidationReport in module skore._reports.cross_validation.report

class CrossValidationReport(skore._reports.base._BaseReport, skore._externals.pandas_accessors.DirNamesMixin)
 |  CrossValidationReport(estimator: 'EstimatorLike', X: 'ArrayLike | None' = None, y: 'ArrayLike | None' = None, data: 'dict | None' = None, pos_label: 'PositiveLabel | None' = None, splitter: 'int | SKLearnCrossValidator | Generator | None' = None, n_jobs: 'int | None' = None) -> 'None'
 |
 |  Provide a report of cross-validation results.
 |
 |  Upon initialization, clones ``es
```

