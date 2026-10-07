# GroupKFold

Source: inspect: sklearn.model_selection.GroupKFold @ 1.9.1
Probed: 2026-10-07

## Signature

```
(n_splits=5, *, shuffle=False, random_state=None)
```

## help() (first 1000 chars)

```
Python Library Documentation: class GroupKFold in module sklearn.model_selection._split

class GroupKFold(GroupsConsumerMixin, _BaseKFold)
 |  GroupKFold(n_splits=5, *, shuffle=False, random_state=None)
 |
 |  K-fold iterator variant with non-overlapping groups.
 |
 |  Each group will appear exactly once in the test set across all folds (the
 |  number of distinct groups has to be at least equal to the number of folds).
 |
 |  The folds are approximately balanced in the sense that the number of
 |  samples is approximately the same in each test fold when `shuffle` is True.
 |
 |  Read more in the :ref:`User Guide <group_k_fold>`.
 |
 |  For visualisation of cross-validation behaviour and
 |  comparison between common scikit-learn split methods
 |  refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`
 |
 |  Parameters
 |  ----------
 |  n_splits : int, default=5
 |      Number of folds. Must be at least 2.
 |
 |      .. versionchanged:: 0.22
 |          ``n_splits``
```

## Usage

- **Call:** `GroupKFold(n_splits=5).split(X, y, groups=patient_id_series)`
- **Don't call:** `GroupKFold().split(X, y)` without groups= — raises
- **Trap:** skore's sklearn path calls `splitter.split(X, y)` without groups, so precompute: `cv_splits = list(GroupKFold(5).split(X, y, groups=groups))`
- **Returns:** generator of (train_idx, test_idx) arrays
