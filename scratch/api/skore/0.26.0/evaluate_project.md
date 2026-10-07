# skore evaluation and project API

Source: inspect: skore @ 0.26.0
Probed: 2026-10-07

## evaluate

### Signature

```
(estimator: 'EstimatorLike | list[EstimatorLike] | dict[str, EstimatorLike]', X: 'ArrayLike | None' = None, y: 'ArrayLike | None' = None, data: 'dict | None' = None, *, splitter: "float | int | Literal['prefit'] | SKLearnCrossValidator | Generator | _DefaultType" = <DEFAULT>, pos_label: 'int | float | bool | str | None' = None, n_jobs: 'int | None' = None) -> 'EstimatorReport | CrossValidationReport | ComparisonReport'
```

### help()

```
Python Library Documentation: function evaluate in module skore._dispatch.evaluate

evaluate(estimator: 'EstimatorLike | list[EstimatorLike] | dict[str, EstimatorLike]', X: 'ArrayLike | None' = None, y: 'ArrayLike | None' = None, data: 'dict | None' = None, *, splitter: "float | int | Literal['prefit'] | SKLearnCrossValidator | Generator | _DefaultType" = <DEFAULT>, pos_label: 'int | float | bool | str | None' = None, n_jobs: 'int | None' = None) -> 'EstimatorReport | CrossValidationReport | ComparisonReport'
    Evaluate one or more estimators on the given data.

    Passing several estimators provides a report to compare them, while the
    ``splitter`` parameter controls whether a train-test split or
    cross-validation is used.

    Parameters
    ----------
    estimator : estimator object, list of estimators, or dict of estimators
        The estimator to evaluate of several estimators to compare. An estimator can
        be one of the following:

        - a scikit-learn compatible estimator as a :class:`~sklearn.base.BaseEstimator`;
        - a skrub :class:`~skrub.DataOp` to preprocess the data;
        - a skrub :class:`~skrub.SkrubLearner` extracted from a :class:`~skrub.DataOp`
          by calling :meth:`~skrub.DataOp.skb.make_learner`.

    X : array-like or None
        Feature matrix shared by all estimators when comparing several models.
        When comparing prefit estimators and no test features are needed,
        pass ``X=None``. To compare estimators evaluated on different feature
        matrices, call :func:`~skore.evaluate` once per estimator, then
        :func:`~skore.compare`.

    y : array-like of shape (n_samples,), or None
        Target vector.

    data : dict or None
        When ``estimator`` is a skrub :class:`~skrub.SkrubLearner`, bindings for
        variables contained in the DataOp that was used to create this learner
        (e.g. ``{"X": X_df, "other_table": df, ...}``).

    splitter : float, int, "prefit", or cross-validation object, default=0.2
        Determines how the data is split. When omitted, a skrub learner whose
        DataOp was configured with an explicit cross-validation splitter via
        :meth:`~skrub.DataOp.skb.mark_as_X` uses that splitter (including
        ``split_kwargs`` such as ``groups``). Otherwise, the default is a
        single 80/20 train-test split:

        - ``float``: perform a single train-test split where the data is shuffled before
          splitting with a fixed seed (``random_state=0``) for reproducibility.
          Pass a :class:`~skore.TrainTestSplit` instance for more control over the
          splitting parameters.
        - ``"prefit"``: the estimator is assumed to be already fitted; ``X``
          and ``y`` are used as the test set.
        - ``int``: number of folds for cross-validation (passed to
          :class:`~skore.CrossValidationReport`).
        - cross-validation splitter (e.g. ``KFold``, ``StratifiedKFold``):
          passed directly to :class:`~skore.CrossValidationReport`.

    pos_label : int, float, bool or str, default=None
        The positive class label for binary classification metrics. Forwarded
        to the underlying report.

    n_jobs : int or None, default=None
        Number of jobs for parallel execution. Forwarded to
        :class:`~skore.CrossValidationReport` or
        :class:`~skore.ComparisonReport`.

    Returns
    -------
    report : :class:`~skore.EstimatorReport`, :class:`~skore.CrossValidationReport`             or :class:`~skore.ComparisonReport`
        The report corresponding to the evaluation strategy.

    See Also
    --------
    :func:`~skore.compare` :
        Compare already evaluated reports.
    :class:`~skore.EstimatorReport` :
        Report for a fitted estimator on a test set.
    :class:`~skore.CrossValidationReport` :
        Report for cross-validation of an estimator.
    :class:`~skore.ComparisonReport` :
        Report comparing several evaluated models.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import evaluate
    >>> X, y = make_classification(random_state=42)

    Default 80/20 train-test split:

    >>> report = evaluate(LogisticRegression(), X, y)

    Cross-validation with 5 folds:

    >>> report = evaluate(LogisticRegression(), X, y, splitter=5)

    Evaluate a pre-fitted estimator:

    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> fitted_model = LogisticRegression().fit(X_train, y_train)
    >>> report = evaluate(fitted_model, X_test, y_test, splitter="prefit")

    Compare several named estimators:

    >>> report = evaluate(
    ...     {"m1": LogisticRegression(), "m2": LogisticRegression(C=2.0)},
    ...     X,
    ...     y,
    ...     splitter=0.2,
    ... )
    >>> list(report.reports_)
    ['m1', 'm2']

```

## Project

### Signature

```
(name: 'str', *, mode: 'ProjectMode' = 'local', **kwargs)
```

### help()

```
Python Library Documentation: class Project in module skore._project.project

class Project(builtins.object)
 |  Project(name: 'str', *, mode: 'ProjectMode' = 'local', **kwargs)
 |
 |  API to manage a collection of key-report pairs.
 |
 |  Its constructor initializes a project by creating a new project or by loading an
 |  existing one.
 |
 |  The class main methods are :func:`~skore.Project.put`,
 |  :func:`~skore.Project.summarize`, :func:`~skore.Project.get`, and
 |  :func:`~skore.Project.sync`, respectively to insert a key-report pair into the
 |  project, obtain the metadata/metrics of the inserted reports, get a specific report
 |  by its id, and synchronize reports between projects.
 |
 |  Three mutually exclusive modes are available and can be configured using the
 |  ``mode`` parameter of the constructor:
 |
 |  .. rubric:: Hub mode
 |
 |  The project is configured to communicate with ``skore hub``.
 |
 |  In this mode, ``workspace`` is a ``skore hub`` concept that must be configured on
 |  the ``skore hub`` interface. It represents an isolated entity managing users,
 |  projects, and resources. It can be a company, organization, or team that operates
 |  independently within the system.
 |
 |  Note: Using Project in ``hub`` mode requires an account on ``skore hub``, with
 |  access rights to the specified workspace. Authentication to ``skore hub`` is done by
 |  running ``skore.login()`` before instantiating the Project.
 |
 |  .. rubric:: Local mode
 |
 |  Otherwise, the project is configured to the ``local`` mode to be persisted on
 |  the user machine in a directory called a ``workspace``.
 |
 |  | The workspace can be set using kwargs or the environment variable
 |    ``SKORE_WORKSPACE``.
 |  | If not, it will be set to a default location:
 |
 |      - we search the current working directory and its parents for a
 |        skore workspace: a directory named 'skore' containing a file named
 |        '.SKORE_WORKSPACE'. If found, use that.
 |      - otherwise if we are in a Git repository, create 'skore' at the root
 |          of the repository.
 |      - otherwise create 'skore' in the current working directory.
 |
 |  .. rubric:: MLflow mode
 |
 |  In this mode, ``name`` is used as the MLflow experiment name. Reports are persisted
 |  as MLflow model artifacts in runs created under this experiment.
 |
 |  Refer to the :ref:`project` section of the user guide for more details.
 |
 |  Parameters
 |  ----------
 |  name : str
 |      The name of the project.
 |  mode : {"hub", "local", "mlflow"}
 |      The mode of the project.
 |  **kwargs : dict
 |      Extra keyword arguments passed to the project, depending on its mode.
 |
 |      workspace : str or Path-like, optional
 |
 |          - If ``mode="hub"``, the Hub workspace name (required);
 |          - If ``mode="local"``, the local persistence directory (optional);
 |          - If ``mode="mlflow"``, ignored.
 |
 |      tracking_uri : str, mode:mlflow only.
 |          The URI of the MLflow tracking server.
 |
 |  Attributes
 |  ----------
 |  name : str
 |      The name of the project.
 |  mode : {"hub", "local", "mlflow"}
 |      The mode of the project.
 |  workspace : Path or str or None
 |      The workspace for ``local`` (``Path``) and ``hub`` (``str``) modes; ``None``
 |      otherwise.
 |  tracking_uri : str or None
 |      The MLflow tracking URI for ``mlflow`` mode; ``None`` otherwise.
 |  ml_task : MLTask
 |      The ML task of the project; unset until a first report is put.
 |
 |  Examples
 |  --------
 |  Construct reports.
 |
 |  >>> from sklearn.datasets import make_regression
 |  >>> from sklearn.linear_model import LinearRegression
 |  >>> from skore import evaluate
 |  >>>
 |  >>> X, y = make_regression(random_state=42)
 |  >>> regressor = LinearRegression()
 |  >>> regressor_report = evaluate(regressor, X, y, splitter=0.2)
 |
 |  Construct the project in local mode, persisted in a temporary directory.
 |
 |  >>> from pathlib import Path
 |  >>> from tempfile import TemporaryDirectory
 |  >>> from skore import Project
 |  >>>
 |  >>> tmpdir = TemporaryDirectory().name
 |  >>> local_project = Project(name="my-xp", mode="local", workspace=Path(tmpdir))
 |
 |  Put reports in the project.
 |
 |  >>> local_project.put("my-simple-regression", regressor_report)
 |
 |  Investigate metadata/metrics to filter the best reports.
 |
 |  >>> summary = local_project.summarize()
 |  >>> summary = summary.query("rmse < 67")
 |  >>> reports = summary.compare()
 |
 |  See Also
 |  --------
 |  :class:`~skore.Summary` :
 |      Tabular view of metadata and metrics for persisted reports.
 |  :func:`~skore.compare` :
 |      Compare reports side by side.
 |  :meth:`Project.summarize` :
 |      Create a summary view to investigate persisted reports' metadata/metrics.
 |
 |  Methods defined here:
 |
 |  __init__(self, name: 'str', *, mode: 'ProjectMode' = 'local', **kwargs)
 |      Initialize a project.
 |
 |      Parameters
 |      ----------
 |      name : str
 |          The name of the project.
 |          For mode:mlflow, this name will be used as the experiment name.
 |      mode : {"hub", "local", "mlflow"}, default "local"
 |          The mode of the project.
 |      **kwargs : dict
 |          Extra keyword arguments passed to the project, depending on its mode.
 |
 |          workspace : str or Path-like, optional
 |              Hub workspace name when ``mode="hub"`` (required). Local persistence
 |              directory when ``mode="local"`` (optional). Ignored when
 |              ``mode="mlflow"``.
 |
 |              For ``mode="local"``:
 |
 |              | The workspace can be shared between all the projects.
 |              | The workspace can be set using kwargs or the environment variable
 |                ``SKORE_WORKSPACE``.
 |              | If not provided, a workspace is found or created according to those
 |                rules (see find_workspace() in this module):
 |
 |                    - if the SKORE_WORKSPACE environment variable is set, use that.
 |                    - otherwise look in the current working directory and its
 |                      parent for a skore workspace: a directory named
 |                      'skore' containing a file named
 |                      '.SKORE_WORKSPACE'. If found, use that.
 |                    - otherwise if we are in a Git repository, create
 |                      'skore' at the root of the repository.
 |                    - otherwise create 'skore' in the current working directory.
 |
 |          tracking_uri : str, mode:mlflow only.
 |              The URI of the MLflow tracking server.
 |
 |      Examples
 |      --------
 |      >>> from pathlib import Path
 |      >>> from tempfile import TemporaryDirectory
 |      >>> from skore import Project
 |      >>> tmpdir = TemporaryDirectory()
 |      >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
 |      >>> project.name
 |      'my-xp'
 |      >>> project.mode
 |      'local'
 |      >>> tmpdir.cleanup()
 |
 |  __repr__(self) -> 'str'
 |      Return repr(self).
 |
 |  get(self, id: 'str') -> 'EstimatorReport | CrossValidationReport'
 |      Get a persisted report by its id.
 |
 |      Report IDs can be found via :meth:`skore.Project.summarize`, which is also the
 |      preferred method of interacting with a ``skore.Project``. The ``id`` passed here
 |      must match the ``id`` column returned by :meth:`Project.summarize`.
 |
 |      Parameters
 |      ----------
 |      id : str
 |          The id of a report already put in the ``project``.
 |
 |      Returns
 |      -------
 |      report : EstimatorReport or CrossValidationReport
 |          The report associated with ``id``.
 |
 |      Examples
 |      --------
 |      >>> from sklearn.datasets import make_regression
 |      >>> from sklearn.linear_model import LinearRegression
 |      >>> from pathlib import Path
 |      >>> from tempfile import TemporaryDirectory
 |      >>> from skore import Project, evaluate
 |      >>> X, y = make_regression(random_state=42)
 |      >>> report = evaluate(LinearRegression(), X, y, splitter=0.2)
 |      >>> tmpdir = TemporaryDirectory()
 |      >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
 |      >>> project.put("my-regression", report)
 |      >>> summary = project.summarize()
 |      >>> report_id = summary.frame().index.get_level_values("id")[0]
 |      >>> retrieved = project.get(report_id)
 |      >>> type(retrieved).__name__
 |      'EstimatorReport'
 |      >>> tmpdir.cleanup()
 |
 |  put(self, key: 'str', report: 'EstimatorReport | CrossValidationReport')
 |      Put a key-report pair to the project.
 |
 |      If the key already exists, its last report is modified to point to this new
 |      report, while keeping track of the report history.
 |
 |      Parameters
 |      ----------
 |      key : str
 |          The key to associate with ``report`` in the project.
 |          Name of the run for mode:mlflow
 |      report : EstimatorReport | CrossValidationReport
 |          The report to associate with ``key`` in the project.
 |
 |      Returns
 |      -------
 |      None
 |          The report is persisted in the project backend.
 |
 |      Examples
 |      --------
 |      >>> from sklearn.datasets import make_regression
 |      >>> from sklearn.linear_model import LinearRegression
 |      >>> from pathlib import Path
 |      >>> from tempfile import TemporaryDirectory
 |      >>> from skore import Project, evaluate
 |      >>> X, y = make_regression(random_state=42)
 |      >>> report = evaluate(LinearRegression(), X, y, splitter=0.2)
 |      >>> tmpdir = TemporaryDirectory()
 |      >>> project = Project(name="my-xp", mode="local", workspace=Path(tmpdir.name))
 |      >>> project.put("my-regression", report)
 |      >>> tmpdir.cleanup()
 |
 |  summarize(self) -> 'Summary'
 |      Obtain metadata/metrics for all persisted reports.
 |
 |      Reports are returned in ascending order of their ``date`` field.
 |
 |      Returns
 |      -------
 |      summary : Summary
 |          Metadata and metrics for every report persisted in the project.
 |
 |      See Also
 |      --------
 |      :class:`~skore.Summary` :
 |          Tabular view with interactive filtering in Jupyter.
 |      :func:`~skore.compare` :
 |          Compare selected reports side by side.
 |
 |  sync(self, other: 'Project | ProjectMode', *, bidirectional: 'bool' = False, dry_run: 'bool' = False, **kwargs: 'Any') -> 'DataFrame'
 |      Copy missing reports to another project.
 |
 |      Reports are matched using the ``report_id`` column returned by
 |      ``Project.summarize().frame()``. Set ``bidirectional=True`` to copy missing
 |      reports in both directions.
 |
 |      Parameters
 |      ----------
 |      other : Project or {"hub", "local", "mlflow"}
 |          Destination project. When a mode is given, build the destination with this
 |          project's name and the mode-specific keyword arguments.
 |      bidirectional : bool, default=False
 |          If ``False``, transfer reports from this project to ``other``. If ``True``,
 |          also transfer reports missing from this project.
 |      dry_run : bool, default=False
 |          Return the planned operations without loading or storing reports.
 |      **kwargs : dict
 |          Mode-specific arguments used to build the destination when ``other`` is a
 |          mode string. For example, pass ``workspace`` for ``"hub"`` or
 |          ``tracking_uri`` for ``"mlflow"``.
 |
 |      Returns
 |      -------
 |      result : pandas.DataFrame
 |          Synchronization status indexed by ``report_id``. The ``direction`` column
 |          is ``"outbound"`` from this project to ``other``, ``"inbound"`` from
 |          ``other`` to this project, or missing for skipped reports. The ``status``
 |          column is ``"planned"``, ``"transferred"``, or ``"skipped"``.
 |
 |  ----------------------------------------------------------------------
 |  Static methods defined here:
 |
 |  delete(name: 'str', *, mode: 'ProjectMode' = 'local', **kwargs)
 |      Delete a project.
 |
 |      Parameters
 |      ----------
 |      name : str
 |          The name of the project.
 |      mode : {"hub", "local", "mlflow"}, default "local"
 |          The mode of the project.
 |      **kwargs : dict
 |          Extra keyword arguments passed to the project, depending on its mode.
 |
 |          workspace : str or Path-like, optional
 |              See the :class:`Project` class docstring for details.
 |
 |          tracking_uri : str, mode:mlflow only.
 |              The URI of the MLflow tracking server.
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |
 |  mode
 |      The mode of the project.
 |
 |  name
 |      The name of the project.
 |
 |  tracking_uri
 |      The MLflow tracking URI for mlflow mode; ``None`` otherwise.
 |
 |  workspace
 |      The workspace for local and hub modes; ``None`` otherwise.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object

```

## login

### Signature

```
(*, mode: Literal['hub', 'local', 'mlflow'] = 'hub', **kwargs)
```

### help()

```
Python Library Documentation: function login in module skore._project.login

login(*, mode: Literal['hub', 'local', 'mlflow'] = 'hub', **kwargs)
    Log in to Skore Hub for the duration of the session (e.g. script).

    This command is only useful if you have an account on Skore Hub and wish
    to push artifacts to it.

    By default, it will open a login screen on your browser. However, this login only
    persists for the lifetime of the Python process (e.g. one run of a script, or one
    Jupyter session), so you will have to authenticate via your browser at every run.
    The recommended way to connect to Skore Hub for repeated script runs is using an
    API key; refer to the Skore Hub documentation for how to create one.

    Parameters
    ----------
    mode : {"hub", "local", "mlflow"}, default="hub"
        The mode of the storage backend to log in. If the mode is not "hub", the
        function is a no-op.

    **kwargs : dict
        Extra keyword arguments passed to the login function, depending on its mode.

        Arguments for ``mode="hub"``:

        timeout : int, default=600
            The time, in seconds, before raising an error if communication with
            Skore Hub fails.

    Returns
    -------
    None
        For ``mode="local"`` and ``mode="mlflow"``. For ``mode="hub"``, the return
        value depends on the hub login plugin.

    Examples
    --------
    >>> from skore import login
    >>> login(mode="local")

    See Also
    --------
    :class:`~skore.Project` :
        Refer to the :ref:`project` section of the user guide for more details.

```

