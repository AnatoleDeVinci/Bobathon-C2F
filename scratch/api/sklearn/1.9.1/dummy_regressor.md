# DummyRegressor

Source: inspect: sklearn.dummy.DummyRegressor @ 1.9.1
Probed: 2026-10-07

## Signature

```
(*, strategy='mean', constant=None, quantile=None)
```

## help()

```
Python Library Documentation: class DummyRegressor in module sklearn.dummy

class DummyRegressor(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, sklearn.base.BaseEstimator)
 |  DummyRegressor(*, strategy='mean', constant=None, quantile=None)
 |
 |  Regressor that makes predictions using simple rules.
 |
 |  This regressor is useful as a simple baseline to compare with other
 |  (real) regressors. Do not use it for real problems.
 |
 |  Read more in the :ref:`User Guide <dummy_estimators>`.
 |
 |  .. versionadded:: 0.13
 |
 |  Parameters
 |  ----------
 |  strategy : {"mean", "median", "quantile", "constant"}, default="mean"
 |      Strategy to use to generate predictions.
 |
 |      * "mean": always predicts the mean of the training set
 |      * "median": always predicts the median of the training set
 |      * "quantile": always predicts a specified quantile of the training set,
 |        provided with the quantile parameter.
 |      * "constant": always predicts a constant value that is provided by
 |        the user.
 |
 |  constant : int or float or array-like of shape (n_outputs,), default=None
 |      The explicit constant as predicted by the "constant" strategy. This
 |      parameter is useful only for the "constant" strategy.
 |
 |  quantile : float in [0.0, 1.0], default=None
 |      The quantile to predict using the "quantile" strategy. A quantile of
 |      0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
 |      maximum.
 |
 |  Attributes
 |  ----------
 |  constant_ : ndarray of shape (1, n_outputs)
 |      Mean or median or quantile of the training targets or constant value
 |      given by the user.
 |
 |  n_features_in_ : int
 |      Number of features seen during :term:`fit`.
 |
 |  feature_names_in_ : ndarray of shape (`n_features_in_`,)
 |      Names of features seen during :term:`fit`. Defined only when `X` has
 |      feature names that are all strings.
 |
 |  n_outputs_ : int
 |      Number of outputs.
 |
 |  See Also
 |  --------
 |  DummyClassifier: Classifier that makes predictions using simple rules.
 |
 |  Examples
 |  --------
 |  >>> import numpy as np
 |  >>> from sklearn.dummy import DummyRegressor
 |  >>> X = np.array([1.0, 2.0, 3.0, 4.0])
 |  >>> y = np.array([2.0, 3.0, 5.0, 10.0])
 |  >>> dummy_regr = DummyRegressor(strategy="mean")
 |  >>> dummy_regr.fit(X, y)
 |  DummyRegressor()
 |  >>> dummy_regr.predict(X)
 |  array([5., 5., 5., 5.])
 |  >>> dummy_regr.score(X, y)
 |  0.0
 |
 |  Method resolution order:
 |      DummyRegressor
 |      sklearn.base.MultiOutputMixin
 |      sklearn.base.RegressorMixin
 |      sklearn.base.BaseEstimator
 |      sklearn.utils._repr_html.base.ReprHTMLMixin
 |      sklearn.utils._repr_html.base._HTMLDocumentationLinkMixin
 |      sklearn.utils._metadata_requests._MetadataRequester
 |      builtins.object
 |
 |  Methods defined here:
 |
 |  __init__(self, *, strategy='mean', constant=None, quantile=None)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |
 |  __sklearn_tags__(self)
 |
 |  fit(self, X, y, sample_weight=None)
 |      Fit the baseline regressor.
 |
 |      Parameters
 |      ----------
 |      X : array-like of shape (n_samples, n_features)
 |          Training data.
 |
 |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          Target values.
 |
 |      sample_weight : array-like of shape (n_samples,), default=None
 |          Sample weights.
 |
 |      Returns
 |      -------
 |      self : object
 |          Fitted estimator.
 |
 |  predict(self, X, return_std=False)
 |      Perform classification on test vectors X.
 |
 |      Parameters
 |      ----------
 |      X : array-like of shape (n_samples, n_features)
 |          Test data.
 |
 |      return_std : bool, default=False
 |          Whether to return the standard deviation of posterior prediction.
 |          All zeros in this case.
 |
 |          .. versionadded:: 0.20
 |
 |      Returns
 |      -------
 |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          Predicted target values for X.
 |
 |      y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          Standard deviation of predictive distribution of query points.
 |
 |  score(self, X, y, sample_weight=None)
 |      Return the coefficient of determination R^2 of the prediction.
 |
 |      The coefficient R^2 is defined as `(1 - u/v)`, where `u` is the
 |      residual sum of squares `((y_true - y_pred) ** 2).sum()` and `v` is the
 |      total sum of squares `((y_true - y_true.mean()) ** 2).sum()`. The best
 |      possible score is 1.0 and it can be negative (because the model can be
 |      arbitrarily worse). A constant model that always predicts the expected
 |      value of y, disregarding the input features, would get a R^2 score of
 |      0.0.
 |
 |      Parameters
 |      ----------
 |      X : None or array-like of shape (n_samples, n_features)
 |          Test samples. Passing None as test samples gives the same result
 |          as passing real test samples, since `DummyRegressor`
 |          operates independently of the sampled observations.
 |
 |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
 |          True values for X.
 |
 |      sample_weight : array-like of shape (n_samples,), default=None
 |          Sample weights.
 |
 |      Returns
 |      -------
 |      score : float
 |          R^2 of `self.predict(X)` w.r.t. y.
 |
 |  set_fit_request(self: sklearn.dummy.DummyRegressor, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.dummy.DummyRegressor from sklearn.utils._metadata_requests.RequestMethod.__get__.<locals>
 |      Configure whether metadata should be requested to be passed to the ``fit`` method.
 |
 |      Note that this method is only relevant when this estimator is used as a
 |      sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
 |      with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
 |      Please check the :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |
 |      The options for each parameter are:
 |
 |      - ``True``: metadata is requested, and passed to ``fit`` if provided. The request is ignored if metadata is not provided.
 |
 |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``fit``.
 |
 |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
 |
 |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
 |
 |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
 |      existing request. This allows you to change the request for some
 |      parameters and not others.
 |
 |      .. versionadded:: 1.3
 |
 |      Parameters
 |      ----------
 |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
 |          Metadata routing for ``sample_weight`` parameter in ``fit``.
 |
 |      Returns
 |      -------
 |      self : object
 |          The updated object.
 |
 |  set_predict_request(self: sklearn.dummy.DummyRegressor, *, return_std: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.dummy.DummyRegressor from sklearn.utils._metadata_requests.RequestMethod.__get__.<locals>
 |      Configure whether metadata should be requested to be passed to the ``predict`` method.
 |
 |      Note that this method is only relevant when this estimator is used as a
 |      sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
 |      with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
 |      Please check the :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |
 |      The options for each parameter are:
 |
 |      - ``True``: metadata is requested, and passed to ``predict`` if provided. The request is ignored if metadata is not provided.
 |
 |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``predict``.
 |
 |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
 |
 |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
 |
 |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
 |      existing request. This allows you to change the request for some
 |      parameters and not others.
 |
 |      .. versionadded:: 1.3
 |
 |      Parameters
 |      ----------
 |      return_std : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
 |          Metadata routing for ``return_std`` parameter in ``predict``.
 |
 |      Returns
 |      -------
 |      self : object
 |          The updated object.
 |
 |  set_score_request(self: sklearn.dummy.DummyRegressor, *, sample_weight: Union[bool, NoneType, str] = '$UNCHANGED$') -> sklearn.dummy.DummyRegressor from sklearn.utils._metadata_requests.RequestMethod.__get__.<locals>
 |      Configure whether metadata should be requested to be passed to the ``score`` method.
 |
 |      Note that this method is only relevant when this estimator is used as a
 |      sub-estimator within a :term:`meta-estimator` and metadata routing is enabled
 |      with ``enable_metadata_routing=True`` (see :func:`sklearn.set_config`).
 |      Please check the :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |
 |      The options for each parameter are:
 |
 |      - ``True``: metadata is requested, and passed to ``score`` if provided. The request is ignored if metadata is not provided.
 |
 |      - ``False``: metadata is not requested and the meta-estimator will not pass it to ``score``.
 |
 |      - ``None``: metadata is not requested, and the meta-estimator will raise an error if the user provides it.
 |
 |      - ``str``: metadata should be passed to the meta-estimator with this given alias instead of the original name.
 |
 |      The default (``sklearn.utils.metadata_routing.UNCHANGED``) retains the
 |      existing request. This allows you to change the request for some
 |      parameters and not others.
 |
 |      .. versionadded:: 1.3
 |
 |      Parameters
 |      ----------
 |      sample_weight : str, True, False, or None,                     default=sklearn.utils.metadata_routing.UNCHANGED
 |          Metadata routing for ``sample_weight`` parameter in ``score``.
 |
 |      Returns
 |      -------
 |      self : object
 |          The updated object.
 |
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |
 |  __annotations__ = {'_parameter_constraints': <class 'dict'>}
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from sklearn.base.MultiOutputMixin:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.base.BaseEstimator:
 |
 |  __dir__(self)
 |      Default dir() implementation.
 |
 |  __getstate__(self)
 |      Helper for pickle.
 |
 |  __repr__(self, N_CHAR_MAX=700)
 |      Return repr(self).
 |
 |  __setstate__(self, state)
 |
 |  __sklearn_clone__(self)
 |
 |  get_params(self, deep=True)
 |      Get parameters for this estimator.
 |
 |      Parameters
 |      ----------
 |      deep : bool, default=True
 |          If True, will return the parameters for this estimator and
 |          contained subobjects that are estimators.
 |
 |      Returns
 |      -------
 |      params : dict
 |          Parameter names mapped to their values.
 |
 |  set_params(self, **params)
 |      Set the parameters of this estimator.
 |
 |      The method works on simple estimators as well as on nested objects
 |      (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
 |      parameters of the form ``<component>__<parameter>`` so that it's
 |      possible to update each component of a nested object.
 |
 |      Parameters
 |      ----------
 |      **params : dict
 |          Estimator parameters.
 |
 |      Returns
 |      -------
 |      self : estimator instance
 |          Estimator instance.
 |
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
 |
 |  get_metadata_routing(self)
 |      Get metadata routing of this object.
 |
 |      Please check :ref:`User Guide <metadata_routing>` on how the routing
 |      mechanism works.
 |
 |      Returns
 |      -------
 |      routing : MetadataRequest
 |          A :class:`~sklearn.utils.metadata_routing.MetadataRequest` encapsulating
 |          routing information.
 |
 |  ----------------------------------------------------------------------
 |  Class methods inherited from sklearn.utils._metadata_requests._MetadataRequester:
 |
 |  __init_subclass__(**kwargs)
 |      Set the ``set_{method}_request`` methods.
 |
 |      This uses PEP-487 [1]_ to set the ``set_{method}_request`` methods. It
 |      looks for the information available in the set default values which are
 |      set using ``__metadata_request__*`` class attributes, or inferred
 |      from method signatures.
 |
 |      The ``__metadata_request__*`` class attributes are used when a method
 |      does not explicitly accept a metadata through its arguments or if the
 |      developer would like to specify a request value for those metadata
 |      which are different from the default ``None``.
 |
 |      References
 |      ----------
 |      .. [1] https://www.python.org/dev/peps/pep-0487

```

