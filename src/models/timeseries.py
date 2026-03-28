import pandas as pd
import pymc as pm
import numpy as np
from pymc_extras.model_builder import ModelBuilder
import arviz as az


def train_test_split_time_series(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float | int = 0.2,
    sort_by_index: bool = True,
):
    """Split timeseries data into chronological train/test partitions.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix, in chronological order or with a time index.
    y : Series or ndarray
        Target vector, aligned with X.
    test_size : float or int, default 0.2
        If float, fraction of samples to reserve for test. If int, number of test samples.
    sort_by_index : bool, default True
        If True and X is DataFrame or Series with datetime index, sort by index before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if isinstance(X, (pd.DataFrame, pd.Series)) and sort_by_index:
        X = X.sort_index()
    if isinstance(y, (pd.Series, pd.DataFrame)) and sort_by_index:
        y = y.sort_index()

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    n = len(X)
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as float must be between 0 and 1")
        n_test = int(np.ceil(n * test_size))
    elif isinstance(test_size, int):
        if test_size <= 0:
            raise ValueError("test_size as int must be > 0")
        n_test = test_size
    else:
        raise TypeError("test_size must be float or int")

    n_test = min(n_test, n - 1)
    n_train = n - n_test

    if isinstance(X, np.ndarray):
        X_train = X[:n_train]
        X_test = X[n_train:]
    else:
        X_train = X.iloc[:n_train]
        X_test = X.iloc[n_train:]

    if isinstance(y, np.ndarray):
        y_train = y[:n_train]
        y_test = y[n_train:]
    else:
        y_train = y.iloc[:n_train]
        y_test = y.iloc[n_train:]

    return X_train, X_test, y_train, y_test


class SalesModel(ModelBuilder):
    _model_type = "sales_model"
    version = "0.1"

    def build_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        category_col: str = "category",
        article_col: str = "article",
        **kwargs,
    ):

        self.category_col = category_col
        self.article_col = article_col

        X_values = X.values
        y_values = y.to_numpy() if isinstance(y, pd.Series) else y

        self._generate_and_preprocess_data(X_values, y_values)

        with pm.Model() as self.model:
            ...

    def _data_setter(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray = None
    ):
        if isinstance(X, pd.DataFrame):
            x_values = X.to_numpy()
        else:
            x_values = X

        with self.model:
            pm.set_data({"x_data": x_values})
            if y is not None:
                pm.set_data(
                    {"y_data": y.to_numpy() if isinstance(y, pd.Series) else y}
                )

    @staticmethod
    def get_default_model_config() -> dict: ...

    @staticmethod
    def get_default_sampler_config() -> dict:
        return {
            "draws": 1000,
            "tune": 1000,
            "target_accept": 0.85,
            "chains": 4,
        }

    @property
    def output_var(self) -> str:
        return "y"

    @property
    def _serializable_model_config(self) -> dict[str, int | float | dict]:
        return self.model_config

    def _save_input_params(self, idata: az.InferenceData):
        pass

    def _generate_and_preprocess_data(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> None:
        self.catgeory_idx, self.categories = pd.factorize(X[self.category_col])
        self.article_idx, self.articles = pd.factorize(X[self.article_col])

        self.y_data = y
