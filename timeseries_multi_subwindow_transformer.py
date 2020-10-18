# common imports
import numpy as np
import logging
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

logger = logging.getLogger()


def add_trend_feature(arr):
    idx = np.array(range(len(arr)))
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def ewma(x):
    a = pd.Series.ewm(pd.Series(x), span=len(x))
    b = (a.mean()).mean()
    return b


class TimeseriesMultiSubWindowTransformer(BaseEstimator):
    def __init__(self, max_lookback_window=10, func=np.mean, sub_window_len_ratios=[0.1, 0.3, 0.6, 0.9]):
        self.max_lookback_window = max_lookback_window
        self.func = func

        self.sub_window_lens = None
        if sub_window_len_ratios:
            self.sub_window_lens = list()
            for r in sub_window_len_ratios:
                self.sub_window_lens.append(int(r * max_lookback_window))

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.sub_window_lens is None:
            raise ValueError('There are no subwindow lens')

        df = pd.DataFrame(X)
        df_list = list()
        col_names = list()
        for l in self.sub_window_lens:
            tmp = df.rolling(window=l).apply(self.func)
            df_list.append(tmp)
            col_names.append(f'subwindow_len_{l}')

        df_all = pd.concat(df_list, axis=1)
        df_all.columns = col_names
        return df_all.values

