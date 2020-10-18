import unittest
import logging
import pandas as pd
import numpy as np
import time

logger = logging.getLogger()
from timeseries_multi_subwindow_transformer import TimeseriesMultiSubWindowTransformer
from timeseries_multi_subwindow_transformer import add_trend_feature, ewma


class TimeseriesMultiSubwindowTransformerTest(unittest.TestCase):
    def setUp(self):
        infile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
        cols = ["ID", "time", "AirPassengers"]
        df = pd.read_csv(infile, names=cols, sep=r',', index_col='ID', engine='python', skiprows=1)
        trainnum = 100
        self.trainset = df.iloc[:trainnum, 1].values
        self.trainset = self.trainset.reshape(-1, 1)
        self.testset = df.iloc[trainnum:, 1].values
        self.testset = self.testset.reshape(-1, 1)

    def check_transformer(self, transformer, X):
        tr = transformer.fit(X)
        self.assertIsNotNone(tr)
        Xt = tr.transform(X)
        self.assertEqual(X.shape[0], Xt.shape[0])
        return Xt

    def test_timeseries_multi_subwindow_transformer(self):
        functions = [
                     np.mean,
                     np.std,
                     np.var,
                     np.max,
                     np.min,
                     np.median,
                     add_trend_feature,
                     ewma,
                    ]

        for func in functions:
            start = time.time()
            transformer = TimeseriesMultiSubWindowTransformer(func=func)
            self.assertIsNotNone(transformer)
            Xt = self.check_transformer(transformer=transformer, X=self.trainset)
            self.assertTrue(Xt.shape[1] > 0)
            end = time.time()
            print(f'TimeseriesMultiSubWindowTransformer for function: {func}, Xt.shape: {Xt.shape}, running time: {end - start}')

        print()

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
