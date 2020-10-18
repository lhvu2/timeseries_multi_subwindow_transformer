# timeseries_multi_subwindow_transformer

This repo contains a column transformer for time series. The transformer receives a list of subwindow length ratios and for each subwindow it transforms the subwindow into a scalar using the input function. Input functions can be standard sklearn or numpy function, or a custom function.

The unittest shows usages for both standard and custom functions.

Prerequisites:
- pandas
- numpy


How to run?
- python test_timeseries_multi_subwindow_transformers.py

Expected output:

TimeseriesMultiSubWindowTransformer for function: <function mean at 0x7fc080135f28>, Xt.shape: (100, 4), running time: 0.06944632530212402

TimeseriesMultiSubWindowTransformer for function: <function std at 0x7fc08013a158>, Xt.shape: (100, 4), running time: 0.0827341079711914

TimeseriesMultiSubWindowTransformer for function: <function var at 0x7fc08013a2f0>, Xt.shape: (100, 4), running time: 0.07758784294128418

TimeseriesMultiSubWindowTransformer for function: <function amax at 0x7fc080135268>, Xt.shape: (100, 4), running time: 0.055271148681640625

TimeseriesMultiSubWindowTransformer for function: <function amin at 0x7fc080135400>, Xt.shape: (100, 4), running time: 0.05540823936462402

TimeseriesMultiSubWindowTransformer for function: <function median at 0x7fc0b0186620>, Xt.shape: (100, 4), running time: 0.046692848205566406

TimeseriesMultiSubWindowTransformer for function: <function add_trend_feature at 0x7fc0500687b8>, Xt.shape: (100, 4), running time: 0.14040803909301758

TimeseriesMultiSubWindowTransformer for function: <function ewma at 0x7fc05022b378>, Xt.shape: (100, 4), running time: 0.19244861602783203

.
----------------------------------------------------------------------
Ran 1 test in 0.822s

OK

