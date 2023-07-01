import numpy as np
import statsmodels.api
import math
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy
import pandas as pd
from arch import arch_model
from arch.__future__ import reindexing
from IV import computing_IV
from EWMA import EWMA_lambda_selection, EWMA_Vol
from confidence_interval_estimation import confidence_interval

def hurst_exponent(q, time_series, max_lag=20):
    lags = range(2, max_lag)

    tau = [np.mean(np.abs(time_series.diff(lag).dropna())**(q)) for lag in lags]

    model = statsmodels.api.OLS(np.log(tau), q* np.log(lags))
    result = model.fit()

    return result

def Z_Score_Study(ticker, asset_data, rolling_windows=20):

    z_score = (asset_data['Close'][-385:] - asset_data['Close'][-385:].rolling(rolling_windows).mean().shift()) / asset_data['Close'][
                                                                                             -385:].rolling(rolling_windows).std().shift()
    z_score_to_be_observed = z_score[-252:]

    plt.plot(z_score_to_be_observed)
    plt.title(f'{rolling_windows}-Days Rolling Z-score on Close Price in Previous Year')
    plt.show()

    # ADF Test for Stability of Close Price Z-Score
    res = statsmodels.api.tsa.stattools.adfuller(z_score_to_be_observed, maxlag=20)
    adf_test_p_value = res[1]

    # Hurst Exponent
    q = 1
    max_lag = 20
    hurst_res = hurst_exponent(q, z_score_to_be_observed, max_lag)
    hurst = hurst_res.params[0]

    print(tabulate([[ticker, adf_test_p_value, hurst]],
                   headers=['ADF Test P Value %', 'Hurst Exponent'],
                   tablefmt='fancy_grid', stralign='center', numalign='center', floatfmt=".2f"))

def daily_return_distribution(ticker, asset_data, look_back_window=20):

    ret_sample = asset_data['Return'][-1*look_back_window:]
    plt.hist(ret_sample, bins=20)
    plt.title(f'Previous {look_back_window} Days Daily Return Distribution')
    plt.show()

    print(tabulate([[ticker, np.mean(ret_sample), np.std(ret_sample), scipy.stats.skew(ret_sample),
                     scipy.stats.kurtosis(ret_sample)]],
                   headers=['Mean %', 'Volatility %', 'Skewness %', 'Kurtosis %'],
                   tablefmt='fancy_grid', stralign='center', numalign='center', floatfmt=".2f"))

def volatility_graph(ticker, asset_data, Implied_Volatility):
    # Below is the forecast using GARCH
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.3, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(asset_data['Return'][-365:].rolling(20).std(), label='Previous Volatility')
    # ax.plot(rolling_predictions, label='Rolling Prediction')
    # ax.plot(forecast['forecast'], label='GARCH Forecast')
    if len(Implied_Volatility) > 0:
        ax.plot(Implied_Volatility['IV'], label='Implied Volatility')
    ax.set_title(f'{ticker} Empirical Volatility and Implied Volatility')
    ax.legend()
    plt.show()

def confidence_interval_next_close_price(asset_data, confidence_level, garch_vol, EWMA_vol, IV):

    # Confidence Interval based on GARCH Model Estimation
    return_interval = confidence_interval(asset_data['Return'][-20:] / 100, garch_vol / 100, confidence_level)
    close_left_bound_garch = (return_interval[0] + 1) * asset_data['Close'].iloc[-1]
    close_right_bound_garch = (return_interval[1] + 1) * asset_data['Close'].iloc[-1]

    # Confidence Interval based on EWMA Model Estimation
    return_interval = confidence_interval(asset_data['Return'][-20:] / 100, EWMA_vol / 100, confidence_level)
    close_left_bound_ewma = (return_interval[0] + 1) * asset_data['Close'].iloc[-1]
    close_right_bound_ewma = (return_interval[1] + 1) * asset_data['Close'].iloc[-1]

    # Confidence Interval based on IV
    return_interval = confidence_interval(asset_data['Return'][-20:] / 100, IV / 100, confidence_level)
    close_left_bound_IV = (return_interval[0] + 1) * asset_data['Close'].iloc[-1]
    close_right_bound_IV = (return_interval[1] + 1) * asset_data['Close'].iloc[-1]

    print(tabulate([['lower bound', close_left_bound_garch, close_left_bound_ewma, close_left_bound_IV],
                    ['upper bound', close_right_bound_garch, close_right_bound_ewma, close_right_bound_IV]],
                   headers=['GARCH Model', 'EWMA Model', 'Implied Volatility'],
                   tablefmt='fancy_grid', stralign='center', numalign='center', floatfmt=".2f"))

def next_volatility_prdiction(ticker, asset_data, selected_l, gm_result):
    ret = asset_data['Return'][-252:]/100

    start_date = asset_data.index[-1].date() + pd.Timedelta(days=1)
    end_date = asset_data.index[-1].date() + pd.Timedelta(days=5)
    forecast_horizon = pd.date_range(start=start_date, end=end_date, freq='B')
    gm_forecast = gm_result.forecast(horizon=len(forecast_horizon), start=start_date)
    garch_vol = np.sqrt(gm_forecast.variance.iloc[-1][0])

    # EWMA Model
    EWMA_vol_ls = EWMA_Vol(selected_l, ret)
    EWMA_vol = np.sqrt((selected_l * EWMA_vol_ls[-1] ** 2 + (1 - selected_l) * ret[-1] ** 2)) * 100

    # IV
    Implied_Volatility = computing_IV(ticker, asset_data.index[-1].date())
    IV = Implied_Volatility.iloc[0][0]

    print(tabulate([[ticker, garch_vol, EWMA_vol, IV]],
                   headers=['GARCH Predicted Volatility %', 'EWMA Predicted Volatility %', 'Implied Volatility %'],
                   tablefmt='fancy_grid', stralign='center', numalign='center', floatfmt=".2f"))

    return garch_vol, EWMA_vol, Implied_Volatility

def EWMA_Model_Study(asset_data):
    ret = asset_data['Return'][-252:] / 100
    selected_l = EWMA_lambda_selection(ret)

    print('EWMA lambda: ', selected_l)

    return selected_l

def GARCH_Model_Study(asset_data, selected_p, selected_q):
    garch_model = arch_model(asset_data['Return'][-252:], p=selected_p, q=selected_q, mean='zero', vol='GARCH', dist='Normal')
    gm_result = garch_model.fit(disp='off')

    return gm_result

def GARCH_Model_Selection(asset_data, p_range = [i for i in range(1,6)], q_range = [i for i in range(1,6)]):

    selected_p = None
    selected_q = None
    best_ic = np.inf

    for p in p_range:
        for q in q_range:
            test_gm_result = GARCH_Model_Study(asset_data, p, q)
            ic = test_gm_result.aic

            if ic < best_ic:
                selected_p = p
                selected_q = q
                best_ic = ic
                gm_result = test_gm_result

    print(gm_result.params)

    print(
        '\n Where: \n mu = mean return \n omega = long-term average \n alpha = short-run volatility \n beta = persistence of volatility')


    print(f"Selected GARCH Model: p = {selected_p}, q = {selected_q}")


    return gm_result

def Empirical_Volatility_Study(ticker, asset_data):
    daily_volatility = asset_data['Return'][-252:].std()
    monthly_volatility = math.sqrt(21) * daily_volatility
    annual_volatility = math.sqrt(252) * daily_volatility

    print(tabulate([[ticker, daily_volatility, monthly_volatility, annual_volatility]],
                   headers=['Daily Volatility %', 'Monthly Volatility %', 'Annual Volatility %'],
                   tablefmt='fancy_grid', stralign='center', numalign='center', floatfmt=".2f"))




