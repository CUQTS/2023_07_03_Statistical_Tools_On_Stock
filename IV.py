import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def computing_IV(ticker, curr_date):

    #obtaining spot price
    stock = yf.Ticker(ticker)
    spot = stock.history(start=(curr_date), end=(curr_date+datetime.timedelta(days = 1)))["Close"][0]

    #obtaining the options data
    expiration_date_list = stock.options
    end_date = datetime.date(curr_date.year + 1, curr_date.month, curr_date.day)

    #if options data is not available, return empty list
    if len(expiration_date_list) == 0 :
        print('Options Data is not available for this input, empty list is returned')
        return []

    #finding expiration
    effective_expiration = []
    for expiration_date in expiration_date_list:
        if expiration_date < end_date.strftime("%Y-%m-%d"):
            effective_expiration.append(expiration_date)
        elif expiration_date >= end_date.strftime("%Y-%m-%d"):
            effective_expiration.append(expiration_date)
            break
    #fidnding IV
    IV_ls = []

    for e in effective_expiration:
        opt = stock.option_chain(e)

        call_list = opt[0]
        call_list.dropna(inplace=True)
        call_IV = sum(call_list['volume'] * call_list['impliedVolatility']) / sum(call_list['volume'])
        call_total_volume = sum(call_list['volume'])

        put_list = opt[1]
        put_list.dropna(inplace=True)
        put_IV = sum(put_list['volume'] * put_list['impliedVolatility']) / sum(put_list['volume'])
        put_total_volume = sum(put_list['volume'])

        estimated_IV = call_IV * call_total_volume / (
                    call_total_volume + put_total_volume) + put_IV * put_total_volume / (
                                   call_total_volume + put_total_volume)

        estimated_IV = 100 * estimated_IV/math.sqrt(252)

        IV_ls.append(estimated_IV)

    #adjustment for the ATM_IV at 1 year tenor
    begin_date = datetime.datetime.strptime(effective_expiration[-2], "%Y-%m-%d").date()
    last_date = datetime.datetime.strptime(effective_expiration[-1], "%Y-%m-%d").date()

    IV_ls[-1] = IV_ls[-2] + (end_date - begin_date).days / (last_date - begin_date).days * (IV_ls[-1] - IV_ls[-2])

    effective_expiration[-1] = end_date.strftime("%Y-%m-%d")

    #preparing output
    within_year_IV = pd.DataFrame()
    within_year_IV.index = effective_expiration
    within_year_IV['IV'] = IV_ls
    index_ls = [pd.Timestamp(idx) for idx in within_year_IV.index]
    within_year_IV.index = index_ls

    return (within_year_IV)


def plot_implied_volatility_surface(ticker, strike_level_bound):
    strike_level_ls = [i for i in range(-1 * strike_level_bound, strike_level_bound + 5, 5)]
    stock = yf.Ticker(ticker)
    curr_date = datetime.date.today()
    end_date = curr_date + datetime.timedelta(days=365)
    spot = stock.history(start=(curr_date - datetime.timedelta(days=3)), end=curr_date)["Close"][0]

    expiration_date_list = stock.options
    effective_expiration = []
    for expiration_date in expiration_date_list:
        if expiration_date <= end_date.strftime("%Y-%m-%d"):
            effective_expiration.append(expiration_date)
        else:
            break

    call_IV_surface_df = pd.DataFrame()
    put_IV_surface_df = pd.DataFrame()

    for e in effective_expiration:
        opt = stock.option_chain(e)
        call_IV_surface_ls = []
        put_IV_surface_ls = []

        for strike_level in strike_level_ls:
            # computing call IV
            target_strike = spot * (1 - strike_level / 100)
            if np.sum(opt[0]['strike']>=target_strike) == 0:
                target_call_lower = opt[0][opt[0]['strike']<=target_strike].iloc[-2]
                target_call_upper = opt[0][opt[0]['strike']<=target_strike].iloc[-1]
            elif np.sum(opt[0]['strike']<=target_strike) == 0:
                target_call_lower = opt[0][opt[0]['strike']>=target_strike].iloc[0]
                target_call_upper = opt[0][opt[0]['strike']>=target_strike].iloc[1]
            else:
                target_call_lower = opt[0][opt[0]['strike']<=target_strike].iloc[-1]
                target_call_upper = opt[0][opt[0]['strike']>=target_strike].iloc[0]
            target_IV = target_call_lower['impliedVolatility'] + (
                        target_call_upper['impliedVolatility'] - target_call_lower['impliedVolatility']) * (
                                    target_strike - target_call_lower['strike']) / (
                                    target_call_upper['strike'] - target_call_lower['strike']) if (target_call_upper[
                                                                                                       'strike'] -
                                                                                                   target_call_lower[
                                                                                                       'strike']) != 0 else \
            target_call_lower['impliedVolatility']

            target_IV = target_IV if target_IV > 0 else 0

            call_IV_surface_ls.append(target_IV)

            # computing put IV
            target_strike = spot * (1 - strike_level / 100)
            if np.sum(opt[1]['strike']>=target_strike) == 0:
                target_call_lower = opt[1][opt[1]['strike']<=target_strike].iloc[-2]
                target_call_upper = opt[1][opt[1]['strike']<=target_strike].iloc[-1]
            elif np.sum(opt[1]['strike']<=target_strike) == 0:
                target_call_lower = opt[1][opt[1]['strike']>=target_strike].iloc[0]
                target_call_upper = opt[1][opt[1]['strike']>=target_strike].iloc[1]
            else:
                target_call_lower = opt[1][opt[1]['strike']<=target_strike].iloc[-1]
                target_call_upper = opt[1][opt[1]['strike']>=target_strike].iloc[0]
            target_IV = target_call_lower['impliedVolatility'] + (
                        target_call_upper['impliedVolatility'] - target_call_lower['impliedVolatility']) * (
                                    target_strike - target_call_lower['strike']) / (
                                    target_call_upper['strike'] - target_call_lower['strike']) if (target_call_upper[
                                                                                                       'strike'] -
                                                                                                   target_call_lower[
                                                                                                       'strike']) != 0 else \
            target_call_lower['impliedVolatility']

            target_IV = target_IV if target_IV>0 else 0

            put_IV_surface_ls.append(target_IV)

        call_IV_surface_df[
            str((datetime.datetime.strptime(e, "%Y-%m-%d") - datetime.datetime.now()).days)] = call_IV_surface_ls
        put_IV_surface_df[
            str((datetime.datetime.strptime(e, "%Y-%m-%d") - datetime.datetime.now()).days)] = put_IV_surface_ls

    call_IV_surface_df.index = strike_level_ls
    put_IV_surface_df.index = [strike*-1 for strike in strike_level_ls]

    # plot call IV surface

    x = np.arange(len(call_IV_surface_df.columns))
    y = np.arange(len(call_IV_surface_df.index))
    X, Y = np.meshgrid(x, y)
    Z = call_IV_surface_df
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    norm = mpl.colors.Normalize(vmin=Z.min().min(), vmax=Z.max().max())
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(Z)),
                    linewidth=0, antialiased=False)
    plt.xticks(x, call_IV_surface_df.columns)
    plt.yticks(y, call_IV_surface_df.index)
    plt.title('Call Option Implied Volatility Surface')
    plt.xlabel('DTE')
    plt.ylabel('Moneyness (%)')

    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array(Z)
    plt.colorbar(m)
    plt.show()
    plt.close()

    # plot put IV surface
    x = np.arange(len(put_IV_surface_df.columns))
    y = np.arange(len(put_IV_surface_df.index))
    X, Y = np.meshgrid(x, y)
    Z = put_IV_surface_df
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    norm = mpl.colors.Normalize(vmin=Z.min().min(), vmax=Z.max().max())
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.jet(norm(Z)),
                    linewidth=0, antialiased=False)
    plt.xticks(x, put_IV_surface_df.columns)
    plt.yticks(y, put_IV_surface_df.index)
    plt.title('Put Option Implied Volatility Surface')
    plt.xlabel('DTE')
    plt.ylabel('Moneyness (%)')

    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array(Z)
    plt.colorbar(m)
    plt.show()
    plt.close()

