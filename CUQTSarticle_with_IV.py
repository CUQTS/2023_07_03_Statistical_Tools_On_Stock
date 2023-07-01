#importing Libraries 
import yfinance as yf
from datetime import date
import Stat_Tool
import IV

if __name__ == "__main__":

    #User Input
    ticker=str(input("Please insert the ticker symbol according to Yahoo Finance: "))
    start_date="2013-01-01"
    end_date=date.today().strftime("%Y-%m-%d")
    asset_data=yf.download(ticker, start=start_date, end=end_date)

    #Calculating Daily, Weekly, and Monthly Volatility
    asset_data['Return'] = 100*(asset_data['Close'].pct_change())
    asset_data.dropna(inplace=True)

    Stat_Tool.Empirical_Volatility_Study(ticker, asset_data)

    #Volatility Prediction
    #GARCH Model
    p_range = [i for i in range(1,6)]
    q_range = [i for i in range(1,6)]
    gm_result = Stat_Tool.GARCH_Model_Selection(asset_data, p_range , q_range)

    #EWMA Model
    selected_l = Stat_Tool.EWMA_Model_Study(asset_data)

    #Next Day Volatility Prediction
    garch_vol, EWMA_vol, Implied_Volatility = Stat_Tool.next_volatility_prdiction(ticker, asset_data, selected_l, gm_result)

    #Coming Year Volatility Prediction from Implied Volatiltiy
    Stat_Tool.volatility_graph(ticker, asset_data, Implied_Volatility)

    #Implied Volatility Surface
    while True:
        strike_level_bound = input('Please enter the strike level bound (range from 5 to 95) (multiple of 5): ')
        try:
            strike_level_bound = int(strike_level_bound)
            if strike_level_bound%5!=0:
                print('Non multiple of 5, please try again')
                continue
            elif strike_level_bound > 95 or strike_level_bound < 5 :
                print('out of range, please try again')
                continue
            break
        except:
            print('Non numeric input, please try again')
            pass

    IV.plot_implied_volatility_surface(ticker, strike_level_bound)

    #Daily Return Distribution
    while True:
        look_back_window = input('Please enter the length of look back period for daily return distribution: ')
        try:
            look_back_window = int(look_back_window)
            break
        except:
            print('Non numeric input, please try again')
            pass

    Stat_Tool.daily_return_distribution(ticker, asset_data, look_back_window)

    #Confidence Interval on Next Day Close Price
    while True:
        confidence_level = input('Please enter the target confidence level (0-1) : ')
        try:
            confidence_level = float(confidence_level)
            if confidence_level<0 or confidence_level>1 :
                continue
            break
        except:
            print('Non numeric input, please try again')
            pass

    Stat_Tool.confidence_interval_next_close_price(asset_data, confidence_level, garch_vol, EWMA_vol, Implied_Volatility.iloc[0][0])

    #Rolling Z-Score on Close Price
    while True:
        rolling_windows = input('Please enter the length of rolling window for Z-Score: ')
        try:
            rolling_windows = int(rolling_windows)
            break
        except:
            print('Non numeric input, please try again')
            pass

    Stat_Tool.Z_Score_Study(ticker, asset_data, rolling_windows)
