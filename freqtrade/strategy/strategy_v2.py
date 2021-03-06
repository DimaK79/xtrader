# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.indicator_helpers import fishers_inverse
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame, DatetimeIndex, merge
import numpy # noqa
# --------------------------------
import talib.abstract as ta


class StrategyV2(IStrategy):
    minimal_roi = {
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.15

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def get_ticker_indicator(self):
        return int(self.ticker_interval[:-1])

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        from technical.util import resample_to_interval
        from technical.util import resampled_merge
        
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        # EMA - Exponential Moving Average
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)       
        # resample our dataframes
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 3)
        dataframe_long  = resample_to_interval(dataframe, self.get_ticker_indicator() * 7)

        # compute our RSI's
        dataframe_short['rsi'] = ta.RSI(dataframe_short, timeperiod=14)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)

        # merge dataframe back together
        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_long)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe.fillna(method='ffill', inplace=True)
        #print("rsi - {:.2f}, rsi_15 - {:.2f}, rsi_35 - {:.2f}".format(dataframe['rsi'].iloc[-1], dataframe['rsi_x'].iloc[-1], dataframe['rsi_y'].iloc[-1]))
        #print(dataframe.iloc[-1])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                # must be bearish
				#(dataframe['plus_di'] < dataframe['minus_di']) &
                (dataframe['ema50'] >= dataframe['ema200']) &
                (dataframe['rsi'] < (dataframe['rsi_y'] - 20)) &
                (dataframe['rsi'] != 0) &
                (dataframe['rsi_x'] != 0) 
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (   (dataframe['rsi'] > dataframe['rsi_x']) &
                (dataframe['rsi'] > dataframe['rsi_y']) |
                (dataframe['rsi'] > 90) &
                (dataframe['rsi_x'] > 90)
            ),
            'sell'] = 1
        return dataframe
    
class StrategyV3(IStrategy):
    minimal_roi = {
        "1440":  0.2
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.15

    # Optimal ticker interval for the strategy
    ticker_interval = '15m'

    def get_ticker_indicator(self):
        return int(self.ticker_interval[:-1])

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        from technical.util import resample_to_interval
        from technical.util import resampled_merge
        
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        # EMA - Exponential Moving Average
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # Stoch
        stoch = ta.STOCH(dataframe, fastk_period= 5, slowk_period= 2, slowk_matype=0, slowd_period= 2, slowd_matype=0)
        dataframe['slowd15'] = stoch['slowd']
        dataframe['slowk15'] = stoch['slowk']
        
        stoch = ta.STOCH(dataframe, fastk_period= 10, slowk_period= 3, slowk_matype=0, slowd_period= 3, slowd_matype=0)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=24)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=24)
        
        dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        
        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)
        dataframe['sma220'] = ta.SMA(dataframe, timeperiod=220)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=28)

        # resample our dataframes
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 3)
        dataframe_long = resample_to_interval(dataframe, self.get_ticker_indicator() * 7)

        # compute our RSI's
        dataframe_short['rsi'] = ta.RSI(dataframe_short, timeperiod=14)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)
        
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        dataframe['mfi'] = ta.MFI(dataframe)
        
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        # merge dataframe back together
        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_long)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe.fillna(method='ffill', inplace=True)
        
                # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo .gl/2JGGoy)
        dataframe['fisher_rsi'] = fishers_inverse(dataframe['rsi'])
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        
        dataframe['resample_rsi_2'] = dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*3)]
        dataframe['resample_rsi_8'] = dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*7)]
        
        dataframe['average'] = (dataframe['close'] + dataframe['open'] + dataframe['high'] + dataframe['low']) / 4
        
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                #(dataframe['plus_di'] > dataframe['minus_di']) &
                #(dataframe['sma10'] > dataframe['ema10']) &
                #(dataframe['rsi'] != 0) &
               # (dataframe['sma10'] > dataframe['sma20']) &
                #(dataframe['bb_lowerband'] > dataframe['close']) &
                #(dataframe['CDLHAMMER'] == 100) &
                #(dataframe['slowk'] > dataframe['slowd']) &
                #(dataframe['slowk'] < 35) &
                #(dataframe['sma3'] <= dataframe['sma20']) &
                #(dataframe['slowk'] < 10) &
                #(dataframe['high'].shift(2) > dataframe['sma3']) &
                (dataframe['ema50'] >= dataframe['ema200']) &
                (dataframe['rsi'] < (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 7)] - 20)) &
                (dataframe['rsi'] != 0) &
                (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*3)] != 0)
                
            ),
            'buy'] = 1
        print(dataframe['rsi'])
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        
        dataframe.loc[
            (
                (

                        (      
                               (dataframe['rsi'] > dataframe['resample_rsi_2']) &
                               (dataframe['rsi'] > dataframe['resample_rsi_8']) |
                               (dataframe['rsi'] > 90) &
                               (dataframe['resample_rsi_2'] > 90)
                        )


                )

            ),
            'sell'] = 1
            
        print(dataframe['rsi'])
        return dataframe

class StrategyV4(IStrategy):
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.03

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # Stoch
        stoch_5  = ta.STOCH(dataframe, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        stoch_15 = ta.STOCH(dataframe, fastk_period=27, slowk_period=9, slowk_matype=0, slowd_period=9, slowd_matype=0)
        
        dataframe['slowd_5'] = stoch_5['slowd']
        dataframe['slowk_5'] = stoch_5['slowk']
        
        dataframe['slowd_15'] = stoch_15['slowd']
        dataframe['slowk_15'] = stoch_15['slowk']
        
        dataframe['stochJ_5'] = (3 * dataframe['slowk_5']) - (2 * dataframe['slowd_5'])
        dataframe['stochJ_15'] = (3 * dataframe['slowk_15']) - (2 * dataframe['slowd_15'])
        
        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        
        

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < 45) &
                (dataframe['slowk_15'] > dataframe['slowd_15']) &
                (dataframe['slowk_15'] < 50) &
                (dataframe['slowk_5'] < 25) &
                (dataframe['stochJ_5'] > dataframe['slowd_5']) &
                (dataframe['fastk'] > dataframe['fastd']) &
                (dataframe['fastk'] >= 0) &
                (dataframe['fastk'] < 50)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['stochJ_15'] >= 100) &
                (dataframe['fisher_rsi'] >= 0.8) & 
                (dataframe['fastk'] >= 100)
            ),
            'sell'] = 1
        return dataframe