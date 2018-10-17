# pragma pylint: disable=W0603
""" Cryptocurrency Exchanges support """
import logging
from random import randint
from typing import List, Dict, Any, Optional
from datetime import datetime
from math import floor, ceil
from decimal import Decimal, getcontext
from binance.client import Client
from typing import Dict, List, NamedTuple, Tuple
import math
import time
import numpy as np
import ccxt
import arrow
import talib as ta

from freqtrade import constants, OperationalException, DependencyException, TemporaryError

logger = logging.getLogger(__name__)

API_RETRY_COUNT = 4


# Urls to exchange markets, insert quote and base with .format()
_EXCHANGE_URLS = {
    ccxt.bittrex.__name__: '/Market/Index?MarketName={quote}-{base}',
    ccxt.binance.__name__: '/tradeDetail.html?symbol={base}_{quote}'
}


def retrier(f):
    def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return f(*args, **kwargs)
        except (TemporaryError, DependencyException) as ex:
            logger.warning('%s() returned exception: "%s"', f.__name__, ex)
            if count > 0:
                count -= 1
                kwargs.update({'count': count})
                logger.warning('retrying %s() still for %s times', f.__name__, count)
                return wrapper(*args, **kwargs)
            else:
                logger.warning('Giving up retrying: %s()', f.__name__)
                raise ex
    return wrapper


class Exchange(object):

    # Current selected exchange
    _api: ccxt.Exchange = None
    _conf: Dict = {}
    _cached_ticker: Dict[str, Any] = {}

    # Holds all open sell orders for dry_run
    _dry_run_open_orders: Dict[str, Any] = {}

    def __init__(self, config: dict) -> None:
        """
        Initializes this module with the given config,
        it does basic validation whether the specified
        exchange and pairs are valid.
        :return: None
        """
        self._conf.update(config)

        if config['dry_run']:
            logger.info('Instance is running with dry_run enabled')

        exchange_config = config['exchange']
        self._api = self._init_ccxt(exchange_config)

        logger.info('Using Exchange "%s"', self.name)

        # Check if all pairs are available
        self.validate_pairs(config['exchange']['pair_whitelist'])

        if config.get('ticker_interval'):
            # Check if timeframe is available
            self.validate_timeframes(config['ticker_interval'])

    def _init_ccxt(self, exchange_config: dict) -> ccxt.Exchange:
        """
        Initialize ccxt with given config and return valid
        ccxt instance.
        """
        # Find matching class for the given exchange name
        name = exchange_config['name']

        if name not in ccxt.exchanges:
            raise OperationalException(f'Exchange {name} is not supported')
        try:
            self.client = Client(exchange_config.get('key'), exchange_config.get('secret'))
            
            api = getattr(ccxt, name.lower())({
                'apiKey': exchange_config.get('key'),
                'secret': exchange_config.get('secret'),
                'password': exchange_config.get('password'),
                'uid': exchange_config.get('uid', ''),
                'enableRateLimit': exchange_config.get('ccxt_rate_limit', True),
            })
        except (KeyError, AttributeError):
            raise OperationalException(f'Exchange {name} is not supported')

        self.set_sandbox(api, exchange_config, name)

        return api

    @property
    def name(self) -> str:
        """exchange Name (from ccxt)"""
        return self._api.name

    @property
    def id(self) -> str:
        """exchange ccxt id"""
        return self._api.id

    def set_sandbox(self, api, exchange_config: dict, name: str):
        if exchange_config.get('sandbox'):
            if api.urls.get('test'):
                api.urls['api'] = api.urls['test']
                logger.info("Enabled Sandbox API on %s", name)
            else:
                logger.warning(self._api.name, "No Sandbox URL in CCXT, exiting. "
                                               "Please check your config.json")
                raise OperationalException(f'Exchange {name} does not provide a sandbox api')

    def validate_pairs(self, pairs: List[str]) -> None:
        """
        Checks if all given pairs are tradable on the current exchange.
        Raises OperationalException if one pair is not available.
        :param pairs: list of pairs
        :return: None
        """

        try:
            markets = self._api.load_markets()
        except ccxt.BaseError as e:
            logger.warning('Unable to validate pairs (assuming they are correct). Reason: %s', e)
            return

        stake_cur = self._conf['stake_currency']
        for pair in pairs:
            # Note: ccxt has BaseCurrency/QuoteCurrency format for pairs
            # TODO: add a support for having coins in BTC/USDT format
            if not pair.endswith(stake_cur):
                raise OperationalException(
                    f'Pair {pair} not compatible with stake_currency: {stake_cur}')
            if pair not in markets:
                raise OperationalException(
                    f'Pair {pair} is not available at {self.name}')

    def validate_timeframes(self, timeframe: List[str]) -> None:
        """
        Checks if ticker interval from config is a supported timeframe on the exchange
        """
        timeframes = self._api.timeframes
        if timeframe not in timeframes:
            raise OperationalException(
                f'Invalid ticker {timeframe}, this Exchange supports {timeframes}')

    def exchange_has(self, endpoint: str) -> bool:
        """
        Checks if exchange implements a specific API endpoint.
        Wrapper around ccxt 'has' attribute
        :param endpoint: Name of endpoint (e.g. 'fetchOHLCV', 'fetchTickers')
        :return: bool
        """
        return endpoint in self._api.has and self._api.has[endpoint]

    def symbol_amount_prec(self, pair, amount: float):
        '''
        Returns the amount to buy or sell to a precision the Exchange accepts
        Rounded down
        '''
        if self._api.markets[pair]['precision']['amount']:
            symbol_prec = self._api.markets[pair]['precision']['amount']
            big_amount = amount * pow(10, symbol_prec)
            amount = floor(big_amount) / pow(10, symbol_prec)
        return amount
    
    def get_symbol_info(self, symbol):
        return self.client.get_symbol_info(symbol)
    
    
    def symbol_amount_prec_str(self, pair, amount: float):
        '''
        Returns the amount to buy or sell to a precision the Exchange accepts
        Rounded down
        '''
        try:
            print('symbol_amount_prec_str')
            print(pair)
            symbol = pair.replace('/','')
            sym = symbol[:-3]
            symbol_info = self.get_symbol_info(symbol)
            symbal = self.client.get_asset_balance(asset=sym)
            quaselltrade = float(symbal['free'])
            symbol_info['filters'] = {item['filterType']: item for item in symbol_info['filters']}
            min_qty = float(symbol_info['filters']['LOT_SIZE']['minQty'])
            step_size = float(symbol_info['filters']['LOT_SIZE']['stepSize'])
            amount = float(step_size * math.floor(amount / step_size))
            quaselltrade = float(step_size * math.floor(quaselltrade / step_size))
            time.sleep(0.3)
            print(min_qty, amount, quaselltrade)
            
            return amount, quaselltrade
        except  OSError as e:
            print(e)
            return 0, 0

    def symbol_price_prec(self, pair, price: float):
        '''
        Returns the price buying or selling with to the precision the Exchange accepts
        Rounds up
        '''

        if self._api.markets[pair]['precision']['price']:
            symbol_prec = self._api.markets[pair]['precision']['price']
            big_price = price * pow(10, symbol_prec)
            price = ceil(big_price) / pow(10, symbol_prec)
            print(symbol_prec, pair, price)
        return price
    
    def symbol_price_prec_str(self, pair, price: float):
        '''
        Returns the price buying or selling with to the precision the Exchange accepts
        Rounds up
        '''

        if self._api.markets[pair]['precision']['price']:
            symbol_prec = self._api.markets[pair]['precision']['price']
            big_price = price * pow(10, symbol_prec)
            price = ceil(big_price) / pow(10, symbol_prec)
            price = "{:0.{}f}".format(price, symbol_prec)
            print(symbol_prec, pair, price)
        return price
    
    def buy(self, pair: str, rate: float, amount: float) -> Dict:
        if self._conf['dry_run']:
            order_id = f'dry_run_buy_{randint(0, 10**6)}'
            self._dry_run_open_orders[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': 'limit',
                'side': 'buy',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed',
                'fee': None
            }
            return {'id': order_id}

        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            symbol = pair.replace('/','')
            amount, quaselltrade = self.symbol_amount_prec_str(pair, amount)
            if (float(amount) > 0):
                rate = self.symbol_price_prec_str(pair, rate)
                print(amount, rate)
                return self.client.order_limit_buy(symbol=symbol, quantity=amount, price=rate)
            else:
                return {'orderId': None}
                print('Error - Buy')
            
            #return self._api.create_limit_buy_order(pair, amount, rate)
        except ccxt.InsufficientFunds as e:
            raise DependencyException(
                f'Insufficient funds to create limit buy order on market {pair}.'
                f'Tried to buy amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not create limit buy order on market {pair}.'
                f'Tried to buy amount {amount} at rate {rate} (total {rate*amount}).'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not place buy order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)
        except  OSError as e:
            return {'orderId': None}
            print(e)
            pass

    def sell(self, pair: str, rate: float, amount: float) -> Dict:
        
        if self._conf['dry_run']:
            order_id = f'dry_run_sell_{randint(0, 10**6)}'
            self._dry_run_open_orders[order_id] = {
                'pair': pair,
                'price': rate,
                'amount': amount,
                'type': 'limit',
                'side': 'sell',
                'remaining': 0.0,
                'datetime': arrow.utcnow().isoformat(),
                'status': 'closed'
            }
            return {'id': order_id}

        try:
            # Set the precision for amount and price(rate) as accepted by the exchange
            

            rate = self.symbol_price_prec_str(pair, rate)
            print('exchange.sell !!!')
            print(amount, rate)
            symbol = pair.replace('/','')
            #print(self._api.create_limit_sell_order(pair, amount, Decimal(format(rate_prec, '.8f'))))
            return self.client.order_limit_sell(symbol=symbol, quantity=amount, price=rate)
            #return self._api.create_limit_sell_order(pair, amount, rate_prec)
        except  OSError as e:
            return {'orderId': None}
            print(e)

    @retrier
    def get_balance(self, currency: str) -> float:
        if self._conf['dry_run']:
            return 999.9

        # ccxt exception is already handled by get_balances
        balances = self.get_balances()
        balance = balances.get(currency)
        if balance is None:
            raise TemporaryError(
                f'Could not get {currency} balance due to malformed exchange response: {balances}')
        return balance['free']

    @retrier
    def get_balances(self) -> dict:
        if self._conf['dry_run']:
            return {}

        try:
            balances = self._api.fetch_balance()
            # Remove additional info from ccxt results
            balances.pop("info", None)
            balances.pop("free", None)
            balances.pop("total", None)
            balances.pop("used", None)

            return balances
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get balance due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_tickers(self) -> Dict:
        try:
            return self._api.fetch_tickers()
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching tickers in batch.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load tickers due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_ticker(self, pair: str, refresh: Optional[bool] = True) -> dict:
        if refresh or pair not in self._cached_ticker.keys():
            try:
                data = self._api.fetch_ticker(pair)
                try:
                    self._cached_ticker[pair] = {
                        'bid': float(data['bid']),
                        'ask': float(data['ask']),
                    }
                except KeyError:
                    logger.debug("Could not cache ticker data for %s", pair)
                return data
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f'Could not load ticker history due to {e.__class__.__name__}. Message: {e}')
            except ccxt.BaseError as e:
                raise OperationalException(e)
        else:
            logger.info("returning cached ticker-data for %s", pair)
            print(self._cached_ticker[pair])
            return self._cached_ticker[pair]

    @retrier
    def get_candle_history(self, pair: str, tick_interval: str,
                           since_ms: Optional[int] = None) -> List[Dict]:
        try:
            # last item should be in the time interval [now - tick_interval, now]
            till_time_ms = arrow.utcnow().shift(
                            minutes=-constants.TICKER_INTERVAL_MINUTES[tick_interval]
                        ).timestamp * 1000
            # it looks as if some exchanges return cached data
            # and they update it one in several minute, so 10 mins interval
            # is necessary to skeep downloading of an empty array when all
            # chached data was already downloaded
            till_time_ms = min(till_time_ms, arrow.utcnow().shift(minutes=-10).timestamp * 1000)

            data: List[Dict[Any, Any]] = []
            while not since_ms or since_ms < till_time_ms:
                data_part = self._api.fetch_ohlcv(pair, timeframe=tick_interval, since=since_ms)
                symbol = pair.replace('/','')
                #data_part = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE,"12 hours ago UTC"),dtype='float')
                #print(data_part)
                # Because some exchange sort Tickers ASC and other DESC.
                # Ex: Bittrex returns a list of tickers ASC (oldest first, newest last)
                # when GDAX returns a list of tickers DESC (newest first, oldest last)
                data_part = sorted(data_part, key=lambda x: x[0])
                #print(data_part)

                if not data_part:
                    break

                logger.debug('Downloaded data for %s time range [%s, %s]',
                             pair,
                             arrow.get(data_part[0][0] / 1000).format(),
                             arrow.get(data_part[-1][0] / 1000).format())

                data.extend(data_part)
                since_ms = data[-1][0] + 1

            return data
        except ccxt.NotSupported as e:
            raise OperationalException(
                f'Exchange {self._api.name} does not support fetching historical candlestick data.'
                f'Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load ticker history due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(f'Could not fetch ticker data. Msg: {e}')

    @retrier
    def cancel_order(self, order_id: str, pair: str) -> None:
        if self._conf['dry_run']:
            return

        try:
            return self._api.cancel_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not cancel order. Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not cancel order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_order(self, order_id: str, pair: str) -> Dict:
        if self._conf['dry_run']:
            order = self._dry_run_open_orders[order_id]
            order.update({
                'id': order_id
            })
            return order
        try:
            return self._api.fetch_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise DependencyException(
                f'Could not get order. Message: {e}')
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get order due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_trades_for_order(self, order_id: str, pair: str, since: datetime) -> List:
        if self._conf['dry_run']:
            return []
        if not self.exchange_has('fetchMyTrades'):
            return []
        try:
            my_trades = self._api.fetch_my_trades(pair, since.timestamp())
            matched_trades = [trade for trade in my_trades if trade['order'] == order_id]

            return matched_trades

        except ccxt.NetworkError as e:
            raise TemporaryError(
                f'Could not get trades due to networking error. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def get_pair_detail_url(self, pair: str) -> str:
        try:
            url_base = self._api.urls.get('www')
            base, quote = pair.split('/')

            return url_base + _EXCHANGE_URLS[self._api.id].format(base=base, quote=quote)
        except KeyError:
            logger.warning('Could not get exchange url for %s', self.name)
            return ""

    @retrier
    def get_markets(self) -> List[dict]:
        try:
            return self._api.fetch_markets()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not load markets due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    @retrier
    def get_fee(self, symbol='ETH/BTC', type='', side='', amount=1,
                price=1, taker_or_maker='maker') -> float:
        try:
            # validate that markets are loaded before trying to get fee
            if self._api.markets is None or len(self._api.markets) == 0:
                self._api.load_markets()

            return self._api.calculate_fee(symbol=symbol, type=type, side=side, amount=amount,
                                           price=price, takerOrMaker=taker_or_maker)['rate']
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f'Could not get fee info due to {e.__class__.__name__}. Message: {e}')
        except ccxt.BaseError as e:
            raise OperationalException(e)

    def get_amount_lots(self, pair: str, amount: float) -> float:
        """
        get buyable amount rounding, ..
        """
        # validate that markets are loaded before trying to get fee
        if not self._api.markets:
            self._api.load_markets()
        return self._api.amount_to_lots(pair, amount)
    
    
    def get_indicators(self, pair)-> Tuple[bool, bool]:
        try:
            symbol = pair.replace('/','')
            candles_5min = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE,"24 hours ago UTC"),dtype='float')
            candles_15min = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE,"48 hours ago UTC"),dtype='float')
            candles_1hr = np.array(self.client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR,"3 days ago UTC"),dtype='float')
            
            close_1hr = candles_1hr[:,4]
            close_15min = candles_15min[:,4]
            close_5min = candles_5min[:,4]
            
            rsi_1 = ta.RSI(close_5min, timeperiod=14)
            rsi_2 = ta.RSI(close_15min, timeperiod=14)
            rsi_3 = ta.RSI(close_1hr, timeperiod=14)             
            
            ema50 = ta.EMA(close_5min, timeperiod=50)
            ema200 = ta.EMA(close_5min, timeperiod=200)
            
            (buy, sell) = (False, False)
            
            if ema50[-1] >= ema200[-1] and rsi_1[-1] < rsi_3[-1] - 20 and rsi_1[-1] !=0 and rsi_2[-1] != 0:
                buy = True
            
            if (rsi_1[-1] > rsi_2[-1] and rsi_1[-1] > rsi_3[-1]) or (rsi_1[-1] > 90 and rsi_2[-1] > 90):
                sell = True
            
            #msg_txt = (buy, sell)
            msg_txt = ("{}, {}, ; pair: {}, ema50: {:.8f}, ema200: {:.8f}, rsi_1 {:.2f}, rsi_2 {:.2f}, rsi_3 {:.2f}".format(buy, sell, symbol, ema50[-1], ema200[-1], rsi_1[-1], rsi_2[-1], rsi_3[-1]))
            print(msg_txt)
        #except  OSError as e:
        #    (buy, sell) = (False, False)
        #    print(e)
        except:
            (buy, sell) = (False, False)
            pass            
                        
        return buy, sell

