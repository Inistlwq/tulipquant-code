# coding:utf-8

from __future__ import absolute_import, print_function, unicode_literals

import datetime
import time
from fastcache import clru_cache as lru_cache
import numpy as np
try:
    import SharedArray
except:
    pass
import jqdata
from jqdata.utils.utils import fixed_round
from jqdata.utils.datetime_utils import to_timestamp, vec2date, to_datetime, to_date
from retrying import retry

class _ShmBlock(object):
    """
    一个共享内存块

    `array`: np.array
    `index`: 行索引, list of timestamps
    `columns`: 列索引, list of strs

    天数据: 一只股票的历史数据是一块
    分钟数据: 一直股票的一天数据是一块
    """

    def __init__(self, array, columns):
        self.array = array
        self.columns = columns
        self.colspos = {col:i for i, col in enumerate(columns)}
        self.index = array[:, 0]
        pass

    def getitem(self, row, col):
        # COPY ALL， 先看看效果，在优化
        return self.array[row, self.colspos[col]]



class ShmDayStore(object):

    def __init__(self, cfg):
        self._cfg = cfg
        pass

    @lru_cache(None)
    def open_shm_block(self, security):
        shm_path = self._cfg.get_day_shm_path(security.code)
        try:
            array = retry(stop_max_attempt_number=3, wait_fixed=1000)(SharedArray.attach)("file://" + shm_path, readonly=True)
        except OSError as e:
            array = np.empty((0, len(security.day_column_names)+1))
        block = _ShmBlock(array, ('date',) + security.day_column_names)
        return block

    def get_trading_days(self, security):
        return self.open_shm_block(security).index

    def have_data(self, security, date):
        """ 当天是否有数据, 当天有交易返回 True, 否则 False """
        index = self.get_trading_days(security)
        ts = to_timestamp(date)
        idx = index.searchsorted(ts, side='right') - 1
        if idx >= 0 and index[idx] == ts:
            return True
        return False

    def _get_idx_by_period(self, security, start_date, end_date, include_end):
        ct = self.open_shm_block(security)
        start_ts = to_timestamp(start_date)
        end_ts = to_timestamp(end_date)
        start_idx = ct.index.searchsorted(start_ts)
        stop_idx = ct.index.searchsorted(end_ts, side='right' if include_end else 'left')
        return (ct, slice(start_idx, stop_idx))

    def _get_idx_by_count(self, security, end_date, count, include_end):
        ct = self.open_shm_block(security)
        end_ts = to_timestamp(end_date)
        stop_idx = ct.index.searchsorted(end_ts, side='right' if include_end else 'left')
        start_idx = stop_idx - count
        if start_idx < 0:
            start_idx = 0
        return (ct, slice(start_idx, stop_idx))

    def get_factor_by_period(self, security, start_date, end_date, include_now=True):
        """
        获取 security [start_date, end_date] 期间的复权因子。

        如果停牌，则返回停牌前的复权因子。
        """
        ct, idx_slice = self._get_idx_by_period(security, start_date, end_date, include_now)
        if idx_slice.start < idx_slice.stop:
            index = ct.getitem(idx_slice, 'date')
            index = vec2date(index)
            data = ct.getitem(idx_slice, 'factor')
            return index, np.copy(data)
        else:
            factor = self.get_factor_by_date(security, start_date)
            index = np.array([start_date])
            data = np.array([factor])
            return index, data

    def get_factor_by_date(self, security, date):
        """
        获取 security 在 date 这一天的复权因子，不存在则返回 1.0
        """
        ct = self.open_shm_block(security)

        idx = ct.index.searchsorted(to_timestamp(date), side='right') - 1
        if idx < 0:
            return 1.0
        if idx < len(ct.index):
            return ct.getitem(idx, 'factor')
        return 1.0

    # 获取天数据
    def get_bar_by_period(self, security, start_date, end_date, fields, include_now=True):
        """
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `start_date`: 开始日期, 例如 datetime.date(2015, 1, 1)。
        `end_date`: 结束日期，例如 datetime.date(2016, 12, 30)。
        `fields`: 行情数据字段。
        """
        ct, idx_slice = self._get_idx_by_period(security, start_date, end_date, include_now)
        data = {name: np.copy(ct.getitem(idx_slice, name)) for name in fields}
        data['date'] = ct.getitem(idx_slice, 'date')
        return data

    def get_bar_by_count(self, security, end_date, count, fields, include_now=True):
        """
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `end_date`: 结束日期，例如 datetime.date(2016, 12, 30)。
        `count`: count条记录, 例如 300。
        `fields`: 行情数据字段。
        """
        ct, idx_slice = self._get_idx_by_count(security, end_date, count, include_now)
        data = {name: np.copy(ct.getitem(idx_slice, name)) for name in fields}
        data['date'] = ct.getitem(idx_slice, 'date')
        return data

    def get_date_by_period(self, security, start_date, end_date, include_now=True):
        ct, idx_slice = self._get_idx_by_period(security, start_date, end_date, include_now)
        return ct.index[idx_slice]

    def get_date_by_count(self, security, end_date, count, include_now=True):
        ct, idx_slice = self._get_idx_by_count(security, end_date, count, include_now)
        return ct.index[idx_slice]

    def get_bar_by_date(self, security, somedate):
        """
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `date`: 如果date天不存在数据，返回上一个交易日的数据。
        """
        ct = self.open_shm_block(security)
        ts = to_timestamp(somedate)
        idx = ct.index.searchsorted(ts, side='right') - 1
        if idx < 0:
            return None
        price_decimals = security.price_decimals
        bar = {}

        for name in ct.columns:
            if name in ['open', 'close', 'high', 'low', 'price', 'avg',
                        'pre_close', 'high_limit', 'low_limit']:
                bar[name] = fixed_round(ct.getitem(idx, name), price_decimals)
            elif name in ['volume', 'money']:
                bar[name] = fixed_round(ct.getitem(idx, name), 0)
            else:
                bar[name] = ct.getitem(idx, name)
        bar['date'] = to_date(bar['date'])
        return bar


class ShmMinuteStore(object):
    """ 
    - 共享内存中分钟数据是一个交易日一个文件。
    - 期货的夜盘所在交易日和当时真实日期的值不一样。
    - 接口中的end_dt可能是今天之后的日期。
    """

    def __init__(self, cfg):
        self._cfg = cfg


    @lru_cache(None)
    def open_shm_block(self, security, date):
        shm_path = self._cfg.get_minute_shm_path(security.code, date.strftime("%Y-%m-%d"))
        try:
            array = retry(stop_max_attempt_number=3, wait_fixed=1000)(SharedArray.attach)("file://" + shm_path, readonly=True)
        except OSError as e:
            array = np.empty((0, len(security.minute_column_names) + 1))
        block = _ShmBlock(array, ('date',) + security.minute_column_names)
        return block

    def get_latest_shm_date(self, security):
        '''
        分钟数据：共享内存中只保存当前一天的数据。
        
        如果用户的end_dt 超出当天，需要调整。
        '''
        cal = jqdata.stores.CalendarStore.instance()
        now_dt = datetime.datetime.now()
        trade_date = cal.get_current_trade_date(security, now_dt)
        return trade_date


    def _get_idx_by_period(self, security, start_dt, end_dt, include_end):
        shm_date = self.get_latest_shm_date(security)
        ct = self.open_shm_block(security, shm_date)
        start_ts = to_timestamp(start_dt)
        end_ts = to_timestamp(end_dt)
        start_idx = ct.index.searchsorted(start_ts)
        stop_idx = ct.index.searchsorted(end_ts, side='right' if include_end else 'left')
        return (ct, slice(start_idx, stop_idx))

    def _get_idx_by_count(self, security, end_dt, count, include_end):
        shm_date = self.get_latest_shm_date(security)
        ct = self.open_shm_block(security, shm_date)
        end_ts = to_timestamp(end_dt)
        stop_idx = ct.index.searchsorted(end_ts, side='right' if include_end else 'left')
        start_idx = stop_idx - count
        if start_idx < 0:
            start_idx = 0
        return (ct, slice(start_idx, stop_idx))

    def get_bar_by_period(self, security, start_dt, end_dt, fields, include_now=True):
        ct, idx_slice = self._get_idx_by_period(security, start_dt, end_dt, include_now)
        data = {name: np.copy(ct.getitem(idx_slice, name)) for name in fields}
        data['date'] = ct.index[idx_slice]
        return data

    def get_bar_by_count(self, security, end_dt, count, fields, include_now=True):
        ct, idx_slice = self._get_idx_by_count(security, end_dt, count, include_now)
        data = {name: np.copy(ct.getitem(idx_slice, name)) for name in fields}
        data['date'] = ct.index[idx_slice]
        return data

    def get_bar_by_dt(self, security, somedt):
        shm_date = self.get_latest_shm_date(security)
        ct = self.open_shm_block(security, shm_date)
        ts = to_timestamp(somedt)
        idx = ct.index.searchsorted(ts, side='right') - 1

        if idx < 0:
            return None

        price_decimals = security.price_decimals
        bar = {}
        for name in ct.columns:
            if name in ['open', 'close', 'high', 'low', 'price', 'avg',
                        'pre_close', 'high_limit', 'low_limit']:
                bar[name] = fixed_round(ct.getitem(idx, name), price_decimals)
            elif name in ['volume', 'money']:
                bar[name] = fixed_round(ct.getitem(idx, name), 0)
            else:
                bar[name] = ct.getitem(idx, name)

        bar['date'] = to_datetime(bar['date'])
        return bar

    def get_minute_by_period(self, security, start_dt, end_dt, include_now=True):
        ct, idx_slice = self._get_idx_by_period(security, start_dt, end_dt, include_now)
        return ct.index[idx_slice]

    def get_minute_by_count(self, security, end_dt, count, include_now=True):
        ct, idx_slice = self._get_idx_by_count(security, end_dt, count, include_now)
        return ct.index[idx_slice]


class MixedMinuteStore(object):

    def __init__(self, cfg):
        from .bcolz_store import get_bcolz_minute_store
        self._cfg = cfg
        self._today = datetime.date.today()
        self._bcolz_store = get_bcolz_minute_store()
        self._shm_store = get_shm_minute_store()
        pass

    def _should_access_shm(self, security, dt):
        # 考虑商品期货的夜盘。
        from jqdata.stores.calendar_store import get_calendar_store
        cal = get_calendar_store()
        trade_date = cal.get_current_trade_date(
            security, dt)
        return self._today <= trade_date

    def get_bar_by_period(self, security, start_dt, end_dt, fields, include_now=True):
        """
        获取一支股票的分钟数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `start_dt`: 开始时间, 例如 datetime.datetime(2015, 1, 1, 0, 0, 0)。
        `end_dt`: 结束时间，例如 datetime.datetime(2016, 12, 30, 0, 0, 0)。
        `fields`: 行情数据字段。
        """
        if self._should_access_shm(security, end_dt):
            b = self._shm_store.get_bar_by_period(
                security, start_dt, end_dt, fields, include_now=include_now)
            if len(b['date']) > 0:
                nw_end_dt = to_datetime(b['date'][0]) - datetime.timedelta(minutes=1)
                include_now = True
            else:
                nw_end_dt = end_dt
                include_now = include_now
            a = self._bcolz_store.get_bar_by_period(
                security, start_dt, nw_end_dt, fields, include_now=include_now)
            ret = {k: np.concatenate((a[k], b[k])) for k in a}
            return ret
        else:
            return self._bcolz_store.get_bar_by_period(
                security, start_dt, end_dt, fields, include_now=include_now)

    def get_bar_by_count(self, security, end_dt, count, fields, include_now=True):
        """
        获取一支股票的分钟数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `end_dt`: 结束日期，例如 datetime.date(2016, 12, 30, 0, 0, 0)。
        `count`: 记录条数。
        `fields`: 行情数据字段。
        """
        if self._should_access_shm(security, end_dt):
            b = self._shm_store.get_bar_by_count(
                security, end_dt, count, fields, include_now=include_now)
            if len(b['date']) > 0:
                nw_end_dt = to_datetime(b['date'][0]) - datetime.timedelta(minutes=1)
                include_now = True
            else:
                nw_end_dt = end_dt
                include_now = include_now
            a = self._bcolz_store.get_bar_by_count(
                security, nw_end_dt, count - len(b['date']), fields)
            ret = {k: np.concatenate((a[k], b[k])) for k in a}
            return ret
        else:
            return self._bcolz_store.get_bar_by_count(
                security, end_dt, count, fields, include_now=include_now)

    def get_bar_by_dt(self, security, somedt):
        u"""
        获取一支股票的分钟数据.

        `security`: 股票代码，例如 '000001.XSHE'。
        `dt`: 如果dt 分钟不存在数据，返回上一个交易分钟的数据。
        """
        if self._should_access_shm(security, somedt):
            bar = self._shm_store.get_bar_by_dt(security, somedt)
            if bar:
                return bar
        return self._bcolz_store.get_bar_by_dt(security, somedt)


    def get_minute_by_period(self, security, start_dt, end_dt, include_now=True):
        if self._should_access_shm(security, end_dt):
            b = self._shm_store.get_minute_by_period(
                security, start_dt, end_dt, include_now=include_now)
            if len(b) > 0:
                nw_end_dt = to_datetime(b[0]) - datetime.timedelta(minutes=1)
                include_now = True
            else:
                nw_end_dt = end_dt
                include_now = include_now

            a = self._bcolz_store.get_minute_by_period(
                security, start_dt, nw_end_dt, include_now=include_now)
            ret = np.concatenate((a, b))
            return ret
        else:
            return self._bcolz_store.get_minute_by_period(
                security, start_dt, end_dt, include_now=include_now)

    def get_minute_by_count(self, security, end_dt, count, include_now=True):
        if self._should_access_shm(security, end_dt):
            b = self._shm_store.get_minute_by_count(
                security, end_dt, count, include_now=include_now)
            if len(b) > 0:
                nw_end_dt = to_datetime(b[0]) - datetime.timedelta(minutes=1)
                include_now = True
            else:
                nw_end_dt = end_dt
                include_now = include_now
            a = self._bcolz_store.get_minute_by_count(
                security, nw_end_dt, count - len(b), include_now=include_now)
            ret = np.concatenate((a, b))
            return ret
        else:
            return self._bcolz_store.get_minute_by_count(
                security, end_dt, count, include_now=include_now)


@lru_cache(None)
def get_shm_day_store():
    cfg = jqdata.get_config()
    return ShmDayStore(cfg)


@lru_cache(None)
def get_shm_minute_store():
    cfg = jqdata.get_config()
    return ShmMinuteStore(cfg)


@lru_cache(None)
def get_mixed_minute_store():
    cfg = jqdata.get_config()
    return MixedMinuteStore(cfg)
