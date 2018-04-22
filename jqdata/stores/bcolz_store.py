#!/usr/bin/env python
#coding:utf-8

import os
import math
import time
from datetime import datetime, date, timedelta, time
from fastcache import clru_cache as lru_cache
import pandas as pd
import numpy as np
import bcolz

import jqdata
from jqdata.utils.utils import fixed_round
from jqdata.utils.datetime_utils import to_timestamp, to_date, vec2date, to_datetime, vec2trimtime
from jqdata.exceptions import ParamsError
from .bcolz_utils import _Table, retry_bcolz_open

__all__ = [
    'get_bcolz_day_store',
    'get_bcolz_minute_store',
    'BcolzDayStore',
    'BcolzMinuteStore',
]

MINUTE_COLUMNS =  tuple('open close high low volume money avg factor'.split())
_COL_POWERS = {col: 10000.0 if col in ('open', 'close', 'high', 'low', 'avg')\
               else 1.0 for col in MINUTE_COLUMNS}
_EMPTY_NP_ARRAY = np.array([])

class BcolzDayStore(object):
    '''bcolz 日行情数据存储'''

    def __init__(self):
        pass

    def __str__(self):
        return 'BcolzDS'

    @staticmethod
    def instance():
        if not hasattr(BcolzDayStore, "_instance"):
            BcolzDayStore._instance = BcolzDayStore()
        return BcolzDayStore._instance

    @staticmethod
    def set_cache():
        cls = BcolzDayStore
        funcs = ['open_bcolz_carray', 'open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if not hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, lru_cache(None)(getattr(cls, func_name)))

    @staticmethod
    def unset_cache():
        cls = BcolzDayStore
        funcs = ['open_bcolz_carray', 'open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, getattr(func_cls, "__wrapped__"))

    @staticmethod
    def clear_cache():
        cls = BcolzDayStore
        funcs = ['open_bcolz_carray', 'open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "cache_clear"):
                func_cls.cache_clear()

    @lru_cache(None)
    def open_bcolz_carray(self, security, col):
        p = jqdata.get_config().get_bcolz_day_path(security)
        return retry_bcolz_open(os.path.join(p, col))


    @lru_cache(None)
    def open_bcolz_table(self, security):
        if security.is_open_fund():
            p = jqdata.get_config().get_bcolz_otcfund_path(security)
        else:
            p = jqdata.get_config().get_bcolz_day_path(security)
        ct = retry_bcolz_open(p)
        return _Table(ct, ct.cols['date'][:])


    @lru_cache(None)
    def get_trading_days(self, security):
        if security.is_open_fund():
            p = jqdata.get_config().get_bcolz_otcfund_path(security)
            cr = retry_bcolz_open(os.path.join(p, 'date'))
        else:
            cr = self.open_bcolz_carray(security, 'date')
        return cr[:]


    def have_data(self, security, date):
        # 是否停牌
        index = self.get_trading_days(security)
        ts = to_timestamp(date)
        idx = index.searchsorted(ts, side='right') - 1
        if idx >= 0 and index[idx] == ts:
            return True
        return False


    def get_factor_by_period(self, security, start_date, end_date):
        '''
        获取 security [start_date, end_date] 期间的复权因子。
        如果停牌，则返回停牌前的复权因子。
        '''
        ct = self.open_bcolz_table(security)

        start_ts = to_timestamp(start_date)
        end_ts = to_timestamp(end_date)
        start_idx = ct.index.searchsorted(start_ts)
        end_idx = ct.index.searchsorted(end_ts, side='right') - 1
        if start_idx <= end_idx:
            index = ct.table.cols['date'][start_idx:end_idx+1]
            index = vec2date(index)
            data = ct.table.cols['factor'][start_idx:end_idx+1]
            return index, data
        else:
            factor = self.get_factor_by_date(security, start_date)
            index = np.array([start_date])
            data = np.array([factor])
            return index, data


    def get_factor_by_date(self, security, date):
        '''获取 security 在 date 这一天的复权因子，不存在则返回 1.0
        '''
        ct = self.open_bcolz_table(security)
        idx = ct.index.searchsorted(to_timestamp(date), side='right') - 1
        if idx < 0:
            return 1.0
        if idx < len(ct.index):
            f = ct.table.cols['factor'][idx]
            assert f > 0, 'security=%s, date=%s, factor=%s' % (security.code, date, f)
            return f
        return 1.0


    # 获取天数据
    def get_bar_by_period(self, security, start_date, end_date, fields, include_now=True):
        '''
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `start_date`: 开始日期, 例如 datetime.date(2015, 1, 1)。
        `end_date`: 结束日期，例如 datetime.date(2016, 12, 30)。
        `fields`: 行情数据字段。
        '''
        ct = self.open_bcolz_table(security)

        start_ts = to_timestamp(start_date)
        end_ts = to_timestamp(end_date)

        start_idx = ct.index.searchsorted(start_ts)
        end_idx = ct.index.searchsorted(end_ts, 'right') - 1
        if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
            end_idx -= 1
        if end_idx < start_idx:
            return {}
        data = {name: ct.table.cols[name][start_idx:end_idx+1] for name in fields}
        data['date'] = ct.table.cols['date'][start_idx:end_idx+1]
        return data


    def get_bar_by_count(self, security, end_date, count, fields, include_now=True):
        '''
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `end_date`: 结束日期，例如 datetime.date(2016, 12, 30)。
        `count`: count条记录, 例如 300。
        `fields`: 行情数据字段。
        '''
        ct = self.open_bcolz_table(security)
        count = int(count)
        end_ts = to_timestamp(end_date)
        end_idx = ct.index.searchsorted(end_ts, side='right') - 1
        if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
            end_idx -= 1

        if end_idx < 0:
            return {}
        start_idx = end_idx - count + 1
        if start_idx < 0:
            start_idx = 0

        data = {name: ct.table.cols[name][start_idx:end_idx+1] for name in fields}
        data['date'] = ct.table.cols['date'][start_idx:end_idx+1]
        return data


    def get_date_by_period(self, security, start_date, end_date, include_now=True):
        ct = self.open_bcolz_table(security)
        start_ts = to_timestamp(start_date)
        end_ts = to_timestamp(end_date)
        start_idx = ct.index.searchsorted(start_ts)
        end_idx = ct.index.searchsorted(end_ts, side='right') - 1
        if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
            end_idx -= 1
        if start_idx <= end_idx:
            return ct.index[start_idx:end_idx+1]
        else:
            return _EMPTY_NP_ARRAY


    def get_date_by_count(self, security, end_date, count, include_now=True):
        count = int(count)
        ct = self.open_bcolz_table(security)
        end_ts = to_timestamp(end_date)
        end_idx = ct.index.searchsorted(end_ts, side='right') - 1
        if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
            end_idx -= 1
        if end_idx < 0:
            return _EMPTY_NP_ARRAY
        start_idx = end_idx - count + 1
        if start_idx < 0:
            start_idx = 0
        return ct.index[start_idx:end_idx+1]


    def get_bar_by_date(self, security, somedate):
        '''
        获取一支股票的天数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `date`: 如果date天不存在数据，返回上一个交易日的数据。
        '''
        ct = self.open_bcolz_table(security)
        ts = to_timestamp(somedate)
        idx = ct.index.searchsorted(ts, side='right') - 1
        if idx < 0:
            return None
        price_decimals = security.price_decimals
        bar = {}
        for name in ct.table.names:
            if name in ['open', 'close', 'high', 'low', 'price', 'avg', 'pre_close', 'high_limit', 'low_limit']:
                bar[name] = fixed_round(ct.table.cols[name][idx], price_decimals)
            elif name in ['volume', 'money']:
                bar[name] = fixed_round(ct.table.cols[name][idx], 0)
            elif name in ['unit', 'acc', 'refactor']:
                bar[name] = fixed_round(ct.table.cols[name][idx], price_decimals)
            else:
                bar[name] = ct.table.cols[name][idx]
        bar['date'] = to_date(bar['date'])
        return bar


def calc_stock_minute_index(security, date_index, dt, side='left', include_now=True):
    '''
    # 计算dt 在分钟数据中的索引（只对股票有效）

    如果 dt 不存在 date_index ，
    side=left 表示返回左边的index，
    side=right 表示返回右边的index
    -1 表示没有

    include_now 表示是否包含 dt
    '''
    if isinstance(dt, datetime):
        ts = to_timestamp(dt.date())
    elif isinstance(dt, pd.Timestamp):
        ts = int(dt.value/(10**9)/86400)*86400
    else:
        raise ParamsError("wrong dt=%s, type(dt)=%s" % (dt, type(dt)))

    total_days = date_index.searchsorted(ts, side='right') - 1
    if total_days < 0:
        return -1
    if date_index[total_days] == ts:
        trading_day = True
    else:
        trading_day = False
        total_days += 1
    # 前面一共有total_days个交易日
    dt_minutes = dt.hour * 60 + dt.minute
    total_minutes = 0
    if trading_day:
        # 9:31 之前
        if dt_minutes < (9*60+31):
            if side == 'left':
                total_minutes = 0
            else:
                total_minutes = 1
        elif dt_minutes < (11*60+31):
            total_minutes = dt_minutes - (9 * 60 + 30)
            if not include_now:
                total_minutes -= 1
        elif dt_minutes < 13*60 + 1:
            if side == 'left':
                total_minutes = 120
            else:
                total_minutes = 121
        elif dt_minutes < 15*60 + 1:
            total_minutes = 120 + dt_minutes - 13 * 60
            if not include_now:
                total_minutes -= 1
        else:
            if side == 'left':
                total_minutes = 240
            else:
                total_minutes = 241
    else:
        if side == 'left':
            total_minutes = 0
        else:
            total_minutes = 1

    # 下标从0开始。 -1 表示 没有。
    return (total_days * 240 + total_minutes) - 1


class BcolzMinuteStore(object):

    def __init__(self):
        pass

    def __str__(self):
        return 'BcolzMS'

    @staticmethod
    def instance():
        if not hasattr(BcolzMinuteStore, "_instance"):
            BcolzMinuteStore._instance = BcolzMinuteStore()
        return BcolzMinuteStore._instance

    @staticmethod
    def set_cache():
        cls = BcolzMinuteStore
        funcs = ['open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if not hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, lru_cache(None)(getattr(cls, func_name)))

    @staticmethod
    def unset_cache():
        cls = BcolzMinuteStore
        funcs = ['open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, getattr(func_cls, "__wrapped__"))

    @staticmethod
    def clear_cache():
        cls = BcolzMinuteStore
        funcs = ['open_bcolz_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "cache_clear"):
                func_cls.cache_clear()

    @lru_cache(None)
    def open_bcolz_table(self, security):
        p = jqdata.get_config().get_bcolz_minute_path(security)
        try:
            ct = retry_bcolz_open(p)
        except Exception as e:
            # 证劵刚刚上市或者还没有上市，所有没有bcolz数据。
            if security.start_date >= date.today():
                return None
            raise e

        # 股指期货缓存date列，其他的缓存天行情的date列
        if security.is_futures():
            return _Table(ct, ct.cols['date'][:])
        else:
            try:
                # index = BcolzDayStore.instance().open_bcolz_carray(security, 'date')[:]
                # attrs['date'] 表示交易日。
                index = np.array(ct.attrs['date'])
            except KeyError as e:
                # 从天bcolz数据中取date列.
                index = BcolzDayStore.instance().open_bcolz_carray(security, 'date')[:]
            return _Table(ct, index)


    def get_bar_by_period(self, security, start_dt, end_dt, fields, include_now=True):
        '''
        获取一支股票的分钟数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `start_dt`: 开始时间, 例如 datetime.datetime(2015, 1, 1, 0, 0, 0)。
        `end_dt`: 结束时间，例如 datetime.datetime(2016, 12, 30, 0, 0, 0)。
        `fields`: 行情数据字段。
        '''

        ct = self.open_bcolz_table(security)
        if ct is None:
            return {k: _EMPTY_NP_ARRAY for k in ('date',) + tuple(fields)}

        if security.is_futures():
            start_ts = to_timestamp(start_dt)
            end_ts = to_timestamp(end_dt)
            start_idx = ct.index.searchsorted(start_ts)
            end_idx = ct.index.searchsorted(end_ts, 'right') - 1
            if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
                end_idx -= 1
        else:
            start_idx = calc_stock_minute_index(security, ct.index, start_dt, side='right')
            end_idx = calc_stock_minute_index(security, ct.index, end_dt, include_now=include_now)

        if start_idx > end_idx:
            return {k: _EMPTY_NP_ARRAY for k in ('date',)+tuple(fields)}

        data = {name: ct.table.cols[name][start_idx:end_idx+1]/_COL_POWERS.get(name, 1)\
                for name in fields}
        if security.is_futures():
            data['date'] = ct.index[start_idx:end_idx+1]
        else:
            data['date'] = ct.table.cols['date'][start_idx:end_idx+1]
        return data


    def get_bar_by_count(self, security, end_dt, count, fields, include_now=True):
        '''
        获取一支股票的分钟数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `end_dt`: 结束日期，例如 datetime.date(2016, 12, 30, 0, 0, 0)。
        `count`: 记录条数。
        `fields`: 行情数据字段。
        '''
        count = int(count)
        ct = self.open_bcolz_table(security)
        if ct is None:
            return {k: _EMPTY_NP_ARRAY for k in ('date',) + tuple(fields)}
        if security.is_futures():
            end_ts = to_timestamp(end_dt)
            end_idx = ct.index.searchsorted(end_ts, 'right') - 1
            if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
                end_idx -= 1
        else:
            end_idx = calc_stock_minute_index(security, ct.index, end_dt, include_now=include_now)
        if end_idx < 0:
            return {k: _EMPTY_NP_ARRAY for k in ('date',)+tuple(fields)}
        start_idx = end_idx + 1 - count
        if start_idx < 0:
            start_idx = 0
        data = {name: ct.table.cols[name][start_idx:end_idx+1]/_COL_POWERS.get(name, 1)\
                for name in fields}
        if security.is_futures():
            data['date'] = ct.index[start_idx:end_idx+1]
        else:
            data['date'] = ct.table.cols['date'][start_idx:end_idx+1]
        return data


    def get_minute_by_period(self, security, start_dt, end_dt, include_now=True):
        ct = self.open_bcolz_table(security)
        if ct is None:
            return _EMPTY_NP_ARRAY

        if security.is_futures():
            start_ts = to_timestamp(start_dt)
            end_ts = to_timestamp(end_dt)
            start_idx = ct.index.searchsorted(start_ts)
            end_idx = ct.index.searchsorted(end_ts, 'right') - 1
            if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
                end_idx -= 1
        else:
            start_idx = calc_stock_minute_index(security, ct.index, start_dt, side='right')
            end_idx = calc_stock_minute_index(security, ct.index, end_dt, include_now=include_now)

        if start_idx > end_idx:
            return _EMPTY_NP_ARRAY

        if security.is_futures():
            return ct.index[start_idx:end_idx+1]
        else:
            return ct.table.cols['date'][start_idx:end_idx+1]


    def get_minute_by_count(self, security, end_dt, count, include_now=True):
        count = int(count)
        ct = self.open_bcolz_table(security)
        if ct is None:
            return _EMPTY_NP_ARRAY
        if security.is_futures():
            end_ts = to_timestamp(end_dt)
            end_idx = ct.index.searchsorted(end_ts, 'right') - 1
            if not include_now and end_idx >= 0 and ct.index[end_idx] == end_ts:
                end_idx -= 1
        else:
            end_idx = calc_stock_minute_index(security, ct.index, end_dt, include_now=include_now)
        if end_idx < 0:
            return _EMPTY_NP_ARRAY
        start_idx = end_idx + 1 - count
        if start_idx < 0:
            start_idx = 0
        if start_idx > end_idx:
            return _EMPTY_NP_ARRAY

        if security.is_futures():
            return ct.index[start_idx:end_idx+1]
        else:
            return ct.table.cols['date'][start_idx:end_idx+1]


    def get_bar_by_dt(self, security, somedt):
        '''
        获取一支股票的分钟数据。

        `security`: 股票代码，例如 '000001.XSHE'。
        `dt`: 如果dt 分钟不存在数据，返回上一个交易分钟的数据。
        '''

        ct = self.open_bcolz_table(security)
        if ct is None:
            return None
        if security.is_futures():
            ts = to_timestamp(somedt)
            idx = ct.index.searchsorted(ts, side='right') - 1
        else:
            idx = calc_stock_minute_index(security, ct.index, somedt)

        if idx < 0:
            return None

        price_decimals = security.price_decimals
        bar = {}
        for name in ct.table.names:
            if name in ['open', 'close', 'high', 'low', 'price', 'avg', 'pre_close', 'high_limit', 'low_limit']:
                bar[name] = fixed_round(ct.table.cols[name][idx]/_COL_POWERS.get(name, 1), price_decimals)
            elif name in ['volume', 'money']:
                bar[name] = fixed_round(ct.table.cols[name][idx]/_COL_POWERS.get(name, 1), 0)
            else:
                bar[name] = ct.table.cols[name][idx]

        bar['date'] = to_datetime(bar['date'])
        return bar


def get_bcolz_day_store():
    return BcolzDayStore.instance()


def get_bcolz_minute_store():
    return BcolzMinuteStore.instance()



