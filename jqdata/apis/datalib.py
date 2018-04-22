#!/usr/bin/env python
#coding:utf-8

'''
取bcolz数据相关接口
'''
import sys
if sys.version_info[0] == 3:
    xrange = range
    
import math
import datetime
import numpy as np

from fastcache import clru_cache as lru_cache

from jqdata.stores.calendar_store import get_calendar_store, CalendarStore
from jqdata.utils.datetime_utils import to_timestamp

from .bar_port import get_minute_bar_by_count, \
    get_minute_bar_by_period, \
    get_daily_bar_by_count, \
    get_daily_bar_by_period, \
    get_factor_by_date, \
    get_date_by_count, \
    get_date_by_period, \
    get_minute_by_count, \
    get_minute_by_period


__all__ = [
    'get_price_daily_single',
    'get_price_minute_single',
]

def copy_if_readonly(a):
    return a if a.flags.writeable else np.copy(a)

nan = float('nan')

def replace_list(arr, from_, to):
    return [x if x != from_ else to for x in arr]

EMPTY_ARRAY = np.empty((0,))

def paused_day_array(pre_close, factor, cols):
    arr = []
    for col in cols:
        if col in ['open', 'close', 'high', 'low', 'avg', 'high_limit', 'low_limit', 'pre_close']:
            arr.append(pre_close)
        elif col in ['volume', 'money']:
            arr.append(0.0)
        elif col in ['paused']:
            arr.append(1.0)
        elif col in ['factor']:
            arr.append(factor)
    return np.array(arr, dtype=float)


def fill_paused(security, cols_dict, index, full_index, fields, index_type='date'):
    '''
    index 是 a的索引
    full_index 是完整的交易索引。

    index_type 表示 dates 和 trade_dates的类型， 'date' 表示日期， 'minute' 表示 分钟
    '''
    # 获取最大有效数据日期, 超过此日期数据为nan
    end_date = security.end_date
    if index_type == 'minute':
        t0 = datetime.datetime.now().replace(second=0, microsecond=0)
        if end_date:
            t1 = datetime.datetime.combine(end_date, datetime.time(14, 59))
            max_valid_date = min(t0, t1)
        else:
            max_valid_date = t0
    elif index_type == 'date':
        t0 = datetime.date.today()
        if end_date:
            max_valid_date = min(t0, end_date)
        else:
            max_valid_date = t0

    else:
        raise Exception("wrong index_type=%s, should be 'date' or 'minute'" % index_type)
    max_valid_ts = to_timestamp(max_valid_date)
    v = np.searchsorted(index, full_index, 'right') - 1
    a = np.column_stack((cols_dict[f] for f in fields))
    nan_const = np.full(a.shape[1], nan)
    b = []
    for i, vi in enumerate(v):
        if vi < 0:
            b.append(nan_const) # 未上市
        else:
            index_ts = index[vi]
            full_index_ts = full_index[i]
            if full_index_ts != index_ts:
                if full_index_ts > max_valid_ts:
                    b.append(nan_const)
                else:
                    # 必须有close字段
                    close_value = cols_dict['close'][vi]
                    if 'factor' in cols_dict:
                        factor_value = cols_dict['factor'][vi]
                    else:
                        factor_value = 1.0
                    b.append(paused_day_array(close_value, factor_value, fields))
            else:
                b.append(a[vi])
    new_a = np.array(b).reshape(len(b), a.shape[1])
    for i, col in enumerate(fields):
        cols_dict[col] = new_a[:,i]
    return cols_dict


def fetch_daily_data(security, end_date, fields, start_date=None, count=None, skip_paused=True, fq=None, include_now=True):
    # 需要返回的行数
    return_count = None
    if skip_paused:
        if count:
            return_count = count
            cols_dict = get_daily_bar_by_count(security, end_date, count, fields, include_now=include_now)
            dates = cols_dict.pop('date', EMPTY_ARRAY)
        else:
            cols_dict = get_daily_bar_by_period(security, start_date, end_date, fields, include_now=include_now)
            dates = cols_dict.pop('date', EMPTY_ARRAY)
            return_count = len(dates)
        if len(dates) == 0:#  and return_count == 0:
            cols_dict = {col:EMPTY_ARRAY for col in fields}
            return cols_dict, EMPTY_ARRAY
    else:
        calendar = get_calendar_store()
        valid_trade_days = calendar.get_trade_days_by_enddt(security, end_date, start_date=start_date,
                                                            count=count, include_now=include_now)
        if count:
            # assert len(valid_trade_days) <= count
            return_count = count
        else:
            return_count = len(valid_trade_days)
        if return_count <= 0:
            cols_dict = {col:EMPTY_ARRAY for col in fields}
            return cols_dict, EMPTY_ARRAY

        max_count = len(valid_trade_days) + 1
        # 需要上一个交易日的close数据填充
        if 'close' not in fields:
            cols_dict = get_daily_bar_by_count(security, end_date, max_count, fields+['close'], include_now=include_now)
        else:
            cols_dict = get_daily_bar_by_count(security, end_date, max_count, fields, include_now=include_now)
        dates = cols_dict.pop('date', [])
        if len(dates) == 0:
            cols_dict = {col:np.full(len(valid_trade_days), nan) for col in fields}
            return cols_dict, valid_trade_days

        mask = np.in1d(valid_trade_days, dates, assume_unique=True)
        # 发生了停牌
        if not np.all(mask):
            cols_dict = fill_paused(security, cols_dict, dates, valid_trade_days, fields)
            dates = valid_trade_days
        else:
            start = np.searchsorted(dates, valid_trade_days[0])
            if start > 0:
                dates = dates[start:]
                for col in cols_dict:
                    cols_dict[col] = cols_dict[col][start:]
    return cols_dict, dates


def calc_daily_indexs(security, end_date, start_date=None, count=None, skip_paused=True, include_now=True):
    '''
    计算日期索引
    '''
    if skip_paused:
        if count:
            dates = get_date_by_count(security, end_date, count)
            assert len(dates) <= count
            if len(dates) == count:
                return dates
            # 补全
            calendar = get_calendar_store()
            miss = count - len(dates)
            prev_dates = [(calendar.first_date_timestamp - i*86400) for i in xrange(miss, 0, -1)]
            dates = np.concatenate((prev_dates, dates))
            return dates
        else:
            dates = get_date_by_period(security, start_date, end_date)
            return dates
    else:
        calendar = get_calendar_store()
        dates = calendar.get_trade_days_by_enddt(security, end_date, start_date=start_date, count=count)
        if count and len(dates) < count:
            miss = count - len(dates)
            prev_dates = [(calendar.first_date_timestamp - i * 86400) for i in xrange(miss, 0, -1)]
            dates = np.concatenate((prev_dates, dates))
            return dates
        else:
            return dates


def fetch_minute_data(security, end_dt, fields, start_dt=None, count=None, skip_paused=True, fq=None, include_now=True):
    # 需要返回的行数
    return_count = 0

    if skip_paused:
        if count:
            return_count = count
            cols_dict = get_minute_bar_by_count(security, end_dt, count, fields, include_now=include_now)
            minutes = cols_dict.pop('date', EMPTY_ARRAY)
            if len(minutes) < count:
                miss = return_count - len(minutes)
                prev = np.full(miss, nan)
                for f in fields:
                    cols_dict[f] = np.concatenate((prev, cols_dict[f]))
                calendar = get_calendar_store()
                prev_minutes = [(calendar.first_date_timestamp - i * 60) for i in xrange(miss, 0, -1)]
                minutes = np.concatenate((prev_minutes, minutes))
        else:
            cols_dict = get_minute_bar_by_period(security, start_dt, end_dt, fields, include_now=include_now)
            minutes = cols_dict.pop('date', EMPTY_ARRAY)
            return_count = len(minutes)

        if len(minutes) == 0 and return_count <= 0:
            cols_dict = {col:EMPTY_ARRAY for col in fields}
            return cols_dict, EMPTY_ARRAY

    else:
        calendar = get_calendar_store()
        valid_trade_minutes = calendar.get_trade_minutes_by_enddt(security, end_dt, start_dt=start_dt, count=count,
                                                                  include_now=include_now)
        if count:
            # assert len(valid_trade_minutes) == count
            return_count = count
        else:
            return_count = len(valid_trade_minutes)

        if return_count <= 0:
            cols_dict = {col:EMPTY_ARRAY for col in fields}
            return cols_dict, EMPTY_ARRAY
        # 需要上一个交易日的close数据填充
        if 'close' not in fields:
            cols_dict = get_minute_bar_by_count(security, end_dt, return_count + 1, fields+['close'], include_now=include_now)
        else:
            cols_dict = get_minute_bar_by_count(security, end_dt, return_count + 1, fields, include_now=include_now)
        minutes = cols_dict.pop('date', EMPTY_ARRAY)
        if len(minutes) == 0:
            cols_dict = {col:np.full(return_count, nan) for col in fields}
            return cols_dict, valid_trade_minutes

        mask = np.in1d(valid_trade_minutes, minutes, assume_unique=True)
        if not np.all(mask):
            cols_dict = fill_paused(security, cols_dict, minutes, valid_trade_minutes, fields)
            minutes = valid_trade_minutes
        else:
            start = np.searchsorted(minutes, valid_trade_minutes[0])
            if start > 0:
                minutes = minutes[start:]
                for col in cols_dict:
                    cols_dict[col] = cols_dict[col][start:]

    return cols_dict, minutes


def calc_minute_indexs(security, end_dt, start_dt=None, count=None, skip_paused=True, include_now=True):
    '''
    计算分钟索引: 返回 numpy.ndarray, 每一项为时间戳
    '''
    if skip_paused:
        if count:
            minutes = get_minute_by_count(security, end_dt, count, include_now=include_now)
            assert len(minutes) <= count
            if len(minutes) == count:
                return minutes
            # 补全
            miss = count - len(minutes)
            calendar = get_calendar_store()
            prev_minutes = [(calendar.first_date_timestamp - i*60) for i in xrange(miss, 0, -1)]
            minutes = np.concatenate((prev_minutes, minutes))
            return minutes
        else:
            minutes = get_minute_by_period(security, start_dt, end_dt, include_now=include_now)
            return minutes
    else:
        calendar = get_calendar_store()
        minutes = calendar.get_trade_minutes_by_enddt(security, end_dt, start_dt=start_dt, count=count, include_now=include_now)
        if count and len(minutes) < count:
            miss = count - len(minutes)
            prev_minutes = [(calendar.first_date_timestamp - i*60) for i in xrange(miss, 0, -1)]
            minutes = np.concatenate((prev_minutes, minutes))
            return minutes
        else:
            return minutes


# 取天数据的核心函数, 取单只股票的天数据
def get_price_daily_single(security, end_date=None, fields=None, start_date=None, count=None,
                           skip_paused=True, fq=None, include_now=True, pre_factor_ref_date=None):
    fields = list(fields)
    if 'price' in fields:
        fields = replace_list(fields, 'price', 'avg')

    fields = list(set(fields))  # 去掉重复的列
    if fq and 'factor' not in fields:
        fields.append('factor')
    if not count:
        assert start_date is not None

    cols_dict, dates = fetch_daily_data(security, end_date, fields,
                                        start_date=start_date,
                                        count=count,
                                        skip_paused=skip_paused,
                                        fq=fq,
                                        include_now=include_now)

    if fq:
        factors = cols_dict['factor']
        if fq == 'pre':
            assert pre_factor_ref_date, '请设置前复权基准日期'
            f = get_factor_by_date(security, pre_factor_ref_date)
            if f and abs(f - 1.0) >= 1e-6:
                factors = copy_if_readonly(factors) / f
        price_decimals = security.price_decimals
        for f in fields:
            if f in ['open', 'close', 'high', 'low', 'price', 'avg', 'pre_close', 'high_limit', 'low_limit']:
                cols_dict[f] *= factors
                np.round(cols_dict[f], price_decimals, cols_dict[f])
            elif f in ['volume']:
                cols_dict[f] /= factors
                np.round(cols_dict[f], 0, cols_dict[f])

    if 'factor' in fields:
        if fq:
            cols_dict['factor'] = factors
        else:
            cols_dict['factor'] = np.full(len(dates), 1.0)

    # 需要返回的行数
    return_dates = calc_daily_indexs(security, end_date,
                                     start_date=start_date, count=count,
                                     skip_paused=skip_paused, include_now=include_now)
    return_count = len(return_dates)
    if len(dates) < return_count:
        # pad nan
        miss = return_count - len(dates)
        prev = np.full(miss, nan)
        for f in fields:
            cols_dict[f] = np.concatenate((prev, cols_dict[f]))
        calendar = get_calendar_store()
        prev_dates = [(calendar.first_date_timestamp - i*86400) for i in xrange(miss, 0, -1)]
        dates = np.concatenate((prev_dates, dates))

    return cols_dict, dates

def get_price_minute_single(security, end_dt=None, fields=None, start_dt=None, count=None,
                            skip_paused=True, fq=None, include_now=True, pre_factor_ref_date=None):
    fields = list(fields)
    if 'price' in fields:
        fields = replace_list(fields, 'price', 'avg')

    if fq and 'factor' not in fields:
        fields.append('factor')
    fields = list(set(fields))  # 去掉重复的列

    minute_fields = [f for f in fields if f in security.minute_column_names]
    day_fields = [f for f in fields if f not in security.minute_column_names]
    if minute_fields:
        minute_a, minute_index = fetch_minute_data(security, end_dt, minute_fields,
                                                   start_dt=start_dt, count=count,
                                                   skip_paused=skip_paused, fq=fq,
                                                   include_now=include_now)
    else:
        minute_index = calc_minute_indexs(security, end_dt, start_dt=start_dt,
                                          count=count, skip_paused=skip_paused,
                                          include_now=include_now)
        minute_a = {}
    if day_fields:
        # 是否包括今天的数据
        if include_now:
            include_day = True
        elif not CalendarStore.instance().is_first_minute(security, end_dt): # 不是第一分钟
            include_day = True
        else:
            include_day = False
        day_a, day_index = fetch_daily_data(security, end_dt.date(), day_fields,
                                            start_date=start_dt.date() if start_dt else None,
                                            count=count,
                                            skip_paused=skip_paused, fq=fq,
                                            include_now=include_day)
        if len(minute_index) > 0 and len(day_index) > 0:
            # 将分钟时间戳转换成当天时间戳
            minute_dates = np.vectorize(lambda ts: int(ts/86400) * 86400)(minute_index)
            row_idxs = np.searchsorted(day_index, minute_dates, 'right')
            # 在最前面插入第一项。
            for f in day_fields:
                day_a[f] = np.insert(day_a[f], 0, nan)[row_idxs]
        if 'close' in day_a:
            day_a.pop('close')
    else:
        day_a = {}
        day_index = EMPTY_ARRAY

    if len(minute_index) == 0:
        cols_dict = {col:EMPTY_ARRAY for col in fields}
        return cols_dict, EMPTY_ARRAY

    if len(minute_index) > 0 and len(day_index) > 0:
        minute_a.update(day_a)
        cols_dict = minute_a
        fields = minute_fields + day_fields
    elif len(minute_index) > 0:
        cols_dict = minute_a
        fields = minute_fields
    elif len(day_index) > 0:
        cols_dict = day_a
        fields = day_fields

    if fq:
        factors = cols_dict['factor']
        if fq == 'pre':
            assert pre_factor_ref_date, '请设置请复权基准日期'
            f = get_factor_by_date(security, pre_factor_ref_date)
            if abs(f-1) > 1e-6:
                factors = factors / f
        price_decimals = security.price_decimals
        for f in fields:
            if f in ['open', 'close', 'high', 'low', 'price', 'avg', 'pre_close', 'high_limit', 'low_limit']:
                cols_dict[f] *= factors
                np.round(cols_dict[f], price_decimals, cols_dict[f])
            elif f in ['volume']:
                cols_dict[f] = cols_dict[f]/factors
                np.round(cols_dict[f], 0, cols_dict[f])

    if 'factor' in fields:
        if fq:
            cols_dict['factor'] = factors
        else:
            cols_dict['factor'] = np.full(len(minute_index), 1.0)

    return cols_dict, minute_index



