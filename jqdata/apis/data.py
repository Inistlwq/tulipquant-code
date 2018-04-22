#!/usr/bin/env python
#coding:utf-8

'''
取数据相关接口
'''

import sys

import six
import numpy as np
import pandas as pd

from fastcache import clru_cache as lru_cache

from .data_utils import *
from .datalib import get_price_daily_single
from .datalib import get_price_minute_single

from jqdata.names import DEFAULT_FIELDS
from jqdata.utils.datetime_utils import vec2datetime, vec2combine, vec2date,\
    vec2timestamp, to_timestamp, to_datetime
from jqdata.stores.calendar_store import CalendarStore
from jqdata.exceptions import ParamsError

__all__ = [
    'get_price',
    'history',
    'attribute_history',
    'get_extras',
    'get_bars',
    'get_ticks',
    'get_current_tick',
]

def _has_stock_with_future(security_list):
    #
    assert is_list(security_list)
    has_stock = False
    has_future = False
    for s in security_list:
        if s.is_stock() or s.is_index() or s.is_fund():
            has_stock = True
        elif s.is_futures():
            has_future = True
    return has_stock and has_future

def get_price(security, start_date=None, end_date=None,
              frequency='daily', fields=None, skip_paused=False,
              fq='pre', count=None, pre_factor_ref_date=None):

    security = convert_security(security)

    if count is not None and start_date is not None:
        raise ParamsError("get_price 不能同时指定 start_date 和 count 两个参数")

    if count is not None:
        count = int(count)

    end_dt = convert_dt(end_date) if end_date else datetime.datetime(2015, 12, 31)
    end_dt = min(end_dt, date2dt(CalendarStore.instance().last_day))

    start_dt = convert_dt(start_date) if start_date else datetime.datetime(2015, 1, 1)
    start_dt = max(start_dt, date2dt(CalendarStore.instance().first_day))


    if pre_factor_ref_date:
        pre_factor_ref_date = convert_date(pre_factor_ref_date)

    if frequency in frequency_compat:
        unit = frequency_compat.get(frequency)
    else:
        unit = frequency

    if fields is not None:
        fields = ensure_str_tuple(fields)
        if 'price' in fields:
            warn_price_as_avg('使用 price 作为 get_price 的 fields 参数', 'getprice')
    else:
        fields = tuple(DEFAULT_FIELDS)


    check_unit_fields(unit, fields)
    fq = ensure_fq(fq)
    skip_paused = bool(skip_paused)
    if is_list(security) and skip_paused:
        raise ParamsError("get_price 取多只股票数据时, 为了对齐日期, 不能跳过停牌")

    if is_list(security) and _has_stock_with_future(security):
        if unit.endswith('m'):
           raise ParamsError("get_price 取分钟数据时，为了对齐数据，不能同时取股票和期货。")


    group = int(unit[:-1])
    res = {}
    for s in (security if is_list(security) else [security]):
        if unit.endswith('d'):
            a, index = get_price_daily_single(
                s,
                end_date=end_dt.date(),
                start_date=start_dt.date() if start_dt else None,
                count=count * group if count is not None else None,
                fields=fields,
                skip_paused=skip_paused,
                fq=fq,
                include_now=True,
                pre_factor_ref_date=pre_factor_ref_date)
        else:
            a, index = get_price_minute_single(
                s,
                end_dt=end_dt,
                start_dt=start_dt,
                count=count * group if count is not None else None,
                fields=fields,
                skip_paused=skip_paused,
                fq=fq,
                include_now=True,
                pre_factor_ref_date=pre_factor_ref_date)

        # group it
        dict_by_column = {f: group_array(a[f if f != 'price' else 'avg'], group, f) for f in fields}
        if index is not None and len(index) > 0:
            index = group_array(index, group, 'index')
            index = vec2datetime(index)
        res[s.code] = dict(index=index, columns=fields, data=dict_by_column)


    if is_list(security):
        fields = fields or DEFAULT_FIELDS
        if len(security) == 0:
            return pd.Panel(items=fields)
        pn_dict = {}
        index = res[security[0].code]['index']
        for f in fields:
            df_dict = {s.code:res[s.code]['data'][f] for s in security}
            pn_dict[f] = pd.DataFrame(index=index, columns=[s.code for s in security], data=df_dict)
        return pd.Panel(pn_dict)
    else:
        return pd.DataFrame(**res[security.code])


def history(end_dt, count, unit='1d', field='avg', security_list=None,
            df=True, skip_paused=False, fq='pre', pre_factor_ref_date=None):
    '''
    只能在回测/模拟中调用，不能再研究中调用。

    参数说明：

    `unit`是'Xd'时, end_dt的类型是datetime.date;
    `unit`是'Xm'时, end_dt的类型是datetime.datetime;

    `count` 必须大于0；

    `security_list`： 必须是 str 或者 tuple，不能是list 否则 lru_cache 会出错。
    '''
    count = int(count)
    assert count > 0 , "history, count必须是一个正整数"
    check_unit_fields(unit, (field, ))
    if security_list is not None:
        security_list = list_or_str(security_list)
    if isinstance(security_list, tuple):
        security_list = list(security_list)
    security_list = convert_security(security_list)

    if field == 'price':
        warn_price_as_avg('使用 price 作为 history 的 field 参数', 'history')
        field = 'avg'

    group = int(unit[:-1])
    total = count * group
    dict_by_column = {}

    _index = None
    df = bool(df)
    skip_paused = bool(skip_paused)
    fq = ensure_fq(fq)
    if pre_factor_ref_date is not None:
        pre_factor_ref_date=convert_date(pre_factor_ref_date)
    need_index = df and not skip_paused

    if is_list(security_list) and _has_stock_with_future(security_list):
        if unit.endswith('m') and need_index:
            raise ParamsError("history 取分钟数据时，为了对齐数据，不能同时取股票和期货。")

    if unit.endswith('d'):
        end_dt = convert_date(end_dt)
        for security in security_list:
            a, _index = get_price_daily_single(
                security,
                end_date=end_dt,
                count=total,
                fields=(field,),
                skip_paused=skip_paused,
                fq=fq,
                include_now=False,
                pre_factor_ref_date=pre_factor_ref_date)

            a = a[field]
            a = group_array(a, group, field)
            dict_by_column[security.code] = a
    else:
        end_dt = convert_dt(end_dt)
        for security in security_list:
            a, _index = get_price_minute_single(
                security,
                end_dt=end_dt,
                count=total,
                fields=(field,),
                skip_paused=skip_paused,
                fq=fq,
                include_now=False,
                pre_factor_ref_date=pre_factor_ref_date)

            # 取第一列
            a = a[field]
            a = group_array(a, group, field)
            dict_by_column[security.code] = a

    if not df:
        return dict_by_column
    else:
        if need_index and _index is not None and len(_index) > 0:
            index = group_array(_index, group, 'index')
            index = vec2datetime(index)
        else:
            index = None
        return pd.DataFrame(index=index,
                            columns=[s.code for s in security_list],
                            data=dict_by_column)


def attribute_history(end_dt, security, count, unit='1d',
                      fields=tuple(DEFAULT_FIELDS),
                      skip_paused=True,
                      df=True,
                      fq='pre',
                      pre_factor_ref_date=None):
    '''
    只能在回测/模拟中调用，不能再研究中调用。
    参数说明：

    `unit`是'Xd'时, end_dt的类型是datetime.date;
    `unit`是'Xm'时, end_dt的类型是datetime.datetime;

    `count` 必须大于0；
    '''
    count = int(count)
    assert count > 0, "attribute_history, count必须是一个正整数"
    fields = ensure_str_tuple(fields)
    check_unit_fields(unit, fields)
    security = convert_security(security)
    if 'price' in fields:
        warn_price_as_avg('使用 price 作为 attribute_history 的 fields 参数', 'attributehistory')

    group = int(unit[:-1])
    total = int(count * group)
    skip_paused = bool(skip_paused)
    df = bool(df)
    fq = ensure_fq(fq)
    if pre_factor_ref_date is not None:
        pre_factor_ref_date = convert_date(pre_factor_ref_date)
    if unit.endswith('d'):
        end_dt = convert_date(end_dt)
        a, index = get_price_daily_single(
            security,
            end_date=end_dt,
            count=total,
            fields=fields,
            skip_paused=skip_paused,
            fq=fq,
            include_now=False,
            pre_factor_ref_date=pre_factor_ref_date)

    else:
        end_dt = convert_dt(end_dt)
        a, index = get_price_minute_single(
            security,
            end_dt=end_dt,
            count=total,
            fields=fields,
            skip_paused=skip_paused,
            fq=fq,
            include_now=False,
            pre_factor_ref_date=pre_factor_ref_date)

    dict_by_column = {f: group_array(a[f if f != 'price' else 'avg'], group, f) for f in fields}

    if not df:
        return dict_by_column
    else:
        if index is not None and len(index) > 0:
            index = group_array(index, group, 'index')
            index = vec2datetime(index)
        return pd.DataFrame(index=index, columns=fields, data=dict_by_column)


def get_extras(info, security_list, start_date=None, end_date=None, df=True, count=None):
    assert info in ('is_st', 'acc_net_value', 'unit_net_value', 'futures_sett_price', 'futures_positions')
    securities = list_or_str(security_list)
    securities = convert_security(securities)
    if start_date and count:
        raise ParamsError("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise ParamsError("count 参数需要大于 0 或者为 None")
    if count is not None:
        count = int(count)
    end_date = convert_date(end_date) if end_date else convert_date('2015-12-31')

    from jqdata.stores import FundStore, StStore, FuturesStore, CalendarStore

    if start_date:
        start_date = convert_date(start_date)
    elif count:
        ix = CalendarStore.instance().get_trade_days_between(datetime.date(2005, 1, 4), end_date)
        start_date = ix[-count]
    else:
        start_date = convert_date('2015-01-01')
    df = bool(df)
    dates = CalendarStore.instance().get_trade_days_between(start_date, end_date)
    values = {}
    if info == 'is_st':
        for s in securities:
            values[s.code] = StStore.instance().query(s, dates)
    elif info in ('acc_net_value', 'unit_net_value'):
        for s in securities:
            values[s.code] = FundStore.instance().query(s, dates, info)
    elif info in ('futures_sett_price', 'futures_positions'):
        for s in securities:
            values[s.code] = FuturesStore.instance().query(s, dates, info)
    if df:
        columns = [s.code for s in securities]
        ret = dict(index=vec2combine(dates), columns=columns, data=values)
        ret = pd.DataFrame(**ret)
        return ret
    else:
        return values


from .bar_port import get_daily_bar_by_count, get_factor_by_date,\
    get_minute_bar_by_count, get_minute_bar_by_period

def copy_if_readonly(a):
    return a if a.flags.writeable else np.copy(a)


FIELD_AGG_FUNCTIONS = {
    'date': 'last',
    'open': 'first',
    'close': 'last',
    'high': np.maximum,
    'low': np.minimum,
    'money': np.add,
    'volume': np.add,
    'factor': 'last',
}

FIELD_SNAP_FUNCTIONS = {
    'date': 'last',
    'open': 'first',
    'close': 'last',
    'high': np.max,
    'low': np.min,
    'money': np.sum,
    'volume': np.sum,
    'factor': 'last',
}

def get_snapshot(security, trade_date, end_dt):
    '''
    获取在一天中的某一分钟看到的 day bar快照，相当于所有当天到end_dt所有分钟bar合并。

    :param security:
    :param end_dt:
    :return: 返回 {} 或者 {k1:v1, k2:v2, .... }
    '''
    # 先实现，再优化。
    fields = (u'date', u'open', u'close', u'high', u'low', u'volume', u'money', u'avg', u'factor')
    if trade_date < security.start_date:
        return {}
    start_dt = CalendarStore.instance().get_open_dt(security, trade_date)
    cols_dict = get_minute_bar_by_period(security, start_dt, end_dt, fields, include_now=True)
    if not cols_dict or len(cols_dict['date']) == 0:
        return {}
    bar = {}
    for f in cols_dict:
        how = FIELD_SNAP_FUNCTIONS.get(f, None)
        if how is None:
            continue
        if how == 'last':
            bar[f] = cols_dict[f][-1]
        elif how == 'first':
            bar[f] = cols_dict[f][0]
        else:
            bar[f] = how(cols_dict[f])
    return bar


def _resample_simple_xm_bars(bars, resample):
    '''
    合并股票(指数，基金)分钟bar, bars除了最后一个其他必须是整齐的。

    :param bars: bars 不能为{}。
    :param resample: (5, 15, 30, 60, 120)。
    :return: 返回合并后的{'open': array[...], 'close': array[...]}
    '''
    assert resample in (5, 15, 30, 60, 120)
    bar_len = len(bars['date'])
    result_len = int(bar_len//resample) + (1 if bar_len % resample else 0)
    result = {f: np.zeros(result_len, dtype=bars[f].dtype) for f in bars}
    if result_len == 0:
        return result

    for f in bars:
        how = FIELD_AGG_FUNCTIONS.get(f)
        if how == 'last':
            result[f][:-1] = bars[f][resample-1:-1:resample]
            result[f][-1] = bars[f][-1]
        elif how == 'first':
            result[f] = bars[f][::resample]
        else:
            result[f] = how.reduceat(bars[f], list(range(0, len(bars[f]), resample)))
    return result

def _to_minute_positions(tsarr):
    '''
    计算分钟dt在一天的交易位置。
    
    :param tsarr: 
    :return: 
    '''
    tsarr = (tsarr - tsarr[0])//60
    tsarr = (tsarr % (24*60))
    return tsarr

def _unique_sorted(arr):
    '''
    去掉重复的。
    :param arr: 
    :return: 
    '''
    res = []
    for i in range(len(arr)-1):
        if arr[i] != arr[i+1]:
            res.append(arr[i])
    res.append(arr[-1])
    return np.array(res)

def _resample_future_xm_bars(bars, resample):
    assert resample in (5, 15, 30, 60, 120)

    bar_len = len(bars['date'])
    minutes = _to_minute_positions(bars['date'])
    # 时间按resample分钟划分刻度。
    time_series = np.arange(0, minutes[-1] + 1, resample)
    #
    indexes = minutes.searchsorted(time_series)
    if len(indexes) == 0:
        return {f:np.zeros(0, dtype=bars[f].dtype) for f in bars}

    indexes = _unique_sorted(minutes.searchsorted(time_series))
    result_len = len(indexes)
    result = {}
    for f in bars:
        result[f] = np.zeros(result_len, dtype=bars[f].dtype)

    for f in bars:
        how = FIELD_AGG_FUNCTIONS[f]
        if how == 'last':
            result[f][:-1] = bars[f][indexes[1:] - 1]
            result[f][-1] = bars[f][-1]
        elif how == 'first':
            result[f] = bars[f][indexes]
        else:
            result[f] = how.reduceat(bars[f], indexes)

    return result

def _is_same_week(day1, day2):
    a = day1.isocalendar()
    b = day2.isocalendar()
    if a[0] == b[0] and a[1] == b[1]:
        return True
    return False

def _is_same_month(d1, d2):
    return d1.year == d2.year and d1.month == d2.month

def _resample_days_bars(bars, unit):
    '''
    将day bar合并成周线和月线。
    :param bars: 
    :param unit: 
    :return: 
    '''
    assert unit in ('1w', '1M')
    cmpfunc = _is_same_week if unit == '1w' else _is_same_month

    if len(bars['date']) == 0:
        return {f:np.zeros(0, dtype=bars[f].dtype) for f in bars}

    dates = vec2datetime(bars['date'])
    indexes = []
    n = len(dates)
    i = j = 0
    while j < n:
        same_unit = cmpfunc(dates[i], dates[j])
        # print(dates[i], dates[j], same_unit)
        if not same_unit:
            indexes.append(i)  # append last day in week
            i = j
        j += 1
    if j == n:
        indexes.append(i)
    indexes = np.array(indexes)
    result_len = len(indexes)
    result = {f: np.zeros(result_len, dtype=bars[f].dtype) for f in bars}
    # print(indexes)
    for f in bars:
        how = FIELD_AGG_FUNCTIONS[f]
        if how == 'last':
            result[f][:-1] = bars[f][indexes[1:] - 1]
            result[f][-1] = bars[f][-1]
        elif how == 'first':
            result[f] = bars[f][indexes]
        else:
            result[f] = how.reduceat(bars[f], indexes)

    return result

def _not_include_now(security, trade_date, end_dt, unit):
    assert unit in ('5m', '15m', '30m', '60m', '120m')
    x = int(unit[:-1])
    trade_minutes = CalendarStore.instance().get_trade_minutes(security, trade_date)
    # 非交易日，直接返回。
    if len(trade_minutes) == 0:
        return end_dt

    if security.is_futures():
        minutes = _to_minute_positions(vec2timestamp(trade_minutes))
        # 时间按resample分钟划分刻度。
        time_series = np.arange(0, minutes[-1] + 1, x)
        indexes = _unique_sorted(minutes.searchsorted(time_series))
        arr = trade_minutes[indexes]
        i = arr.searchsorted(end_dt)
        if i > 0:
            return arr[i-1]
        return arr[0] - datetime.timedelta(minutes=x)
    else:
        arr = trade_minutes[x-1::x]
        i = arr.searchsorted(end_dt)
        if i > 0:
            return arr[i-1]
        return arr[0] - datetime.timedelta(minutes=x)

def _pre_fq(security, cols_dict, pre_factor_ref_date=None):
    # 期货没有复权。
    # 周线和月线必须先复权
    if not security.is_futures():
        if pre_factor_ref_date is not None:
            pre_factor_ref_date = convert_date(pre_factor_ref_date)
            f = get_factor_by_date(security, pre_factor_ref_date)
            factors = cols_dict['factor']
            if abs(f - 1.0) > 1e-6:
                factors = copy_if_readonly(factors) / f
            price_decimals = security.price_decimals
            for f in cols_dict.keys():
                if f in ['open', 'close', 'high', 'low', 'avg']:
                    cols_dict[f] *= factors
                    np.round(cols_dict[f], price_decimals, cols_dict[f])
                elif f in ['volume']:
                    cols_dict[f] /= factors
                    np.round(cols_dict[f], 0, cols_dict[f])
    return cols_dict


def get_bars(end_dt, security, count, unit='1d',
            fields=('open','high','low','close'),
            include_now=False,
            fq=None,
            pre_factor_ref_date=None):
    '''
    :param end_dt: 截止日期
    :param security: 标的
    :param count:   bar个数
    :param unit: 频率，'1d'表示1天，'xm'表示x分钟。
    :param fields:
    :param include_now:
    :param fq: 'pre'表示前复权, 'post'表示后复权, None表示真实价格。
    :param pre_factor_ref_date: 前复权基准日期，这一天的价格为真实价格。None则表示全部取真实价格。
    :return:
    '''
    valid_bar_fields = ('date', 'open', 'close', 'high', 'low', 'volume', 'money')
    if isinstance(fields, (list, tuple)):
        for f in fields:
            assert f in valid_bar_fields, "get_bars 只支持 %s 字段" % (valid_bar_fields)
        str_field = False
    elif isinstance(fields, six.string_types):
        assert fields in valid_bar_fields, "get_bars 只支持 %s 字段" % (valid_bar_fields)
        str_field = True
    else:
        raise ParamsError("fields 应该是字符串或者list")

    if str_field:
        new_fields = [fields]
    else:
        new_fields = [i for i in fields]
    if 'factor' not in fields:
        new_fields.append('factor')
    end_dt = convert_dt(end_dt)
    valid_unit = ('1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M')
    assert unit in valid_unit, 'get_bars, unit必须是 %s 中一种' % valid_unit
    count = int(count)
    assert count > 0, "get_bars, count必须是一个正整数"
    fq = ensure_fq(fq)
    security = convert_security(security)
    include_now = bool(include_now)

    end_trade_date = CalendarStore.instance().get_current_trade_date(security, end_dt)

    def ensure_not_empty(cols_dict):
        if cols_dict == {}:
            ret = {}
            for f in valid_bar_fields:
                ret[f] = np.zeros(0)
            return ret
        return cols_dict

    if unit == '1d':
        if include_now:
            # 获取当天的snapshot
            snapshot = get_snapshot(security, end_trade_date, end_dt)
            if snapshot:
                cols_dict = get_daily_bar_by_count(security, end_trade_date, count - 1 ,
                                                   new_fields,
                                                   include_now=False)
                cols_dict = ensure_not_empty(cols_dict)
                for f in cols_dict:
                    cols_dict[f] = np.append(cols_dict[f], snapshot[f])
            else:
                cols_dict = get_daily_bar_by_count(security, end_trade_date, count,
                                                   new_fields,
                                                   include_now=False)
                cols_dict = ensure_not_empty(cols_dict)
        else:
            cols_dict = get_daily_bar_by_count(security, end_trade_date, count,
                                                new_fields,
                                                include_now=False)
            cols_dict = ensure_not_empty(cols_dict)
    elif unit == '1m':
        end_dt = convert_dt(end_dt)
        cols_dict = get_minute_bar_by_count(security, end_dt, count,
                                            new_fields,
                                            include_now=include_now)
        cols_dict = ensure_not_empty(cols_dict)
    elif unit in ('5m', '15m', '30m', '60m', '120m'):
        x = int(unit[:-1])
        if security.is_futures():
            trade_days = CalendarStore.instance().get_all_trade_days(security)
            trade_days = trade_days[(trade_days >= security.start_date )&\
                                    (trade_days <= end_trade_date)&\
                                    (trade_days <= security.end_date)]
            cols_dict = {f:np.zeros(0) for f in new_fields}

            for idx in range(len(trade_days)-1, -1, -1):
                open_dt = CalendarStore.instance().get_open_dt(security, trade_days[idx])
                if trade_days[idx] == end_trade_date:
                    if not include_now:
                        close_dt = _not_include_now(security, end_trade_date, end_dt, unit)
                    else:
                        close_dt = end_dt
                else:
                    close_dt = CalendarStore.instance().get_close_dt(security, trade_days[idx])
                tmp_dict = get_minute_bar_by_period(security, open_dt, close_dt, new_fields, include_now=True)
                if not tmp_dict or len(tmp_dict[new_fields[0]] == 0):
                    continue
                tmp_dict = _resample_future_xm_bars(tmp_dict, x)
                for col in cols_dict:
                    cols_dict[col] = np.append(tmp_dict[col], cols_dict[f])
                if len(cols_dict[new_fields[0]]) >= count:
                    break
            for f in cols_dict:
                cols_dict[f] = cols_dict[f][-count:]
        else:
            cols_dict = {f:np.zeros(0) for f in new_fields}
            open_dt = CalendarStore.instance().get_open_dt(security, end_dt.date())
            if not include_now:
                close_dt = _not_include_now(security, end_trade_date, end_dt, unit)
            else:
                close_dt = end_dt
            tmp_dict = get_minute_bar_by_period(security, open_dt, close_dt, new_fields, include_now=True)
            if tmp_dict and len(tmp_dict[new_fields[0]]) > 0:
                tmp_dict = _resample_simple_xm_bars(tmp_dict, x)
                for col in cols_dict:
                    cols_dict[col] = np.append(tmp_dict[col], cols_dict[col])
            need_count = count - len(cols_dict[new_fields[0]])
            if need_count > 0:
                tmp_dict = get_minute_bar_by_count(security, open_dt, need_count * x,
                                                    new_fields,
                                                    include_now=False)
                tmp_dict = _resample_simple_xm_bars(tmp_dict, x)
                for col in cols_dict:
                    cols_dict[col] = np.append(tmp_dict[col], cols_dict[col])
            for f in cols_dict:
                cols_dict[f] = cols_dict[f][-count:]

    # 周线和月线必须先复权，然后 resample
    elif unit == '1w':
        if include_now:
            snapshot = get_snapshot(security, end_trade_date, end_dt)
            cols_dict = get_daily_bar_by_count(security, end_trade_date, count * 5,
                                           new_fields,
                                           include_now=False)
            cols_dict = ensure_not_empty(cols_dict)
            if snapshot:
                for f in cols_dict:
                    cols_dict[f] = np.append(cols_dict[f], snapshot[f])
        else:
            # monday == 0 ... Sunday == 6
            weekday = end_trade_date.weekday()
            last_sunday = end_trade_date - datetime.timedelta(weekday+1)
            cols_dict = get_daily_bar_by_count(security, last_sunday, count * 5,
                                           new_fields,
                                           include_now=False)
            cols_dict = ensure_not_empty(cols_dict)
        cols_dict = _pre_fq(security, cols_dict, pre_factor_ref_date)
        cols_dict = _resample_days_bars(cols_dict, unit)
        for f in cols_dict:
            cols_dict[f] = cols_dict[f][-count:]
    elif unit == '1M':
        if include_now:
            snapshot = get_snapshot(security, end_trade_date, end_dt)
            cols_dict = get_daily_bar_by_count(security, end_trade_date, count*31,
                                               new_fields,
                                               include_now=False)
            cols_dict = ensure_not_empty(cols_dict)
            if snapshot:
                for f in cols_dict:
                    cols_dict[f] = np.append(cols_dict[f], snapshot[f])
        else:
            end_date = end_trade_date.replace(day=1)-datetime.timedelta(days=1)
            cols_dict = get_daily_bar_by_count(security, end_date, count*31,
                                               new_fields,
                                               include_now=False)
            cols_dict = ensure_not_empty(cols_dict)
        cols_dict = _pre_fq(security, cols_dict, pre_factor_ref_date)
        cols_dict = _resample_days_bars(cols_dict, unit)
        for f in cols_dict:
            cols_dict[f] = cols_dict[f][-count:]
    else:
        raise ParamsError("get_bars 支持 '1m', '1d'")

    # 将时间戳转换成datetime 或者 date。
    if 'date' in cols_dict:
        if unit in ('1d','1w', '1M'):
            cols_dict['date'] = vec2date(cols_dict['date'])
        else:
            cols_dict['date'] = vec2datetime(cols_dict['date'])

    # 期货没有复权。
    # 周线和月线必须先复权(同一个周期内复权因子可能不同）
    if not security.is_futures() and unit not in ('1w', '1M'):
        if pre_factor_ref_date is not None:
            cols_dict = _pre_fq(security, cols_dict, pre_factor_ref_date)

    if str_field:
        dtype = np.dtype([(fields, cols_dict[fields].dtype)])
        cols = [cols_dict[fields]]
        result = np.rec.fromarrays(cols, dtype=dtype).view(np.ndarray)
    else:
        # numpy bug: name 不能为unicode。
        dtype = np.dtype([(str(name), cols_dict[name].dtype) for name in fields])
        cols = [cols_dict[name] for name in fields]
        result = np.rec.fromarrays(cols, dtype=dtype).view(np.ndarray)
    return result

def get_ticks(security, end_dt, start_dt=None, count=None,
              fields=['time', 'current', 'high','low', 'volume', 'money']):
    from jqdata.stores.tick_store import get_tick_store
    from jqdata.utils.security import convert_security
    from jqdata.utils.datetime_utils import parse_datetime

    security = convert_security(security)
    end_dt = parse_datetime(end_dt)
    if start_dt is None and count is None:
        raise ParamsError("start_dt和count不能同时为None")
    elif start_dt is not None and count is not None:
        raise ParamsError("start_dt和count只能有一个不为None")
    if start_dt is not None:
        start_dt = parse_datetime(start_dt)
    if count is not None:
        count = int(count)
        assert count > 0, "get_ticks, count必须是一个正整数"

    store = get_tick_store()
    table = store.get_table(security)
    idx = table.find_great_or_equal(end_dt)
    if start_dt is not None:
        if start_dt > end_dt:
            raise ParamsError("start_dt 必须小于等于 end_dt")
        start = table.find_great_or_equal(start_dt)
    elif count is not None:
        start = max(0, idx-count)

    arr = table.array[start:idx]

    ret = {}
    for f in fields:
        if f in ('current', 'high', 'low', 'a1_p', 'b1_p'):
            ret[f] = arr[f]/10000.
        elif f == 'time':
            ret[f] = arr[f]/1000.
        else:
            ret[f] = arr[f]/1.0
    dtype = np.dtype([(str(f), ret[f].dtype) for f in fields])
    cols = [ret[f] for f in fields]
    result = np.rec.fromarrays(cols, dtype=dtype).view(np.ndarray)
    return result


def get_current_tick(security, current_dt):
    from jqdata.stores.tick_store import get_tick_store
    from jqdata.utils.security import convert_security
    from jqdata.utils.datetime_utils import parse_datetime
    from jqdata.models.tick import Tick

    security = convert_security(security)
    assert isinstance(current_dt, datetime.datetime)
    # current_dt = parse_datetime(current_dt)
    store = get_tick_store()
    table = store.get_table(security)
    if table is None:
        raise Exception("找不到%s 的tick数据" % security.code)
    idx = table.find_less_or_equal(current_dt)
    price_fields = ('current', 'high', 'low',
                    'a1_p', 'a2_p', 'a3_p', 'a4_p', 'a5_p',
                    'b1_p', 'b2_p', 'b3_p', 'b4_p', 'b5_p')
    if idx < table.len and idx >= 0:
        data = table.array[idx]
        ret = {}
        if hasattr(data, 'dtype'):
            for f in data.dtype.names:
                if f in price_fields:
                    ret[f] = data[f] / 10000.
                elif f == 'time':
                    ret[f] = data[f] / 1000.
                else:
                    ret[f] = data[f]
        else:
            # data is tuple
            names = table.array.dtype.names
            for i in range(0, len(names)):
                f = names[i]
                if f in price_fields:
                    ret[f] = data[i] / 10000.
                elif f == 'time':
                    ret[f] = data[i] / 1000.
                else:
                    ret[f] = data[i]
        return Tick(security, ret)
    return None


def get_field(security, field='close', unit='1d'):
    '''
    获取bcolz某一列, 归因分析取数据专用。

    :param security:
    :param unit:
    :param field:
    :return:
    '''

    assert unit == '1d', '暂时只支持1d'
    from jqdata.stores.bcolz_store import get_bcolz_day_store
    store = get_bcolz_day_store()
    security = convert_security(security)
    cr = store.open_bcolz_carray(security, field)
    return cr[:]
