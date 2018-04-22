#!/usr/bin/env python
#-*- coding: utf-8 -*-

# all common third-party modules
import os,re,sys,json

import datetime
from datetime import timedelta

import collections
import math
import time

import warnings
warnings.simplefilter('once', UserWarning)

import contextlib

import six

from six.moves import xrange

IS_LINUX = sys.platform.startswith('linux')

if IS_LINUX:
    import SharedArray

import numpy as np

nan = NaN = np.nan
isnan = np.isnan

in_user_space = 'KUANKE_USER_SPACE' in os.environ

if in_user_space:
    import pandas as pd

from fastcache import clru_cache as lru_cache

single_cache = lru_cache(None)

from .time_utils import *

DEFAULT_FIELDS = ('open', 'close', 'high', 'low', 'volume', 'money')
ALLOED_FIELDS = ('open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'price')
HistoryOptions = collections.namedtuple('HistoryOptions', 'skip_paused fq df')

def my_assert(condition, e=None):
    if not condition:
        if not e:
            e = AssertionError()
        elif isinstance(e, str) or isinstance(e, unicode):
            e = AssertionError(e)
        elif not isinstance(e, Exception):
            e = AssertionError(str(e))
        raise e


def is_research():
    return 'RESEARCH_CLIENT' in os.environ

class UserKnownException(Exception):
    pass

def get_day_index_not_after(a, idate):
    return np.searchsorted(a, idate, 'right') - 1

def lmap(func, iterable):
    return [func(x) for x in iterable]

def filter_dict(dictionary, keys):
    """Filters a dict by only including certain keys."""
    key_set = set(keys) & set(dictionary.keys())
    return {key: dictionary[key] for key in key_set}

def is_str(s):
    return type(s) in (str, six.text_type)

def is_list(l):
    return type(l) in (list, tuple)

def list_or_str(alist):
    if is_str(alist):
        return (alist,)
    else:
        alist = tuple(alist)
        check_string_list(alist)
        return alist

def check_string(s):
    my_assert(is_str(s), "参数必须是个字符串, 实际是:" + str(s))

def check_list(l):
    my_assert(type(l) in (tuple, list), "参数必须是tuple或者list, 实际是:" + str(l))

def check_string_list(security_list):
    check_list(security_list)
    for security in security_list:
        check_string(security)
    pass

check_security = check_string

check_security_list = check_string_list

def dict2df(dict):
    import pandas as pd
    return pd.DataFrame(**dict)

def rindex_array(array):
    return {v:k for k, v in enumerate(array)}

def replace_list(list, from_, to):
    return [x if x != from_ else to for x in list]

def merge_dict(*dicts):
    r = {}
    for d in dicts:
        r.update(d)
    return r

def is_nparray(a):
    return isinstance(a, np.ndarray)

def concat_array(a, b):
    return np.concatenate((a, b)) if (is_nparray(a) or is_nparray(b)) else a + b

pow10 = (1, 10, 100, 1000, 10000)

def fixed_round(x, n):
    return round(x * pow10[n]) / pow10[n]

def round_price(x, n=2):
    return fixed_round(x, n)

round_money = round_price

def round_volume(x):
    return int(round(x)) if not isnan(x) else NaN

def group_array(a, group, field):
    if group > 1:
        new_int_index = xrange(0, len(a), group)
        grouper = {
            'open': (lambda v: [v[i] for i in new_int_index]),
            'close': (lambda v: [v[min(i+group-1, len(v)-1)] for i in new_int_index]),
            'high': (lambda v: [max(v[i:i+group]) for i in new_int_index]),
            'low': (lambda v: [min(v[i:i+group]) for i in new_int_index]),
            'volume': (lambda v: [sum(v[i:i+group]) for i in new_int_index]),
            'money': (lambda v: [sum(v[i:i+group]) for i in new_int_index]),
            # same to close
            'index': (lambda v: [v[min(i+group-1, len(v)-1)] for i in new_int_index]),
        }
        npa = is_nparray(a)
        a = grouper[field](a)
        if npa:
            a = np.array(a)
    return a

@lru_cache(None)
def get_all_day_data(security):
    shm_path = 'get_all_day_data_raw-%s' % (security)
    try:
        a = SharedArray.attach(shm_path, readonly=True)
    except (OSError, IOError):
        api_proxy.prepare_all_day_data(security)
        a = SharedArray.attach(shm_path, readonly=True)
    return a

def load_np(s):
    import io
    s = io.BytesIO(s)
    return np.load(s)

# 内存 >= 100G, 分钟数据使用共享内存
USE_MINUTE_SHM = get_total_memory() > (100 * 1024**3)

def get_minute_data_of_day(security, date):
    if is_research():
        func = get_minute_data_of_day_copied
    elif USE_MINUTE_SHM and date.year >= 2011:
        func = get_minute_data_of_day_shm_cached
    elif api_proxy.context.sim_trade and (date - get_today()).days <= MINUTE_SHM_MAX_DAYS:
        func = get_minute_data_of_day_shm_cached
    else:
        func = get_minute_data_of_day_copied_cached
    return func(security, date)

def get_minute_data_of_day_shm(security, date):
    shm_path = 'get_minute_data_of_day_raw-%s-%s' % (security, date)
    try:
        a = SharedArray.attach(shm_path, readonly=True)
    except (OSError, IOError):
        api_proxy.prepare_minute_data(security, date)
        a = SharedArray.attach(shm_path, readonly=True)
    return a

get_minute_data_of_day_shm_cached = lru_cache(50000)(get_minute_data_of_day_shm)

def get_minute_data_of_day_copied(security, date):
    s = api_proxy.get_minute_data_of_day(security, date)
    a = load_np(s)
    a.flags.writeable = False
    return a

get_minute_data_of_day_copied_cached = lru_cache(10000)(get_minute_data_of_day_copied)

DAY_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
FUTURES_DAY_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor settlement open_interest'.split())
FUND_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
DAY_CLOSE_COLUMN = DAY_COLUMNS.index('close')
DAY_FACTOR_COLUMN = DAY_COLUMNS.index('factor')
DAY_PAUSED_COLUMN = DAY_COLUMNS.index('paused')
TICK_COLUMNS = MINUTE_COLUMNS = 'open close high low volume money avg'.split()

# 共享内存只存放最近10天的分钟数据
MINUTE_SHM_MAX_DAYS = 10

def paused_day_array(pre_close, factor):
    open = close = high = low = avg = high_limit = low_limit = pre_close
    volume = money = 0.0
    paused = 1.0
    return np.array([open, close, high, low, volume, money, pre_close, high_limit, low_limit, paused, avg, factor], dtype=float)

def fq_price(p, f, price_decimals, inplace=True):
    if inplace:
        p *= f
        return np.round(p, price_decimals, p)
    else:
        return np.round(p*f, price_decimals)

def fq_volume(p, f, price_decimals, inplace=True):
    if inplace:
        p /= f
        return np.round(p, 0, p)
    else:
        return np.round(p/f, 0)

fq_funcs = {
    'open': fq_price,
    'close': fq_price,
    'high': fq_price,
    'low': fq_price,
    'price': fq_price,
    'avg': fq_price,
    'pre_close': fq_price,
    'high_limit': fq_price,
    'low_limit': fq_price,
    'volume': fq_volume,
}

def get_factor(security, date):
    a = get_all_day_data(security)
    i = np.searchsorted(a[:, 0], date2int(date), 'right') - 1
    # 如果在 date 之前没有上市, 取上次第一天的 factor
    i = max(i, 0)
    return a[i, 1+DAY_FACTOR_COLUMN] if i < len(a) else 1.0

def pad_date_index(index, count):
    if len(index) < count:
        miss = count - len(index)
        first_date = get_first_day()
        pad = [first_date-i*DELTA_DAY for i in xrange(miss, 0, -1)]
        return concat_array(pad, index)
    return index

def pad_minute_index(index, count):
    if len(index) < count:
        miss = count - len(index)
        first_date = get_first_day()
        first_dt = date2dt(first_date)
        pad = [first_dt-i*DELTA_MINUTE for i in xrange(miss, 0, -1)]
        return concat_array(pad, index)
    return index

def pad_nan(a, count):
    if len(a) >= count:
        return a
    else:
        miss = count - len(a)
        pad = np.full(miss, nan)
        return concat_array(pad, a)

get_now = datetime.datetime.now
get_today = datetime.date.today
combine_dt = datetime.datetime.combine

def copy_if_readonly(a):
    return a if a.flags.writeable else np.copy(a)

class Object(object):
    def copy(self, **kwargs):
        import copy
        o = copy.copy(self)
        o.__dict__.update(kwargs)
        return o

    @classmethod
    def from_dict(cls, dict):
        o = cls()
        o.__dict__.update(dict)
        return o
    pass

class UserObject(Object):
    # copy object from us to user space
    def copy(self, o):
        if o is None:
            return self
        for k in self.__dict__:
            setattr(self, k, getattr(o, k, None))
        return self

    @classmethod
    def copy_of(cls, o):
        if o is None:
            return None
        self = cls()
        return self.copy(o) or self

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, str(self.__dict__))

    if 'KUANKE_TEST' in os.environ:
        def __eq__(self, other):
            return self.__dict__ == other.__dict__

class SecurityInfo(UserObject):
    def __init__(self):
        self.name = None
        self.display_name = None
        self.start_date = None
        self.end_date = None
        self.type = None
        self.parent = None

    def copy(self, o):
        UserObject.copy(self, o)
        self.type = o.display_type

def dict2dict(dict):
    import numpy as np
    data = dict['data']
    return {k:np.array(v) for k, v in six.iteritems(data)}

def convert_dict(dict, df):
    return (df and dict2df or dict2dict)(dict)

DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2015-12-31'

def check_unit_fields(unit, fields):
    count = int(unit[:-1])
    my_assert(unit[-1] in ('d', 'm') and count > 0, 'unit应该是1d/1m, 5d/5m这种数字+d/m的形式')
    if fields:
        if count > 1:
            my_assert(all([(f in DEFAULT_FIELDS) for f in fields]), "查询多天/多分钟的历史数据时, field必须是"+str(DEFAULT_FIELDS)+"中的一个")
        else:
            my_assert(all([(f in ALLOED_FIELDS) for f in fields]), "field必须是"+str(ALLOED_FIELDS)+"中的一个")

def check_fq(fq):
    my_assert(fq in (None, 'pre', 'post'), "fq 参数应该是下列选项之一: None, 'pre', 'post'")

# common api used by backtest and research

frequency_compat = {
    'daily': '1d',
    'minute': '1m',
}

def get_date_list(end_date, count):
    a = get_all_trade_days_list()
    e = np.searchsorted(a, end_date, 'right')
    return a[max(e-count, 0):e]

def fill_paused(security, a, adates, dates):
    # 获取最大有效数据日期, 超过此日期数据为nan
    max_valid_date = date2int(min(get_today(), get_security_meta(security, 'end_date')))

    ncolumns = a.shape[1]
    v = np.searchsorted(adates, dates, 'right') - 1
    b = [None]*(len(v))
    for i, vi in enumerate(v):
        if vi < 0:
            # n未上市
            x = np.full(ncolumns, nan)
        else:
            adate = adates[vi]
            date = dates[i]
            if date != adate:
                if date > max_valid_date:
                    x = np.full(ncolumns, nan)
                else:
                    # 停牌
                    x = paused_day_array(a[vi][DAY_CLOSE_COLUMN], a[vi][DAY_FACTOR_COLUMN])
            else:
                x = a[vi]
        b[i] = x
    return np.array(b).reshape(len(b), ncolumns)

def is_continous(a):
    for i in range(1, len(a)):
        if a[i] != a[i-1] + 1:
            return False
    return True

def count_days_between(start_date, end_date):
    days = get_all_trade_days_list()
    start = np.searchsorted(days, start_date)
    # dont include end
    stop = np.searchsorted(days, end_date, 'right')
    return stop - start

def get_days_between(start_date, end_date):
    days = get_all_trade_days_list()
    start = np.searchsorted(days, start_date)
    # dont include end
    stop = np.searchsorted(days, end_date, 'right')
    return days[start:stop] if stop > start else []

def get_unpaused_days_between(security, start_date, end_date):
    a = get_all_day_data(security)
    idates = a[:, 0]
    start = np.searchsorted(idates, date2int(start_date))
    stop = np.searchsorted(idates, date2int(end_date), 'right')
    return lmap(int2date, idates[start:stop]) if stop > start else []

# def count_minutes_between(start_dt, end_dt):
#     day_minutes = get_all_minutes_of_day()
#     start = np.searchsorted(day_minutes, start_dt.time())
#     stop = np.searchsorted(day_minutes, end_dt.time(), 'right')
#     count = stop - start

#     days = count_days_between(start_dt.date(), end_dt.date())
#     count += (days - 1) * len(day_minutes)
#     return count

# def count_unpaused_minutes_between(security, start_dt, end_dt):
#     if is_paused(security, start_dt.date()):
#         # 开始日期停牌, 则不包含开始日期
#         start_dt = start_dt.replace(hour=23)
#     if is_paused(security, end_dt.date()):
#         # 结束日期停牌, 则不包含结束日期
#         end_dt = end_dt.replace(hour=0)

#     day_minutes = get_all_minutes_of_day()
#     start_minute_index = np.searchsorted(day_minutes, start_dt.time())
#     stop_minute_index = np.searchsorted(day_minutes, end_dt.time(), 'right')
#     count = stop_minute_index - start_minute_index

#     a = get_all_day_data(security)
#     idates = a[:, 0]
#     start = np.searchsorted(idates, date2int(start_dt.date()))
#     stop = np.searchsorted(idates, date2int(end_dt.date()), 'right')
#     days = stop - start
#     count += (days - 1) * len(day_minutes)
#     return count

# 取天数据的核心函数, 取单只股票的天数据
def get_price_daily_single(security, end_date, fields, options, start_date=None, count=None, need_index=False):
    if 'price' in fields:
        fields = replace_list(fields, 'price', 'avg')

    skip_paused = options.skip_paused
    fq = options.fq

    a = get_all_day_data(security)
    idates = a[:, 0]
    a = a[:, 1:]
    index = None

    row_end = get_day_index_not_after(idates, date2int(end_date))
    if count is None:
        my_assert(start_date is not None)
        if skip_paused:
            row_start = np.searchsorted(idates, date2int(start_date))
            count = row_end + 1 - row_start
        else:
            count = count_days_between(start_date, end_date)
        if count < 0:
            return np.empty((0, len(fields))), []

    row_start = max(row_end + 1 - count, 0)
    a = a[row_start:row_end+1]
    idates = idates[row_start:row_end+1]

    if skip_paused:
        if need_index:
            index = lmap(int2date, idates)
    else:
        index = get_date_list(end_date, count)
        iindex = lmap(date2int, index)
        if (len(idates) == len(iindex) and len(idates) > 0 and idates[-1] == iindex[-1] and idates[0] == iindex[0]):
            # 没有停牌
            pass
        else:
            a = fill_paused(security, a, idates, iindex)

    if fq:
        factors = a[:, DAY_COLUMNS.index('factor')]

    if fq == 'pre':
        f = get_factor(security, get_power_rate_ref_date())
        factors = factors / f

    columns = [DAY_COLUMNS.index(field) for field in fields]
    if is_continous(columns):
        # 下面的方式不会copy
        a = a[:, columns[0]:columns[0]+len(columns)]
    else:
        a = a[:, columns]

    if fq:
        price_decimals = get_security_price_decimals(security)
        for i, f in enumerate(fields):
            if f in fq_funcs:
                a = copy_if_readonly(a)
                fq_funcs[f](a[:, i], factors, price_decimals)

    if 'factor' in fields:
        a = copy_if_readonly(a)
        i = fields.index('factor')
        if fq:
            a[:, i] = factors
        else:
            a[:, i] = 1.0

    if len(a) < count:
        # pad nan
        miss = count-len(a)
        prev = np.full((miss, len(fields)), nan)
        if need_index:
            prev_index = pad_date_index([], miss)
        a = np.concatenate((prev, a))
        if need_index:
            index = concat_array(prev_index, index)
    return a, index

# 取天数据, 能兼容单只和多只股票
def get_price_daily(security, start_date, end_date, count, unit, fields, options):
    if is_list(security):
        return {s: get_price_daily_single_grouped(s,
                    start_date=start_date,
                    end_date=end_date,
                    count=count,
                    unit=unit,
                    fields=fields,
                    options=options) for s in security}
    else:
        return get_price_daily_single_grouped(security,
                    start_date=start_date,
                    end_date=end_date,
                    count=count,
                    unit=unit,
                    fields=fields,
                    options=options)

# 取单只股票的数据, 把结果按传入的 unit (比如 3d, 5d) 规整(group)一下
def get_price_daily_single_grouped(security, start_date, end_date, count, unit, fields, options):
    group = int(unit[:-1])
    a, index = get_price_daily_single(security,
        start_date=start_date,
        end_date=end_date,
        count=count * group if count is not None else None,
        fields=fields,
        options=options,
        need_index=options.df)

    dict_by_column = {f: group_array(a[:, i], group, f) for i, f in enumerate(fields)}
    if options.df:
        index = group_array(index, group, 'index')

    if not options.df:
        return dict_by_column
    else:
        return dict(index=lmap(date2dt, index), columns=fields, data=dict_by_column)
    pass

# API接口: get_price
def get_price(security, start_date=None, end_date=None, \
            frequency='daily', fields=None, skip_paused=False, fq='pre', count=None):
    """
    {JQ_WEB_SERVER}/api?f=home&m=memu#getprice
    """.format(JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''))

    import pandas as pd
    my_assert(is_str(security) or is_list(security))

    if count is not None and start_date is not None:
        raise Exception("get_price 不能同时指定 start_date 和 count 两个参数")

    if count is not None:
        count = int(count)

    start_dt = convert_dt(start_date) if start_date else datetime.datetime(2015, 1, 1)
    end_dt = convert_dt(end_date) if end_date else datetime.datetime(2015, 12, 31)

    start_dt = max(start_dt, date2dt(get_first_day()))
    end_dt = min(end_dt, date2dt(get_last_day()))

    if fields is not None:
        fields = list_or_str(fields)
        if 'price' in fields:
            warn_price_as_avg('使用 price 作为 get_price 的 fields 参数', 'getprice')
    else:
        fields = DEFAULT_FIELDS

    if frequency in frequency_compat:
        frequency = frequency_compat.get(frequency)

    check_unit_fields(frequency, fields)

    check_fq(fq)

    if is_list(security) and skip_paused:
        raise Exception("get_price 取多只股票数据时, 为了对齐日期, 不能跳过停牌")

    # 分钟数据, 不能超过
    if frequency.endswith('m'):
        securities_count = len(security) if is_list(security) else 1
        days = count if count is not None else count_days_between(start_dt.date(), end_dt.date())
        if not securities_count * days <= 3000:
            raise UserKnownException("分钟数据, 股票数量乘以交易天数不能超过3000条")

    if frequency.endswith('d'):
        res = get_price_daily(security, start_dt.date(), end_dt.date(), count, frequency, fields,
                HistoryOptions(df=True, skip_paused=bool(skip_paused), fq=fq))
    elif frequency.endswith('m'):
        res = get_price_minute(security, start_dt, end_dt, count, frequency, fields,
                HistoryOptions(df=True, skip_paused=bool(skip_paused), fq=fq))

    def to_df(d):
        index = d['index']
        if len(index) > 0 and is_str(index[0]):
            d['index'] = lmap(parse_dt, index)
        return pd.DataFrame(**d)

    if is_list(security):
        fields = fields or DEFAULT_FIELDS
        if len(security) == 0:
            return pd.Panel(items=fields)
        pn_dict = {}
        index = res[security[0]]['index']
        if len(index) > 0 and is_str(index[0]):
            index = lmap(parse_dt, index)
        for f in fields:
            df_dict = {s:res[s]['data'][f] for s in security}
            pn_dict[f] = pd.DataFrame(index=index, columns=security, data=df_dict)
        return pd.Panel(pn_dict)
    else:
        return to_df(res)

def get_all_securities(types=[], date=None):
    """
    获取平台支持的所有股票、基金、指数信息，这里请在使用时注意防止未来函数。

    **参数**
    types: list, 用来过滤securities的类型, list元素可选: 'stock', 'fund', 'index'. types为空时返回所有股票, 不包括基金和指数(因为指数不可以交易)
    date: 日期, 一个字符串或者 datetime.datetime/datetime.date 对象, 用于获取某日期还在上市的股票信息

    **返回**
    [pandas.DataFrame]
    """
    types = list_or_str(types)
    for t in types:
        my_assert(t in ('stock', 'index', 'fund', 'futures', 'etf', 'lof', 'fja', 'fjb'))
    if date:
        date = convert_date(date)
    import pandas as pd
    df = pd.DataFrame(**api_proxy.get_all_securities_df_dict(types, date))
    if is_research():
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
    return df

def get_index_stocks(index_symbol, date=None):
    """
    获取一个指数的成分股列表
    index_symbol: 指数代码
    返回股票代码的list

    示例:
    ```python
    # 获取所有沪深300的股票, 设为股票池
    stocks = get_index_stocks('000300.XSHG')
    ```

    所有指数列表见: {JQ_WEB_SERVER}/help/api/index#getindexstocks
    """.format(JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''))

    check_security(index_symbol)
    if date:
        date = convert_date(date)
    return api_proxy.get_index_stocks(index_symbol, date)

def get_industry_stocks(industry_code, date=None):
    """
    获取一个行业的所有股票
    industry_code: 行业编码
    返回股票代码的list

    示例:
    ```python
    # 获取所有计算机/互联网行业的股票, 设为股票池
    stocks = get_industry_stocks('I64')
    ```

    所有行业代码列表: {JQ_WEB_SERVER}/help/api/index#getindustrystocks
    """.format(JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''))

    check_string(industry_code)
    if date:
        date = convert_date(date)
    return api_proxy.get_industry_stocks(industry_code, date)

def get_concept_stocks(concept_code, date=None):
    """获取概念板块成分股

    参数:
        concept_code: 板块代码
        date: 日期

    返回一个股票代码的 list

    示例:

    ```python
    # 获取所有移动互联网概念板块的股票, 设为股票池
    stocks = get_industry_stocks('GN099')
    ```
    """
    check_string(concept_code)
    if date:
        date = convert_date(date)
    return api_proxy.get_concept_stocks(concept_code, date)

def get_margincash_stocks():
    day = api_proxy.context.current_dt.date()
    return api_proxy.get_margincash_stocks(day)

def get_marginsec_stocks():
    day = api_proxy.context.current_dt.date()
    return api_proxy.get_marginsec_stocks(day)

def get_extras(info, security_list, start_date=None, end_date=None, df=True, count=None):
    my_assert(info in ('is_st', 'acc_net_value', 'unit_net_value', 'futures_sett_price', 'futures_positions'))
    security_list = list_or_str(security_list)
    if start_date and count:
        raise Exception("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise Exception("count 参数需要大于 0 或者为 None")
    end_date = convert_date(end_date) if end_date else convert_date(DEFAULT_END_DATE)
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(datetime.date(2005, 1, 4), end_date)[-count] if count else convert_date(DEFAULT_START_DATE))
    dict = api_proxy.get_extras(info, security_list, start_date, end_date)
    frame = convert_dict(dict, df)
    if is_research() and df:
        import pandas as pd
        frame.index = pd.to_datetime(frame.index)
    return frame

def get_security_info(code):
    check_security(code)
    return get_all_securities_info()[code]

@single_cache
def get_all_securities_info():
    def info_from_dict(d):
        if is_research():
            d['start_date'] = convert_date(d['start_date'])
            d['end_date'] = convert_date(d['end_date'])
        return SecurityInfo.from_dict(d)

    d = api_proxy.get_all_securities_info()
    d = {code: info_from_dict(dict) for code, dict in six.iteritems(d)}
    return d

def get_security_price_decimals(security):
    type = get_all_securities_info()[security].type
    return type == 'stock' and 2 or 4

def get_security_meta(security, field):
    info = get_all_securities_info()[security]
    return getattr(info, field)

def get_power_rate_ref_date():
    if is_research():
        if 'POWER_RATE_REF_DATE' in os.environ:
            return convert_date(os.environ['POWER_RATE_REF_DATE'])
        return get_trade_day_before(get_today())
    else:
        return api_proxy.context.power_rate_ref_date
    pass

class ApiProxy(UserObject):
    pass

api_proxy = ApiProxy()

class KeyedDefaulitDict(dict):
    """docstring for KeyedDefaulitDict"""
    def __init__(self, *args, **kwargs):
        self.default_factory = kwargs.pop('default_factory') if 'default_factory' in kwargs else None
        self.add_missing = kwargs.pop('add_missing') if 'add_missing' in kwargs else True
        super(KeyedDefaulitDict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.default_factory:
            v = self.default_factory(key)
            if self.add_missing:
                self[key] = v
            return v
        else:
            raise KeyError(key)

def warn_price_as_avg(msg, doc_section, stacklevel=2):
    warnings.warn("{msg} 已经废弃, 请使用 avg 或者 close 代替, 具体请看文档: '{JQ_WEB_SERVER}/api#{doc_section}'".format(\
            JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''), msg=msg, doc_section=doc_section), UserWarning, stacklevel+1)

def get_unpaused_day_before(security, date):
    idates = get_all_day_data(security)[:, 0]
    i = np.searchsorted(idates, date2int(date)) - 1
    return int2date(idates[i]) if i >= 0 else None

def is_listed(security, date):
    info = get_all_securities_info()[security]
    return info.start_date <= date <= info.end_date

# 停牌或者不在市
def is_paused(security, date):
    idate = date2int(date)
    a = get_all_day_data(security)
    idates = a[:, 0]
    i = np.searchsorted(idates, idate)
    return idates[i] != idate if i < len(idates) else True

def get_day_array(security, date):
    a = get_all_day_data(security)
    idate = date2int(date)
    idates = a[:,0]
    a = a[:, 1:]
    i = get_day_index_not_after(idates, idate)
    if i < 0:
        # 未上市
        return np.full(a.shape[1], nan)
    if idates[i] != idate:
        # 判断是否已经退市或者想拉取后面的数据
        if date > get_security_meta(security, 'end_date') or date > get_today():
            return np.full(a.shape[1], nan)
        # 停牌
        return paused_day_array(a[i][DAY_CLOSE_COLUMN], a[i][DAY_FACTOR_COLUMN])
    else:
        return a[i]

# 取单只股票分钟数据, 能取多天的数据
def get_price_minute_single(security, end_dt, fields, options, count=None, need_index=False, ref_factor=None):
    if 'price' in fields:
        fields = replace_list(fields, 'price', 'avg')

    arrays = []
    indexs = []
    while count > 0:
        if end_dt is not None:
            a, index = get_price_minute_single_security_single_date(security,
                    end_dt=end_dt, fields=fields, options=options,
                    count=count, need_index=need_index)
        else:
            a = np.full((count, len(fields)), nan)
            index = pad_minute_index([], count) if need_index else None

        arrays.append(a)
        indexs.append(index)
        count -= len(a)

        if count > 0:
            if options.skip_paused:
                prev_date = get_unpaused_day_before(security, end_dt.date())
            else:
                prev_date = get_trade_day_before(end_dt.date())

            end_dt = get_last_minute_dt(security, prev_date) if prev_date else None

    # 空数组不能执行 np.concatenate
    if len(arrays) > 0:
        arrays.reverse()
        a = np.concatenate(arrays)
        if need_index:
            indexs.reverse()
            index = np.concatenate(indexs)
        else:
            index = None
    else:
        a = np.empty((0, len(fields)))
        index = []
    return a, index

# 取单只股票分钟数据, 只能取一天之内的, 核心函数
def get_price_minute_single_security_single_date(security, end_dt, fields, options, count=None, need_index=False, ref_factor=None):
    date = end_dt.date()
    fq = options.fq
    skip_paused = options.skip_paused

    minutes = get_minutes_of_day(security, date)
    row_end = np.searchsorted(minutes, end_dt.time(), 'right') - 1
    my_assert(count is not None)

    row_start = max(row_end - count + 1, 0)
    index = None

    paused = is_paused(security, date)
    if paused:
        if skip_paused:
            a = np.empty((0, len(fields)))
            index = []
        else:
            if need_index:
                index = minutes[row_start:row_end+1]
                index = [combine_dt(date, t) for t in index]

            a = get_day_array(security, date)
            columns = [DAY_COLUMNS.index(f) for f in fields]
            a = a[columns]
            a = a.reshape(1, len(a)).repeat(row_end + 1 - row_start, axis=0)
    else:
        if need_index:
            index = minutes[row_start:row_end+1]
            index = [combine_dt(date, t) for t in index]

        minute_fields = [f for f in fields if f in MINUTE_COLUMNS]
        day_fields = [f for f in fields if f not in MINUTE_COLUMNS]
        minute_a = None
        day_a = None

        if minute_fields:
            # 取分钟数据
            a = get_minute_data_of_day(security, date)
            columns = [MINUTE_COLUMNS.index(f) for f in minute_fields]
            minute_a = a[row_start:row_end+1, columns]

        if day_fields:
            # 取天数据, 重复N遍
            a = get_day_array(security, date)
            columns = [DAY_COLUMNS.index(f) for f in day_fields]
            a = a[columns]
            day_a = a.reshape(1, len(a)).repeat(row_end + 1 - row_start, axis=0)

        if day_a is not None and minute_a is not None:
            # 组装分钟和日数据
            # 取得各分钟数据列
            d = {f: minute_a[:, i] for i, f in enumerate(minute_fields)}
            # 天数据列
            d.update({f: day_a[:, i] for i, f in enumerate(day_fields)})
            # 把各列拼装成一个np.array
            a = np.column_stack((d[f] for f in fields))
        else:
            a = minute_a if minute_a is not None else day_a

    if fq:
        # 取得后复权因子
        factor = get_factor(security, date)

    if fq == 'pre':
        # 后复权转换成前复权
        if ref_factor is None:
            ref_factor = get_factor(security, get_power_rate_ref_date())
        factor /= ref_factor

    if fq and abs(factor-1) > 1e-6:
        # 如果有复权, 分属性复权
        price_decimals = get_security_price_decimals(security)
        for i, f in enumerate(fields):
            if f in fq_funcs:
                a = copy_if_readonly(a)
                fq_funcs[f](a[:, i], factor, price_decimals)

    # 如果需要返回复权因子
    if 'factor' in fields:
        a = copy_if_readonly(a)
        i = fields.index('factor')
        if fq:
            a[:, i] = factor
        else:
            # 不复权, 返回复权因子都是1
            a[:, i] = 1.0
    return a, index

# 取分钟数据, 能兼容单只和多只股票
def get_price_minute(security, start_dt, end_dt, count, unit, fields, options):
    if is_list(security):
        return {s: get_price_minute_single_grouped(s,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    count=count,
                    unit=unit,
                    fields=fields,
                    options=options) for s in security}
    else:
        return get_price_minute_single_grouped(security,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    count=count,
                    unit=unit,
                    fields=fields,
                    options=options)

# 取单只股票分钟数据, 把结果按unit重整
def get_price_minute_single_grouped(security, start_dt, end_dt, count, unit, fields, options):
    group = int(unit[:-1])

    if count is not None:
        total = count * group
        a, index = get_price_minute_single(security,
                end_dt=end_dt, count=total, fields=fields,
                options=options, need_index=options.df)
    else:
        start_date = start_dt.date()
        end_date = end_dt.date()
        dates = get_unpaused_days_between(security, start_date, end_date) if options.skip_paused \
                else get_days_between(start_date, end_date)

        array = []
        indexs = []
        last_date_index = len(dates) - 1
        for i, date in enumerate(dates):

            day_minutes = get_minutes_of_day(security, date)
            NMINUTES = len(day_minutes)

            start_minute_index = 0
            stop_minute_index = NMINUTES
            if i == 0 and date == start_date:
                start_minute_index = np.searchsorted(day_minutes, start_dt.time())
            if i == last_date_index and date == end_date:
                stop_minute_index = np.searchsorted(day_minutes, end_dt.time(), 'right')

            if start_minute_index >= stop_minute_index:
                continue

            a, index = get_price_minute_single(security,
                end_dt=combine_dt(date, day_minutes[stop_minute_index-1]),
                count=stop_minute_index - start_minute_index,
                fields=fields,
                options=options,
                need_index=options.df)
            array.append(a)
            indexs.append(index)

        a = np.concatenate(array) if array else np.empty((0, len(fields)))
        if options.df:
            index = np.concatenate(indexs) if indexs else []

    # group it
    dict_by_column = {f: group_array(a[:, i], group, f) for i, f in enumerate(fields)}
    if options.df:
        index = group_array(index, group, 'index')

    if not options.df:
        return dict_by_column
    else:
        return dict(index=index, columns=fields, data=dict_by_column)
    pass

class cached_property_nonan(object):
    """
    等同于cached_property, 但是不cache nan, 直到取到非nan值才cache
    """  # noqa

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.func(obj)
        if isnan(value):
            return value
        obj.__dict__[self.func.__name__] = value
        return value

def normalize_code(code):
    '''
    上海证券交易所证券代码分配规则
    https://biz.sse.com.cn/cs/zhs/xxfw/flgz/rules/sserules/sseruler20090810a.pdf

    深圳证券交易所证券代码分配规则
    http://www.szse.cn/main/rule/bsywgz/39744233.shtml
    '''
    if isinstance(code, int):
        suffix = 'XSHG' if code >= 500000 else 'XSHE'
        return '%06d.%s' % (code, suffix)
    elif is_str(code):
        code = code.upper()
        if code[-5:] in ('.XSHG', '.XSHE', '.CCFX'):
            return code
        suffix = None
        match = re.search(r'[0-9]{6}', code)
        if match is None:
            raise Exception("没找到六位数字")
        number = match.group(0)
        if 'SH' in code:
            suffix = 'XSHG'
        elif 'SZ' in code:
            suffix = 'XSHE'

        if suffix is None:
            suffix = 'XSHG' if int(number) >= 500000 else 'XSHE'
        return '%s.%s' % (number, suffix)
    else:
        raise Exception("normalize_code 的参数必须是字符串或者整数")
