#!/usr/bin/env python
#coding:utf-8

'''
取数据相关接口
'''
import sys
if sys.version_info[0] == 3:
    xrange = range
    
import warnings
import datetime
import collections
import os
import six

import pandas as pd
import numpy as np

from jqdata.exceptions import ParamsError


HistoryOptions = collections.namedtuple('HistoryOptions', 'skip_paused fq df')




EMPTY_ARRAY = np.empty((0,))
nan = NAN = float('nan')




def warn_price_as_avg(msg, doc_section, stacklevel=2):
    warnings.warn("{msg} 已经废弃, 请使用 avg 或者 close 代替, 具体请看文档: '{JQ_WEB_SERVER}/api#{doc_section}'".format(
        JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''), msg=msg, doc_section=doc_section), UserWarning, stacklevel+1)


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
        a = grouper[field](a)
        if not isinstance(a, np.ndarray):
            a = np.array(a)
    return a



DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2015-12-31'

frequency_compat = {
    'daily': '1d',
    'minute': '1m',
}



def my_assert(condition, e=None):
    if not condition:
        if not e:
            e = AssertionError()
        elif isinstance(e, six.string_types):
            e = AssertionError(e)
        elif not isinstance(e, Exception):
            e = AssertionError(str(e))
        raise e



def check_unit_fields(unit, fields):
    count = int(unit[:-1])
    my_assert(unit[-1] in ('d', 'm') and count > 0, 'unit应该是1d/1m, 5d/5m这种数字+d/m的形式')
    if fields:
        if count > 1:
            my_assert(all([(f in DEFAULT_FIELDS) for f in fields]), "查询多天/多分钟的历史数据时, field必须是"+str(DEFAULT_FIELDS)+"中的一个")
        else:
            my_assert(all([(f in ALLOED_FIELDS) for f in fields]), "field必须是"+str(ALLOED_FIELDS)+"中的一个")


def lmap(func, iterable):
    return [func(x) for x in iterable]


def is_str(s):
    return isinstance(s, six.string_types)

def is_list(l):
    return isinstance(l, (list, tuple))

import six
from jqdata.models.security import Security
from jqdata.stores.security_store import SecurityStore

def convert_security(s):
    if isinstance(s, six.string_types):
        t = SecurityStore.instance().get_security(s)
        if not t:
            raise ParamsError("找不到标的{}".format(s))
        return t
    elif isinstance(s, Security):
        return s
    elif isinstance(s, (list, tuple)):
        res = []
        for i in range(len(s)):
            if isinstance(s[i], Security):
                res.append(s[i])
            elif isinstance(s[i], six.string_types):
                t = SecurityStore.instance().get_security(s[i])
                if not t:
                    raise ParamsError("找不到标的{}".format(s[i]))
                res.append(t)
            else:
                raise ParamsError("找不到标的{}".format(s[i]))
        return res
    else:
        raise ParamsError('security 必须是一个Security实例或者数组')


def check_string(s):
    my_assert(is_str(s), "参数必须是个字符串, 实际是:" + str(s))


def check_list(l):
    my_assert(type(l) in (tuple, list), "参数必须是tuple或者list, 实际是:" + str(l))


def list_or_str(alist):
    if is_str(alist):
        return [alist]
    else:
        alist = list(alist)
        check_string_list(alist)
        return alist

def ensure_str_tuple(args):
    if is_str(args):
        return (args,)
    else:
        atuple = tuple(args)
        for i in atuple:
            assert isinstance(i, six.string_types)
        return atuple



def check_string_list(security_list):
    check_list(security_list)
    for security in security_list:
        assert isinstance(security, (six.string_types, Security)), "security必须是字符串或者Security对象"


check_security_list = check_string_list

def ensure_fq(fq):
    my_assert(fq in (None, 'pre', 'post', 'none'), "fq 参数应该是下列选项之一: None, 'pre', 'post', 'none'")
    if fq == 'none':
        return None
    else:
        return fq


def replace_list(arr, from_, to):
    return [x if x != from_ else to for x in arr]



def date2dt(date):
    return datetime.datetime.combine(date, datetime.time.min)


def convert_dt(dt):
    """
    >>> convert_dt(datetime.date(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt(datetime.datetime(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt('2015-1-1')
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt('2015-01-01 09:30:00')
    datetime.datetime(2015, 1, 1, 9, 30)

    >>> convert_dt(datetime.datetime(2015, 1, 1, 9, 30))
    datetime.datetime(2015, 1, 1, 9, 30)
    """
    if is_str(dt):
        if ':' in dt:
            return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.strptime(dt, '%Y-%m-%d')
    elif isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, datetime.date):
        return date2dt(dt)
    raise ParamsError("date 必须是datetime.date, datetime.datetime或者如下格式的字符串:'2015-01-05'")


def convert_date(date):
    """
    >>> convert_date('2015-1-1')
    datetime.date(2015, 1, 1)

    >>> convert_date('2015-01-01 00:00:00')
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.datetime(2015, 1, 1))
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.date(2015, 1, 1))
    datetime.date(2015, 1, 1)
    """
    if is_str(date):
        if ':' in date:
            date = date[:10]
        return datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    raise ParamsError("date 必须是datetime.date, datetime.datetime或者如下格式的字符串:'2015-01-05'")


DEFAULT_FIELDS = ['open', 'close', 'high', 'low', 'volume', 'money']
ALLOED_FIELDS = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'price']

DAY_COLUMNS =         tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
FUTURES_DAY_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor settlement open_interest'.split())
FUND_COLUMNS =        tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
MINUTE_COLUMNS =      tuple('open close high low volume money avg factor'.split())
TICK_COLUMNS =        tuple('open close high low volume money avg'.split())

DAY_CLOSE_COLUMN = DAY_COLUMNS.index('close')
DAY_FACTOR_COLUMN = DAY_COLUMNS.index('factor')
DAY_PAUSED_COLUMN = DAY_COLUMNS.index('paused')

