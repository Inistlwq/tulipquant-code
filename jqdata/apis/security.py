# coding: utf-8

from __future__ import absolute_import, print_function, unicode_literals

import os
import re
import six
import datetime
import pandas
import re
import jqdata.stores
from jqdata.models.security import Security
from jqdata.stores.index_store import get_index_store
from jqdata.stores.industry_store import get_industry_store
from jqdata.stores.concept_store import get_concept_store
from jqdata.stores.security_store import get_security_store
from jqdata.stores.open_fund_store import get_open_fund_store
from jqdata.stores.calendar_store import get_calendar_store
from jqdata.stores.dominant_future_store import get_dominant_future_store
from jqdata.utils.utils import is_str,convert_date,convert_dt,check_string
from jqdata.exceptions import ParamsError


__all__ = [
    'get_index_stocks',
    'get_industry_stocks',
    'get_concept_stocks',
    'get_all_securities',
    'get_security_info',
    'get_fund_info',
    'get_history_name',
    'normalize_code',
    'get_dominant_future',
    'get_future_contracts',
]


def get_index_stocks(index_symbol, date):
    store = get_index_store()
    check_string(index_symbol)
    date = convert_date(date)
    return store.get_index_stocks(index_symbol, str(date))


def get_industry_stocks(industry_code, date):
    store = get_industry_store()
    check_string(industry_code)
    date = convert_date(date)
    return store.get_industry_stocks(industry_code, str(date))


def get_concept_stocks(concept_code, date):
    store = get_concept_store()
    check_string(concept_code)
    date = convert_date(date)
    return store.get_concept_stocks(concept_code, str(date))


def get_all_securities(types=[], date=None):
    if is_str(types):
        types = [types]
    store = get_security_store()
    return store.get_all_securities(types, date)


def get_security_info(code):
    if isinstance(code, Security):
        return code
    check_string(code)
    store = get_security_store()
    return store.get_security(code)


def get_fund_info(security, date=None):
    if isinstance(security, Security):
        return security
    check_string(security)
    if date is None:
        date = datetime.date.today()
    store = get_open_fund_store()
    return store.get_fund_info(security, date)
    pass


def get_history_name(code, date):
    ins = jqdata.stores.HisnameStore.instance()
    return ins.get_history_name(code, date)


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
    elif isinstance(code, six.string_types):
        code = code.upper()
        if code[-5:] in ('.XSHG', '.XSHE', '.CCFX'):
            return code
        suffix = None
        match = re.search(r'[0-9]{6}', code)
        if match is None:
            raise ParamsError(u"wrong code={}".format(code))
        number = match.group(0)
        if 'SH' in code:
            suffix = 'XSHG'
        elif 'SZ' in code:
            suffix = 'XSHE'

        if suffix is None:
            suffix = 'XSHG' if int(number) >= 500000 else 'XSHE'
        return '%s.%s' % (number, suffix)
    else:
        raise ParamsError(u"normalize_code(code=%s) 的参数必须是字符串或者整数" % code)


def get_dominant_future(dt, underlying_symbol):
    """获取某一期货品种策略当前日期的主力合约代码"""
    dt = convert_dt(dt)
    check_string(underlying_symbol)
    store = get_dominant_future_store()
    calendar_store = get_calendar_store()
    current_trade_date = calendar_store.get_current_trade_date(
        get_security_store().get_security('AU9999.XSGE'), dt)
    return store.get_dominant_code(current_trade_date, underlying_symbol)


def get_future_contracts(dt, underlying_symbol):
    """期货可交易合约列表"""
    dt = convert_dt(dt)
    check_string(underlying_symbol)
    underlying_symbol = underlying_symbol.upper()
    store = get_security_store()
    calendar_store = get_calendar_store()
    current_trade_date = calendar_store.get_current_trade_date(
        get_security_store().get_security('AU9999.XSGE'), dt)
    futures = store.get_all_securities(['futures'], current_trade_date)
    all_code = list(futures.index)
    all_code.sort()
    code = []
    for n in all_code:
        if '9999' not in n and '8888' not in n:
            res = re.findall(r"(.*)[0-9]{4}\.[a-zA-Z]+$", n)
            if len(res) == 1 and underlying_symbol == res[0].upper():
                code.append(n.upper())
    return code
