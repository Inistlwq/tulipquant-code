# coding:utf-8

import datetime
from jqdata.stores.calendar_store import get_calendar_store
from jqdata.utils.datetime_utils import parse_date
from jqdata.exceptions import ParamsError
from jqdata.stores.concept_store import get_concept_store
from jqdata.stores.industry_store import get_industry_store
import pandas as pd
from .data_utils import *

__all__ = [
    'get_trade_days',
    'get_all_trade_days',
    'get_concepts',
    'get_industries'
]


def get_trade_days(start_date=None, end_date=None, count=None):
    if start_date and count:
        raise ParamsError("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise ParamsError("count 参数需要大于 0 或者为 None")

    if not end_date:
        end_date = datetime.date.today()
    else:
        end_date = parse_date(end_date)

    store = get_calendar_store()
    if start_date:
        start_date = parse_date(start_date)
        return store.get_trade_days_between(start_date, end_date)
    elif count is not None:
        return store.get_trade_days_by_count(end_date, count)
    else:
        raise ParamsError("start_date 参数与 count 参数必须输入一个")


def get_all_trade_days():
    store = get_calendar_store()
    return store.get_all_trade_days()


def get_concepts():
    store = get_concept_store()
    ret = store.get_concepts()
    fields = ['name', 'start_date']
    index = [r['code'] for r in ret]
    dict_by_column = {}
    for f in fields:
        dict_by_column[f] = [r[f] for r in ret]
    df = pd.DataFrame(index=index, columns=fields, data=dict_by_column)
    df['start_date'] = pd.to_datetime(df['start_date'])
    return df
    pass


def get_industries(name):
    store = get_industry_store()
    ret = store.get_industries(name)
    fields = ['name', 'start_date']
    index = [r['code'] for r in ret]
    dict_by_column = {}
    for f in fields:
        dict_by_column[f] = [r[f] for r in ret]
    df = pd.DataFrame(index=index, columns=fields, data=dict_by_column)
    df['start_date'] = pd.to_datetime(df['start_date'])
    return df
    pass
