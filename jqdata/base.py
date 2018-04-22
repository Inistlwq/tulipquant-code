# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import os,re,json,requests,datetime,contextlib,abc
from fastcache import clru_cache as lru_cache
from retrying import retry

import numpy as np

DATA_SERVER = os.environ.get('JQDATA_API_SERVER', 'http://jqdata:8000')

from .utils import *

@lru_cache(None)
def get_trade_days_from_json():
    this_dir = os.path.dirname(__file__)
    with open(this_dir + '/data/all_trade_days.json') as f:
        days = json.load(f)
    return np.array([convert_date(day) for day in days])

def request_data_no_retry(url, payload=None, method="GET"):
    with contextlib.closing(requests.get(url, params=payload) if method == "GET" \
            else requests.post(url, data=payload)) as req:
        req.raise_for_status()
        res = req.json()
        if int(res["code"]) != 0:
            raise Exception(req.text)
        data = res.get("data", None)
        return data

request_data = retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)(request_data_no_retry)

def json_serial_fallback(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return str(obj)
    raise TypeError ("%s not serializable" % obj)

def is_trade_day(date):
    return date in get_trade_days_from_json()

def get_prev_trade_day(date):
    """获取前一个交易日

    如果 date 不是交易日, 则返回该日期之前的最后一个交易日
    """
    convert_date(date)
    days = get_trade_days_from_json()
    if not (days[0] <= date <= days[-1]):
        raise Exception("%s 不在规定的日期范围内" % date)
    pos = np.searchsorted(days, date)
    return days[pos-1] if pos >= 1 else None
