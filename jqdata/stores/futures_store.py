#!/usr/bin/env python
#coding:utf-8

from fastcache import clru_cache as lru_cache
import bcolz

import jqdata
from jqdata.exceptions import ParamsError
from jqdata.utils.datetime_utils import to_timestamp, to_date, vec2date, to_datetime
import numpy as np

from .bcolz_utils import _Table, retry_bcolz_open

__all__ = [
    'get_futures_store',
    'FuturesStore',
]

class FuturesStore(object):

    def __init__(self):
        pass


    def __str__(self):
        return 'BcolzFS'

    @staticmethod
    def instance():
        if not hasattr(FuturesStore, "_instance"):
            FuturesStore._instance = FuturesStore()
        return FuturesStore._instance

    @staticmethod
    def set_cache():
        cls = FuturesStore
        funcs = ['open_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if not hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, lru_cache(None)(getattr(cls, func_name)))

    @staticmethod
    def unset_cache():
        cls = FuturesStore
        funcs = ['open_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "__wrapped__"):
                setattr(cls, func_name, getattr(func_cls, "__wrapped__"))

    @staticmethod
    def clear_cache():
        cls = FuturesStore
        funcs = ['open_table']
        for func_name in funcs:
            func_cls = getattr(cls, func_name)
            if hasattr(func_cls, "cache_clear"):
                func_cls.cache_clear()

    @lru_cache(None)
    def open_table(self, security):
        p = jqdata.get_config().get_bcolz_day_path(security)
        ct = retry_bcolz_open(p)
        return _Table(ct, ct.cols['date'][:])


    def query(self, security, dates, field):
        n = len(dates)
        if n == 0:
            return np.array([])

        ct = self.open_table(security)
        start_ts = to_timestamp(dates[0])
        end_ts = to_timestamp(dates[-1])

        start_idx = ct.index.searchsorted(start_ts)
        end_idx = ct.index.searchsorted(end_ts, 'right') - 1
        if end_idx < start_idx:
            return np.array([np.nan] * n)
        if field == 'futures_sett_price':
            name = 'settlement'
        elif field == 'futures_positions':
            name = 'open_interest'
        else:
            raise ParamsError("filed should in (futures_sett_price, open_interest)")

        ret = np.round(ct.table.cols[name][start_idx:end_idx+1], security.price_decimals)

        if len(ret) < n:
            st = to_date(ct.table.cols['date'][start_idx])
            et = to_date(ct.table.cols['date'][end_idx])
            for i in range(0, n):
                if dates[i] >= st:
                    break
            if i > 0:
                ret = np.concatenate([np.array([np.nan]*i), ret])
            for i in range(n-1, -1, -1):
                if dates[i] <= et:
                    break
            if i < n - 1:
                ret = np.concatenate([ret, np.array([np.nan] * (n - i - 1))])
        return ret


def get_futures_store():
    return FuturesStore.instance()


