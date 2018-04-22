#!/usr/bin/env python
#coding:utf-8
import os
import math
import time
from fastcache import clru_cache as lru_cache
import numpy as np
import bcolz

try:
    import SharedArray
except:
    pass

import jqdata
from jqdata.utils.datetime_utils import to_timestamp, to_date
from jqdata.utils.utils import fixed_round

from .bcolz_utils import retry_bcolz_open, _BenchTable
from .calendar_store import CalendarStore
__all__ = [
    'BenchmarkStore',
]


class BenchmarkStore(object):
    '''缓存benchmark close 和 factor, 快速计算benchmark的值。'''

    def __init__(self, security, db='bcolz', freq='day'):
        assert db in ['bcolz', 'shm']
        assert freq in ['day', 'minute']

        self.security = security
        self.db = db
        self.freq = freq

        if db == 'bcolz':
            if freq == 'day':
                p = jqdata.get_config().get_bcolz_day_path(security)
            else:
                p = jqdata.get_config().get_bcolz_minute_path(security)
            ct = retry_bcolz_open(p)
            self.table =  _BenchTable(ct.cols['close'][:],
                                      ct.cols['factor'][:],
                                      ct.cols['date'][:])
        else:
            if freq == 'day':
                p = jqdata.get_config().get_day_shm_path(security.code)
                arr = SharedArray.attach("file://" + p, readonly=True)
                date_idx = 0
                close_idx = security.day_column_names.index('close') + 1
                factor_idx = security.day_column_names.index('factor') + 1
                self.table = _BenchTable(arr[:, close_idx],
                                         arr[:, factor_idx],
                                         arr[:, date_idx])
            else:
                raise Exception("unsupport db=shm, freq=minute")


    def get_price(self, somedt):
        '''
        获取一支股票的后复权价格

        `security`: 股票代码，例如 '000001.XSHE'。
        `date`: 如果date天不存在数据，返回上一个交易日的数据。
        '''
        ct = self.table
        ts = to_timestamp(somedt)
        idx = ct.index.searchsorted(ts, side='right') - 1
        if idx < 0:
            return np.nan
        bar_date = to_date(ct.index[idx])
        if bar_date == somedt.date():
            open_dt = CalendarStore.instance().get_open_dt(self.security, somedt.date())
            if somedt < open_dt:
                idx = idx -1
                if idx < 0:
                    return np.nan

        return fixed_round(ct.closes[idx] * ct.factors[idx],
                           self.security.price_decimals)






