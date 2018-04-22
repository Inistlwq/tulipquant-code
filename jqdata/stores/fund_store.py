#!/usr/bin/env python
#coding:utf-8

from fastcache import clru_cache as lru_cache
import numpy as np
import bcolz

import jqdata
from jqdata.utils.datetime_utils import to_timestamp, to_date, vec2date, to_datetime
from jqdata.exceptions import ParamsError

from .bcolz_utils import _Table, retry_bcolz_open
__all__ = [
    'get_fund_store',
    'FundStore',
]



class FundStore(object):

    def __init__(self):
        pass


    @staticmethod
    def instance():
        if not hasattr(FundStore, "_instance"):
            FundStore._instance = FundStore()
        return FundStore._instance


    @lru_cache(None)
    def open_table(self, security):
        if security.is_open_fund():
            p = jqdata.get_config().get_bcolz_otcfund_path(security)
        else:
            p = jqdata.get_config().get_bcolz_fund_path(security)
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
        if field == 'acc_net_value':
            name = 'acc'
        elif field == 'unit_net_value':
            name = 'unit'
        else:
            raise ParamsError("field should in (acc_net_value, unit_net_value)")
        ret = np.round(ct.table.cols[name][start_idx:end_idx + 1], security.price_decimals)

        if len(ret) < n:
            dates = list(dates)
            raw_list = [np.nan] * n
            ret_idx = 0
            for idx in range(start_idx, end_idx + 1):
                date = to_date(ct.table.cols['date'][idx])
                day_idx = dates.index(date)
                raw_list[day_idx] = ret[ret_idx]
                ret_idx += 1
            ret = np.array(raw_list)
        return ret


def get_fund_store():
    return FundStore.instance()
