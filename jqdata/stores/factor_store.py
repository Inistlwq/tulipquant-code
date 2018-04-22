#!/usr/bin/env python
#coding:utf-8

import pickle
import six
import datetime
import jqdata

from jqdata.exceptions import ParamsError

__all__ = [
    'get_factor_store',
    'FactorStore',
]

class FactorStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(FactorStore, "_instance"):
            FactorStore._instance = FactorStore(jqdata.get_config().get_factor_pk())
        return FactorStore._instance


    def get_factor_by_dates(self, code, dates):
        ret = [self.get_factor_by_date(code, date) for date in dates]
        return ret


    def get_factor_by_date(self, code, date):
        factors = self._dic.get(code)
        if not factors:
            return 1.0
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
        for i in range(len(factors) - 1, -1, -1):
            if factors[i][0] <= date_s:
                return factors[i][1]
        return 1.0
        

def get_factor_store():
    return FactorStore.instance()
