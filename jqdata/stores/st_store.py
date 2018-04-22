#!/usr/bin/env python

import os
import pickle
import numpy as np

from fastcache import clru_cache as lru_cache
import jqdata
from jqdata.utils.datetime_utils import parse_date


__all__ = ['get_st_store', 'StStore']

class StStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)
            for code in self._dic.keys():
                self._dic[code] = [parse_date(s) for s in self._dic[code]]

    @staticmethod
    def instance():
        if not hasattr(StStore, '_instance'):
            StStore._instance = StStore(jqdata.get_config().get_st_pk())
        return StStore._instance


    def get_periods(self, code):
        data = self._dic.get(code, [])
        if len(data) == 0:
            return []
        ret = []
        for i in range(0, len(data)-1, 2):
            st = data[i]
            et = data[i+1]
            ret.append([str(st), str(et)])
        return ret


    def is_st(self, code, date):
        periods = self._dic.get(code, [])
        if not periods:
            return False
        for i in range(0, len(periods)-1, 2):
            st = periods[i]
            et = periods[i+1]
            if date < st:
                return False
            if st <= date and date <= et:
                return True
        return False

    def query(self, security, dates):
        ret = [self.is_st(security.code, date) for date in dates]
        return np.array(ret)


def get_st_store():
    return StStore.instance()

