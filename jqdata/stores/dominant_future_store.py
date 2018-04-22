#!/usr/bin/env python
# coding: utf-8
import six
import os
import datetime
import pickle
import pandas as pd
import numpy as np
from fastcache import clru_cache as lru_cache
import jqdata
from .security_store import *
from .calendar_store import get_calendar_store
from ..utils.utils import convert_date

__all__ = [
    'DominantFutureStore',
    'get_dominant_future_store'
]


class DominantFutureStore(object):

    def __init__(self, f):
        if os.path.exists(f):
            with open(f, "rb") as store:
                self._dic = pickle.load(store)
                self._f = f
        else:
            self._dic = {}
            self._f = f
        dates = np.array([convert_date(i) for i in self._dic.keys()])
        self._dic_dates = dates[np.argsort(dates)]

    def set_dominant_code(self, day, codes):
        vcode = {}
        for k, v in codes.items():
            vcode[k] = self.add_suffix(v)
        self._dic[day] = vcode

    def get_dominant_code(self, day, symbol):
        date = str(day)
        data = self.get_recent_data(date)
        if data:
            ret = data.get(symbol, {})
            if ret:
                return ret
        return ""

    def get_recent_data(self, date):
        data = self._dic.get(date, {})
        if data:
            return data
        else:
            idx = self._dic_dates.searchsorted(convert_date(date))
            if idx == 0:
                rng = range(1, len(self._dic_dates))
            else:
                rng = range(1, idx)[::-1]
            for i in rng:
                date = str(self._dic_dates[i - 1])
                data = self._dic.get(date, {})
                if data:
                    return data
        return None
        pass

    def save(self):
        with open(self._f, "wb") as fp:
            pickle.dump(self._dic, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def add_suffix(self, code):
        for f in get_security_store().get_all_futures():
            if code == f[:-5]:
                return f
        return code
        pass

    @staticmethod
    def instance():
        if not hasattr(DominantFutureStore, '_instance'):
            DominantFutureStore._instance = DominantFutureStore(
                jqdata.get_config().get_dominant_future_pk())
        return DominantFutureStore._instance


def get_dominant_future_store():
    return DominantFutureStore.instance()
