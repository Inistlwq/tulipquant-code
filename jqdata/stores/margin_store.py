#!/usr/bin/env python
#coding: utf-8
import os
import pickle
import datetime
import six

from fastcache import clru_cache as lru_cache
import jqdata
import json
from jqdata.exceptions import ParamsError
from ..pk_tables import MarginStockEntity, get_session

__all__ = [
    'get_margin_store',
    'MarginStore_Pk',
    'MarginStore_Sqlite',
]

class MarginStore_Pk(object):

    def __init__(self, f):
        if os.path.exists(f):
            with open(f, "rb") as store:
                self._dic = pickle.load(store)
                self._f = f
        else:
            self._dic = {}
            self._f = f

    @staticmethod
    def instance():
        if not hasattr(MarginStore_Pk, "_instance"):
            cfg = jqdata.get_config()
            MarginStore_Pk._instance = MarginStore_Pk(cfg.get_margin_pk())
        return MarginStore_Pk._instance

    def get_margin_stocks(self, day):
        res = self._dic.get(day, [])
        if res:
            return res
        ks = sorted(self._dic.keys())
        for i in range(len(ks) - 1, -1, -1):
            k = ks[i]
            res = self._dic.get(k)
            if res:
                return res
        return []

    def set_margin_stocks(self, day, stocks):
        self._dic[day] = stocks

    def save(self):
        with open(self._f, "wb") as fp:
            pickle.dump(self._dic, fp, protocol=pickle.HIGHEST_PROTOCOL)



class MarginStore_Sqlite(object):

    def __init__(self):
        pass

    @staticmethod
    def instance():
        if not hasattr(MarginStore_Sqlite, "_instance"):
            MarginStore_Sqlite._instance = MarginStore_Sqlite()
        return MarginStore_Sqlite._instance


    def get_margin_stocks(self, day):
        if isinstance(day, (datetime.date, datetime.datetime)):
            day_s = day.strftime("%Y-%m-%d")
        elif isinstance(day, six.string_types):
            day_s = day
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
        session = get_session()
        ms = session.query(MarginStockEntity).filter(MarginStockEntity.margin_date <= day_s).order_by(
            MarginStockEntity.margin_date.desc()).first()
        if ms is not None:
            res = json.loads(ms.margin_json)
        else:
            res = None
        if res:
            return res
        return []



@lru_cache(None)
def get_margin_store():
    import os
    if os.path.exists(jqdata.get_config().get_sqlite_pk()):
        return MarginStore_Sqlite.instance()
    else:
        return MarginStore_Pk.instance()
