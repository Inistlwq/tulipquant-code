#!/usr/bin/env python
#coding:utf-8
import datetime
import pickle
import six
from fastcache import clru_cache as lru_cache
import jqdata
import json
from jqdata.exceptions import ParamsError

from .security_store import SecurityStore
from ..pk_tables import IndexEntity, get_session

__all__ = [
    'get_index_store',
    'IndexStore_Pk',
    'IndexStore_Sqlite',
]

class IndexStore_Pk(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(IndexStore_Pk, "_instance"):
            IndexStore_Pk._instance = IndexStore_Pk(jqdata.get_config().get_index_pk())
        return IndexStore_Pk._instance


    @lru_cache(1024)
    def get_index_stocks(self, index_symbol, date):
        assert isinstance(index_symbol, six.string_types)
        index_symbol = index_symbol.upper()
        stocks = self._dic.get(index_symbol, [])
        if not stocks:
            if index_symbol not in SecurityStore.instance().get_all_indexs().keys():
                raise ParamsError("指数'%s'不存在" % index_symbol)
            else:
                return []
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
        ret = []
        for code, periods in stocks:
            for i in range(0, len(periods) - 1, 2):
                # 不包括最后一天
                if periods[i] <= date_s and date_s < periods[i + 1]:
                    ret.append(code)
                    break
        return ret


class IndexStore_Sqlite(object):

    def __init__(self):
        pass

    @staticmethod
    def instance():
        if not hasattr(IndexStore_Sqlite, "_instance"):
            IndexStore_Sqlite._instance = IndexStore_Sqlite()
        return IndexStore_Sqlite._instance


    @lru_cache(1024)
    def get_index_stocks(self, index_symbol, date):
        assert isinstance(index_symbol, six.string_types)
        index_symbol = index_symbol.upper()
        session = get_session()
        idx = session.query(IndexEntity).get(index_symbol)
        if idx is not None:
            stocks = json.loads(idx.index_json)
        else:
            stocks = []
        if not stocks:
            if index_symbol not in SecurityStore.instance().get_all_indexs().keys():
                raise ParamsError("指数'%s'不存在" % index_symbol)
            else:
                return []
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
        ret = []
        for code, periods in stocks:
            for i in range(0, len(periods) - 1, 2):
                # 不包括最后一天
                if periods[i] <= date_s and date_s < periods[i + 1]:
                    ret.append(code)
                    break
        return ret


def get_index_store():
    import os
    if os.path.exists(jqdata.get_config().get_sqlite_pk()):
        return IndexStore_Sqlite.instance()
    else:
        return IndexStore_Pk.instance()
