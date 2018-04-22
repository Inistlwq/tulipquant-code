#!/usr/bin/env python
# coding:utf-8
import pickle
import six
import datetime
import jqdata

from jqdata.exceptions import ParamsError

from .security_store import SecurityStore
from ..pk_tables import ConceptEntity, get_session

__all__ = [
    'get_concept_store',
    'ConceptStore',
    'ConceptStore_Sqlite',
]


class ConceptStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(ConceptStore, "_instance"):
            ConceptStore._instance = ConceptStore(jqdata.get_config().get_concept_pk())
        return ConceptStore._instance

    def get_concept_stocks(self, concept_code, date):
        assert isinstance(concept_code, six.string_types)
        concept_code = concept_code.upper()
        stocks = self._dic['stocks'].get(concept_code, [])
        if not stocks:
            if concept_code not in self._dic['name']:
                raise ParamsError("概念板块 '%s' 不存在" % concept_code)
            return []
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")

        ret = []
        for code, periods in stocks:
            if not SecurityStore.instance().exists(code):
                continue
            for i in range(0, len(periods) - 1, 2):
                # 不包括最后一天
                if periods[i] <= date_s and date_s < periods[i + 1]:
                    ret.append(code)
                    break
        return ret

    def get_concepts(self):
        return self._dic['list']
        pass


class ConceptStore_Sqlite(object):

    def __init__(self):
        pass

    @staticmethod
    def instance():
        if not hasattr(ConceptStore_Sqlite, "_instance"):
            ConceptStore_Sqlite._instance = ConceptStore_Sqlite()
        return ConceptStore_Sqlite._instance

    def get_concept_stocks(self, concept_code, date):
        assert isinstance(concept_code, six.string_types)
        concept_code = concept_code.upper()
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")

        session = get_session()
        count = session.query(ConceptEntity.stock).filter(ConceptEntity.code == concept_code).count()
        if count == 0:
            raise ParamsError("概念板块 '%s' 不存在" % concept_code)
        stocks = session.query(ConceptEntity.stock).filter(ConceptEntity.code == concept_code).filter(
            ConceptEntity.name != '').filter(ConceptEntity.stock_startdate <= date_s).filter(
            ConceptEntity.stock_enddate > date_s).distinct().all()
        ret = []
        for s, in stocks:
            ret.append(s)
        return ret

    def get_concepts(self):
        session = get_session()
        concepts = session.query(ConceptEntity.code, ConceptEntity.name,
                                 ConceptEntity.start_date).filter(ConceptEntity.name != '').distinct().all()
        ret = []
        for code, name, start_date in concepts:
            item = {}
            item['code'] = code
            item['name'] = name
            item['start_date'] = start_date
            ret.append(item)
        return ret
        pass

    def get_stock_concepts(self, stock):
        session = get_session()
        concepts = session.query(ConceptEntity.code, ConceptEntity.name, ConceptEntity.start_date,
                                 ConceptEntity.stock_startdate, ConceptEntity.stock_enddate
                                 ).filter(ConceptEntity.stock == stock).distinct().all()
        ret = []
        for code, name, start_date, stock_startdate, stock_enddate in concepts:
            item = {}
            item['code'] = code
            item['name'] = name
            item['start_date'] = start_date
            item['stock_startdate'] = stock_startdate
            item['stock_enddate'] = stock_enddate
            ret.append(item)
        return ret
        pass


def get_concept_store():
    import os
    if os.path.exists(jqdata.get_config().get_sqlite_pk()):
        return ConceptStore_Sqlite.instance()
    else:
        return ConceptStore.instance()
