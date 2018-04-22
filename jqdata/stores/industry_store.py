#!/usr/bin/env python
# coding:utf-8

import datetime
import pickle
import six
from fastcache import clru_cache as lru_cache
import jqdata
from jqdata.exceptions import ParamsError

from .security_store import SecurityStore
from ..pk_tables import IndustryEntity, get_session

__all__ = [
    'get_industry_store',
    'IndustryStore',
    'IndustryStore_Sqlite'
]


def check_date(date):
    if isinstance(date, (datetime.date, datetime.datetime)):
        date_s = date.strftime("%Y-%m-%d")
    elif isinstance(date, six.string_types):
        date_s = date
    else:
        raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
    return date_s
    pass


class IndustryStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(IndustryStore, "_instance"):
            IndustryStore._instance = IndustryStore(jqdata.get_config().get_industry_pk())
        return IndustryStore._instance

    def get_industry_stocks(self, industry_code, date):
        '''获取行业代码在指定日期的股票列表'''
        assert isinstance(industry_code, six.string_types)
        stocks = self._dic['stocks'].get(industry_code.upper())
        if not stocks:
            if industry_code not in self._dic['codes']['csrc'] and \
               industry_code not in self._dic['codes']['wind'] and \
               industry_code not in self._dic['codes']['sw']:
                raise ParamsError(u"行业'%s'不存在" % industry_code)
            return []
        ret = []
        date_s = check_date(date)
        for code, periods in stocks:
            if not SecurityStore.instance().exists(code):
                continue
            for i in range(0, len(periods) - 1, 2):
                # 不包括最后一天
                if periods[i] <= date_s and date_s < periods[i + 1]:
                    ret.append(code)
                    break

        return ret

    def get_security_industry(self, code, date):
        '''获取股票代码在指定日期的行业'''
        from bisect import bisect_left
        date_s = check_date(date)
        for industry_code in self._dic['codes']['csrc'].keys():
            stocks = self.get_industry_stocks(industry_code, date_s)
            stocks.sort()
            i = bisect_left(stocks, code)
            if i < len(stocks) and stocks[i] == code:
                return industry_code
        return None

    def get_industries(self, name):
        assert isinstance(name, six.string_types)
        if name in self._dic['lists']:
            return self._dic['lists'][name]
        else:
            raise ParamsError("name 参数必须是zjw/jq_l1/jq_l2/sw_l1/sw_l2/sw_l3中的一种")
        pass


class IndustryStore_Sqlite(object):

    def __init__(self):
        pass

    @staticmethod
    def instance():
        if not hasattr(IndustryStore_Sqlite, "_instance"):
            IndustryStore_Sqlite._instance = IndustryStore_Sqlite()
        return IndustryStore_Sqlite._instance

    def get_industry_stocks(self, industry_code, date):
        '''获取行业代码在指定日期的股票列表'''
        assert isinstance(industry_code, six.string_types)
        industry_code = industry_code.upper()
        date_s = check_date(date)
        session = get_session()
        count = session.query(IndustryEntity.code).filter(IndustryEntity.code == industry_code).count()
        if count == 0:
            raise ParamsError("行业板块 '%s' 不存在" % industry_code)
        stocks = session.query(IndustryEntity.stock).filter(IndustryEntity.code == industry_code).filter(
            IndustryEntity.stock_startdate <= date_s).filter(IndustryEntity.stock_enddate > date_s).filter(
            IndustryEntity.stock != '').distinct().all()
        ret = []
        for s, in stocks:
            ret.append(s)
        return ret

    def get_security_industry(self, code, date):
        '''获取股票代码在指定日期的行业'''
        date_s = check_date(date)
        session = get_session()
        industry = session.query(IndustryEntity.code).filter(IndustryEntity.stock == code).filter(
            IndustryEntity.stock_startdate <= date_s).filter(IndustryEntity.stock_enddate > date_s).distinct().all()
        ret = []
        for s, in industry:
            ret.append(s)
        if len(ret) > 0:
            return ret[0]
        else:
            return None

    def get_security_industry_pair(self, stock_list, start_date, end_date, industry_type):
        # 参数校验
        if not isinstance(stock_list, (list, tuple)):
            if stock_list is not None:
                stock_list = [stock_list]
        start_date = check_date(start_date)
        end_date = check_date(end_date)
        session = get_session()
        sec_ids_pair = session.query(IndustryEntity.stock, IndustryEntity.code).filter(
            IndustryEntity.type_ == industry_type).filter(IndustryEntity.stock.in_(stock_list)).filter(
            IndustryEntity.stock_startdate <= start_date).filter(
            IndustryEntity.stock_enddate >= end_date).distinct().all()
        result = {}
        for s, c in sec_ids_pair:
            result[s] = c
        for sec in stock_list:
            if sec not in result:
                result[sec] = 'NID'
        return result
        pass

    def get_security_industry_date(self, stock_list, industries):
        # 参数校验
        if not isinstance(stock_list, (list, tuple)):
            if stock_list is not None:
                stock_list = [stock_list]
        session = get_session()
        sec_ids_pair = session.query(IndustryEntity.stock, IndustryEntity.code, IndustryEntity.stock_startdate, IndustryEntity.stock_enddate).filter(
            IndustryEntity.stock.in_(stock_list)).filter(IndustryEntity.code.in_(industries)).distinct().all()
        result = []
        for sec, code, stock_startdate, stock_enddate in sec_ids_pair:
            item = {}
            item['stock'] = sec
            item['code'] = code
            item['stock_startdate'] = stock_startdate
            item['stock_enddate'] = stock_enddate
            result.append(item)
        return result
        pass

    def get_industries(self, name):
        assert isinstance(name, six.string_types)
        session = get_session()
        industries = session.query(IndustryEntity.code, IndustryEntity.name, IndustryEntity.start_date
                                   ).filter(IndustryEntity.type_ == name).distinct().all()
        ret = []
        for code, name, start_date in industries:
            item = {}
            item['code'] = code
            item['name'] = name
            item['start_date'] = start_date
            ret.append(item)
        return ret
        pass

    def get_stock_industry(self, stock):
        session = get_session()
        industry = session.query(IndustryEntity.code, IndustryEntity.name, IndustryEntity.type_,
                                 IndustryEntity.start_date, IndustryEntity.stock_startdate,
                                 IndustryEntity.stock_enddate).filter(IndustryEntity.stock == stock
                                                                      ).distinct().all()
        ret = []
        for code, name, type_, start_date, stock_startdate, stock_enddate in industry:
            item = {}
            item['code'] = code
            item['name'] = name
            item['type'] = type_
            item['start_date'] = start_date
            item['stock_startdate'] = stock_startdate
            item['stock_enddate'] = stock_enddate
            ret.append(item)
        return ret
        pass


def get_industry_store():
    import os
    if os.path.exists(jqdata.get_config().get_sqlite_pk()):
        return IndustryStore_Sqlite.instance()
    else:
        return IndustryStore.instance()
