#!/usr/bin/env python
# coding: utf-8
import six
import datetime
import pickle
import pandas as pd
from fastcache import clru_cache as lru_cache
import jqdata
from jqdata.models.security import Security

__all__ = [
    'SecurityStore',
    'get_security_store'
]


class SecurityStore(object):

    def __init__(self, f):
        self._dic = {}
        self._otcdic = {}
        self.load(f)

    def load(self, f):
        with open(f, "rb") as store:
            dic = pickle.load(store)
            for i in dic:
                self._dic[i] = Security(**dic[i])

    def load_open_fund(self):
        f = jqdata.get_config().get_open_fund_security_pk()
        with open(f, "rb") as store:
            dic = pickle.load(store)
            for i in dic:
                self._otcdic[i] = Security(**dic[i])
        pass

    @staticmethod
    def instance():
        if not hasattr(SecurityStore, '_instance'):
            SecurityStore._instance = SecurityStore(jqdata.get_config().get_security_pk())
        return SecurityStore._instance

    def get_all_securities(self, types=[], date=None):
        '''
        所有证劵（不包括场外基金）
        :param types:
        :param date:
        :return:
        '''
        if not types:
            r = self.get_all_stocks()
        else:
            maps = {
                'stock': self.get_all_stocks,
                'index': self.get_all_indexs,
                'fund': self.get_all_funds,
                'futures': self.get_all_futures,
                'etf': self.get_all_etf,
                'lof': self.get_all_lof,
                'fja': self.get_all_fja,
                'fjb': self.get_all_fjb,
                'open_fund': self.get_all_otcfunds,
                'bond_fund': self.get_all_bondfunds,
                'stock_fund': self.get_all_stockfunds,
                'QDII_fund': self.get_all_QDIIfunds,
                'money_market_fund': self.get_all_money_market_funds,
                'mixture_fund': self.get_all_mixturefunds
            }
            r = {}
            for t in types:
                assert (t in ('stock', 'index', 'fund', 'futures', 'etf', 'lof', 'fja', 'fjb', 'open_fund',
                              'bond_fund', 'stock_fund', 'QDII_fund', 'money_market_fund', 'mixture_fund'))
                r.update(maps[t]())
        if date:
            from jqdata.utils.datetime_utils import parse_date
            if isinstance(date, six.string_types) and ":" in date:
                date = date[:10]
            date = parse_date(date)
            ss = [v for v in r.values() if (v.start_date <= date and date <= v.end_date)]
        else:
            ss = r.values()
        ss = list(ss)
        ss.sort(key=lambda s: s.code)
        columns = ('display_name', 'name', 'start_date', 'end_date', 'type')
        res = dict(index=[s.code for s in ss],
                   columns=columns,
                   data=[[(getattr(s, c)) for c in columns] for s in ss])
        df = pd.DataFrame(**res)
        return df

    @lru_cache(None)
    def get_all_indexs(self):
        ret = {}
        for i in self._dic:
            if self._dic[i].is_index():
                ret[i] = self._dic[i]
        return ret

    @lru_cache(None)
    def get_all_futures(self):
        ret = {}
        for i in self._dic:
            if self._dic[i].is_futures():
                ret[i] = self._dic[i]
        return ret

    @lru_cache(None)
    def get_commodities(self, underlying):
        '''
        :param underlying:
        :return:
        '''
        d = self.get_all_commodity_futures()
        res = {}
        for i in d:
            try:
                if d[i]._extra['underlying'] == underlying:
                    res[i] = d[i]
            except:
                pass
        return res

    @lru_cache(None)
    def get_all_commodity_futures(self):
        d = self.get_all_futures()
        ret = {}
        for i in d:
            if d[i].is_commodity_futures():
                ret[i] = d[i]
        return ret

    @lru_cache(None)
    def get_all_index_futures(self):
        d = self.get_all_futures()
        ret = {}
        for i in d:
            if d[i].is_index_futures():
                ret[i] = d[i]
        return ret

    @lru_cache(None)
    def get_all_stocks(self):
        ret = {}
        for i in self._dic:
            if self._dic[i].is_stock():
                ret[i] = self._dic[i]
        return ret

    @lru_cache(None)
    def get_all_otcfunds(self):
        '''
        获取所有场外基金
        :return:
        '''
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_open_fund():
                ret[i] = self._otcdic[i]
        return ret

    @lru_cache(None)
    def get_all_bondfunds(self):
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_bond_fund():
                ret[i] = self._otcdic[i]
        return ret
        pass

    @lru_cache(None)
    def get_all_stockfunds(self):
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_stock_fund():
                ret[i] = self._otcdic[i]
        return ret
        pass

    @lru_cache(None)
    def get_all_QDIIfunds(self):
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_QDII_fund():
                ret[i] = self._otcdic[i]
        return ret
        pass

    @lru_cache(None)
    def get_all_money_market_funds(self):
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_money_market_fund():
                ret[i] = self._otcdic[i]
        return ret
        pass

    @lru_cache(None)
    def get_all_mixturefunds(self):
        if len(self._otcdic) == 0:
            self.load_open_fund()
        ret = {}
        for i in self._otcdic:
            if self._otcdic[i].is_mixture_fund():
                ret[i] = self._otcdic[i]
        return ret
        pass

    @lru_cache(None)
    def get_all_funds(self):
        '''
        获取所有场内基金
        :return:
        '''
        ret = {}
        for i in self._dic:
            if self._dic[i].is_fund():
                ret[i] = self._dic[i]
        return ret

    @lru_cache(None)
    def get_all_etf(self):
        funds = self.get_all_funds()
        ret = {}
        for i in funds:
            if funds[i].subtype == 'etf':
                ret[i] = funds[i]
        return ret

    @lru_cache(None)
    def get_all_lof(self):
        funds = self.get_all_funds()
        ret = {}
        for i in funds:
            if funds[i].subtype == 'lof':
                ret[i] = funds[i]
        return ret

    @lru_cache(None)
    def get_all_fja(self):
        funds = self.get_all_funds()
        ret = {}
        for i in funds:
            if funds[i].subtype == 'fja':
                ret[i] = funds[i]
        return ret

    @lru_cache(None)
    def get_all_fjb(self):
        funds = self.get_all_funds()
        ret = {}
        for i in funds:
            if funds[i].subtype == 'fjb':
                ret[i] = funds[i]
        return ret

    def get_security(self, code):
        if code.endswith('.OF'):
            if len(self._otcdic) == 0:
                self.load_open_fund()
            return self._otcdic.get(code)
        else:
            return self._dic.get(code)

    def get_all_codes(self):
        if len(self._otcdic) == 0:
                self.load_open_fund()
        return self._dic.keys() + self._otcdic.keys()

    def get_all_values(self):
        if len(self._otcdic) == 0:
                self.load_open_fund()
        return self._dic.values() + self._otcdic.values()

    def exists(self, code):
        if code.endswith('.OF'):
            if len(self._otcdic) == 0:
                self.load_open_fund()
            return code in self._otcdic
        else:
            return code in self._dic


def get_security_store():
    return SecurityStore.instance()
