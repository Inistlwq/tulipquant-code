#!/usr/bin/env python
#coding:utf-8

import pickle
import datetime
import six

import jqdata
from jqdata.utils.datetime_utils import parse_date

__all__ = [
    'get_dividend_store',
    'DividendStore',
]


class DividendStore(object):

    def __init__(self, f):
        self._dic = {}
        self._otcdic = {}
        self.load(f)

    def load(self, f):
        with open(f, "rb") as store:
            dic = pickle.load(store)
            for i in dic:
                self._dic[i] = dic[i]

    def load_open_fund(self):
        cfg = jqdata.get_config()
        f = cfg.get_open_fund_dividend_pk()
        with open(f, "rb") as store:
            dic = pickle.load(store)
            for i in dic:
                self._otcdic[i] = dic[i]

    @staticmethod
    def instance():
        if not hasattr(DividendStore, "_instance"):
            cfg = jqdata.get_config()
            DividendStore._instance = DividendStore(cfg.get_dividend_pk())
        return DividendStore._instance

    def get_split_dividend(self, code, start_date, end_date):
        ret = []
        # 普通股票分红
        infos = self._dic.get(code)
        if infos:
            start_date_s = parse_date(start_date).strftime("%Y-%m-%d")
            end_date_s = parse_date(end_date).strftime("%Y-%m-%d")
            for info in infos:
                if start_date_s <= info[0] <= end_date_s:
                    # 每十股: 送股，转增股，派现税前，派现税后
                    date_s, stock_paid, into_shares, bonus_pre_tax = info[:4]
                    ret.append({
                        'date': parse_date(date_s),
                        'bonus_pre_tax': bonus_pre_tax / 10.0,
                        'scale_factor': (stock_paid + into_shares) / 10.0 + 1.,
                    })
            return ret
        # 场外基金分红
        if code.endswith('.OF'):
            if len(self._otcdic) == 0:
                self.load_open_fund()
        else:
            return ret
        otc_infos = self._otcdic.get(code)
        if otc_infos:
            start_date_s = parse_date(start_date).strftime("%Y-%m-%d")
            end_date_s = parse_date(end_date).strftime("%Y-%m-%d")
            for info in otc_infos:
                ex_date = info['ex_date']
                otc_ex_date = info['otc_ex_date']
                date = None
                if ex_date < '2200-01-01':
                    date = ex_date
                else:
                    date = otc_ex_date
                if start_date_s <= date <= end_date_s:
                    proportion = info['proportion']
                    if proportion:
                        proportion = float(proportion)
                    else:
                        proportion = 0
                    split_ratio = info['split_ratio']
                    if split_ratio:
                        split_ratio = float(split_ratio)
                    else:
                        split_ratio = 0
                    ret.append({
                        'date': parse_date(date),
                        'proportion': proportion,
                        'split_ratio': split_ratio,
                    })
            return ret
        return ret


def get_dividend_store():
    return DividendStore.instance()
