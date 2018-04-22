#!/usr/bin/env python
# coding: utf-8
import six
import datetime
import pickle
import pandas as pd
from fastcache import clru_cache as lru_cache
import jqdata
from jqdata.finance_table import (
    FundShareInfo,
    FundPortfolioStock,
    FundPortfolioBond,
    get_session)

__all__ = [
    'OpenFundStore',
    'get_open_fund_store'
]


class OpenFundStore(object):

    def __init__(self, f):
        self._dic = {}
        self.load(f)

    def load(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(OpenFundStore, '_instance'):
            OpenFundStore._instance = OpenFundStore(jqdata.get_config().get_open_fund_info_pk())
        return OpenFundStore._instance

    def get_fund_info(self, security, date):
        # 某些字段懒加载
        code = str(security).split('.')[0]
        ret = self._dic.get(security)
        if ret:
            session = get_session()
            # 获取最近日期
            day_info = session.query(FundShareInfo.code, FundShareInfo.pub_date).filter(FundShareInfo.code == code).filter(FundShareInfo.pub_date <= date).order_by(FundShareInfo.pub_date.desc()).limit(1).all()
            if day_info:
                share_info = session.query(FundShareInfo.end_share, FundShareInfo.pub_date).filter(FundShareInfo.code == code).filter(FundShareInfo.pub_date == day_info[0].pub_date).distinct().all()
                if len(share_info) > 0:
                    ret['fund_share'] = float(share_info[0].end_share)
            stock_day = session.query(FundPortfolioStock.code, FundPortfolioStock.pub_date).filter(FundPortfolioStock.code == code).filter(FundPortfolioStock.pub_date <= date).order_by(FundPortfolioStock.pub_date.desc()).limit(1).all()
            if stock_day:
                stock_share = session.query(FundPortfolioStock.symbol, FundPortfolioStock.proportion).filter(FundPortfolioStock.code == code).filter(FundPortfolioStock.pub_date == stock_day[0].pub_date).distinct().all()
                if stock_share:
                    ret['heavy_hold_stocks'] = [str(ss.symbol) for ss in stock_share]
                    ret['heavy_hold_stocks_proportion'] = sum([float(ss.proportion) for ss in stock_share])
            bond_day = session.query(FundPortfolioBond.code, FundPortfolioBond.pub_date).filter(FundPortfolioBond.code == code).filter(FundPortfolioBond.pub_date <= date).order_by(FundPortfolioBond.pub_date.desc()).limit(1).all()
            if bond_day:
                bond_share = session.query(FundPortfolioBond.symbol, FundPortfolioBond.proportion).filter(FundPortfolioBond.code == code).filter(FundPortfolioBond.pub_date == bond_day[0].pub_date).distinct().all()
                if bond_share:
                    ret['heavy_hold_bond'] = [str(bs.symbol) for bs in bond_share]
                    ret['heavy_hold_bond_proportion'] = sum([float(bs.proportion) for bs in bond_share])
        return ret


def get_open_fund_store():
    return OpenFundStore.instance()