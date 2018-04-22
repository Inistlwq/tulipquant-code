# -*- coding: utf-8 -*-
import pandas as pd
from jqdata.apis.data import get_price
from jqdata.apis import (
    get_fundamentals,
    query,
    valuation)
from datetime import date
from jqdata.stores.calendar_store import get_calendar_store
from jqfactor.fundamentals import finance
from jqfactor.utils import convert_date
from jqdata.apis.base import get_industries
from jqdata.stores.industry_store import IndustryStore_Sqlite
industry_store = IndustryStore_Sqlite.instance()


class Factor(object):
    """factor definition"""
    name = ''
    max_window = None
    dependencies = []

    def calc(self, df):
        """df: a pd.DataFrame, index: date, columns: factors"""
        pass


def factor_sort(fac_dep_dic, fac_raw):
    '''
    fac_dep_dic:因子的依赖字典
    fac_raw:需要排序的因子列表
    逻辑：将需要排序的因子和它们的依赖因子比对，将不存在于依赖中的因子加在list后面，依赖因子继续前面的逻辑
    '''
    if len(fac_raw) == 0:
        return []
    depends = []
    for fac in fac_raw:
        if fac not in fac_dep_dic:
            raise Exception("invalid factor")
        else:
            depends = depends + fac_dep_dic[fac]
    depends = list(set(depends))
    diff = list(set(fac_raw).difference(set(depends)))
    if len(diff) > 0:
        return factor_sort(fac_dep_dic, depends) + diff
    elif set(fac_raw) == set(depends):
        raise Exception("factor dependencies infinite loop")
    pass


def calc_factors(securities, factors, start_date, end_date):
    """
    securities: a list of str, security code
    factors: a list of Factor objects
    return a pd.Panel, axes: ['security', 'date', 'factor']
    """
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    indus_type = ['sw_l1', 'sw_l2', 'sw_l3', 'zjw', 'jq_l1', 'jq_l2']
    industry_raw = []
    for it in indus_type:
        ids = get_industries(it)
        for ds in ids.index:
            industry_raw.append(ds)
    markets_raw = ['open', 'close', 'high', 'low', 'volume', 'money']
    name_fac_dic = {f.name: f for f in factors}
    industry = []
    markets = ['close']
    fundamentals = {}
    # 将因子排序
    # 将依赖的market和finance/industry因子筛选出
    fac_dep_dic = {}
    for f in factors:
        dep_list = []
        for fd in f.dependencies:
            if fd in industry_raw:
                industry.append(fd)
            elif fd in markets_raw:
                markets.append(fd)
            elif fd in set(finance.keys()):
                fundamentals[fd] = finance[fd]
            else:
                dep_list.append(fd)
        fac_dep_dic[f.name] = dep_list
    fac_name_sorted = factor_sort(fac_dep_dic, fac_dep_dic.keys())
    industry = list(set(industry))
    markets = list(set(markets))
    # 计算最大窗口
    fac_length = {}
    for n in fac_name_sorted:
        fac_window = name_fac_dic[n].max_window
        fac_max = fac_window
        for d in name_fac_dic[n].dependencies:
            if d in fac_length:
                if fac_window + fac_length[d] > fac_max:
                    fac_max = fac_window + fac_length[d]
        fac_length[n] = fac_max
    len_value = fac_length.values()
    max_len = max(len_value)
    # 获取基础数据
    cal_store = get_calendar_store()
    days = cal_store.get_trade_days_between(start_date, end_date)
    days = cal_store.get_trade_days_by_count(end_date, len(days) + max_len)
    if len(fundamentals) > 0:
        q = query(valuation.code, *fundamentals.values()).filter(valuation.code.in_(securities))
        d = {day: get_fundamentals(q, date=day) for day in days}
        for df in d.values():
            df.index = df.code.values
            del df['code']
        pnl = pd.Panel(d)
        pnl = pnl.transpose(2, 0, 1)
        if len(markets) == 0:
            data = pnl
    if len(markets) > 0:
        p = get_price(securities, days[0], days[-1], '1d', markets, fq='post', pre_factor_ref_date=days[-1])
        p.major_axis = [date(x.year, x.month, x.day) for x in p.major_axis]
        if len(fundamentals) == 0:
            data = p
    if len(fundamentals) * len(markets) > 0:
        data = pd.concat([pnl, p])

    # 获取行业数据-每个行业作为一个因子
    fac_dict = {}
    if len(industry) > 0:
        for ds in industry:
            fac_dict[ds] = pd.DataFrame(index=[dt for dt in days], columns=securities)
            fac_dict[ds] = fac_dict[ds].fillna(0)
        sec_code = industry_store.get_security_industry_date(securities, industry)
        for dd in days:
            for sc in sec_code:
                if str(dd) >= sc['stock_startdate'] and str(dd) <= sc['stock_enddate']:
                    fac_dict[sc['code']][sc['stock']][dd] = 1

    # 填充dict
    for item in data.items:
        fac_dict[item] = data[item]
    for fac in fac_name_sorted:
        fac_dict[fac] = pd.DataFrame(index=[dt for dt in days], columns=securities)
    # 转换index格式为timestamp
    for df in fac_dict:
        fac_dict[df].index = pd.to_datetime(fac_dict[df].index)
    # 因子计算
    for fac in fac_name_sorted:
        factor = name_fac_dic[fac]
        for dix in range(max_len, len(days)):
            dic_tmp = {df: fac_dict[df][dix - factor.max_window:dix] for df in fac_dict}
            calc_f = factor.calc(dic_tmp)
            if calc_f is not None:
                fac_dict[fac].iloc[dix] = calc_f
    fac_dict = {df: fac_dict[df][max_len:] for df in fac_dict}
    return fac_dict
    pass
