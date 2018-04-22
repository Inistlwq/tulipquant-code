# -*- coding: utf-8 -*-

import re
import warnings

import pandas as pd
from cached_property import cached_property

from jqdata.apis.data import get_price
from jqdata.apis import get_money_flow
from jqdata.apis.security import normalize_code
from jqdata.apis import get_fundamentals, valuation, query as jqdata_query
from jqdata.stores.calendar_store import get_calendar_store
from jqdata.stores.security_store import get_security_store
from jqdata.stores.index_store import get_index_store
from jqdata.stores.industry_store import IndustryStore_Sqlite
from jqdata.pk_tables import (IndustryEntity, ConceptEntity, IndexEntity,
                              get_session as jqdata_get_session)

from jqfactor.when import Quarter, convert_date
from jqfactor.logger import redirect_stdout_to_userlog
from jqfactor.fundamentals import finance
from jqfactor.extend import ExtendedDataGetter


class MultiFactorCalculator():
    """多因子计算"""

    def __init__(self):
        self.cal_store = get_calendar_store()
        self.sec_store = get_security_store()
        self.idx_store = get_index_store()
        self.idst_store = IndustryStore_Sqlite()
        self.extended_store = ExtendedDataGetter()

    @cached_property
    def industries_raw(self):
        session = jqdata_get_session()
        indus_types = ['sw_l1', 'sw_l2', 'sw_l3', 'zjw', 'jq_l1', 'jq_l2']
        codes = session.query(
            IndustryEntity.code
        ).filter(
            IndustryEntity.type_.in_(indus_types)
        ).distinct().all()
        return {cs[0] for cs in codes}

    @cached_property
    def markets_raw(self):
        return {'open', 'close', 'high', 'low', 'volume', 'money'}

    @cached_property
    def fundamentals_raw(self):
        return set(finance.keys())

    @cached_property
    def concepts_raw(self):
        session = jqdata_get_session()
        codes = session.query(
            ConceptEntity.code
        ).filter(
            ConceptEntity.name != ''
        ).distinct().all()
        return {cs[0] for cs in codes}

    @cached_property
    def indexes_raw(self):
        session = jqdata_get_session()
        idxes = session.query(
            IndexEntity.code
        ).filter(
            IndexEntity.index_json != ''
        ).distinct().all()
        return {item[0] for item in idxes}

    @cached_property
    def money_flows_raw(self):
        return {
            "change_pct", "net_amount_main", "net_pct_main", "net_amount_xl",
            "net_pct_xl", "net_amount_l", "net_pct_l", "net_amount_m",
            "net_pct_m", "net_amount_s", "net_pct_s"
        }

    @cached_property
    def extended_raw(self):
        return self.extended_store.get_names()

    def _sort_factors(self, factor_dependencies_dict, factor_raw):
        """排序依赖因子

        将需要排序的因子和其依赖因子比对，将不存在于依赖中的因子加在 list 后面

        参数：
            factor_dependencies_dict: 因子的依赖字典
            factor_raw: 需要排序的因子列表

        如果存在循环依赖则抛出异常
        """
        if len(factor_raw) == 0:
            return []
        depends = []
        for fac in factor_raw:
            if fac not in factor_dependencies_dict:
                raise Exception("invalid factor")
            else:
                depends = depends + factor_dependencies_dict[fac]
        depends = list(set(depends))
        diff = list(set(factor_raw).difference(set(depends)))
        if len(diff) > 0:
            return self._sort_factors(factor_dependencies_dict, depends) + diff
        elif set(factor_raw) == set(depends):
            raise Exception("factor dependencies infinite loop")

    def _filter_factors(self, factors, init_markets=None):
        """筛选出依赖的基础因子筛选出"""
        markets = init_markets or []
        industries = []
        concepts = []
        indexes = []
        fundamentals = {}
        money_flows = []
        derived_fundamentals = {}
        extended = []

        dependencies = {}
        dfunds_ptn = re.compile(r"^_[1-8]$")
        for fact in factors:
            dep_list = []
            for factd in fact.dependencies:
                if factd in self.industries_raw:
                    industries.append(factd)
                elif factd in self.concepts_raw:
                    concepts.append(factd)
                elif factd in self.indexes_raw:
                    indexes.append(factd)
                elif factd in self.markets_raw:
                    markets.append(factd)
                elif factd in self.fundamentals_raw:
                    fundamentals[factd] = finance[factd]
                elif factd in self.money_flows_raw:
                    money_flows.append(factd)
                elif factd in self.extended_raw:
                    extended.append(factd)
                elif dfunds_ptn.search(factd[-2:]) and factd[:-2] in self.fundamentals_raw:
                    derived_fundamentals[factd] = finance[factd[:-2]]
                else:
                    # TODO 过滤不存在的因子
                    dep_list.append(factd)
            dependencies[fact.name] = dep_list

        industries = list(set(industries))
        markets = list(set(markets))

        # 将因子排序，把被其他因子依赖的因子放在前面先计算
        ordered_factors = self._sort_factors(dependencies, dependencies.keys())

        return {
            "markets": markets,
            "industries": industries,
            "concepts": concepts,
            "indexes": indexes,
            "fundamentals": fundamentals,
            "money_flows": money_flows,
            "derived_fundamentals": derived_fundamentals,
            "extended": extended,
            "ordered_factors": ordered_factors,
        }

    @staticmethod
    def _calc_max_window_len(factors):
        """计算最大窗口"""
        fact_len = {}
        for name in factors.keys():
            fact_window = factors[name].max_window
            fact_max = fact_window
            for dep in factors[name].dependencies:
                if dep in fact_len:
                    window = fact_window + fact_len[dep]
                    if window > fact_max:
                        fact_max = window
            fact_len[name] = fact_max
        len_values = fact_len.values()
        return max(len_values)

    @staticmethod
    def _get_base_data(securities, dates, markets, fundamentals):
        """获取基础数据"""
        data = []

        markets_len = len(markets)
        if markets_len:
            prices = get_price(securities, dates[0], dates[-1], '1d', markets, fq='post',
                               pre_factor_ref_date=dates[-1])
            prices.major_axis = prices.major_axis.map(lambda dt: dt.date)
            data.append(prices)

        fundamentals_len = len(fundamentals)
        if fundamentals_len:
            q = jqdata_query(
                valuation.code, *fundamentals.values()
            ).filter(
                valuation.code.in_(securities)
            )
            fund_dict = {day: get_fundamentals(q, date=day).set_index("code") for day in dates}
            funds = pd.Panel(fund_dict)
            funds = funds.transpose(2, 0, 1)
            data.append(funds)

        data_len = len(data)
        if data_len == 0:
            return pd.Panel()
        elif data_len == 1:
            return data[0]
        else:
            return pd.concat(data)

    def _get_industry_data(self, securities, dates, industries):
        """获取行业数据，每个行业作为一个因子"""
        if not len(industries):
            return {}

        _df = pd.DataFrame(data=0, index=dates, columns=securities)
        data = {idst: _df.copy() for idst in industries}
        sec_codes = self.idst_store.get_security_industry_date(securities, industries)
        for date in dates:
            for sc in sec_codes:
                if str(date) >= sc['stock_startdate'] and str(date) <= sc['stock_enddate']:
                    data[sc['code']].loc[date, sc['stock']] = 1

        return data

    def _get_concept_data(self, securities, dates, concepts):
        """获取概念数据"""
        if not len(concepts):
            return {}

        session = jqdata_get_session()
        con_codes = session.query(
            ConceptEntity.code,
            ConceptEntity.stock,
            ConceptEntity.stock_startdate,
            ConceptEntity.stock_enddate
        ).filter(
            ConceptEntity.stock.in_(securities)
        ).filter(
            ConceptEntity.code.in_(concepts)
        ).distinct().all()
        con_codes = pd.DataFrame(data=con_codes,
                                 columns=["code", "stock", "stock_startdate", "stock_enddate"])

        _df = pd.DataFrame(data=0, index=dates, columns=securities)
        data = {con: _df.copy() for con in concepts}
        for date in dates:
            for _, cc in con_codes.iterrows():
                if str(date) >= cc.stock_startdate and str(date) <= cc.stock_enddate:
                    data[cc.code].loc[date, cc.stock] = 1

        return data

    def _get_index_data(self, securities, dates, indexes):
        """获取指数数据"""
        data = {}
        for idx in indexes:
            df = pd.DataFrame(data=0, index=dates, columns=securities)
            for date in dates:
                stocks = self.idx_store.get_index_stocks(idx, date)
                for sec in securities:
                    if sec in stocks:
                        df.loc[date, sec] = 1
            data[idx] = df
        return data

    def _get_money_flow_data(self, securities, dates, money_flows):
        """获取资金流数据"""
        data = {}
        if not len(money_flows):
            return data

        fields = ["date", "sec_code"] + money_flows
        df = get_money_flow(securities, dates[0], dates[-1], fields)
        df["sec_code"] = df["sec_code"].apply(normalize_code)
        df.set_index(["date", "sec_code"], inplace=True)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            for flow in money_flows:
                data[flow] = df[flow].unstack()

        return data

    @staticmethod
    def _get_derived_fundamental_data(securities, dates, derived_fundamentals):
        """获取基本面因子的衍生数据"""
        data = {}
        if not len(derived_fundamentals):
            return data

        for factor_name, fund_col in derived_fundamentals.items():
            q = jqdata_query(
                valuation.code, fund_col
            ).filter(
                valuation.code.in_(securities)
            )
            col, pre_qs = factor_name.rsplit("_", 1)
            pre_qs = int(pre_qs)
            fund_dict = {
                date: get_fundamentals(
                    q,
                    statDate=str(Quarter(date) - 2 - pre_qs)
                ).set_index("code") for date in dates
            }
            funds = pd.Panel(fund_dict)
            funds = funds.transpose(2, 0, 1)
            data[factor_name] = funds[col]

        return data

    def __call__(self, securities, factors, start_date, end_date,
                 init_markets=None, redirect_calc_output=False):
        """计算多个因子

        init_markets 为需要额外计算的量价因子

        返回给定因子及其依赖因子，结果为 dict，key 因子名，value 为因子计算结果
        """
        start_date = convert_date(start_date)
        end_date = convert_date(end_date)

        tidied_factors = self._filter_factors(factors, init_markets)

        ordered_factors = tidied_factors["ordered_factors"]
        name_fact_dict = {f.name: f for f in factors}
        max_len = self._calc_max_window_len(name_fact_dict)

        dates = self.cal_store.get_trade_days_between(start_date, end_date)
        dates = self.cal_store.get_trade_days_by_count(end_date, len(dates) + max_len)

        fundamentals = tidied_factors["fundamentals"]
        markets = tidied_factors["markets"]
        base_data = self._get_base_data(securities, dates, markets, fundamentals)

        industries = tidied_factors["industries"]
        industry_data = self._get_industry_data(securities, dates, industries)

        concepts = tidied_factors["concepts"]
        concept_data = self._get_concept_data(securities, dates, concepts)

        indexes = tidied_factors["indexes"]
        index_data = self._get_index_data(securities, dates, indexes)

        money_flows = tidied_factors["money_flows"]
        money_flow_data = self._get_money_flow_data(securities, dates, money_flows)

        derived_fundamentals = tidied_factors["derived_fundamentals"]
        d_fund_data = self._get_derived_fundamental_data(securities, dates,
                                                         derived_fundamentals)

        extended = tidied_factors["extended"]
        extended_data = self.extended_store.get_data(securities, dates, extended)

        fact_dict = {}
        fact_dict.update(industry_data)
        fact_dict.update(concept_data)
        fact_dict.update(index_data)
        fact_dict.update(money_flow_data)
        fact_dict.update(d_fund_data)
        fact_dict.update(extended_data)
        fact_dict.update({item: base_data[item] for item in base_data.items})
        fact_dict.update({fact: pd.DataFrame(index=dates, columns=securities)
                          for fact in ordered_factors})

        for df in fact_dict.values():
            df.index = pd.to_datetime(df.index)

        for fact in ordered_factors:
            factor = name_fact_dict[fact]
            for idx in range(max_len, len(dates)):
                start_idx = idx - factor.max_window
                data = {fn: df[start_idx:idx] for fn, df in fact_dict.items()
                        if fn in factor.dependencies}
                if redirect_calc_output:
                    with redirect_stdout_to_userlog():
                        f_row = factor.calc(data)
                else:
                    f_row = factor.calc(data)
                if f_row is not None:
                    fact_dict[fact].iloc[idx] = f_row

        return {name: df[max_len:] for name, df in fact_dict.items()}


calc_multiple_factors = MultiFactorCalculator()
