# coding: utf-8

import numpy as np
import pandas as pd
import alphalens as al
from cached_property import cached_property
from fastcache import lru_cache

from jqdata.apis import get_industries
from jqdata.stores.industry_store import IndustryStore_Sqlite


class FactorAnalyzer(object):
    """单因子分析"""

    def __init__(self, start_date, end_date, stocks, industry_type, factor,
                 pricing_data):
        self._start_date = start_date
        self._end_date = end_date
        self._stocks = stocks
        self._industry_type = industry_type
        self._factor = factor
        self._pricing_data = pricing_data

    @cached_property
    def factor_data(self):
        industry_store = IndustryStore_Sqlite.instance()
        groupby = industry_store.get_security_industry_pair(self._stocks,
                                                            self._start_date,
                                                            self._end_date,
                                                            self._industry_type)

        industries = get_industries(self._industry_type)
        ids_labels = industries["name"].copy()
        ids_labels = ids_labels.to_dict()
        ids_labels['NID'] = u'不在指定行业内'

        return al.utils.get_clean_factor_and_forward_returns(
            self._factor,
            self._pricing_data,
            groupby=groupby,
            by_group=False,
            quantiles=None,
            bins=5,
            periods=(1, 5, 10),
            filter_zscore=20,
            groupby_labels=ids_labels
        )

    def mean_return_by_quantile(self):
        """收益分析

        用来画分位数收益的柱状图

        返回 pandas.DataFrame, index 是 factor_quantile, 值是(1，2，3，4，5),
        column 是 period 的值 (1，5，10)
        """
        mean_ret_quantile, _ = al.performance.mean_return_by_quantile(self.factor_data,
                                                                      by_group=False,
                                                                      demeaned=False)
        mean_compret_quantile = mean_ret_quantile.apply(al.utils.rate_of_return, axis=0)
        return mean_compret_quantile

    # 用来画各个分位数的累积收益曲线（cumulative return by quantile）， period 分别传入 1，5，10
    def cumulative_return_by_quantile(self, period=1):
        '''
        返回值：
            DataFrame
            index 是时间， column 是分位数
        '''

        mean_ret_quant_daily, std_quant_daily = al.performance.mean_return_by_quantile(self.factor_data,
                                                                                       by_date=True,
                                                                                       by_group=False,
                                                                                       demeaned=False)
        ret_wide = mean_ret_quant_daily[period].reset_index() \
            .pivot(index='date', columns='factor_quantile', values=period)

        if period > 1:
            compound_returns = lambda ret, period: ((np.nanmean(ret) + 1) ** (1. / period)) - 1
            ret_wide = pd.rolling_apply(ret_wide, period, compound_returns,
                                        min_periods=1, args=(period,))

        cum_ret = ret_wide.add(1).cumprod()
        cum_ret = cum_ret.loc[:, ::-1]

        return cum_ret

    # 多空组合收益， period 参数分别传入 1，5，10 获得 1， 5， 10期的分位数收益。
    def long_short_cum_return(self, period=1):
        '''
        返回值
            一个 Series， index 是时间
        '''
        factor_returns = al.performance.factor_returns(self.factor_data)[period]

        if period > 1:
            compound_returns = lambda ret, period: ((np.nanmean(ret) + 1) ** (1. / period)) - 1
            factor_returns = pd.rolling_apply(factor_returns, period, compound_returns,
                                              min_periods=1, args=(period,))

        return factor_returns.add(1).cumprod()

    @lru_cache(None)
    def mean_return_by_factor_quantile_and_group(self):
        """分行业的分位数收益

        返回值：
            Multi_index 的 DataFrame
            index 分别是分位数、 行业名称， column 是 period （1，5，10）
        """
        returns = al.performance.mean_return_by_quantile(self.factor_data,
                                                         by_date=False,
                                                         by_group=True,
                                                         demeaned=True)
        mean_return_quantile_group, mean_return_quantile_group_std_err = returns
        mean_compret_quantile_group = mean_return_quantile_group.apply(al.utils.rate_of_return, axis=0)
        return mean_compret_quantile_group

    # ic 分析
    # 日度 ic
    def ic_ts(self):
        '''
        返回值：
        DataFrame
            index 是时间， column 是 period 的值（1，5，10）

        '''
        ts = al.performance.factor_information_coefficient(self.factor_data)
        return ts, pd.rolling_mean(ts, window=22, min_periods=22)

    #  行业 ic
    def sector_ic(self):
        return al.performance.mean_information_coefficient(self.factor_data,
                                                           by_group=True)

    # 换手率分析
    def top_and_bottom_quantile_turn_over(self):
        '''
        返回值
        dict, key是 period, value 是一个 DataFrame，  DF 的 index 是日期， column 是分位数
        '''

        turnover_periods = al.utils.get_forward_returns_columns(self.factor_data.columns)
        quantile_factor = self.factor_data['factor_quantile']

        quantile_turnover_rate = {
            p: pd.concat([al.performance.quantile_turnover(quantile_factor, q, p)
                          for q in range(1, int(quantile_factor.max()) + 1)],
                         axis=1)
            for p in turnover_periods
        }

        return quantile_turnover_rate

    @cached_property
    def _involved_industries(self):
        # 分行业的分位数收益
        fq_industry_mr = self.mean_return_by_factor_quantile_and_group()
        return list(fq_industry_mr.to_panel().minor_axis)

    @property
    def _fq_stats_dict(self):
        """收益曲线"""
        fq_stats = self.cumulative_return_by_quantile(1)
        fq_stats.index = fq_stats.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_stats = fq_stats.to_dict(orient='split')
        return fq_stats

    @property
    def _fq_return_dict(self):
        """收益分析"""
        fq_mean_return = self.mean_return_by_quantile()
        fq_mean_return = fq_mean_return.to_dict(orient='split')

        fq_single_return = self.cumulative_return_by_quantile(1)
        fq_single_return.index = fq_single_return.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_single_return = fq_single_return.to_dict(orient='split')

        fq_5q_return = self.cumulative_return_by_quantile(5)
        fq_5q_return.index = fq_5q_return.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_5q_return = fq_5q_return.to_dict(orient='split')

        fq_10q_return = self.cumulative_return_by_quantile(10)
        fq_10q_return.index = fq_10q_return.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_10q_return = fq_10q_return.to_dict(orient='split')

        fq_single_com_return = self.long_short_cum_return(1)
        fq_single_com_return.index = fq_single_com_return.index.map(lambda x: x.strftime("%Y-%m-%d"))

        fq_5q_com_return = self.long_short_cum_return(5)
        fq_5q_com_return.index = fq_5q_com_return.index.map(lambda x: x.strftime("%Y-%m-%d"))

        fq_10q_com_return = self.long_short_cum_return(10)
        fq_10q_com_return.index = fq_10q_com_return.index.map(lambda x: x.strftime("%Y-%m-%d"))

        # 分行业的分位数收益
        fq_industry_mr = self.mean_return_by_factor_quantile_and_group()
        fq_industry_mr = fq_industry_mr.swaplevel(0, 1).sortlevel()
        fq_industry_mr.reset_index(level=1, drop=True, inplace=True)
        fq_industry_mr_dict = {}
        for m in self._involved_industries:
            if m is not u'不在指定行业内':
                mdf = fq_industry_mr.loc[m]
                mdf = mdf.to_dict(orient='split')
                fq_industry_mr_dict[m] = list(zip(mdf['index'], mdf['data']))

        return {
            'fq_mean_return': list(zip(fq_mean_return['index'], fq_mean_return['data'])),
            'fq_single_return': list(zip(fq_single_return['index'], fq_single_return['data'])),
            'fq_5q_return': list(zip(fq_5q_return['index'], fq_5q_return['data'])),
            'fq_10q_return': list(zip(fq_10q_return['index'], fq_10q_return['data'])),
            'fq_single_com_return': list(zip(list(fq_single_com_return.index), list(fq_single_com_return))),
            'fq_5q_com_return': list(zip(list(fq_5q_com_return.index), list(fq_5q_com_return))),
            'fq_10q_com_return': list(zip(list(fq_10q_com_return.index), list(fq_10q_com_return))),
            'fq_industry_mr': fq_industry_mr_dict,
        }

    @property
    def _fq_ic_dict(self):
        """IC 分析"""
        # ic 的时间序列
        fq_ic, fq_ic_mma = self.ic_ts()
        fq_ic.index = fq_ic.index.map(lambda x: x.strftime("%Y-%m-%d"))

        # 行业 ic
        fq_group_ic = self.sector_ic()
        if u'不在指定行业内' in self._involved_industries:
            fq_group_ic = fq_group_ic.drop(u'不在指定行业内')
        fq_group_ic = fq_group_ic.to_dict(orient='split')

        return {
            'fq_single_ic': list(zip(list(fq_ic.index), list(fq_ic[1]), list(fq_ic_mma[1]))),
            'fq_5q_ic': list(zip(list(fq_ic.index), list(fq_ic[5]), list(fq_ic_mma[1]))),
            'fq_10q_ic': list(zip(list(fq_ic.index), list(fq_ic[10]), list(fq_ic_mma[1]))),
            'fq_group_ic': list(zip(fq_group_ic['index'], fq_group_ic['data'])),
        }

    @property
    def _fq_turnover_dict(self):
        """换手率"""
        fq_turnover = self.top_and_bottom_quantile_turn_over()

        fq_single_turnover = fq_turnover[1][[1, 5]]
        fq_single_turnover.index = fq_single_turnover.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_single_turnover = fq_single_turnover.to_dict(orient='split')

        fq_5q_turnover = fq_turnover[5][[1, 5]]
        fq_5q_turnover.index = fq_5q_turnover.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_5q_turnover = fq_5q_turnover.to_dict(orient='split')

        fq_10q_turnover = fq_turnover[10][[1, 5]]
        fq_10q_turnover.index = fq_10q_turnover.index.map(lambda x: x.strftime("%Y-%m-%d"))
        fq_10q_turnover = fq_10q_turnover.to_dict(orient='split')

        return {
            'fq_single_turnover': list(zip(fq_single_turnover['index'], fq_single_turnover['data'])),
            'fq_5q_turnover': list(zip(fq_5q_turnover['index'], fq_5q_turnover['data'])),
            'fq_10q_turnover': list(zip(fq_10q_turnover['index'], fq_10q_turnover['data'])),
        }

    def analyze(self):
        return {
            'fq_stats': list(zip(self._fq_stats_dict["index"], self._fq_stats_dict["data"])),
            'return': self._fq_return_dict,
            'IC': self._fq_ic_dict,
            'turnover': self._fq_turnover_dict,
        }
