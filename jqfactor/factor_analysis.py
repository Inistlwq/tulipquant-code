# coding: utf-8

import pandas as pd
import numpy as np
import alphalens as al


# 读取因子数据
# raw = pd.read_csv('/Users/yubozhang/jqdata/factor_analysis/bp_factor.csv',dtype={'factor':np.float64})
# raw['date'] = raw['date'].map(lambda x:pd.to_datetime(x,utc=True,yearfirst=True))
# # 读取价格数据
# price = pd.read_csv('/Users/yubozhang/jqdata/factor_analysis/hs300_price.csv',infer_datetime_format=True,index_col=0)
# price.index = pd.to_datetime(price.index,utc=True)
#
# factor = raw.set_index(['date','asset'])['factor']
# factor.head()
#
#
# factor_data = al.utils.get_clean_factor_and_forward_returns(factor,price)


class factor_analysis(object):
    def __init__(self, factor_data):
        self.factor_data = factor_data

    # 收益分析
    # 用来画分位数收益的柱状图
    def mean_return_by_quantile(self):
        '''
        返回值：
        DataFrame
            index 是 factor_quantile, 值是（1，2，3，4，5）, column 是 period 的值（1，5，10）
        '''
        mean_ret_quantile, std_quantile = al.performance.mean_return_by_quantile(self.factor_data,
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
    def long_short_cum_return(self,period=1):
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

    # 分行业的分位数收益：
    def mean_return_by_factor_quantile_and_group(self):
        '''
        返回值：
            Multi_index 的 DataFrame
            index 分别是分位数、 行业名称， column 是 period （1，5，10）
        '''
        mean_return_quantile_group, mean_return_quantile_group_std_err = al.performance.mean_return_by_quantile(self.factor_data,
                                                                                                      by_date=False,
                                                                                                      by_group=True,
                                                                                                      demeaned=True)

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
        return ts, pd.rolling_mean(ts,window=22,min_periods=22)

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

        quantile_turnover_rate = {p: pd.concat([al.performance.quantile_turnover(quantile_factor, q, p) for q in range(1, int(quantile_factor.max()) + 1)],
                                               axis=1) for p in turnover_periods}

        return quantile_turnover_rate
