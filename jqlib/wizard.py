#-*- coding: UTF-8 -*-

# functions which are in common use
from __future__ import division
try:
    from kuanke.user_space_api import *
except:
    try:
        from kuanke.research.research_api import *
    except:
        try:
            from jqresearch.api import *
        except:
            pass

from jqdata import *
import numpy as np
import pandas as pd
import talib
import datetime
from numpy import mean
import traceback
import os
import six
from functools import reduce


def wizard_version():
    '''
    V1.0 基础版本
    V1.1 新增KDJ
    V1.2 新增资金流数据
    V1.3 新增融资融券数据、振幅数据；宏观择时模块；
    V1.4 新增股价相对涨幅、波动率、成交量比数据
    V1.5 新增龙虎榜、限售股数据
    v1.6 新增指数权重、高管增持、业绩预告、重大事项违规处罚数据
    v1.7 修正限售股解禁单位
    '''
    return 'v1.8 修正卖出预买进股票重复添加的错误'

def is_gtja():
    # return os.environ.get('JQCUSTOMER') == 'guotaijunan'
    return False

##################################  风控函数群 ##################################
def open_sell_func(context, security, open_sell_securities=[]):
    if security in context.portfolio.positions.keys():
        log.info(get_current_data()[security].name+'('+security+') '+'未卖出成功，已标记，下一个交易日继续卖出')
        open_sell_securities.append(security)
        open_sell_securities = list(set(open_sell_securities))

## 持仓止损
def portfolio_stoploss(context,loss=1,open_sell_securities=[]):
    if len(context.portfolio.positions)>0:
        all_avg_cost = get_all_avg_cost(context)
        positions_value = context.portfolio.positions_value
        returns = (positions_value/all_avg_cost) - 1.0
        if -returns >= loss:
            log.info('持仓价值总亏损达止损线，清仓止损！')
            # log.info(all_avg_cost, positions_value,returns)
            for security in context.portfolio.positions.keys():
                order_target_value(security, 0)
                # 如果没卖出，加入 open_sell_securities
                open_sell_func(context, security, open_sell_securities)
            # 设定今天策略止损信号
            g.daily_risk_management = False

## 持仓止盈
def portfolio_stopprofit(context,profit=1,open_sell_securities=[]):
    if len(context.portfolio.positions)>0:
        all_avg_cost = get_all_avg_cost(context)
        positions_value = context.portfolio.positions_value
        returns = (positions_value/all_avg_cost) - 1.0
        if returns >= profit:
            log.info('持仓价值总盈利达止盈线，清仓止盈！')
            for security in context.portfolio.positions.keys():
                order_target_value(security, 0)
                # 如果没卖出，加入 open_sell_securities
                open_sell_func(context, security, open_sell_securities)
            # 设定今天策略止损信号
            g.daily_risk_management = False

## 策略最大亏损
def portfolio_max_stoploss(context,loss=1,open_sell_securities=[]):
    returns = context.portfolio.returns
    if -returns >= loss:
        log.info('策略亏损达最大止损线，策略停止交易！')
        for security in context.portfolio.positions.keys():
            order_target_value(security, 0)
            # 如果没卖出，加入 open_sell_securities
            open_sell_func(context, security, open_sell_securities)
            g.risk_management_signal = False

## 策略最大盈利
def portfolio_max_stopprofit(context,profit=1,open_sell_securities=[]):
    returns = context.portfolio.returns
    if returns >= profit:
        log.info('策略盈利达最大止盈线，策略停止交易！')
        for security in context.portfolio.positions.keys():
            order_target_value(security, 0)
            # 如果没卖出，加入 open_sell_securities
            open_sell_func(context, security, open_sell_securities)
            g.risk_management_signal = False

## 个股止损
def security_stoploss(context,loss=0.1,open_sell_securities=[]):
    if len(context.portfolio.positions)>0:
        for security in context.portfolio.positions.keys():
            avg_cost = context.portfolio.positions[security].avg_cost
            current_price = context.portfolio.positions[security].price
            if 1 - current_price/avg_cost >= loss:
                log.info(str(security) + '  跌幅达个股止损线，平仓止损！')
                order_target_value(security, 0)
                # 如果没卖出，加入 g.open_sell_securities
                open_sell_func(context, security, open_sell_securities)
                # if stock in context.portfolio.positions.keys():
                #     g.open_sell_securities.append(stock)

## 个股止盈
def security_stopprofit(context,profit=0.1,open_sell_securities=[]):
    if len(context.portfolio.positions)>0:
        for security in context.portfolio.positions.keys():
            avg_cost = context.portfolio.positions[security].avg_cost
            current_price = context.portfolio.positions[security].price
            if current_price/avg_cost - 1 >= profit:
                log.info(str(security) + '  涨幅达个股止盈线，平仓止盈！')
                order_target_value(security, 0)
                # 如果没卖出，加入 g.open_sell_securities
                open_sell_func(context, security, open_sell_securities)
                # if stock in context.portfolio.positions.keys():
                #     g.open_sell_securities.append(stock)

## 大盘止损
# 止损方法1：根据大盘指数N日均线进行止损
def index_stoploss_sicha(context,n=60,open_sell_securities=[],stoploss_index=None):
    '''
    当大盘N日均线(默认60日)与昨日收盘价构成“死叉”，则清仓止损
    '''
    if stoploss_index==None:
        hist = attribute_history(g.index, n+2, '1d', 'close', df=False)
    else:
        hist = attribute_history(stoploss_index, n+2, '1d', 'close', df=False)
    temp1 = mean(hist['close'][1:-1])
    temp2 = mean(hist['close'][0:-2])
    close1 = hist['close'][-1]
    close2 = hist['close'][-2]
    if (close2 > temp2) and (close1 < temp1):
#         log.info('大盘触及止损线，清仓！')
        if len(context.portfolio.positions)>0:
            log.info('大盘触及止损线，清仓！')
            for security in context.portfolio.positions.keys():
                order_target_value(security, 0)
                # 如果没卖出，加入 g.open_sell_securities
                open_sell_func(context, security, open_sell_securities)
                # if stock in context.portfolio.positions.keys():
                #     g.open_sell_securities.append(stock)
            # 设定今天策略止损信号
            g.daily_risk_management = False
        else:
            log.info('大盘触及止损线，今日不开仓！')
            # 设定今天策略止损信号
            g.daily_risk_management = False
# 止损方法2：根据大盘指数跌幅进行止损
def index_stoploss_diefu(context,n=10,zs=0.03,open_sell_securities=[],stoploss_index=None):
    '''
    当大盘N日内跌幅超过zs，则清仓止损
    '''
    if stoploss_index==None:
        hist = attribute_history(g.index, n+2, '1d', 'close', df=False)
    else:
        hist = attribute_history(stoploss_index, n+2, '1d', 'close', df=False)
    if ((1-float(hist['close'][-1]/hist['close'][0])) >= zs):
#         log.info('大盘触及止损线，清仓！')
        if len(context.portfolio.positions)>0:
            log.info('大盘触及止损线，清仓！')
            for security in context.portfolio.positions.keys():
                order_target_value(security, 0)
                # 如果没卖出，加入 g.open_sell_securities
                open_sell_func(context, security, open_sell_securities)
                # if stock in context.portfolio.positions.keys():
                #     g.open_sell_securities.append(stock)
            # 设定今天策略止损信号
            g.daily_risk_management = False
        else:
            log.info('大盘触及止损线，今日不开仓！')
            # 设定今天策略止损信号
            g.daily_risk_management = False


##################################  出入场附加条件函数群 ##################################

## 获取所有持仓股票的成本总和
def get_all_avg_cost(context):
    all_avg_cost = 0
    for security in context.portfolio.positions.keys():
        all_avg_cost += context.portfolio.positions[security].total_amount * context.portfolio.positions[security].avg_cost
    return all_avg_cost

## 判断个股最大持仓比重
def judge_security_max_proportion(context,security,value,security_max_proportion=1):
    if is_gtja():
        if security in context.portfolio.positions.keys():
            position_value = context.portfolio.positions[security].total_amount * context.portfolio.positions[security].avg_cost
            hold_proportion = position_value/context.portfolio.starting_cash
            relative_proportion = security_max_proportion - hold_proportion
            if relative_proportion > 0:
                buy_value = relative_proportion * context.portfolio.starting_cash
                return buy_value
            else:
                return 0
        else:
            return min(value, context.portfolio.starting_cash*security_max_proportion)

    else:
        if security in context.portfolio.positions.keys():
            position_value = context.portfolio.positions[security].total_amount * context.portfolio.positions[security].avg_cost
            total_value = get_all_avg_cost(context)+context.portfolio.available_cash # 新
            hold_proportion = position_value/context.portfolio.total_value
            relative_proportion = security_max_proportion - hold_proportion
            if relative_proportion > 0:
                buy_value = relative_proportion * context.portfolio.total_value
                return buy_value
            else:
                return 0
        else:
            return min(value, context.portfolio.total_value*security_max_proportion)

## 单只最大买入股数或金额
def max_buy_value_or_amount(security,value,max_buy_value=None,max_buy_amount=None):
    if max_buy_value is not None:
        cash = min(value, max_buy_value)
        current_price = get_current_data()[security].day_open
        try:
            amount = int((cash/current_price)/100)*100
        except:
            amount = 0
        return amount
    elif max_buy_amount is not None:
        current_price = get_current_data()[security].day_open
        try:
            buy_amount = int(value/current_price/100)*100
        except:
            amount = 0
        amount = min(buy_amount, max_buy_amount)
        return amount
    else:
        current_price = get_current_data()[security].day_open
        try:
            amount = int((value/current_price)/100)*100
        except:
            amount = 0
        return amount

## 固定出仓数量或百分比或不限制
def sell_by_amount_or_percent_or_none(context,security,amount=None,percent=None,open_sell_securities=[]):
    if amount is not None:
        if context.portfolio.positions[security].total_amount > 100:
            order(security, -amount)
        else:
            order_target_value(security, 0)
            # 如果没卖出，加入 open_sell_securities
            open_sell_func(context, security, open_sell_securities)
    elif percent is not None:
        if context.portfolio.positions[security].total_amount > 100:
            a = int(percent*context.portfolio.positions[security].total_amount/100)*100
            order(security, -a)
        else:
            order_target_value(security, 0)
            # 如果没卖出，加入 open_sell_securities
            open_sell_func(context, security, open_sell_securities)
    else:
        order_target_value(security, 0)
        # 如果没卖出，加入 open_sell_securities
        open_sell_func(context, security, open_sell_securities)

## 买入委托类型
def order_style(context,security_list, max_hold_stocknum=5, style='by_cap_mean', amount=100):
    '''
    style:
        设置数量:  'by_amount', amount
        等资金:    'by_cap_mean'
        按总市值比例:'by_market_cap_percent'
        按流通市值比例:'by_circulating_market_cap_percent'
    '''
    # 国泰环境
    if is_gtja():
        result = {}
        if style == 'by_amount':
            for stock in security_list:
                current_price = get_current_data()[stock].day_open
                if np.isnan(current_price):
                    result[stock] = current_price*0
                else:
                    result[stock] = current_price*amount
        elif style == 'by_cap_mean':
            Cash = context.portfolio.starting_cash/max_hold_stocknum
            for stock in security_list:
                result[stock] = Cash
        elif style == 'by_market_cap_percent':
            df = get_fundamentals(query(valuation.code,valuation.market_cap).filter(valuation.code.in_(security_list)))
            # 如果空仓使用初始资金计算，如果不空仓则使用初始资金减去已使用总成本后的资金计算
            if len(context.portfolio.positions) == 0:
                for i in range(len(security_list)):
                    Cash = context.portfolio.starting_cash*(df.market_cap[i]/df.market_cap.sum())
                    result[df.code[i]] = Cash
            else:
                used_cash = get_all_avg_cost(context)
                for i in range(len(security_list)):
                    Cash = (context.portfolio.starting_cash-used_cash)*(df.market_cap[i]/df.market_cap.sum())
                    result[df.code[i]] = Cash
        elif style == 'by_circulating_market_cap_percent':
            df = get_fundamentals(query(valuation.code,valuation.circulating_market_cap).filter(valuation.code.in_(security_list)))
            # 如果空仓使用初始资金计算，如果不空仓则使用初始资金减去已使用总成本后的资金计算
            if len(context.portfolio.positions) == 0:
                for i in range(len(security_list)):
                    Cash = context.portfolio.starting_cash*(df.circulating_market_cap[i]/df.circulating_market_cap.sum())
                    result[df.code[i]] = Cash
            else:
                used_cash = get_all_avg_cost(context)
                for i in range(len(security_list)):
                    Cash = (context.portfolio.starting_cash-used_cash)*(df.circulating_market_cap[i]/df.circulating_market_cap.sum())
                    result[df.code[i]] = Cash
        # 返回结果
        return result

    # 聚宽环境
    else:
        result = {}
        if style == 'by_amount':
            for stock in security_list:
                current_price = get_current_data()[stock].day_open
                if np.isnan(current_price):
                    result[stock] = current_price*0
                else:
                    result[stock] = current_price*amount
        elif style == 'by_cap_mean':
            total_value = get_all_avg_cost(context)+context.portfolio.available_cash # 新
            Cash = context.portfolio.total_value/max_hold_stocknum
            for stock in security_list:
                result[stock] = Cash
        elif style == 'by_market_cap_percent':
            df = get_fundamentals(query(valuation.code,valuation.market_cap).filter(valuation.code.in_(security_list)))
            # 如果空仓使用初始资金计算，如果不空仓则使用初始资金减去已使用总成本后的资金计算
            for i in range(len(security_list)):
                Cash = context.portfolio.available_cash*(df.market_cap[i]/df.market_cap.sum())
                result[df.code[i]] = Cash
        elif style == 'by_circulating_market_cap_percent':
            df = get_fundamentals(query(valuation.code,valuation.circulating_market_cap).filter(valuation.code.in_(security_list)))
            # 如果空仓使用初始资金计算，如果不空仓则使用初始资金减去已使用总成本后的资金计算
            for i in range(len(security_list)):
                Cash = context.portfolio.available_cash*(df.circulating_market_cap[i]/df.circulating_market_cap.sum())
                result[df.code[i]] = Cash
        # 返回结果
        return result


##################################  入场附加条件函数群 ##################################




##################################  出场附加条件函数群 ##################################

## 过滤调仓日交易时涨停股票
def limit_filter(security):
    try:
        current_data = get_current_data()
        if current_data[security].day_open<=0.98*current_data[security].high_limit:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 过滤调仓前日交易时涨停股票
def per_limit_filter(security):
    try:
        security_data = attribute_history(security, 1, '1d',['close','high_limit'] , df=False)
        close = security_data['close'][-1]
        high_limit = security_data['high_limit'][-1]
        if close <= 0.98*high_limit:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 持有天数不足N天不卖出
def not_sell_in_n_days(context, security, n=10):
    try:
        trade_days = get_trade_days(start_date=context.portfolio.positions[security].init_time, end_date=context.current_dt)
        hold_days = len(trade_days)
        if hold_days <= n:
            return False
        return True
    except Exception as e:
        log.error(traceback.format_exc())
        return True

# 最长持有天数
def max_hold_days(context, security, n=60):
    try:
        trade_days = get_trade_days(start_date=context.portfolio.positions[security].init_time, end_date=context.current_dt)
        hold_days = len(trade_days)
        if hold_days > n:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

##################################  排序函数群 ##################################
## 返回元数据的 DataFrame
def get_sort_dataframe(security_list, search, sort_weight):
    if search in ['open','close']:
        df = get_price(security_list, fields=search, count=1).iloc[:,0]
        if sort_weight[0] == 'asc':# 升序
            df = df.rank(ascending=False,pct=True) * sort_weight[1]
        elif sort_weight[0] == 'desc':# 降序
            df = df.rank(ascending=True,pct=True) * sort_weight[1]
    else:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list))
        # 生成股票列表
        df = get_fundamentals(q)
        df.set_index(['code'],inplace=True)
        if sort_weight[0] == 'asc':# 升序
            df = df.rank(ascending=False,pct=True) * sort_weight[1]
        elif sort_weight[0] == 'desc':# 降序
            df = df.rank(ascending=True,pct=True) * sort_weight[1]
    return df

##################################  行情数据筛选 ##################################

## 行情大于筛选
def situation_filter_dayu(security, search, value):
    try:
        security_data = attribute_history(security, 1,'1d',fields=search, df=False)
        if security_data[search][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情小于筛选
def situation_filter_xiaoyu(security, search, value):
    try:
        security_data = attribute_history(security, 1,'1d',fields=search, df=False)
        if security_data[search][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情区间筛选
def situation_filter_qujian(security, search, value):
    try:
        security_data = attribute_history(security, 1,'1d',fields=search, df=False)
        a,b = value
        if a < security_data[search][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 行情大于均值筛选
def situation_filter_dayu_ma(security, search, value):
    try:
        security_data = attribute_history(security, value,'1d',fields=search, df=False)
        ma_data = security_data[search].mean()
        if security_data[search][-1] > ma_data:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 行情小于均值筛选
def situation_filter_xiaoyu_ma(security, search, value):
    try:
        security_data = attribute_history(security, value,'1d',fields=search, df=False)
        ma_data = security_data[search].mean()
        if security_data[search][-1] < ma_data:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 行情均值区间筛选
def situation_filter_qujian_ma(security, search, value):
    try:
        security_data = attribute_history(security, max(value),'1d',fields=search, df=False)
        ma_data_short = security_data[search][-min(value):].mean()
        ma_data_long = security_data[search].mean()
        if ma_data_long < security_data[search][-1] < ma_data_short:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 获取N日涨幅
def get_n_day_chg(security, n, include_now=False):
    try:
        security_data = get_bars(security, n+1,'1d', 'close', include_now)
        chg = (security_data['close'][-1]/security_data['close'][0]) - 1
        return chg
    except Exception as e:
        log.error(traceback.format_exc())

## N日涨幅大于筛选
def n_day_chg_dayu(security, n, value):
    try:
        security_data = attribute_history(security, n+1,'1d', 'close', skip_paused=False, df=False)
        chg = (security_data['close'][-1]/security_data['close'][0]) - 1
        if chg > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅小于筛选
def n_day_chg_xiaoyu(security, n, value):
    try:
        security_data = attribute_history(security, n+1,'1d', 'close', skip_paused=False, df=False)
        chg = (security_data['close'][-1]/security_data['close'][0]) - 1
        if chg < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅区间筛选
def n_day_chg_qujian(security, n, value):
    try:
        security_data = attribute_history(security, n+1,'1d', 'close', skip_paused=False, df=False)
        chg = (security_data['close'][-1]/security_data['close'][0]) - 1
        a,b = value
        if a < chg < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 上市天数大于筛选
def ipo_days_dayu(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        if long_days > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 上市天数小于筛选
def ipo_days_xiaoyu(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        if long_days < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 上市天数区间筛选
def ipo_days_qujian(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        a,b = value
        if a < long_days < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


##################################  财务数据筛选 ##################################

## 财务大于筛选
def financial_data_filter_dayu(security_list, search, value):
    # 生成查询条件
    q = query(valuation.code,search).filter(valuation.code.in_(security_list), search>value)
    # 生成股票列表
    df = get_fundamentals(q)
    security_list =list(df['code'])
    return security_list

## 财务小于筛选
def financial_data_filter_xiaoyu(security_list, search, value):
    # 生成查询条件
    q = query(valuation.code,search).filter(valuation.code.in_(security_list), search<value)
    # 生成股票列表
    df = get_fundamentals(q)
    security_list =list(df['code'])
    return security_list

## 财务区间筛选
def financial_data_filter_qujian(security_list, search, value):
    # 生成查询条件
    a,b = value
    q = query(valuation.code,search).filter(valuation.code.in_(security_list), search.between(a,b))
    # 生成股票列表
    df = get_fundamentals(q)
    security_list =list(df['code'])
    return security_list

###################################  技术指标公用函数群 ##################################
# MACD
def MACD(security_list, fastperiod=12, slowperiod=26, signalperiod=9):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 MACD
    security_data = history(slowperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    macd_DIF = {}; macd_DEA = {}; macd_HIST = {}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            macd_DIF[stock] = np.array([np.nan]*slowperiod)
            macd_DEA[stock] = np.array([np.nan]*slowperiod)
            macd_HIST[stock]= np.array([np.nan]*slowperiod)
        else:
            macd_DIF[stock], macd_DEA[stock], macd = talib.MACDEXT(security_data[stock], fastperiod=fastperiod, fastmatype=1, slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod, signalmatype=1)
            macd_HIST[stock] = macd * 2
    return macd_DIF, macd_DEA, macd_HIST

# MA
def MA(security_list, timeperiod=5):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 MA
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    ma = {}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            ma[stock] = np.array([np.nan]*timeperiod)
        else:
            ma[stock] = talib.MA(security_data[stock], timeperiod)
    return ma

# SMA
def SMA(security_list, timeperiod=5) :
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 SMA
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    sma = {}
    for stock in security_list:
        close = np.nan_to_num(security_data[stock])
        sma[stock] = reduce(lambda x, y: ((timeperiod - 1) * x + y) / timeperiod, close)
    return sma


# KDJ-随机指标
def KDJ(security_list, N =9, M1=3, M2=3):
    # 计算close的N日移动平均，权重默认为1
    def SMA_CN(close, N):
        close = np.nan_to_num(close)
        return reduce(lambda x, y: ((N - 1) * x + y) / N, close)

    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 KDJ
    n = max(N , M1, M2)
    k = {}; d = {}; j = {}
    for stock in security_list:
        security_data = attribute_history(stock, n*10, '1d', ['high', 'low', 'close'], df=False)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            k[stock] = np.array([np.nan]*n)
            d[stock] = np.array([np.nan]*n)
            j[stock] = np.array([np.nan]*n)
        else:
            high_KDJ = security_data['high']
            low_KDJ = security_data['low']
            close_KDJ = security_data['close']

            # 使用talib中的STOCHF函数求 RSV
            kValue, dValue = talib.STOCHF(high_KDJ, low_KDJ, close_KDJ, N , M2, fastd_matype=0)
            # 求K值(等于RSV的M1日移动平均)
            kValue = np.array(list(map(lambda x : SMA_CN(kValue[:x], M1), range(1, len(kValue) + 1))))
            # 求D值(等于K的M2日移动平均)
            dValue = np.array(list(map(lambda x : SMA_CN(kValue[:x], M2), range(1, len(kValue) + 1))))
            # 求J值
            jValue = 3 * kValue - 2 * dValue

            k[stock] = kValue
            d[stock] = dValue
            j[stock] = jValue
    return k, d, j

# RSI
def RSI(security_list, timeperiod=14):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 MA
    security_data = history(timeperiod*20, '1d', 'close' , security_list, df=False, skip_paused=True)
    rsi = {}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            rsi[stock] = np.array([np.nan]*timeperiod)
        else:
            rsi[stock] = talib.RSI(security_data[stock], timeperiod)[-1:]
    return rsi

# CCI
def CCI(security_list, timeperiod=14):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 CCI
    cci = {}
    for stock in security_list:
        security_data = attribute_history(stock, timeperiod*2, '1d',['close','high','low'] , df=False)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            cci[stock] = np.array([np.nan]*timeperiod)
        else:
            close_CCI = security_data['close']
            high_CCI = security_data['high']
            low_CCI = security_data['low']
            cci[stock] = talib.CCI(high_CCI, low_CCI, close_CCI, timeperiod)
    return cci

# ATR
def ATR(security_list, timeperiod=14):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 ATR
    atr = {}
    for stock in security_list:
        security_data = attribute_history(stock, timeperiod*2, '1d',['close','high','low'] , df=False)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            atr[stock] = np.array([np.nan]*timeperiod)
        else:
            close_ATR = security_data['close']
            high_ATR = security_data['high']
            low_ATR = security_data['low']
            atr[stock] = talib.ATR(high_ATR, low_ATR, close_ATR, timeperiod)
    return atr

# 布林线
def Bollinger_Bands(security_list, timeperiod=5, nbdevup=2, nbdevdn=2):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 Bollinger Bands
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    upperband={}; middleband={}; lowerband={}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            upperband[stock] = np.array([np.nan]*timeperiod)
            middleband[stock] = np.array([np.nan]*timeperiod)
            lowerband[stock] = np.array([np.nan]*timeperiod)
        else:
            upperband[stock], middleband[stock], lowerband[stock] = talib.BBANDS(security_data[stock], timeperiod, nbdevup, nbdevdn)
    return upperband, middleband, lowerband

# 平均成交额
def MA_MONEY(security_list, timeperiod=5):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 N 日平均成交额
    security_data = history(timeperiod*2, '1d', 'money' , security_list, df=False, skip_paused=True)
    mamoney={}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            mamoney[stock] = np.array([np.nan]*timeperiod)
        else:
            mamoney[stock] = talib.MA(security_data[stock], timeperiod)
    return mamoney

# 平均成交量
def MA_VOLUME(security_list, timeperiod=5):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 N 日平均成交量
    security_data = history(timeperiod*2, '1d', 'volume' , security_list, df=False, skip_paused=True)
    mavol={}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            mavol[stock] = np.array([np.nan]*timeperiod)
        else:
            mavol[stock] = talib.MA(security_data[stock], timeperiod)
    return mavol


# BIAS
def BIAS(security_list, timeperiod=5):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 BIAS
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    bias = {}
    for stock in security_list:
        average_price = security_data[stock][-timeperiod:].mean()
        current_price = security_data[stock][-1]
        bias[stock]=(current_price-average_price)/average_price
    return bias

# BBI
def BBI(security_list, timeperiod1=3, timeperiod2=6, timeperiod3=12, timeperiod4=24):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 BBI
    security_data = history(timeperiod4*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    bbi={}
    for stock in security_list:
        x = security_data[stock]
        d = (x[-timeperiod1:].mean()+x[-timeperiod2:].mean()+x[-timeperiod3:].mean()+x[-timeperiod4:].mean())/4.0
        bbi[stock] = d
    return bbi

# TRIX
def TRIX(security_list, timeperiod=30):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 TRIX
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    trix={}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            trix[stock] = np.array([np.nan]*timeperiod)
        else:
            trix[stock] = talib.TRIX(security_data[stock], timeperiod)
    return trix

# EMA
def EMA(security_list, timeperiod=30):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 EMA
    security_data = history(timeperiod*2, '1d', 'close' , security_list, df=False, skip_paused=True)
    ema={}
    for stock in security_list:
        nan_count = list(np.isnan(security_data[stock])).count(True)
        if nan_count == len(security_data[stock]):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市或刚上市，返回 NaN 值数据。" %stock)
            ema[stock] = np.array([np.nan]*timeperiod)
        else:
            ema[stock] = talib.EMA(security_data[stock], timeperiod)
    return ema

##################################  技术指标判断函数群 ##################################

######  MACD 判断函数 ######
# MACD_DIF 大于
def MACD_DIF_judge_dayu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DIF[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_DIF 小于
def MACD_DIF_judge_xiaoyu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DIF[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_DIF 区间
def MACD_DIF_judge_qujian(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        a,b = value
        if a < macd_DIF[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_DEA 大于
def MACD_DEA_judge_dayu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DEA[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_DEA 小于
def MACD_DEA_judge_xiaoyu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DEA[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_DEA 区间
def MACD_DEA_judge_qujian(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        a,b = value
        if a < macd_DEA[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_HIST 大于
def MACD_HIST_judge_dayu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_HIST[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_HIST 小于
def MACD_HIST_judge_xiaoyu(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_HIST[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD_HIST 区间
def MACD_HIST_judge_qujian(security, value, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        a,b = value
        if a < macd_HIST[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 金叉
def MACD_judge_jincha(security, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if (macd_DIF[security][-1] > macd_DEA[security][-1])and(macd_DIF[security][-2] < macd_DEA[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 死叉
def MACD_judge_sicha(security, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if (macd_DIF[security][-1] < macd_DEA[security][-1])and(macd_DIF[security][-2] > macd_DEA[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 多头
def MACD_judge_duotou(security, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DIF[security][-1] > macd_DEA[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 空头
def MACD_judge_kongtou(security, fastperiod=12, slowperiod=26, signalperiod=9):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod)
        if macd_DIF[security][-1] < macd_DEA[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


######  MA 判断函数 ######
# MA 大于
def MA_judge_dayu(security, value, timeperiod=5):
    try:
        ma = MA(security, timeperiod)
        if ma[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 小于
def MA_judge_xiaoyu(security, value, timeperiod=5):
    try:
        ma = MA(security, timeperiod)
        if ma[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 区间
def MA_judge_qujian(security, value, timeperiod=5):
    try:
        ma = MA(security, timeperiod)
        a,b = value
        if a < ma[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 金叉
def MA_judge_jincha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod)
        ma_long = MA(security, timeperiod=long_timeperiod)
        if (ma_short[security][-1] > ma_long[security][-1])and(ma_short[security][-2] < ma_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 死叉
def MA_judge_sicha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod)
        ma_long = MA(security, timeperiod=long_timeperiod)
        if (ma_short[security][-1] < ma_long[security][-1])and(ma_short[security][-2] > ma_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 多头
def MA_judge_duotou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod)
        ma_long = MA(security, timeperiod=long_timeperiod)
        if ma_short[security][-1] > ma_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 空头
def MA_judge_kongtou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod)
        ma_long = MA(security, timeperiod=long_timeperiod)
        if ma_short[security][-1] < ma_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


######  平均成交额 判断函数 ######
# MA_MONEY 大于
def MA_MONEY_judge_dayu(security, value, timeperiod=5):
    try:
        mamoney = MA_MONEY(security, timeperiod)
        if mamoney[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 小于
def MA_MONEY_judge_xiaoyu(security, value, timeperiod=5):
    try:
        mamoney = MA_MONEY(security, timeperiod)
        if mamoney[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 区间
def MA_MONEY_judge_qujian(security, value, timeperiod=5):
    try:
        mamoney = MA_MONEY(security, timeperiod)
        a,b = value
        if a < mamoney[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 金叉
def MA_MONEY_judge_jincha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mamoney_short = MA_MONEY(security, short_timeperiod)
        mamoney_long = MA_MONEY(security, long_timeperiod)
        if (mamoney_short[security][-1] > mamoney_long[security][-1])and(mamoney_short[security][-2] < mamoney_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 死叉
def MA_MONEY_judge_sicha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mamoney_short = MA_MONEY(security, short_timeperiod)
        mamoney_long = MA_MONEY(security, long_timeperiod)
        if (mamoney_short[security][-1] < mamoney_long[security][-1])and(mamoney_short[security][-2] > mamoney_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 多头
def MA_MONEY_judge_duotou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mamoney_short = MA_MONEY(security, short_timeperiod)
        mamoney_long = MA_MONEY(security, long_timeperiod)
        if mamoney_short[security][-1] > mamoney_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_MONEY 空头
def MA_MONEY_judge_kongtou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mamoney_short = MA_MONEY(security, short_timeperiod)
        mamoney_long = MA_MONEY(security, long_timeperiod)
        if mamoney_short[security][-1] < mamoney_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  平均成交量 判断函数 ######
# MA_VOLUME 大于
def MA_VOLUME_judge_dayu(security, value, timeperiod=5):
    try:
        mavol = MA_VOLUME(security, timeperiod)
        if mavol[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 小于
def MA_VOLUME_judge_xiaoyu(security, value, timeperiod=5):
    try:
        mavol = MA_VOLUME(security, timeperiod)
        if mavol[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 区间
def MA_VOLUME_judge_qujian(security, value, timeperiod=5):
    try:
        mavol = MA_VOLUME(security, timeperiod)
        a,b = value
        if a < mavol[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 金叉
def MA_VOLUME_judge_jincha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mavol_short = MA_VOLUME(security, short_timeperiod)
        mavol_long = MA_VOLUME(security, long_timeperiod)
        if (mavol_short[security][-1] > mavol_long[security][-1])and(mavol_short[security][-2] < mavol_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 死叉
def MA_VOLUME_judge_sicha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mavol_short = MA_VOLUME(security, short_timeperiod)
        mavol_long = MA_VOLUME(security, long_timeperiod)
        if (mavol_short[security][-1] < mavol_long[security][-1])and(mavol_short[security][-2] > mavol_long[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 多头
def MA_VOLUME_judge_duotou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mavol_short = MA_VOLUME(security, short_timeperiod)
        mavol_long = MA_VOLUME(security, long_timeperiod)
        if mavol_short[security][-1] > mavol_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA_VOLUME 空头
def MA_VOLUME_judge_kongtou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        mavol_short = MA_VOLUME(security, short_timeperiod)
        mavol_long = MA_VOLUME(security, long_timeperiod)
        if mavol_short[security][-1] < mavol_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  布林线 判断函数 ######
# 布林上线 大于
def BBands_upperband_judge_dayu(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        if upperband[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# 布林上线 小于
def BBands_upperband_judge_xiaoyu(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        if upperband[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# 布林上线 区间
def BBands_upperband_judge_qujian(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        a,b = value
        if a < upperband[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 布林下线 大于
def BBands_lowerband_judge_dayu(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        if lowerband[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# 布林下线 小于
def BBands_lowerband_judge_xiaoyu(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        if lowerband[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# 布林下线 区间
def BBands_lowerband_judge_qujian(security, value, timeperiod=5, nbdevup=2, nbdevdn=2):
    try:
        upperband, middleband, lowerband = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn)
        a,b = value
        if a < lowerband[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  ATR 判断函数 ######
# ATR 大于
def ATR_judge_dayu(security, value, timeperiod=14):
    try:
        atr = ATR(security, timeperiod)
        if atr[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# ATR 小于
def ATR_judge_xiaoyu(security, value, timeperiod=14):
    try:
        atr = ATR(security, timeperiod)
        if atr[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# ATR 区间
def ATR_judge_qujian(security, value, timeperiod=14):
    try:
        atr = ATR(security, timeperiod)
        a,b = value
        if a < atr[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  CCI 判断函数 ######
# CCI 大于
def CCI_judge_dayu(security, value, timeperiod=14):
    try:
        cci = CCI(security, timeperiod)
        if cci[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# CCI 小于
def CCI_judge_xiaoyu(security, value, timeperiod=14):
    try:
        cci = CCI(security, timeperiod)
        if cci[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# CCI 区间
def CCI_judge_qujian(security, value, timeperiod=14):
    try:
        cci = CCI(security, timeperiod)
        a,b = value
        if a < cci[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  RSI 判断函数 ######
# RSI 大于
def RSI_judge_dayu(security, value, timeperiod=14):
    try:
        rsi = RSI(security, timeperiod)
        if rsi[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# RSI 小于
def RSI_judge_xiaoyu(security, value, timeperiod=14):
    try:
        rsi = RSI(security, timeperiod)
        if rsi[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# RSI 区间
def RSI_judge_qujian(security, value, timeperiod=14):
    try:
        rsi = RSI(security, timeperiod)
        a,b = value
        if a < rsi[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  BIAS 判断函数 ######
# BIAS 大于
def BIAS_judge_dayu(security, value, timeperiod=5):
    try:
        bias = BIAS(security, timeperiod)
        if bias[security] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# BIAS 小于
def BIAS_judge_xiaoyu(security, value, timeperiod=5):
    try:
        bias = BIAS(security, timeperiod)
        if bias[security] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# BIAS 区间
def BIAS_judge_qujian(security, value, timeperiod=5):
    try:
        bias = BIAS(security, timeperiod)
        a,b = value
        if a < bias[security] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  BBI 判断函数 ######
# BBI 多头
def BBI_judge_duotou(security, timeperiod1=3, timeperiod2=6, timeperiod3=12, timeperiod4=24):
    try:
        current_price = get_current_data()[security].day_open
        bbi = BBI(security, timeperiod1, timeperiod2, timeperiod3, timeperiod4)
        if current_price > bbi[security]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# BBI 空头
def BBI_judge_kongtou(security, timeperiod1=3, timeperiod2=6, timeperiod3=12, timeperiod4=24):
    try:
        current_price = get_current_data()[security].day_open
        bbi = BBI(security, timeperiod1, timeperiod2, timeperiod3, timeperiod4)
        if current_price < bbi[security]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  DMA 判断函数 ######
# DMA 金叉
def DMA_judge_jincha(security,fastperiod=5,slowperiod=60,amaperiod=20):
    # 计算 DMA
    try:
        security_data = attribute_history(security, slowperiod*2, '1d', ['close'], skip_paused=True, df=False)['close']
        m1 = map(lambda i: security_data[i-fastperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        m2 = map(lambda i: security_data[i-slowperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        MA1 = map(lambda x: mean(x), m1)
        MA2 = map(lambda x: mean(x), m2)
        DMA = np.array(MA1) - np.array(MA2)
        # 计算 AMA
        a1 = map(lambda i: DMA[i-amaperiod+1:i+1],range(len(DMA))[-2:])
        AMA = map(lambda x: mean(x), a1)
        # 判断
        if (DMA[-1] > AMA[-1])and(DMA[-2] < AMA[-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# DMA 死叉
def DMA_judge_sicha(security,fastperiod=5,slowperiod=60,amaperiod=20):
    # 计算 DMA
    try:
        security_data = attribute_history(security, slowperiod*2, '1d', ['close'], skip_paused=True, df=False)['close']
        m1 = map(lambda i: security_data[i-fastperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        m2 = map(lambda i: security_data[i-slowperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        MA1 = map(lambda x: mean(x), m1)
        MA2 = map(lambda x: mean(x), m2)
        DMA = np.array(MA1) - np.array(MA2)
        # 计算 AMA
        a1 = map(lambda i: DMA[i-amaperiod+1:i+1],range(len(DMA))[-2:])
        AMA = map(lambda x: mean(x), a1)
        # 判断
        if (DMA[-1] < AMA[-1])and(DMA[-2] > AMA[-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# DMA 多头
def DMA_judge_duotou(security,fastperiod=5,slowperiod=60,amaperiod=20):
    # 计算 DMA
    try:
        security_data = attribute_history(security, slowperiod*2, '1d', ['close'], skip_paused=True, df=False)['close']
        m1 = map(lambda i: security_data[i-fastperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        m2 = map(lambda i: security_data[i-slowperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        MA1 = map(lambda x: mean(x), m1)
        MA2 = map(lambda x: mean(x), m2)
        DMA = np.array(MA1) - np.array(MA2)
        # 计算 AMA
        a1 = map(lambda i: DMA[i-amaperiod+1:i+1],range(len(DMA))[-2:])
        AMA = map(lambda x: mean(x), a1)
        # 判断
        if DMA[-1] > AMA[-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# DMA 空头
def DMA_judge_kongtou(security,fastperiod=5,slowperiod=60,amaperiod=20):
    # 计算 DMA
    try:
        security_data = attribute_history(security, slowperiod*2, '1d', ['close'], skip_paused=True, df=False)['close']
        m1 = map(lambda i: security_data[i-fastperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        m2 = map(lambda i: security_data[i-slowperiod+1:i+1],range(len(security_data))[len(security_data)-slowperiod:])
        MA1 = map(lambda x: mean(x), m1)
        MA2 = map(lambda x: mean(x), m2)
        DMA = np.array(MA1) - np.array(MA2)
        # 计算 AMA
        a1 = map(lambda i: DMA[i-amaperiod+1:i+1],range(len(DMA))[-2:])
        AMA = map(lambda x: mean(x), a1)
        # 判断
        if DMA[-1] < AMA[-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  EMA 判断函数 ######
# EMA 金叉
def EMA_judge_jincha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ema_short = EMA(security, timeperiod=short_timeperiod)
        ema_long = EMA(security, timeperiod=long_timeperiod)
        if (ema_short[security][-2] < ema_long[security][-2])and(ema_short[security][-1] > ema_long[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# EMA 死叉
def EMA_judge_sicha(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ema_short = EMA(security, timeperiod=short_timeperiod)
        ema_long = EMA(security, timeperiod=long_timeperiod)
        if (ema_short[security][-2] > ema_long[security][-2])and(ema_short[security][-1] < ema_long[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# EMA 多头
def EMA_judge_duotou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ema_short = EMA(security, timeperiod=short_timeperiod)
        ema_long = EMA(security, timeperiod=long_timeperiod)
        if ema_short[security][-1] > ema_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# EMA 空头
def EMA_judge_kongtou(security, short_timeperiod=5, long_timeperiod=10):
    try:
        ema_short = EMA(security, timeperiod=short_timeperiod)
        ema_long = EMA(security, timeperiod=long_timeperiod)
        if ema_short[security][-1] < ema_long[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  TRIX 判断函数 ######
# TRIX 金叉
def TRIX_judge_jincha(security, trix_timeperiod=60, matrix_timeperiod=5):
    try:
        trix = TRIX(security, trix_timeperiod)
        matrix_0 = trix[security][-matrix_timeperiod:].mean()
        matrix_1 = trix[security][-matrix_timeperiod-1:-1].mean()
        if (trix[security][-1] > matrix_0)and(trix[security][-2] < matrix_1):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# TRIX 死叉
def TRIX_judge_sicha(security, trix_timeperiod=60, matrix_timeperiod=5):
    try:
        trix = TRIX(security, trix_timeperiod)
        matrix_0 = trix[security][-matrix_timeperiod:].mean()
        matrix_1 = trix[security][-matrix_timeperiod-1:-1].mean()
        if (trix[security][-1] < matrix_0)and(trix[security][-2] > matrix_1):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# TRIX 多头
def TRIX_judge_duotou(security, trix_timeperiod=60, matrix_timeperiod=5):
    try:
        trix = TRIX(security, trix_timeperiod)
        matrix_0 = trix[security][-matrix_timeperiod:].mean()
        if trix[security][-1] > matrix_0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# TRIX 空头
def TRIX_judge_kongtou(security, trix_timeperiod=60, matrix_timeperiod=5):
    try:
        trix = TRIX(security, trix_timeperiod)
        matrix_0 = trix[security][-matrix_timeperiod:].mean()
        matrix_1 = trix[security][-matrix_timeperiod-1:-1].mean()
        if trix[security][-1] < matrix_0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

######  KDJ 判断函数 ######
# KDJ_K 大于
def KDJ_K_judge_dayu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_K[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_K 小于
def KDJ_K_judge_xiaoyu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_K[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_K 区间
def KDJ_K_judge_qujian(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        a,b = value
        if a < KDJ_K[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ_D 大于
def KDJ_D_judge_dayu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_D[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_D 小于
def KDJ_D_judge_xiaoyu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_D[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_D 区间
def KDJ_D_judge_qujian(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        a,b = value
        if a < KDJ_D[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ_J 大于
def KDJ_J_judge_dayu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_J[security][-1] > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_J 小于
def KDJ_J_judge_xiaoyu(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        if KDJ_J[security][-1] < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# KDJ_J 区间
def KDJ_J_judge_qujian(security, value, N=9, M1=3, M2=3):
    try:
        KDJ_K, KDJ_D, KDJ_J = KDJ(security, N, M1, M2)
        a,b = value
        if a < KDJ_J[security][-1] < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-金叉
def KDJ_judge_jincha(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # 金叉：K线上穿D线
        if (kdj_K[security][-1] > kdj_D[security][-1]) and (kdj_K[security][-2] < kdj_D[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-死叉
def KDJ_judge_sicha(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # 死叉：K线下穿D线
        if (kdj_K[security][-1] < kdj_D[security][-1]) and (kdj_K[security][-2] > kdj_D[security][-2]) :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-多头
def KDJ_judge_duotou(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # 多头：K线位于D线上方
        if kdj_K[security][-1] > kdj_D[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-空头
def KDJ_judge_kongtou(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # 空头：K线位于D线下方
        if kdj_K[security][-1] < kdj_D[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-买入信号
def KDJ_judge_mrxh(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # K,D < 20 并且 K线上穿D线
        if (kdj_K[security][-1] < 20) and (kdj_K[security][-1] > kdj_D[security][-1]) and (kdj_K[security][-2] < kdj_D[security][-2]):
        # if (kdj_K[security][-1] < 20 and kdj_D[security][-1] < 20) and (kdj_K[security][-2] < kdj_D[security][-2]) and (kdj_K[security][-1] > kdj_D[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-卖出信号
def KDJ_judge_mcxh(security, N=9, M1=3, M2=3):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2)
        # K,D > 80 并且 K线下穿D线
        if (kdj_K[security][-1] > 80) and (kdj_K[security][-1] < kdj_D[security][-1]) and (kdj_K[security][-2] > kdj_D[security][-2]):
        # if (kdj_K[security][-1] > 80 and kdj_D[security][-1] > 80) and (kdj_K[security][-2] > kdj_D[security][-2]) and (kdj_K[security][-1] < kdj_D[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

##################################  形态指标筛选 ##################################

# 十字星 买
def CDLDOJISTAR_judge(security):
    # 计算 十字星
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLDOJISTAR = talib.CDLDOJISTAR(open, high, low, close)
        if CDLDOJISTAR[-1] > 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# 十字星 卖
def CDLDOJISTAR_judge(security):
    # 计算 十字星
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLDOJISTAR = talib.CDLDOJISTAR(open, high, low, close)
        if CDLDOJISTAR[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 三只乌鸦
def CDL3BLACKCROWS_judge(security):
    # 计算 三只乌鸦
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDL3BLACKCROWS = talib.CDL3BLACKCROWS(open, high, low, close)
        if CDL3BLACKCROWS[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


# 两只乌鸦
def CDL2CROWS_judge(security):
    # 计算 两只乌鸦
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDL2CROWS = talib.CDL2CROWS(open, high, low, close)
        if CDL2CROWS[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 红三兵
def CDL3WHITESOLDIERS_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDL3WHITESOLDIERS = talib.CDL3WHITESOLDIERS(open, high, low, close)
        if CDL3WHITESOLDIERS[-1] > 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


# 乌云盖顶
def CDLDARKCLOUDCOVER_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLDARKCLOUDCOVER = talib.CDLDARKCLOUDCOVER(open, high, low, close)
        if CDLDARKCLOUDCOVER[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


# 黄昏之星
def CDLEVENINGSTAR_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLEVENINGSTAR = talib.CDLEVENINGSTAR(open, high, low, close)
        if CDLEVENINGSTAR[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


# 早晨之星
def CDLMORNINGSTAR_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLMORNINGSTAR = talib.CDLMORNINGSTAR(open, high, low, close)
        if CDLMORNINGSTAR[-1] > 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 锤
def CDLHAMMER_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLHAMMER = talib.CDLHAMMER(open, high, low, close)
        if CDLHAMMER[-1] > 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 倒锤
def CDLINVERTEDHAMMER_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLINVERTEDHAMMER = talib.CDLINVERTEDHAMMER(open, high, low, close)
        if CDLINVERTEDHAMMER[-1] > 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


# 流星
def CDLSHOOTINGSTAR_judge(security):
    # 计算 红三兵
    try:
        security_data =  attribute_history(security, 40, unit='1d', fields=('close', 'volume', 'open', 'high', 'low'), skip_paused=True, df=False)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        CDLSHOOTINGSTAR = talib.CDLSHOOTINGSTAR(open, high, low, close)
        if CDLSHOOTINGSTAR[-1] < 0 :
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

############## 资金流##############

## 资金流数据大于
def money_flow_dayu(context, security, search, value, which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = jqdata.get_money_flow(security, end_date=check_date, fields=search, count=1).iloc[0,0]
        if security_data > value:
            return True
        return False
    except Exception as e:
        # log.error(traceback.format_exc())
        log.warn('标的 %s 没有资金流数据；(可能原因:查询日期该标的处于停牌、退市或未上市状态)' %security)
        return False

## 资金流数据小于
def money_flow_xiaoyu(context, security, search, value, which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = jqdata.get_money_flow(security, end_date=check_date, fields=search, count=1).iloc[0,0]
        if security_data < value:
            return True
        return False
    except Exception as e:
        # log.error(traceback.format_exc())
        log.warn('标的 %s 没有资金流数据；(可能原因:查询日期该标的处于停牌、退市或未上市状态)' %security)
        return False

## 资金流数据区间
def money_flow_qujian(context, security, search, value, which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = jqdata.get_money_flow(security, end_date=check_date, fields=search, count=1).iloc[0,0]
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        # log.error(traceback.format_exc())
        log.warn('标的 %s 没有资金流数据；(可能原因:查询日期该标的处于停牌、退市或未上市状态)' %security)
        return False

############## 振幅 ##############

## 振幅大于筛选
def amplitude_dayu(security, value, include_now=False):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 振幅小于筛选
def amplitude_xiaoyu(security, value, include_now=False):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 振幅大于等于筛选
def amplitude_dayudengyu(security, value, include_now=False):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 振幅小于等于筛选
def amplitude_xiaoyudengyu(security, value, include_now=False):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 振幅等于筛选
def amplitude_dengyu(security, value, include_now=False):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg == value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

############## 融资融券 ##############
"""
融资融券余额
"""
def fin_sec_dayu(context, security, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=1)
        if len(data[search]) > 0:
            if float(data[search]) > float(value) * 1e4:
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}的{}数据！".format(
                security, context.previous_date, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def fin_sec_xiaoyu(context, security, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=1)
        if len(data[search]) > 0:
            if float(data[search]) < float(value) * 1e4:
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}的{}数据！".format(
                security, context.previous_date, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def fin_sec_qujian(context, security, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=1)
        if len(data[search]) > 0:
            a, b = value
            if float(a) * 1e4 < float(data[search]) < float(b) * 1e4:
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}的{}数据！".format(
                security, context.previous_date, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

"""融资融券增长率"""

def n_day_fin_sec_chg_dayu(context, security, n, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=n)
        if len(data[search]) > 0 and data[search].iloc[0] != 0:
            rate = float((data[search].iloc[
                         -1] - data[search].iloc[0])) / abs(data[search].iloc[0]) * 100
            if float(rate) > float(value):
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}日的前{}日{}数据！".format(
                security, context.previous_date, n, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def  n_day_fin_sec_chg_xiaoyu(context, security, n, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=n)
        if len(data[search]) > 0 and data[search].iloc[0] != 0:
            rate = float((data[search].iloc[
                         -1] - data[search].iloc[0])) / abs(data[search].iloc[0]) * 100
            if float(rate) < float(value):
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}日的前{}日{}数据！".format(
                security, context.previous_date, n, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def  n_day_fin_sec_chg_qujian(context, security, n, search, value):
    import jqdata
    try:
        data = jqdata.get_mtss(
            security, end_date=context.previous_date, fields=search, count=n)
        if len(data[search]) > 0 and data[search].iloc[0] != 0:
            rate = float((data[search].iloc[
                         -1] - data[search].iloc[0])) / abs(data[search].iloc[0]) * 100
            a, b = value
            if float(a) < float(rate) < float(b):
                return True
            else:
                return False
        else:
            log.error("无法获取{}在{}日的前{}日{}数据！".format(
                security, context.previous_date, n, search))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

############## 股价相对涨幅 ##############

# 获取N日涨幅
def get_n_day_chg(security, n, include_now=False):
    try:
        security_data = get_bars(security, n + 1, '1d', 'close', include_now)
        chg = (security_data['close'][-1] / security_data['close'][0]) - 1
        return chg
    except Exception as e:
        log.error(traceback.format_exc())

# 股价相对于benchmark的相对涨幅
def get_rp(security, n, index, include_now=False):
    # 股价相对于benchmark的相对涨幅
    try:
        security_data = get_bars(security, n, '1d', 'close', include_now=include_now)
        bench_date = get_bars(index, n, '1d', 'close', include_now=include_now)
        if len(security_data) > 0 and len(bench_date) > 0 and security_data["close"][0] * bench_date["close"][0] != 0:
            sec_rt = float(security_data["close"][-1]) / security_data["close"][0] - 1
            ben_rt = float(bench_date["close"][-1]) / bench_date["close"][0] - 1
            rp = (sec_rt / ben_rt - 1)
            return rp
        else:
            log.error("无法获取{}和{}{}日前的价格数据！".format(security, index, n))
            return None
    except:
        log.error(traceback.format_exc())
        return None

# N日相对涨幅大于
def relative_index_chg_dayu(security, n, index, value, include_now=False):
    try:
        rp = get_rp(security, n, index, include_now=include_now)
        if float(rp) > float(value):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日相对涨幅小于
def relative_index_chg_xiaoyu(security, n, index, value, include_now=False):
    try:
        rp = get_rp(security, n, index, include_now=include_now)
        if float(rp) < float(value):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日相对涨幅区间
def relative_index_chg_qujian(security, n, index, value, include_now=False):
    try:
        rp = get_rp(security, n, index, include_now=include_now)
        a, b = value
        if float(a) < float(rp) < float(b):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

############## 波动率 ##############
# N日波动率
def get_vol(security, n, include_now=False):
    # N日波动率
    try:
        close_data = get_bars(security, n + 1, '1d','close', include_now=include_now)
        if len(close_data) > 0:
            rtn = np.diff(close_data["close"]) / close_data["close"][:-1]
            return rtn.std() * sqrt(n)
        else:
            log.error("无法获取{}{}日前的价格数据！".format(security, n))
            return None
    except:
        log.error(traceback.format_exc())
        return None

# N日波动率大于
def volatility_dayu(security, n, value, include_now=False):
    try:
        vol = get_vol(security, n, include_now)
        if float(vol) > float(value) :
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日波动率小于
def volatility_xiaoyu(security, n, value, include_now=False):
    try:
        vol = get_vol(security, n, include_now)
        if float(vol) < float(value) :
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日波动率区间
def volatility_qujian(security, n, value, include_now=False):
    try:
        vol = get_vol(security, n, include_now)
        a, b = value
        if float(a) < float(vol) < float(b):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False


############## 成交量比 ##############

# N日M日成交量比
def get_vol_ratio(security, n, m, include_now=False):
    # N日M日成交量比
    try:
        Ndata = get_bars(security, n, '1d', 'volume', include_now=include_now)
        Mdata = get_bars(security, m, '1d', 'volume', include_now=include_now)
        if len(Ndata) > 0 and len(Mdata) > 0:
            return float(Ndata["volume"].mean()) / Mdata["volume"].mean()
        else:
            log.error("无法获取{} {}和{}日的成交量数据！".format(security, n, m))
            return None
    except:
        log.error(traceback.format_exc())
        return None

# N日M日成交量比大于
def volume_proportion_dayu(security, n, m, value, include_now=False):
    try:
        vr = get_vol_ratio(security, n, m, include_now)
        if float(vr) > float(value):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日M日成交量比小于
def volume_proportion_xiaoyu(security, n, m, value, include_now=False):
    try:
        vr = get_vol_ratio(security, n, m, include_now)
        if float(vr) < float(value):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

# N日M日成交量比区间
def volume_proportion_qujian(security, n, m, value, include_now=False):
    try:
        vr = get_vol_ratio(security, n, m, include_now)
        a, b = value
        if float(a) < float(vr) < float(b):
            return True
        else:
            return False
    except:
        log.error(traceback.format_exc())
        return False

############## 龙虎榜 ##############

## 获取龙虎榜名单
def get_long_hu_list(check_date):
    '''
    获取龙虎榜名单
    check_date:查询日期
    '''
    df = get_billboard_list(stock_list=None, start_date=check_date, end_date=check_date, count=None)
    security_list = list(set(df['code']))
    return security_list

## 判断是否在龙虎榜
def in_long_hu_list(context, security, which_day='previous'):
    '''
    判断是否在龙虎榜
    security:查询标的
    which_day: 'previous'-前一交易日；'today'-当前交易日
    '''

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        long_hu_list = get_long_hu_list(check_date)
        if security in long_hu_list:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## 判断是否不在龙虎榜
def not_in_long_hu_list(context, security, which_day='previous'):
    '''
    判断是否不在龙虎榜
    security:查询标的
    which_day: 'previous'-前一交易日；'today'-当前交易日
    '''

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        long_hu_list = get_long_hu_list(check_date)
        if security in long_hu_list:
            return False
        else:
            return True
    except Exception as e:
        log.error(traceback.format_exc())
        return True

## 转为float
def to_float(stock):
    return float(stock)

## 前五买入卖出净额
def get_lh_net_cash(security,check_date,trade_type,value_type):
    '''
    获取龙虎榜前五买入卖出净额
    security：  查询标的代码
    check_date: 查询日期
    trade_type: BUY-买入；SELL-卖出; ALL-汇总
    value_type：buy_value-买入金额； buy_rate-买入金额占比 （买入金额/市场总成交额)；
                sell_value-卖出金额 ；sell_rate-卖出金额占比（卖出金额/市场总成交额)；
                net_value-净额 （买入金额 - 卖出金额；
    '''
    df = get_billboard_list(stock_list=security, start_date=check_date, end_date=check_date, count=None)

    net_cash = df[df['abnormal_type'] == trade_type][value_type].values
    # net_cash = map(to_float, net_cash)
    net_cash = sum(net_cash)#求前五总净额
    return net_cash

## 前五买入卖出净额大于
def lh_net_cash_dayu(context,security,value,trade_type,value_type,which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = get_lh_net_cash(security,check_date,trade_type,value_type)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 前五买入卖出净额小于
def lh_net_cash_xiaoyu(context,security,value,trade_type,value_type,which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = get_lh_net_cash(security,check_date,trade_type,value_type)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 前五买入卖出净额区间
def lh_net_cash_qujian(context,security,value,trade_type,value_type,which_day='previous'):
    import jqdata

    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()

    try:
        security_data = get_lh_net_cash(security,check_date,trade_type,value_type)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

############## 限售解禁 ##############

## 当日解禁股数量
def lifting_stock_amount(security,check_date):
    '''
    当日解禁股数量
    security：查询标的
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=1)
    # 解禁股数量
    if len(df)>0:
        amount = df['num'].iloc[0]
    else:
        amount = 0
    return amount/10000.0

## 当日新增可售A股占上期末总股本比例(%)
def lifting_total_stock_percentage(security,check_date):
    '''
    当日新增可售A股占上期末总股本比例(%)
    security：查询标的
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=1)
    # 本次新增可售A股占上期末已流通A股比例(%)
    if len(df)>0:
        percentage = df['rate1'].iloc[0]
    else:
        percentage = 0
    return percentage

## 当日新增可售A股占上期末已流通总股本比例(%)
def lifting_stock_percentage(security,check_date):
    '''
    当日新增可售A股占上期末已流通总股本比例(%)
    security：查询标的
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=1)
    # 本次新增可售A股占上期末已流通A股比例(%)
    if len(df)>0:
        percentage = df['rate2'].iloc[0]
    else:
        percentage = 0
    return percentage

## 未来 N 日解禁股数量
def future_n_days_lifting_stock_amount(security, n, check_date):
    '''
    未来 N 日解禁股数量
    security：查询标的
    n：未来几天
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=n)
    # 解禁股数量
    amount = df['num'].sum()
    return amount/10000.0

## 未来 N 日新增可售A股占上期末总股本比例(%)
def future_n_days_lifting_total_stock_percentage(security, n, check_date):
    '''
    未来 N 日新增可售A股占上期末总股本比例(%)
    security：查询标的
    n： 未来几天
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=n)
    # 解禁股数量
    percentage = df['rate1'].sum()
    return percentage

## 未来 N 日新增可售A股占上期末已流通总股本比例(%)
def future_n_days_lifting_stock_percentage(security, n, check_date):
    '''
    未来 N 日新增可售A股占上期末已流通总股本比例(%)
    security：查询标的
    n： 未来几天
    check_date：查询日期
    '''
    df = get_locked_shares(stock_list=security, start_date=check_date, end_date=None, forward_count=n)
    # 解禁股数量
    percentage = df['rate2'].sum()
    return percentage

# 当日解禁股数量大于
def lifting_stock_amount_dayu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_amount(security,check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日解禁股数量大于等于
def lifting_stock_amount_dayudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_amount(security,check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日解禁股数量小于
def lifting_stock_amount_xiaoyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_amount(security,check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日解禁股数量小于等于
def lifting_stock_amount_xiaoyudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_amount(security,check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日解禁股数量区间
def lifting_stock_amount_qujian(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_amount(security,check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末总股本比例(%)大于
def lifting_total_stock_percentage_dayu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_total_stock_percentage(security,check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末总股本比例(%)大于等于
def lifting_total_stock_percentage_dayudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_total_stock_percentage(security,check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末总股本比例(%)小于
def lifting_total_stock_percentage_xiaoyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_total_stock_percentage(security,check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末总股本比例(%)小于等于
def lifting_total_stock_percentage_xiaoyudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_total_stock_percentage(security,check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末总股本比例(%)区间
def lifting_total_stock_percentage_qujian(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_total_stock_percentage(security,check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末已流通A股比例(%)大于
def lifting_stock_percentage_dayu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_percentage(security,check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末已流通A股比例(%)大于等于
def lifting_stock_percentage_dayudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_percentage(security,check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末已流通A股比例(%)小于
def lifting_stock_percentage_xiaoyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_percentage(security,check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末已流通A股比例(%)小于等于
def lifting_stock_percentage_xiaoyudengyu(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_percentage(security,check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 当日新增可售A股占上期末已流通A股比例(%)区间
def lifting_stock_percentage_qujian(context,security,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = lifting_stock_percentage(security,check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日解禁股数量大于
def future_n_days_lifting_stock_amount_dayu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_amount(security, n, check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日解禁股数量大于等于
def future_n_days_lifting_stock_amount_dayudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_amount(security, n, check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日解禁股数量小于
def future_n_days_lifting_stock_amount_xiaoyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_amount(security, n, check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日解禁股数量小于等于
def future_n_days_lifting_stock_amount_xiaoyudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_amount(security, n, check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日解禁股数量区间
def future_n_days_lifting_stock_amount_qujian(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_amount(security, n, check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末总股本比例(%)大于
def future_n_days_lifting_total_stock_percentage_dayu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_total_stock_percentage(security, n, check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末总股本比例(%)大于等于
def future_n_days_lifting_total_stock_percentage_dayudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_total_stock_percentage(security, n, check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末总股本比例(%)小于
def future_n_days_lifting_total_stock_percentage_xiaoyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_total_stock_percentage(security, n, check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末总股本比例(%)小于等于
def future_n_days_lifting_total_stock_percentage_xiaoyudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_total_stock_percentage(security, n, check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末总股本比例(%)区间
def future_n_days_lifting_total_stock_percentage_qujian(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_total_stock_percentage(security, n, check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末已流通A股比例(%)大于
def future_n_days_lifting_stock_percentage_dayu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_percentage(security, n, check_date)
        if security_data > value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末已流通A股比例(%)大于等于
def future_n_days_lifting_stock_percentage_dayudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_percentage(security, n, check_date)
        if security_data >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末已流通A股比例(%)小于
def future_n_days_lifting_stock_percentage_xiaoyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_percentage(security, n, check_date)
        if security_data < value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末已流通A股比例(%)小于等于
def future_n_days_lifting_stock_percentage_xiaoyudengyu(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_percentage(security, n, check_date)
        if security_data <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 未来 N 日新增可售A股占上期末已流通A股比例(%)区间
def future_n_days_lifting_stock_percentage_qujian(context,security, n,value,which_day='today'):
    try:
        if which_day == 'previous':
            check_date = context.previous_date
        elif which_day == 'today':
            check_date = context.current_dt.date()

        security_data = future_n_days_lifting_stock_percentage(security, n, check_date)
        a,b = value
        if a < security_data < b:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

##########指数成分权重#########


def get_index_weight(context, security, index):
    # 获取该股票在指数中的权重
    dt = context.current_dt.date()
    df = fintest.run_query(query(fintest.IDX_WEIGHT_MONTH
                            ).filter(fintest.IDX_WEIGHT_MONTH.index_code == index, fintest.IDX_WEIGHT_MONTH.stock_code == security
                            ).order_by(fintest.IDX_WEIGHT_MONTH.end_date.desc()))
    if not df.empty:
        df["start_date"] = df["end_date"].shift(-1)
        # 判断是否为最新
        LastDate = df["end_date"].max()
        if dt >= LastDate:
            return df["weight"][0]
        else:
            for row in df.iterrows():
                if row.start_date < dt <= row.end_date:
                    return row.weight
            else:
                return None
    else:
        return None

def index_stocks_weight_dayu(context, security, search, value):
    try:
        weight = get_index_weight(context, security, search)
        if weight:
            if weight > value:
                return True
            else:
                return False
        else:
            log.error("无法获取标的{}在指数{}中的权重数据，请检查！".format(security, search))
            return False
    except:
        log.error(traceback.format_exc())
        return False

def index_stocks_weight_xiaoyu(context, security, search, value):
    try:
        weight = get_index_weight(context, security, search)
        if weight:
            if weight < value:
                return True
            else:
                return False
        else:
            log.error("无法获取标的{}在指数{}中的权重数据，请检查！".format(security, search))
            return False
    except:
        log.error(traceback.format_exc())
        return False


def index_stocks_weight_qujian(context, security, search, value):
    try:
        weight = get_index_weight(context, security, search)
        if weight:
            a, b = value
            if a < weight < b:
                return True
            else:
                return False
        else:
            log.error("无法获取标的{}在指数{}中的权重数据，请检查！".format(security, search))
            return False
    except:
        log.error(traceback.format_exc())
        return False

##########高管增持#########

def get_share_change_info(context, security, which_day):
    if which_day == 'previous':
        check_date = context.previous_date
    elif which_day == 'today':
        check_date = context.current_dt.date()
    df = fintest.run_query(query(fintest.STK_SHARE_CHANGE).
                           filter(fintest.STK_SHARE_CHANGE.code == security, fintest.STK_SHARE_CHANGE.change_date == check_date))
    return df


def get_ndays_share_change_info(context, security, n, which_day):
    if which_day == 'previous':
        dt = context.previous_date
    elif which_day == 'today':
        dt = context.current_dt.date()
    check_date = dt - datetime.timedelta(days=n)
    df = fintest.run_query(query(fintest.STK_SHARE_CHANGE).
                           filter(fintest.STK_SHARE_CHANGE.code == security, fintest.STK_SHARE_CHANGE.change_date < check_date))
    return df


def share_holder_num_change_increase(context, security, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        if not df.empty and df["changing_shares"].sum() > 0:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_holder_num_change_decrease(context, security, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        if not df.empty and df["changing_shares"].sum() < 0:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_increase_perc_dayu(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] > 0]
            if data["change_perc"].sum() and data["change_perc"].sum() > value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_increase_perc_xiaoyu(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] > 0]
            if data["change_perc"].sum() and data["change_perc"].sum() < value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_increase_perc_qujian(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        a, b = value
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] > 0]
            if data["change_perc"].sum() and a < data["change_perc"].sum() < b:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_decrease_perc_dayu(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] < 0]
            if data["change_perc"].sum() and data["change_perc"].sum() > value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_decrease_perc_xiaoyu(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] < 0]
            if data["change_perc"].sum() and data["change_perc"].sum() < value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def share_change_decrease_perc_qujian(context, security, value, which_day='previous'):
    try:
        df = get_share_change_info(context, security, which_day)
        a, b = value
        # 判定字段不为空
        if not df.empty:
            data = df[df["change_perc"] < 0]
            if data["change_perc"].sum() and a < data["change_perc"].sum() < b:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动比例为空，请检查！".format(security))
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def n_day_share_holder_num_change_increase(context, security, n, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] > 0]
            return len(data)
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
            return None
    except Exception as e:
        log.error(traceback.format_exc())
        return None


def n_day_share_holder_num_change_decrease(context, security, n, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] < 0]
            return len(data)
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
            return None
    except Exception as e:
        log.error(traceback.format_exc())
        return None

def n_day_share_change_increase_perc_dayu(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] > 0]
            if not data.empty and data["change_perc"].sum() > value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def n_day_share_change_increase_perc_xiaoyu(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] > 0]
            if not data.empty and data["change_perc"].sum() < value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def n_day_share_change_increase_perc_qujian(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        a, b = value
        if not df.empty:
            data = df[df["changing_shares"] > 0]
            if not data.empty and a < data["change_perc"].sum() < b:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def n_day_share_change_decrease_perc_dayu(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] < 0]
            if not data.empty and data["change_perc"].sum() > value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def n_day_share_change_decrease_perc_xiaoyu(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        if not df.empty:
            data = df[df["changing_shares"] < 0]
            if not data.empty and data["change_perc"].sum() < value:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

def n_day_share_change_decrease_perc_qujian(context, security, n, value, which_day='previous'):
    try:
        df = get_ndays_share_change_info(context, security, n, which_day)
        a, b = value
        if not df.empty:
            data = df[df["changing_shares"] < 0]
            if not data.empty and a < data["change_perc"].sum() < b:
                return True
            else:
                return False
        else:
            log.error("标的{}的股份变动数据为空，请检查！".format(security))
    except Exception as e:
        log.error(traceback.format_exc())
        return False

#######业绩预告#######

def get_forecast_info(context, security):
    # 获取最新业绩预告信息
    df = fintest.run_query(query(fintest.STK_FIN_FORCAST).
                           filter(fintest.STK_FIN_FORCAST.code == security, fintest.STK_FIN_FORCAST.report_date < context.current_dt).
                           order_by(fintest.STK_FIN_FORCAST.report_date.desc()).limit(1))
    return df


def forecast_dayu(context, security, search, value):
    try:
        df = get_forcast_info(context, security)
        if not df.empty and df[search][0] > value:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def forecast_xiaoyu(context, security, search, value):
    try:
        df = get_forcast_info(context, security)
        if not df.empty and df[search][0] < value:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def forecast_qujian(context, security, search, value):
    try:
        df = get_forcast_info(context, security)
        a, b = value
        if not df.empty and a < df[search][0] < b:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

#######重大违规事项########
# 当日是否有重大违规标记
def violation(context, security):
    try:
        df = fintest.run_query(query(fintest.STK_VIOLATION).
                               filter(fintest.STK_VIOLATION.code == security).
                               filter(fintest.STK_VIOLATION.declare_date == context.current_dt))
        if df.empty:
            return False
        else:
            return True
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# N日重大违规标记
def get_n_days_violation_info(context, security, n):
    check_date = context.current_dt - datetime.timedelta(days=n)
    df = fintest.run_query(query(fintest.STK_VIOLATION).
                           filter(fintest.STK_VIOLATION.code == security).
                           filter(fintest.STK_VIOLATION.declare_date >= check_date))
    return df


def n_day_violation_dayu(context, security, n, value):
    try:
        df = get_n_days_violation_info(context, security, n)
        if len(df) > value:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def n_day_violation_xiaoyu(context, security, n, value):
    try:
        df = get_n_days_violation_info(context, security, n)
        if len(df) < value:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


def n_day_violation_qujian(context, security, n, value):
    try:
        df = get_n_days_violation_info(context, security, n)
        a, b = value
        if a < len(df) < b:
            return True
        else:
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False



