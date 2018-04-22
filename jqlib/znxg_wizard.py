#-*- coding: UTF-8 -*-

from __future__ import division
from kuanke.user_space_api import *
from jqdata import *
import numpy as np
import pandas as pd
import talib
import datetime
from numpy import mean
import traceback
import os
import six


# 获取股票股票池
def get_security_universe(context, security_universe_index):
    temp_index = []
    for s in security_universe_index:
        if s == 'all_a_securities':
            temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
        else:
            temp_index += get_index_stocks(s)
    return  sorted(list(set(temp_index)))

# 行业过滤
def industry_filter(context, security_list, industry_list):
    if len(industry_list) == 0:
        # 返回股票列表
        return security_list
    else:
        securities = []
        for s in industry_list:
            temp_securities = get_industry_stocks(s)
            securities += temp_securities
        security_list = [stock for stock in security_list if stock in securities]
        # 返回股票列表
        return security_list

##################################  行情数据筛选 ##################################

## 行情大于筛选
def situation_filter_dayu(security, search, value, include_now=True):
    try:
        if search in ['open','close','high','low','money','volume']:
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] > security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] > security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-1] > value:
                    return True
            return False
        else:
            search = search[4:]
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] > security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] > security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-2] > value:
                    return True
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情小于筛选
def situation_filter_xiaoyu(security, search, value, include_now=True):
    try:
        if search in ['open','close','high','low','money','volume']:
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] < security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] < security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-1] < value:
                    return True
            return False
        else:
            search = search[4:]
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] < security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] < security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-2] < value:
                    return True
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情大于等于筛选
def situation_filter_dayudengyu(security, search, value, include_now=True):
    try:
        if search in ['open','close','high','low','money','volume']:
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] >= security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] >= security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-1] >= value:
                    return True
            return False
        else:
            search = search[4:]
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] >= security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] >= security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-2] >= value:
                    return True
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情小于等于筛选
def situation_filter_xiaoyudengyu(security, search, value, include_now=True):
    try:
        if search in ['open','close','high','low','money','volume']:
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] <= security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] <= security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-1] <= value:
                    return True
            return False
        else:
            search = search[4:]
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] <= security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] <= security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-2] <= value:
                    return True
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 行情等于筛选
def situation_filter_dengyu(security, search, value, include_now=True):
    try:
        if search in ['open','close','high','low','money','volume']:
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] == security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-1] == security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-1] == value:
                    return True
            return False
        else:
            search = search[4:]
            if value in ['open','close','high','low','money','volume']:
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] == security_data[value][-1]:
                    return True
            elif value in ['lastopen','lastclose','lasthigh','lastlow','lastmoney','lastvolume']:
                value = value[4:]
                if search == value:
                    security_data = get_bars(security, 2, '1d', [search], include_now)
                else:
                    security_data = get_bars(security, 2, '1d', [search,value], include_now)
                if security_data[search][-2] == security_data[value][-2]:
                    return True
            else:
                security_data = get_bars(security, 2, '1d', [search], include_now)
                if security_data[search][-2] == value:
                    return True
            return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

## N日涨幅大于筛选
def n_day_chg_dayu(security, n, value, n2=5, include_now=True):
    try:
        if value in['nrise']:
            security_data = get_bars(security, n2+1, '1d', ['close'], include_now)
            chg1 = (security_data['close'][-1]/security_data['close'][-2]) - 1
            chg2 = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg1 > chg2:
                return True
        else:
            security_data = get_bars(security, n+1, '1d', ['close'], include_now)
            chg = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg > value:
                return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅小于筛选
def n_day_chg_xiaoyu(security, n, value, n2=5, include_now=True):
    try:
        if value in['nrise']:
            security_data = get_bars(security, n2+1, '1d', ['close'], include_now)
            chg1 = (security_data['close'][-1]/security_data['close'][-2]) - 1
            chg2 = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg1 < chg2:
                return True
        else:
            security_data = get_bars(security, n+1, '1d', ['close'], include_now)
            chg = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg < value:
                return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅大于等于筛选
def n_day_chg_dayudengyu(security, n, value, n2=5, include_now=True):
    try:
        if value in['nrise']:
            security_data = get_bars(security, n2+1, '1d', ['close'], include_now)
            chg1 = (security_data['close'][-1]/security_data['close'][-2]) - 1
            chg2 = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg1 >= chg2:
                return True
        else:
            security_data = get_bars(security, n+1, '1d', ['close'], include_now)
            chg = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg >= value:
                return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅小于等于筛选
def n_day_chg_xiaoyudengyu(security, n, value, n2=5, include_now=True):
    try:
        if value in['nrise']:
            security_data = get_bars(security, n2+1, '1d', ['close'], include_now)
            chg1 = (security_data['close'][-1]/security_data['close'][-2]) - 1
            chg2 = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg1 <= chg2:
                return True
        else:
            security_data = get_bars(security, n+1, '1d', ['close'], include_now)
            chg = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg <= value:
                return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## N日涨幅等于筛选
def n_day_chg_dengyu(security, n, value, n2=5, include_now=True):
    try:
        if value in['nrise']:
            security_data = get_bars(security, n2+1, '1d', ['close'], include_now)
            chg1 = (security_data['close'][-1]/security_data['close'][-2]) - 1
            chg2 = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg1 == chg2:
                return True
        else:
            security_data = get_bars(security, n+1, '1d', ['close'], include_now)
            chg = (security_data['close'][-1]/security_data['close'][0]) - 1
            if chg == value:
                return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False


## 振幅大于筛选
def amplitude_dayu(security, value, include_now=True):
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
def amplitude_xiaoyu(security, value, include_now=True):
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
def amplitude_dayudengyu(security, value, include_now=True):
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
def amplitude_xiaoyudengyu(security, value, include_now=True):
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
def amplitude_dengyu(security, value, include_now=True):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low','close'], include_now)
        chg = ((security_data['high'][-1]-security_data['low'][-1])/security_data['close'][-2])
        if chg == value:
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
## 上市天数大于等于筛选
def ipo_days_dayudengyu(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        if long_days >= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 上市天数小于等于筛选
def ipo_days_xiaoyudengyu(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        if long_days <= value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
## 上市天数等于筛选
def ipo_days_dengyuyu(context, security, value):
    try:
        now = context.current_dt.date()
        ipo_day = get_security_info(security).start_date
        long_days = (now-ipo_day).days
        if long_days == value:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
################################### 财务数据函数群  ####################################
## 财务大于筛选
def financial_data_filter_dayu(context,security_list, search, value):
    ss = str(search).split('.')[-1]
    if ss in ['pe_ratio','pb_ratio']:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list), search>value)
        # 生成股票列表
        df = get_fundamentals(q)
        security_list = list(df['code'])
    elif ss in ['ttl_oper_inc','ttl_oper_cost','oper_inc','ttl_prof','net_prof']:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_inc_stmt_nas.code,
            func.max(mysqldb.comcn_inc_stmt_nas.statDate).label("statDate"),#报告期,
            mysqldb.comcn_inc_stmt_nas.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_inc_stmt_nas.code.in_(security_list),
            mysqldb.comcn_inc_stmt_nas.pubDate < context.current_dt.date(),
            search > value,
            mysqldb.comcn_inc_stmt_nas.data_flag == 102,
        ).group_by(
            mysqldb.comcn_inc_stmt_nas.code,
        ))
        security_list = list(df['code'])
    else:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_fin_anal.code,
            func.max(mysqldb.comcn_fin_anal.statDate).label("statDate"),#报告期,
            mysqldb.comcn_fin_anal.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_fin_anal.code.in_(security_list),
            mysqldb.comcn_fin_anal.pubDate < context.current_dt.date(),
            search > value,
            mysqldb.comcn_fin_anal.data_flag == 102,
        ).group_by(
            mysqldb.comcn_fin_anal.code,
        ))
        security_list = list(df['code'])
        
    return security_list

## 财务小于筛选
def financial_data_filter_xiaoyu(context,security_list, search, value):
    ss = str(search).split('.')[-1]
    if ss in ['pe_ratio','pb_ratio']:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list), search<value)
        # 生成股票列表
        df = get_fundamentals(q)
        security_list = list(df['code'])
    elif ss in ['ttl_oper_inc','ttl_oper_cost','oper_inc','ttl_prof','net_prof']:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_inc_stmt_nas.code,
            func.max(mysqldb.comcn_inc_stmt_nas.statDate).label("statDate"),#报告期,
            mysqldb.comcn_inc_stmt_nas.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_inc_stmt_nas.code.in_(security_list),
            mysqldb.comcn_inc_stmt_nas.pubDate < context.current_dt.date(),
            search < value,
            mysqldb.comcn_inc_stmt_nas.data_flag == 102,
        ).group_by(
            mysqldb.comcn_inc_stmt_nas.code,
        ))
        security_list = list(df['code'])
    else:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_fin_anal.code,
            func.max(mysqldb.comcn_fin_anal.statDate).label("statDate"),#报告期,
            mysqldb.comcn_fin_anal.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_fin_anal.code.in_(security_list),
            mysqldb.comcn_fin_anal.pubDate < context.current_dt.date(),
            search < value,
            mysqldb.comcn_fin_anal.data_flag == 102,
        ).group_by(
            mysqldb.comcn_fin_anal.code,
        ))
        security_list = list(df['code'])
        
    return security_list

## 财务大于等于筛选
def financial_data_filter_dayudengyu(context,security_list, search, value):
    ss = str(search).split('.')[-1]
    if ss in ['pe_ratio','pb_ratio']:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list), search>=value)
        # 生成股票列表
        df = get_fundamentals(q)
        security_list = list(df['code'])
    elif ss in ['ttl_oper_inc','ttl_oper_cost','oper_inc','ttl_prof','net_prof']:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_inc_stmt_nas.code,
            func.max(mysqldb.comcn_inc_stmt_nas.statDate).label("statDate"),#报告期,
            mysqldb.comcn_inc_stmt_nas.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_inc_stmt_nas.code.in_(security_list),
            mysqldb.comcn_inc_stmt_nas.pubDate < context.current_dt.date(),
            search >= value,
            mysqldb.comcn_inc_stmt_nas.data_flag == 102,
        ).group_by(
            mysqldb.comcn_inc_stmt_nas.code,
        ))
        security_list = list(df['code'])
    else:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_fin_anal.code,
            func.max(mysqldb.comcn_fin_anal.statDate).label("statDate"),#报告期,
            mysqldb.comcn_fin_anal.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_fin_anal.code.in_(security_list),
            mysqldb.comcn_fin_anal.pubDate < context.current_dt.date(),
            search >= value,
            mysqldb.comcn_fin_anal.data_flag == 102,
        ).group_by(
            mysqldb.comcn_fin_anal.code,
        ))
        security_list = list(df['code'])
        
    return security_list


## 财务小于等于筛选
def financial_data_filter_xiaoyudengyu(context,security_list, search, value):
    ss = str(search).split('.')[-1]
    if ss in ['pe_ratio','pb_ratio']:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list), search<=value)
        # 生成股票列表
        df = get_fundamentals(q)
        security_list = list(df['code'])
    elif ss in ['ttl_oper_inc','ttl_oper_cost','oper_inc','ttl_prof','net_prof']:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_inc_stmt_nas.code,
            func.max(mysqldb.comcn_inc_stmt_nas.statDate).label("statDate"),#报告期,
            mysqldb.comcn_inc_stmt_nas.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_inc_stmt_nas.code.in_(security_list),
            mysqldb.comcn_inc_stmt_nas.pubDate < context.current_dt.date(),
            search <= value,
            mysqldb.comcn_inc_stmt_nas.data_flag == 102,
        ).group_by(
            mysqldb.comcn_inc_stmt_nas.code,
        ))
        security_list = list(df['code'])
    else:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_fin_anal.code,
            func.max(mysqldb.comcn_fin_anal.statDate).label("statDate"),#报告期,
            mysqldb.comcn_fin_anal.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_fin_anal.code.in_(security_list),
            mysqldb.comcn_fin_anal.pubDate < context.current_dt.date(),
            search <= value,
            mysqldb.comcn_fin_anal.data_flag == 102,
        ).group_by(
            mysqldb.comcn_fin_anal.code,
        ))
        security_list = list(df['code'])
        
    return security_list

## 财务等于筛选
def financial_data_filter_dengyu(context,security_list, search, value):
    ss = str(search).split('.')[-1]
    if ss in ['pe_ratio','pb_ratio']:
        # 生成查询条件
        q = query(valuation.code,search).filter(valuation.code.in_(security_list), search==value)
        # 生成股票列表
        df = get_fundamentals(q)
        security_list = list(df['code'])
    elif ss in ['ttl_oper_inc','ttl_oper_cost','oper_inc','ttl_prof','net_prof']:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_inc_stmt_nas.code,
            func.max(mysqldb.comcn_inc_stmt_nas.statDate).label("statDate"),#报告期,
            mysqldb.comcn_inc_stmt_nas.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_inc_stmt_nas.code.in_(security_list),
            mysqldb.comcn_inc_stmt_nas.pubDate < context.current_dt.date(),
            search == value,
            mysqldb.comcn_inc_stmt_nas.data_flag == 102,
        ).group_by(
            mysqldb.comcn_inc_stmt_nas.code,
        ))
        security_list = list(df['code'])
    else:
        from jqdata.tp.db import mysqldb
        from sqlalchemy.sql import func
        df = mysqldb.run_query(query(
            mysqldb.comcn_fin_anal.code,
            func.max(mysqldb.comcn_fin_anal.statDate).label("statDate"),#报告期,
            mysqldb.comcn_fin_anal.pubDate,#披露日期
            search
        ).filter(
            mysqldb.comcn_fin_anal.code.in_(security_list),
            mysqldb.comcn_fin_anal.pubDate < context.current_dt.date(),
            search == value,
            mysqldb.comcn_fin_anal.data_flag == 102,
        ).group_by(
            mysqldb.comcn_fin_anal.code,
        ))
        security_list = list(df['code'])
        
    return security_list

###################################  技术指标公用函数群 ##################################
# MACD
def MACD(security_list, fastperiod=12, slowperiod=26, signalperiod=9, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 MACD
    macd_DIF = {}; macd_DEA = {}; macd_HIST = {}
    for stock in security_list:
        security_data = get_bars(stock, slowperiod*2, '1d', ['close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            macd_DIF[stock] = np.array([np.nan])
            macd_DEA[stock] = np.array([np.nan])
            macd_HIST[stock]= np.array([np.nan])
        else:
            macd_DIF[stock], macd_DEA[stock], macd = talib.MACDEXT(security_data['close'], fastperiod=fastperiod, fastmatype=1, slowperiod=slowperiod, slowmatype=1, signalperiod=signalperiod, signalmatype=1)
            macd_HIST[stock] = macd * 2
    return macd_DIF, macd_DEA, macd_HIST

# MA
def MA(security_list, timeperiod=5, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 MA
    ma = {}
    for stock in security_list:
        security_data = get_bars(stock, timeperiod*2, '1d', ['close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            ma[stock] = np.array([np.nan])
        else:
            ma[stock] = talib.MA(security_data['close'], timeperiod)
    return ma


# KDJ-随机指标
def KDJ(security_list, N =9, M1=3, M2=3, include_now=True):
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
        security_data = get_bars(stock, n*10, '1d', ['high', 'low', 'close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            k[stock] = np.array([np.nan])
            d[stock] = np.array([np.nan])
            j[stock] = np.array([np.nan])
        else:
            high_KDJ = security_data['high']
            low_KDJ = security_data['low']
            close_KDJ = security_data['close']

            # 使用talib中的STOCHF函数求 RSV
            kValue, dValue = talib.STOCHF(high_KDJ, low_KDJ, close_KDJ, N , M2, fastd_matype=0)
            # 求K值(等于RSV的M1日移动平均)
            kValue = np.array(map(lambda x : SMA_CN(kValue[:x], M1), range(1, len(kValue) + 1)))
            # 求D值(等于K的M2日移动平均)
            dValue = np.array(map(lambda x : SMA_CN(kValue[:x], M2), range(1, len(kValue) + 1)))
            # 求J值
            jValue = 3 * kValue - 2 * dValue

            k[stock] = kValue
            d[stock] = dValue
            j[stock] = jValue
    return k, d, j

# RSI
def RSI(security_list, timeperiod=6, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 RSI
    rsi = {}
    for stock in security_list:
        security_data = get_bars(stock, timeperiod*20, '1d', ['close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            rsi[stock] = np.array([np.nan])
        else:
            rsi[stock] = talib.RSI(security_data['close'], timeperiod)
    return rsi

# CCI
def CCI(security_list, timeperiod=14, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 CCI
    cci = {}
    for stock in security_list:
        security_data = get_bars(stock, timeperiod*2, '1d', ['close','high','low'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            cci[stock] = np.array([np.nan])
        else:
            close_CCI = security_data['close']
            high_CCI = security_data['high']
            low_CCI = security_data['low']
            cci[stock] = talib.CCI(high_CCI, low_CCI, close_CCI, timeperiod)
    return cci


# 布林线
def Bollinger_Bands(security_list, timeperiod=20, nbdevup=2, nbdevdn=2, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 Bollinger Bands
    upperband={}; middleband={}; lowerband={}
    for stock in security_list:
        security_data = get_bars(stock, timeperiod*2, '1d', ['close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            upperband[stock] = np.array([np.nan])
            middleband[stock] = np.array([np.nan])
            lowerband[stock] = np.array([np.nan])
        else:
            upperband[stock], middleband[stock], lowerband[stock] = talib.BBANDS(security_data['close'], timeperiod, nbdevup, nbdevdn)
    return upperband, middleband, lowerband

# BIAS
def BIAS(security_list, N1=6, N2=12, N3=24, include_now=True):
    # 修复传入为单只股票的情况
    if isinstance(security_list, six.string_types):
        security_list = [security_list]
    # 计算 BIAS
    maxN = max(N1, N2, N3)
    bias_bias6 = {}
    bias_bias12 = {}
    bias_bias24 = {}
    for stock in security_list:
        security_data = get_bars(stock, maxN*3, '1d', ['close'], include_now)
        nan_count = list(np.isnan(security_data['close'])).count(True)
        if nan_count == len(security_data['close']):
            log.info("股票 %s 输入数据全是 NaN，该股票可能已退市、未上市或刚上市，返回 NaN 值数据。" %stock)
            bias_bias6[stock] = np.array([np.nan])
            bias_bias12[stock] = np.array([np.nan])
            bias_bias24[stock] = np.array([np.nan])
        else:
            close_BIAS = np.array(security_data['close'])
            average_price = talib.MA(close_BIAS, N1)
            bias6 = [(curr - aver) / aver * 100.0 for curr, aver in zip(close_BIAS, average_price)]
            bias_bias6[stock]= np.array(bias6)

            average_price = talib.MA(close_BIAS, N2)
            bias12 = [(curr - aver) / aver * 100.0 for curr, aver in zip(close_BIAS, average_price)]
            bias_bias12[stock]= np.array(bias12)

            average_price = talib.MA(close_BIAS, N3)
            bias24 = [(curr - aver) / aver * 100.0 for curr, aver in zip(close_BIAS, average_price)]
            bias_bias24[stock]= np.array(bias24)
    return bias_bias6, bias_bias12, bias_bias24



###################################  技术指标形态判断 ##################################
# 获得斜率的角度
def get_degree(list_data, num):
    '''
    计算公式：
        以0-(length-1)为下标，list_data为源数据，经过数据标准化和最小二乘法的拟合得到拟合后的直线，进而获取其斜率和角度
    输入：
        list_data：数据列表
        num: 拟合曲线需要的源数据长度num
    输出：
        degree：角度
    输出结果类型：
        float
    注：
        本程序不对数据长度做处理，所以用户自己先确定输入数据的长度
    '''
    import statsmodels.api as sm
    # 为拟合直线，需要num个数据
    list_data = list_data[-num:]
    # 下标为 0--(num-1)
    ind = np.array([i  for i in range(0, num)])
    val = np.array(list_data)
    # 数据标准化和数据类型转换
    ind = pd.Series(preprocessing.scale(ind))
    val = pd.Series(preprocessing.scale(val))
    # 获得以最小二乘法拟合后的函数模型
    model = sm.OLS(val,ind)
    results = model.fit()
    # 获得拟合后的直线的斜率
    slope = results.params[0]
    # 获得斜率的角度
    degree = math.degrees(math.atan(slope))
    return degree

# MA 金叉
def MA_JX(security, short_timeperiod=5, long_timeperiod=10, include_now=True):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod, include_now=include_now)
        ma_long = MA(security, timeperiod=long_timeperiod, include_now=include_now)
        if (ma_short[security][-2] < ma_long[security][-2])and(ma_short[security][-1] > ma_long[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 死叉
def MA_SX(security, short_timeperiod=5, long_timeperiod=10, include_now=True):
    try:
        ma_short = MA(security, timeperiod=short_timeperiod, include_now=include_now)
        ma_long = MA(security, timeperiod=long_timeperiod, include_now=include_now)
        if (ma_short[security][-2] > ma_long[security][-2])and(ma_short[security][-1] < ma_long[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 多头
def MA_DT(security, ma1=5, ma2=10, ma3=20, ma4=60, include_now=True):
    try:
        security_data = get_bars(security, ma4*2, '1d', ['close'], include_now)
        MA1 = talib.MA(security_data['close'], ma1)[-1]
        MA2 = talib.MA(security_data['close'], ma2)[-1]
        MA3 = talib.MA(security_data['close'], ma3)[-1]
        MA4 = talib.MA(security_data['close'], ma4)[-1]
        if MA1>MA2>MA3>MA4:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MA 空头
def MA_KT(security, ma1=5, ma2=10, ma3=20, ma4=60, include_now=True):
    try:
        security_data = get_bars(security, ma4*2, '1d', ['close'], include_now)
        MA1 = talib.MA(security_data['close'], ma1)[-1]
        MA2 = talib.MA(security_data['close'], ma2)[-1]
        MA3 = talib.MA(security_data['close'], ma3)[-1]
        MA4 = talib.MA(security_data['close'], ma4)[-1]
        if MA1<MA2<MA3<MA4:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# MACD 金叉
def MACD_JX(security, fastperiod=12, slowperiod=26, signalperiod=9, include_now=True):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod, include_now)
        if (macd_DIF[security][-1] > macd_DEA[security][-1])and(macd_DIF[security][-2] < macd_DEA[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 死叉
def MACD_SX(security, fastperiod=12, slowperiod=26, signalperiod=9, include_now=True):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod, include_now)
        if (macd_DIF[security][-1] < macd_DEA[security][-1])and(macd_DIF[security][-2] > macd_DEA[security][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 多头
def MACD_DT(security, fastperiod=12, slowperiod=26, signalperiod=9, include_now=True):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod, include_now)
        if macd_DIF[security][-1] > macd_DEA[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False
# MACD 空头
def MACD_KT(security, fastperiod=12, slowperiod=26, signalperiod=9, include_now=True):
    try:
        macd_DIF, macd_DEA, macd_HIST = MACD(security, fastperiod, slowperiod, signalperiod, include_now)
        if macd_DIF[security][-1] < macd_DEA[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BIAS金叉
def BIAS_JX(security, N1=6, N2=12, N3=24, include_now=True):
    try:
        bias6, bias12, bias24 = BIAS(security, N1, N2, N3, include_now)
        # bias6 上穿 bias24
        if bias6[security][-2] < bias24[security][-2] and bias6[security][-1] > bias24[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BIAS买入信号
def BIAS_MRXH(security, N1=6, N2=12, N3=24, include_now=True):
    try:
        bias6, bias12, bias24 = BIAS(security, N1, N2, N3, include_now)
        # bias12 > 0 并且 bias6 上穿 bias12
        if (bias12[security][-1] > 0) and (bias6[security][-2] < bias12[security][-2] and bias6[security][-1] > bias12[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BIAS超卖
def BIAS_CM(security, N1=6, N2=12, N3=24, include_now=True):
    try:
        bias6, bias12, bias24 = BIAS(security, N1, N2, N3, include_now)
        # bias6 < -10
        if bias6[security][-1] < -10.0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BIAS底背离
def BIAS_DBL(security, N1=6, N2=12, N3=24, num=5, deg=10.0, include_now=True):
    ''' 
    输入：
        security:股票列表
        N：统计的天数 N
        num: 拟合曲线需要的源数据长度num
        deg: 判断背离的角度
    输出：
        底背离与否
    输出结果类型：
        布尔类型
    '''
    res = False
    try:
        # 获取rows天的收盘价
        rows = max(N1, N2, N3)* 2
        close_BIAS = get_bars(security, rows, '1d', ['close'], include_now)
        # 获取num天收盘价的拟合曲线的角度
        degree_close = get_degree(close_BIAS['close'], num)
        # 获取BIAS24值
        bias6, bias12, bias24 = BIAS(security, N1, N2, N3)
        # 获取num天BIAS24值的拟合曲线的角度
        degree_bias24 = get_degree(bias24[security], num)
        # 股价创出新低，BIAS24指标却不断上行
        res = degree_close < -deg and degree_bias24 > deg
        return res
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BOLL突破上轨
def BOLL_TPSG(security, timeperiod=20, nbdevup=2, nbdevdn=2, include_now=True):
    try:
        up, mid, low = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn, include_now)
        # 获取收盘价
        close_BOLL = get_bars(security, timeperiod*2, '1d', ['close'], include_now)
        # 日K上穿上轨
        if close_BOLL['close'][-2] < up[security][-2] and close_BOLL['close'][-1] > up[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BOLL突破中轨
def BOLL_TPZG(security, timeperiod=20, nbdevup=2, nbdevdn=2, include_now=True):
    try:
        up, mid, low = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn, include_now)
        # 获取收盘价
        close_BOLL = get_bars(security, timeperiod*2, '1d', ['close'], include_now)
        # 日K上穿中轨
        if close_BOLL['close'][-2] < mid[security][-2] and close_BOLL['close'][-1] > mid[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# BOLL突破下轨
def BOLL_TPXG(security, timeperiod=20, nbdevup=2, nbdevdn=2, include_now=True):
    try:
        up, mid, low = Bollinger_Bands(security, timeperiod, nbdevup, nbdevdn, include_now)
        # 获取收盘价
        close_BOLL = get_bars(security, timeperiod*2, '1d', ['close'], include_now)
        # 日K上穿下轨
        if close_BOLL['close'][-2] < low[security][-2] and close_BOLL['close'][-1] > low[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# CCI买入信号
def CCI_MRXH(security, timeperiod=14, include_now=True):
    try:
        cci = CCI(security, timeperiod, include_now)
        # 买入信号：CCI上穿100
        if cci[security][-2] < 100.0 and cci[security][-1] > 100.0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# CCI超卖
def CCI_CM(security, timeperiod=14, include_now=True):
    try:
        cci = CCI(security, timeperiod, include_now)
        # 超卖：CCI < -100
        if cci[security][-1] < -100.0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# CCI底背离
def CCI_DBL(security, N=14, num=5, deg=10.0, include_now=True):
    ''' 
    输入：
        security:股票列表
        N：统计的天数 N
        num: 拟合曲线需要的源数据长度num
        deg: 判断背离的角度
    输出：
        底背离与否
    输出结果类型：
        布尔类型
    '''
    res = False
    try:
        # 获取rows天的收盘价
        rows = N * 2
        close_CCI = get_bars(security, rows, '1d', ['close'], include_now)
        # 获取num天收盘价的拟合曲线的角度
        degree_close = get_degree(close_CCI['close'], num)
        # 获取CCI值
        cci = CCI(security, N)
        # 获取num天CCI值的拟合曲线的角度
        degree_cci = get_degree(cci[security], num)
        # 股价创出新低，CCI指标却不断上行
        res = degree_close < -deg and degree_cci > deg
        return res
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-金叉
def KDJ_JX(security, N=9, M1=3, M2=3, include_now=True):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2, include_now)
        # 金叉：K线上穿D线
        if (kdj_K[security][-2] < kdj_D[security][-2]) and (kdj_K[security][-1] > kdj_D[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ-买入信号
def KDJ_MRXH(security, N=9, M1=3, M2=3, include_now=True):
    try:
        kdj_K, kdj_D, kdj_J = KDJ(security, N, M1, M2, include_now)
        # K,D < 20 并且 K线上穿D线
        if (kdj_K[security][-1] < 20 and kdj_D[security][-1] < 20) and (kdj_K[security][-2] < kdj_D[security][-2]) and (kdj_K[security][-1] > kdj_D[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# KDJ底背离
def KDJ_DBL(security, N=9, M1=3, M2=3, num = 5, deg = 10.0, include_now=True):
    '''
    输入：
        security:股票列表
        N：统计的天数 N
        M1：统计的天数 M1
        M2：统计的天数 M2
        num: 拟合曲线需要的源数据长度num
        deg: 判断背离的角度
    输出：
        底背离与否
    输出结果类型：
        布尔类型
    '''
    res = False
    grid = {}
    try:
        # 获取rows天的收盘价
        rows = max(N, M1, M2) * 2
        close_KDJ = get_bars(security, rows, '1d', ['close'], include_now)
        # 获取num天收盘价的拟合曲线的角度
        degree_close = get_degree(close_KDJ['close'], num)
        # 获取K,D,J值
        grid['k'], grid['d'], grid['j'] = KDJ(security, N, M1, M2)
        # 获取num天k,d值的拟合曲线的角度
        degree_k = get_degree(grid['k'][security], num)
        degree_d = get_degree(grid['d'][security], num)
        # 股价创出新低，k，d却不断上行
        res = degree_close < -deg and degree_d > deg and degree_k > deg        
        return res
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# RSI金叉
def RSI_JX(security, N1=6, N2=12, include_now=True):
    try:
        rsi6 = RSI(security, N1, include_now)
        rsi12 = RSI(security, N2, include_now)
        # rsi6 上穿 rsi12
        if rsi6[security][-2] < rsi12[security][-2] and rsi6[security][-1] > rsi12[security][-1]:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# RSI买入
def RSI_MRXH(security, N1=6, N2=12, include_now=True):
    try:
        rsi6 = RSI(security, N1, include_now)
        rsi12 = RSI(security, N2, include_now)
        # rsi6>80 并且 rsi6 上穿 rsi12 rsi 以rsi6来替换
        if rsi6 > 80 and (rsi6[security][-2] < rsi12[security][-2] and rsi6[security][-1] > rsi12[security][-1]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# RSI超卖
def RSI_CM(security, N1=6, include_now=True):
    try:
        rsi6 = RSI(security, N1, include_now)
        # rsi6 < 20
        if rsi6[security][-1] < 20.0:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# RSI底背离
def RSI_DBL(security, N1=6, N2=12, num = 5, deg = 10.0, include_now=True):
    '''
    输入：
        security:股票列表
        N1：统计的天数 N1
        N2：统计的天数 N2
        num: 拟合曲线需要的源数据长度num
        deg: 判断背离的角度
    输出：
        底背离与否
    输出结果类型：
        布尔类型
    '''
    res = False
    grid = {}
    try:
        # 获取rows天的收盘价
        rows = max(N1, N2) * 2
        close_RSI = get_bars(security, rows, '1d', ['close'], include_now)
        # 获取num天收盘价的拟合曲线的角度
        degree_close = get_degree(close_RSI['close'], num)
        # 获取rsi6的值
        rsi6 = RSI(security, N1)
        # 获取num天rsi6值的拟合曲线的角度
        degree_rsi6 = get_degree(rsi6[security], num)
        # 股价创出新低，RSI却不断上行
        res = degree_close < -deg and degree_rsi6 > deg
        return res
    except Exception as e:
        log.error(traceback.format_exc())
        return False

###################################   形态指标筛选   ##################################
# N天阴线
def C_GREENN(security, N=4, include_now=True):
    try:
        security_data = get_bars(security, N, '1d', ['close','open'], include_now)
        close_data = security_data['close']
        open_data = security_data['open']
        Num = [close_data[i]<open_data[i] for i in range(len(close_data))].count(True)
        if Num == N:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# N天阳线
def C_REDN(security, N=4, include_now=True):
    try:
        security_data = get_bars(security, N, '1d', ['close','open'], include_now)
        close_data = security_data['close']
        open_data = security_data['open']
        Num = [close_data[i]>open_data[i] for i in range(len(close_data))].count(True)
        if Num == N:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 连跌N天
def C_DOWNN(security, N=4, include_now=True):
    try:
        security_data = get_bars(security, N+1, '1d', ['close'], include_now)['close']
        Num = [security_data[i]<security_data[i-1] for i in range(1,len(security_data))[::-1]].count(True)
        if Num == N:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 连涨N天
def C_UPN(security, N=4, include_now=True):
    try:
        security_data = get_bars(security, N+1, '1d', ['close'], include_now)['close']
        Num = [security_data[i]>security_data[i-1] for i in range(1,len(security_data))[::-1]].count(True)
        if Num == N:
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 剑
def C_SWORD(security, include_now=True):
    """
    计算公式：
        AA:=VOL>REF(VOL,1)||VOL>CAPITAL;
        BB:=OPEN>=(REF(HIGH,1))&&REF(HIGH,1)>(REF(HIGH,2)*1.06);
        CC:=CLOSE>(REF(CLOSE,1))-(REF(CLOSE,1)*0.01);
        DD:=CLOSE<(HIGH*0.965) && HIGH>(OPEN*1.05);
        EE:=LOW<OPEN && LOW<CLOSE &&  HIGH>(REF(CLOSE,1)*1.06);
        FF:=(HIGH-(MAX(OPEN,CLOSE)))/2>(MIN(OPEN,CLOSE))-LOW;
        GG:=(ABS(OPEN-CLOSE))/2<(MIN(OPEN,CLOSE)-LOW);
        STAR:AA&&BB&&CC&&DD&&EE&&FF&&GG;
        AA赋值:成交量(手)>1日前的成交量(手)或者成交量(手)>当前流通股本(手)
        BB赋值:开盘价>=(1日前的最高价)并且1日前的最高价>(2日前的最高价*1.06)
        CC赋值:收盘价>(1日前的收盘价)-(1日前的收盘价*0.01)
        DD赋值:收盘价<(最高价*0.965) 并且 最高价>(开盘价*1.05)
        EE赋值:最低价<开盘价 并且 最低价<收盘价 并且  最高价>(1日前的收盘价*1.06)
        FF赋值:(最高价-(开盘价和收盘价的较大值))/2>(开盘价和收盘价的较小值)-最低价
        GG赋值:(开盘价-收盘价的绝对值)/2<(开盘价和收盘价的较小值-最低价)
        输出STAR:AA并且BB并且CC并且DD并且EE并且FF并且GG
    输入：
        security: 单只股票代码
        include_now：
    输出：
        STAR 的值
    输出结果类型：
        布尔类型
    """
    import jqdata
    try:
        specificCount = 5
        # 获取specificCount天的收盘价，开盘价，交易量，最高价和最低价数据
        security_data = get_bars(security, specificCount, '1d', ['close','open', 'volume', 'high', 'low'], include_now)
        high_data = security_data['high']
        low_data = security_data['low']
        close_data = security_data['close']
        open_data = security_data['open']
        # 交易量以股为单位
        volume_data = security_data['volume']

        # 查询code的市值数据中的流通股本,单位是万股，时间是day
        q = query(valuation.circulating_cap).filter(valuation.code == security)
        # 获取最近specificCount个交易日的流通股本,变换单位，万股->股，使得流通股本的单位和交易量的单位统一起来，能够相互比较
        circulating_cap = [get_fundamentals(q)['circulating_cap'][0]* 10000]

        aa = (volume_data[-1] > volume_data[-2]) or (volume_data[-1] > circulating_cap[-1])
        bb = aa and (open_data[-1] >= high_data[-2]) and (high_data[-2] > high_data[-3] * 1.06)
        cc = bb and (close_data[-1] > close_data[-2] * (1.0 - 0.01))
        dd = cc and (close_data[-1] < high_data[-1] * 0.965) and (high_data[-1] > open_data[-1] * 1.05)
        ee = dd and (low_data[-1] < open_data[-1]) and (low_data[-1] < close_data[-1]) and (high_data[-1] > close_data[-2] * 1.06)
        ff = ee and (high_data[-1] - max(open_data[-1], close_data[-1])) / 2.0 > min(open_data[-1], close_data[-1]) - low_data[-1]
        gg = ff and abs(open_data[-1] - close_data[-1]) / 2.0 < min(open_data[-1], close_data[-1]) - low_data[-1]

        star = gg
        return star
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 跳空缺口
def C_TKQK(security, include_now=True):
    try:
        security_data = get_bars(security, 2, '1d', ['open','high','low'], include_now)
        if (security_data['open'][-1] > security_data['high'][-2]) \
            or (security_data['open'][-1] < security_data['low'][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 出水芙蓉
def C_CSFR(security,S=20, M=40, N=60, include_now=True):
    """
    计算公式：
        AAA:=CLOSE>OPEN;
        BBB:=AAA&&CLOSE>MA(CLOSE,S)&&CLOSE>MA(CLOSE,M)&&CLOSE>MA(CLOSE,N);
        CCC:=BBB&&OPEN<MA(CLOSE,M)&&OPEN<MA(CLOSE,N);
        CSFR:CCC&&(CLOSE-OPEN)>0.0618*CLOSE;
    输入：
        security: 单只股票代码
        S：统计的天数 S
        M：统计的天数 M
        N：统计的天数 N
        include_now：
    输出：
        CSFR 的值
    输出结果类型：
        布尔类型
    """
    try:
        count = max(S, N, M) + 2
        # 获取count天的收盘价和开盘价数据
        security_data = get_bars(security, count, '1d', ['close','open'], include_now)
        close_data = security_data['close']
        open_data = security_data['open']
        # 分别取收盘价S,M,N天的平均值
        ma_s = np.mean(close_data[-S:])
        ma_m = np.mean(close_data[-M:])
        ma_n = np.mean(close_data[-N:])
        # 分别判断AAA，BBB，CCC，CSFR
        aaa = close_data[-1] > open_data[-1]
        bbb = aaa and close_data[-1] > ma_s and close_data[-1] > ma_m and close_data[-1] > ma_n
        ccc = bbb and open_data[-1] < ma_m and open_data[-1] < ma_n
        csfr = ccc and (close_data[-1] - open_data[-1]) > 0.618 * close_data[-1]
        return csfr
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 身怀六甲
def C_SHLJ(security, include_now=True):
    try:
        security_data = get_bars(security, 2, '1d', ['open','close','high','low'], include_now)
        if (min(security_data['open'][-2], security_data['close'][-2]) < min(security_data['open'][-1], security_data['close'][-1])) \
            and (max(security_data['open'][-2], security_data['close'][-2]) > max(security_data['open'][-1], security_data['close'][-1])):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False 

# 穿头破脚
def C_CTPJ(security, include_now=True):
    try:
        security_data = get_bars(security, 2, '1d', ['high','low'], include_now)
        if (security_data['high'][-1] > security_data['high'][-2]) \
            and (security_data['low'][-1] < security_data['low'][-2]):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 射击之星
def C_SJZX(security, include_now=True):
    try:
        security_data = get_bars(security, 5, '1d', ['open','close','high','low'], include_now)
        syx = security_data['high'][-1] - max(security_data['open'][-1], security_data['close'][-1])
        xyx = min(security_data['open'][-1], security_data['close'][-1]) - security_data['low'][-1]
        st = abs(security_data['open'][-1] - security_data['close'][-1])
        ma5 = security_data['close'].mean()
        
        if (xyx<st*2) and (syx/st > 2.0) and (security_data['close'][-1] > ma5):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 吊颈
def C_DJ(security, include_now=True):
    try:
        security_data = get_bars(security, 5, '1d', ['open','close','high','low'], include_now)
        syx = security_data['high'][-1] - max(security_data['open'][-1], security_data['close'][-1])
        xyx = min(security_data['open'][-1], security_data['close'][-1]) - security_data['low'][-1]
        st = abs(security_data['open'][-1] - security_data['close'][-1])
        ma5 = security_data['close'].mean()
        
        if (syx<st*2) and (xyx/st > 2.0) and (security_data['close'][-1] > ma5):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 红三兵
def C_HSB(security, include_now=True):
    # 计算 红三兵
    try:
        security_data = get_bars(security, 3, '1d', ['open','close','high','low'], include_now)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        if (close[-1]>close[-2]>close[-3]) and (open[-1]<close[-1]) and (open[-2]<close[-2]) and (open[-3]<close[-3]): 
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 三只乌鸦
def C_SZWY(security, include_now=True):
    # 计算 三只乌鸦
    try:
        security_data = get_bars(security, 3, '1d', ['open','close','high','low'], include_now)
        open = security_data['open']
        high = security_data['high']
        low = security_data['low']
        close = security_data['close']
        if (close[-1]<close[-2]<close[-3]) and (open[-1]>close[-1]) and (open[-2]>close[-2]) and (open[-3]>close[-3]): 
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 锤
def C_HAMMER(security, include_now=True):
    # 计算 锤
    try:
        security_data = get_bars(security, 5, '1d', ['open','close','high','low'], include_now)
        syx = security_data['high'][-1] - max(security_data['open'][-1], security_data['close'][-1])
        xyx = min(security_data['open'][-1], security_data['close'][-1]) - security_data['low'][-1]
        st = abs(security_data['open'][-1] - security_data['close'][-1])
        ma5 = security_data['close'].mean()
        
        if (syx/(syx+st+xyx)<=0.02) and (xyx/st > 2.0) and (security_data['close'][-1] < ma5):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 倒锤
def C_HAMMER_R(security, include_now=True):
    # 计算 倒锤
    try:
        security_data = get_bars(security, 5, '1d', ['open','close','high','low'], include_now)
        syx = security_data['high'][-1] - max(security_data['open'][-1], security_data['close'][-1])
        xyx = min(security_data['open'][-1], security_data['close'][-1]) - security_data['low'][-1]
        st = abs(security_data['open'][-1] - security_data['close'][-1])
        ma5 = security_data['close'].mean()
        
        if (xyx/(syx+st+xyx)<=0.02) and (syx/st > 2.0) and (security_data['close'][-1] < ma5):
        # if (security_data['low'][-1]==min(security_data['open'][-1], security_data['close'][-1])) and (syx/st > 2.0) and (security_data['close'][-1] < ma5):
            return True
        return False
    except Exception as e:
        log.error(traceback.format_exc())
        return False

# 黄昏之星
def C_ESTAR(security, include_now=True):
    import talib
    # 计算 黄昏之星
    try:
        security_data = get_bars(security, 40, '1d', ['close', 'volume', 'open', 'high', 'low'], include_now)
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
def C_MSTAR(security, include_now=True):
    import talib
    # 计算 早晨之星
    try:
        security_data = get_bars(security, 40, '1d', ['close', 'volume', 'open', 'high', 'low'], include_now)
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

