# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import numpy as np
from .base import *


__all__ = ['get_mtss', 'get_all_trade_days', 'get_trade_days', 'get_money_flow']


def get_mtss(security_list, start_date=None, end_date=None, fields=None, count=None):
    """
    获取融资融券信息
    security_list: 股票代码或者 list
    start_date: 开始日期, **与 count 二选一, 不可同时使用**. str/datetime.date/datetime.datetime 对象, 默认为平台提供的数据的最早日期
    end_date: 结束日期, str/datetime.date/datetime.datetime 对象, 默认为 datetime.date.today()
    fields: 字段名或者 list, 可选, 默认全部字段
    count: 数量, **与 start_date 二选一，不可同时使用**. 表示返回 end_date 之前 count 个交易日的数据, 包含 end_date

    返回pd.DataFrame, columns:
    日期,股票代码, 融资余额,融资买入额,融资偿还额,融券余额，融资卖出额，融资偿还额，融资融券余额
    date, sec_code, fin_value, fin_buy_value, fin_refund_value, sec_value, sec_sell_value, sec_refund_value, fin_sec_value
    """
    if start_date and count:
        raise Exception("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise Exception("count 参数需要大于 0 或者为 None")

    security_list = obj_to_tuple(security_list)
    check_string_list(security_list)

    end_date = convert_date(end_date) if end_date else datetime.date.today()
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(end_date=end_date, count=count)[0] if count else TRADE_MIN_DATE)

    keys = ["sec_code", "date", "fin_value", "fin_buy_value", "fin_refund_value", \
        "sec_value", "sec_sell_value", "sec_refund_value", "fin_sec_value"]
    nkeys = len(keys)
    if fields:
        fields = obj_to_tuple(fields)
        check_string_list(fields)
        check_fields(keys, fields)
    else:
        fields = ["date", "sec_code"] + keys[2:]

    request_path = "/stock/mtss/query"
    request_params = {
        "code": "",
        "startDate": start_date,
        "endDate": end_date,
    }

    lists = []
    convert_funcs = [str, convert_dt, float, float, float, float, float, float, float]
    for security in security_list:
        request_params["code"] = security
        data = request_data(DATA_SERVER+request_path, request_params)
        for d in data:
            values = [item.strip() for item in d.split(",", nkeys)[:nkeys]]
            values = [convert_funcs[i](v) for i, v in enumerate(values)]
            sec_dict = dict(zip(keys, values))
            sec_dict["sec_code"] = normalize_stock_code(sec_dict["sec_code"])
            lists.append(filter_dict_values(sec_dict, fields))

    import pandas as pd
    df = pd.DataFrame(columns=fields, data=lists)
    return df

def test_get_lhb(start_date, end_date, fields=None):
    """
    获取龙虎榜, 返回pd.DataFrame, columns:
    日期, 股票代码,当日涨幅,龙虎榜成交额,龙虎榜买入，买入占总成交比例，龙虎榜卖出,卖出占总成交比例，上榜原因代码
    date, sec_code, change_pct, turnover_value, buy_value, buy_pct, sell_value, sell_pct, onboard_reason
    """

    keys = ["date", "sec_code", "change_pct", "turnover_value", "buy_value", "buy_pct", \
        "sell_value", "sell_pct", "onboard_reason"]
    nkeys = len(keys)
    if fields:
        fields = obj_to_tuple(fields)
        check_string_list(fields)
        check_fields(keys, fields)
    else:
        fields = keys

    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    dates = date_range(start_date, end_date)

    lists = []
    request_path = "/stock/lhb/get"
    convert_funcs = [convert_dt, str, float, float, float, float, float, float, str]
    for date in dates:
        request_params = {"date": date}
        data = request_data(DATA_SERVER+request_path, request_params)
        for d in data:
            values = [item.strip() for item in d.split(",", nkeys - 1)[:nkeys - 1]]
            values = [convert_funcs[i](v) for i, v in enumerate([date] + values)]
            lhb_dict = dict(zip(keys, values))
            lists.append(filter_dict_values(lhb_dict, fields))

    import pandas as pd
    df = pd.DataFrame(columns=fields, data=lists)
    return df

def get_all_trade_days():
    """
    获取所有交易日，返回 numpy.ndarray.
    """
    return get_trade_days_from_json()

def get_trade_days(start_date=None, end_date=None, count=None):
    """获取指定日期范围内的所有交易日

    参数 start_date: 开始日期, **与 count 二选一, 不可同时使用**. str/datetime.date/datetime.datetime 对象
    参数 end_date: 结束日期, str/datetime.date/datetime.datetime 对象
    参数 count: 数量, **与 count 二选一, 不可同时使用**, 必须大于 0. 表示取 end_date 往前的 count 个交易日

    返回 numpy.ndarray, 包含指定的 start_date 和 end_date, 默认返回所有交易日
    """
    if start_date and count:
        raise Exception("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise Exception("count 参数需要大于 0 或者为 None")

    days = get_trade_days_from_json()
    start = 0
    if start_date:
        start_date = convert_date(start_date)
        if start_date < days[0]:
            raise Exception("日期不在规定的范围内")
        start = np.searchsorted(days, start_date)
    end_date = convert_date(end_date) if end_date else datetime.date.today()
    if end_date > days[-1]:
        raise Exception("日期不在规定的范围内")
    end = np.searchsorted(days, end_date, side="right")

    return days[end-count:end] if count else days[start:end]

def get_money_flow(security_list, start_date=None, end_date=None, fields=None, count=None):
    """
    获取资金流向数据
    security_list: 股票代码或者 list
    start_date: 开始日期, **与 count 二选一，不可同时使用**. str/datetime.date/datetime.datetime 对象, 默认为平台提供的数据的最早日期
    end_date: 结束日期, str/datetime.date/datetime.datetime 对象, 默认为 datetime.date.today()
    fields: 字段名或者 list, 可选, 默认全部字段
    count: 数量, **与 start_date 二选一，不可同时使用**. 表示返回 end_date 之前 count 个交易日的数据, 包含 end_date

    返回pd.DataFrame, columns:
    日期, 股票代码, 涨跌幅(%), 主力净额(万), 主力净占比(%), 超大单净额(万), 超大单净占比（%）, 大单净额(万), 大单净占比(%), 中单净额(万), 中单净占比(%), 小单净额(万), 小单净占比（%）
    date, sec_code, change_pct, net_amount_main, net_pct_main, net_amount_xl, net_pct_xl, net_amount_l, net_pct_l, net_amount_m, net_pct_m, net_amount_s, net_pct_s
    """
    if start_date and count:
        raise Exception("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise Exception("count 参数需要大于 0 或者为 None")

    security_list = obj_to_tuple(security_list)
    check_string_list(security_list)

    end_date = convert_date(end_date) if end_date else datetime.date.today()
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(end_date=end_date, count=count)[0] if count else TRADE_MIN_DATE)

    keys = ["date", "sec_code", "change_pct", "net_amount_main", "net_pct_main", "net_amount_xl", \
        "net_pct_xl", "net_amount_l", "net_pct_l", "net_amount_m", "net_pct_m", "net_amount_s", "net_pct_s"]
    nkeys = len(keys)
    if fields:
        fields = obj_to_tuple(fields)
        check_string_list(fields)
        check_fields(keys, fields)
    else:
        fields = keys

    request_path = "/stock/fundflow/query"
    request_params = {
        "code": "",
        "startDate": start_date,
        "endDate": end_date,
    }

    lists = []
    convert_funcs = [convert_dt, str] + [float_or_nan] * 11
    for security in security_list:
        request_params["code"] = security
        data = request_data(DATA_SERVER+request_path, request_params)
        for d in data:
            values = [item.strip() for item in d.split(",", nkeys - 1)[:nkeys - 1]]
            values.insert(1, security)
            values = [convert_funcs[i](v) for i, v in enumerate(values)]
            sec_dict = dict(zip(keys, values))
            lists.append(filter_dict_values(sec_dict, fields))

    import pandas as pd
    df = pd.DataFrame(columns=fields, data=lists)
    return df
