# -*- coding: UTF-8 -*-

import datetime
from pprint import pprint

import pytest
import numpy as np
import pandas as pd

from jqdata.utils import convert_date, TRADE_MIN_DATE
from jqdata.api import *


def test_get_mtss():
    df = get_mtss('000001.XSHE', '2016-01-01', '2016-04-01')
    assert isinstance(df, pd.DataFrame)

    fields = ["date", "sec_code", "fin_value", "fin_buy_value"]
    df = get_mtss('000001.XSHE', '2016-01-01', '2016-04-01', fields=fields)
    assert df.columns.tolist() == fields

    assert len(get_mtss('000001.XSHE', '2016-01-01', '2016-04-01', fields="sec_sell_value"))
    assert len(get_mtss('000001.XSHE', convert_date('2016-01-01'), '2016-04-01', fields="sec_sell_value"))
    assert len(get_mtss('000001.XSHE', '2016-01-01', convert_date('2016-04-01'), fields="sec_sell_value"))

    assert len(get_mtss(['000001.XSHE', '000002.XSHE', '000099.XSHE'], '2015-03-25', '2016-01-25'))
    assert len(get_mtss(['000001.XSHE', '000002.XSHE', '000099.XSHE'], '2015-03-25', '2016-01-25', fields=["date", "sec_code", "sec_value", "fin_buy_value", "sec_sell_value"]))

    assert len(get_mtss('000001.XSHE', start_date="2016-01-01")) > 0
    assert len(get_mtss('000001.XSHE', end_date="2013-10-01")) == 849
    assert len(get_mtss('000001.XSHE', count=20)) in (19, 20)
    assert len(get_mtss('000001.XSHE', end_date="2013-10-01", count=20)) == 20

    df1 = get_mtss(['000006.XSHE', '000007.XSHE', '000008.XSHE', '000009.XSHE', '000010.XSHE'])
    df2 = get_mtss(['000006.XSHE', '000007.XSHE', '000008.XSHE', '000009.XSHE', '000010.XSHE'], count=10)
    assert df1.tail(5)['sec_code'].tolist() == df2[-5:]['sec_code'].tolist()
    assert {convert_date(date) for date in df2['date'].tolist()[:9]} == set(get_trade_days(count=10)[:9])

    with pytest.raises(Exception, message="start_date 参数与 count 参数只能二选一"):
        get_mtss('000001.XSHE', start_date='2016-01-01', count=10)

    with pytest.raises(Exception, message="count 参数需要大于 0 或者为 None"):
        get_mtss('000001.XSHE', count=0)

    df = get_mtss(['000001.XSHE', '000002.XSHE'], convert_date('2016-02-01'), convert_date('2016-02-02'), fields="sec_sell_value")
    assert len(df.index) == 2 * 2
    assert len(df.columns) == 1
    df["sec_sell_value"]

    df = get_mtss("000001.XSHE", "2016-02-01", "2016-02-01")
    assert len(df.index) == 1
    assert len(df.columns) == 9
    assert df.shape[0] == 1
    column_types = []
    check_types = [pd.tslib.Timestamp, str, float, float, float, float, float, float, float]
    for index, rows in df[0:1].iterrows():
        for col_name in df.columns:
            column_types.append(type(rows[col_name]))
    check_type_ret = [column_type == check_type for column_type, check_type in zip(column_types, check_types)]
    assert all(check_type_ret)

"""
def test_get_lhb():
    assert len(get_lhb('2016-02-01', '2016-02-04'))
    assert len(get_lhb(convert_date('2016-02-01'), '2016-02-04'))
    assert len(get_lhb('2016-02-01', convert_date('2016-02-04')))
    assert len(get_lhb('2016-02-01', '2016-02-01', fields=["date", "sec_code", "change_pct", "turnover_value", "buy_value", "buy_pct", "sell_value", "sell_pct", "onboard_reason"]))
    assert len(get_lhb('2016-02-01', '2016-02-01', fields=["date", "sec_code", "change_pct", "turnover_value"]))
    assert len(get_lhb('2016-02-01', '2016-02-01', fields=["buy_value", "buy_pct", "sell_value", "sell_pct", "onboard_reason"]))
    assert len(get_lhb('2016-02-01', '2016-02-01', fields=["buy_value", "sec_code", "date"]))
    assert len(get_lhb('2016-02-01', '2016-02-01', fields=["date"]))

    df = get_lhb(convert_date('2016-02-01'), convert_date('2016-02-01'))
    assert len(df.columns) == 9
    assert len(df.columns) > 0
    column_types = []
    check_types = [pd.tslib.Timestamp, str, float, float, float, float, float, float, str]
    for index, rows in df[0:1].iterrows():
        for col_name in df.columns:
            column_types.append(type(rows[col_name]))
    check_type_ret = [column_type == check_type for column_type, check_type in zip(column_types, check_types)]
    assert all(check_type_ret)
"""

def test_get_all_trade_days():
    days = get_all_trade_days()
    assert type(days) == np.ndarray
    assert len(days) == 2915
    pprint(days)

def test_get_money_flow():
    df = get_money_flow('000001.XSHE', '2016-02-01', '2016-02-04')
    assert isinstance(df, pd.DataFrame)

    fields = ["date", "sec_code", "change_pct", "net_amount_main", "net_pct_l", "net_amount_m"]
    df = get_money_flow(['000001.XSHE'], '2010-01-01', '2010-01-30', fields=fields)
    assert df.columns.tolist() == fields

    df = get_money_flow(['000001.XSHE', '000040.XSHE', '000099.XSHE'], '2016-04-01', '2016-04-01')
    assert len(df.index) == 3
    assert len(df.columns) == 13

    df = get_money_flow('000001.XSHE', convert_date('2016-01-01'), convert_date('2016-06-30'), fields="date")
    assert len(df.columns) == 1
    assert len(df) > 0

    assert len(get_money_flow('000001.XSHE', start_date="2016-01-01")) > 0
    assert len(get_money_flow('000001.XSHE', end_date="2013-10-01")) == 837
    assert len(get_money_flow('000001.XSHE', count=20)) in (19, 20)
    assert len(get_money_flow('000001.XSHE', end_date="2013-10-01", count=20)) in (19, 20)

    df1 = get_money_flow(['000006.XSHE', '000007.XSHE', '000008.XSHE', '000009.XSHE', '000010.XSHE'])
    df2 = get_money_flow(['000006.XSHE', '000007.XSHE', '000008.XSHE', '000009.XSHE', '000010.XSHE'], count=10)
    assert df1.tail(5)['sec_code'].tolist() == df2[-5:]['sec_code'].tolist()
    assert {convert_date(date) for date in df2['date'].tolist()[:9]} == set(get_trade_days(count=10)[:9])

    with pytest.raises(Exception, message="start_date 参数与 count 参数只能二选一"):
        get_money_flow('000001.XSHE', start_date='2016-01-01', count=10)

    with pytest.raises(Exception, message="count 参数需要大于 0 或者为 None"):
        get_money_flow('000001.XSHE', count=0)

    df = get_money_flow("600000.XSHG", "2016-02-01", "2016-02-01")
    assert df.shape[0] == 1
    assert df.shape[1] == 13
    column_types = []
    check_types = [pd.tslib.Timestamp, str, float, float, float, float, float, float, float, \
        float, float, float, float]
    for index, rows in df[0:1].iterrows():
        for col_name in df.columns:
            column_types.append(type(rows[col_name]))
    check_type_ret = [column_type == check_type for column_type, check_type in zip(column_types, check_types)]
    assert all(check_type_ret)

def test_get_money_flow2():
    startdate = datetime.date(2010, 2, 1)
    enddate = datetime.date(2016, 2, 4)
    all_money_flow = get_money_flow('000831.XSHE', start_date=startdate, \
        end_date=enddate, fields=["net_amount_main", "net_amount_xl", \
        "net_amount_l"])
    pprint(all_money_flow)

def test_get_trade_days():
    assert get_trade_days()[0] == TRADE_MIN_DATE
    from jqdata.base import is_trade_day
    if is_trade_day(datetime.date.today()):
        assert get_trade_days()[-1] == datetime.date.today()
        assert get_trade_days(start_date="2016-06-01")[-1] == datetime.date.today()
        assert get_trade_days(end_date=datetime.date.today())[-1] == datetime.date.today()
        assert get_trade_days(count=1)[0] == datetime.date.today()
        assert get_trade_days(count=5)[-1] == datetime.date.today()

    days = get_trade_days(count=8)
    assert len(days) == 8
    from jqdata.base import get_prev_trade_day
    assert days[-1] in (datetime.date.today(), get_prev_trade_day(datetime.date.today()))

    days = get_trade_days(end_date="2016-01-04", count=6)
    assert days[0] == days[-6]
    assert days[-1] == datetime.date(2016, 1, 4)

    days = get_trade_days(end_date="2016-06-26")
    assert days[0] == TRADE_MIN_DATE
    assert days[-1] == convert_date("2016-06-24")

    days = get_trade_days(start_date="2016-01-01")
    assert days[0] == convert_date("2016-01-04")

    days = get_trade_days(start_date="2016-01-01", end_date="2016-05-02")
    assert days[0] == convert_date("2016-01-04")
    assert days[-1] == convert_date("2016-04-29")
