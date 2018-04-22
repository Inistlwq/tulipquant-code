# -*- coding: utf-8 -*-

# *************************************************************
#  Copyright (c) JoinQuant Development Team
#
#  Author: Huayong Kuang <kuanghuayong@joinquant.com>
#  CreateTime: 2017-08-01 16:53:10 Tuesday
# *************************************************************

import os
import sys
import json
import datetime
import contextlib

import six
import requests


def convert_date(date):
    if isinstance(date, (str, six.text_type)):
        if ':' in date:
            date = date[:10]
        try:
            return datetime.date(*map(int, date.split('-')))
        except Exception:
            raise
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    raise ValueError("日期必须是一个 datetime.date, datetime.datetime "
                     "或者类似 '2015-01-05' 的字符串")


def get_full_path(path):
    return os.path.join(os.environ['HOME'], path)


def json_serial_fallback(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return str(obj)
    raise TypeError("%s not serializable" % obj)


def request_data(path, data={}):
    url = os.environ.get('RESEARCH_API_SERVER', 'http://research-service:5000') + path
    data = json.dumps(data, default=json_serial_fallback)
    with contextlib.closing(requests.post(url, data=data)) as resp:
        body = resp.content.decode('utf-8-sig')
        if resp.status_code != requests.codes.ok:
            raise Exception(body)
        return body


def request_json(path, data={}):
    return json.loads(request_data(path, data))


def get_logger():
    import logging
    format = '%(asctime)-15s %(levelname)s %(message)s'
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(format))
    log = logging.getLogger('jqresearch')
    log.warn = log.warning
    log.addHandler(stream_handler)
    log.setLevel(logging.DEBUG)
    return log


class Backtest(object):
    """回测详情"""

    def __init__(self, backtest_id):
        self.backtest_id = backtest_id

    def __request_json(self, info_type):
        params = {
            "id": self.backtest_id,
            "info_type": info_type,
            'user': os.environ.get('JQ_USER'),
            'token': os.environ.get('RESEARCH_API_TOKEN'),
        }

        return request_json("/get_backtest", params)

    def get_params(self):
        """得到回测参数

        返回一个 dict, 包含调用 create_backtest 时传入的所有信息

        注： algorithm_id，initial_positions，extras 只有在研究中创建的回测才能取到
        """
        return self.__request_json("params")

    def get_status(self):
        """获取回测状态

        返回一个字符串，其含义分别为：

            none      未开始
            running   正在进行
            done      完成
            failed    失败
            canceled  取消
            paused    暂停
            deleted   已删除
        """
        return self.__request_json("status")

    def get_results(self):
        """收益曲线

        返回一个 list，每个元素是一个 dict，键的含义为：

            time: 时间
            returns: 收益
            benchmark_returns: 基准收益

        如果没有收益则返回一个空的 list.
        """
        return self.__request_json("results")

    def get_positions(self):
        """所有持仓列表

        返回一个 list，每个元素为一个 dict，键的含义为：

            time: 时间
            security: 证券代码
            security_name: 证券名称
            amount: 持仓数量
            price: 股票价格
            avg_cost: 买入股票平均每股所花的钱
            closeable_amount: 可平仓数量

        如果没有持仓则返回一个空的 list.
        """
        return self.__request_json("positions")

    def get_orders(self):
        """交易列表

        返回一个 list，每个元素为一个 dict，键的含义为：

            time: 时间
            security: 证券代码
            security_name: 证券名称
            action: 交易类型, 开仓('open')/平仓('close')
            amount: 下单数量
            filled: 成交数量
            price: 平均成交价格
            commission: 交易佣金

        如果没有交易则返回一个空的 list.
        """
        return self.__request_json("orders")

    def get_records(self):
        """所有 record 记录

        返回一个 list，每个元素为一个 dict，键是 time 以及调用 record() 函数时设置的值
        """
        return self.__request_json("records")

    def get_risk(self):
        """总的风险指标

        返回一个 dict，键是各类收益指标数据，如果没有风险指标则返回一个空的 dict.
        """
        return self.__request_json("risk")

    def get_period_risks(self):
        """分月计算的风险指标

        返回一个 dict，键是各类指标, 值为一个 pandas.DataFrame.

        如果没有风险指标则返回一个空的 dict.
        """
        period_risks = self.__request_json("period_risks")
        if not period_risks:
            return {}

        months = [date[:7] for date in period_risks["months"]]
        targets = ["algorithm_return", "benchmark_return", "alpha", "beta", "sharpe", "sortino",
                   "information", "algorithm_volatility", "benchmark_volatility", "max_drawdown"]

        import pandas as pd

        def period_risks_to_df(period):
            nan = float('nan')
            risks = [[risks.get(k, nan) for k in targets] if risks else [nan] * len(targets)
                     for risks in period_risks[period]]
            return pd.DataFrame(risks, index=months, columns=targets)

        periods = ["one_month", "three_month", "six_month", "twelve_month"]
        period_risks = [period_risks_to_df(period) for period in periods]

        return {k: pd.DataFrame([risks[k] for risks in period_risks], index=periods).T for k in targets}

    def __repr__(self):
        return 'Backtest(id="%s")' % self.backtest_id

    def __str__(self):
        return self.__repr__()
