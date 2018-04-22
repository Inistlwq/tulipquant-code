# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import sys
import json
import logging
import inspect
import argparse
import datetime
import traceback
from pprint import pprint

import six

from jqcommon.service import request_web
from jqdata.apis.security import get_index_stocks

from jqfactor import Factor
from jqfactor.logger import user_log, user_err, setup as logger_setup
from jqfactor.store import get_store
from jqfactor.when import parse_date
from jqfactor.analyze import FactorAnalyzer
from jqfactor.calculate import calc_multiple_factors
from jqfactor.loader import CodeLoader, BinaryLoader, LocalLoader


if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')

log = logging.getLogger("jqfactor.entry")


def notify(task_id, status):
    """通知 web 因子运行状态

    状态说明:
        0: 未开始
        1: 正在运行
        2: 运行完成
        3: 运行失败

    成功请求 web 返回 True，否则返回 False
    """
    url = '/algorithm/factorAnaly/notify'
    data = dict(id=task_id, status=status)
    try:
        request_web('post', url, data=data)
        return True
    except Exception as e:
        log.exception("request web failed: {}".format(e))
        return False


def fetch_factors_from_code(user_code_path, scope):
    if os.path.isdir(user_code_path):
        loader = LocalLoader(user_code_path)
    else:
        suf = os.path.splitext(user_code_path)[1]
        source = open(user_code_path).read()
        if suf == '.py':
            loader = CodeLoader(user_code_path, source)
        elif suf == '.so':
            loader = BinaryLoader(user_code_path, source)
        else:
            raise Exception('Code file should be .py or .so file: {!r}'.format(user_code_path))

    mod = loader.load()
    fact_list = []
    for c in mod.__dict__:
        if c != 'Factor':
            fact = getattr(mod, c)
            if inspect.isclass(fact) and issubclass(fact, Factor):
                fact_list.append(fact())

    scope[mod.__name__] = mod
    return fact_list


def _run(args):
    # 配置 web server
    web_server = os.getenv('WEB_SERVER')
    if web_server:
        import jqcommon.config
        jqcommon.config.WEB_SERVERS = [web_server]

    # 通知 web 正在运行
    if not args.debug:
        notify(args.task_id, 1)

    # 解析参数
    user_log.info('参数初始化(1/4)')
    start_date = parse_date(args.start_time)
    end_date = parse_date(args.end_time)
    user_code_path = args.path

    if args.stock == 'hs300':
        stocks = get_index_stocks('000300.XSHG', datetime.date.today())
    elif args.stock == 'zz500':
        stocks = get_index_stocks('000905.XSHG', datetime.date.today())
    else:
        raise Exception("no support stock pool: '%s'" % args.stock)

    if args.industry == 'joinquant':
        industry_type = 'jq_l1'
    elif args.industry == 'shenwan':
        industry_type = 'sw_l1'
    else:
        raise Exception("no support industry type: '%s'" % args.industry)

    # 单因子计算
    user_log.info('因子数据计算(2/4)')
    factors = fetch_factors_from_code(user_code_path, locals())
    factors_len = len(factors)
    if factors_len == 0:
        raise Exception("not defined Factor subclass")
    elif factors_len > 1:
        raise Exception('only support single factor')
    fact_dict = calc_multiple_factors(stocks, factors, start_date, end_date,
                                      init_markets=["close"],
                                      redirect_calc_output=True)
    factor = fact_dict[factors[0].name]

    # 单因子分析数据准备
    user_log.info('因子分析初始化(3/4)')
    fa = FactorAnalyzer(start_date=start_date,
                        end_date=end_date,
                        stocks=stocks,
                        industry_type=industry_type,
                        factor=factor.stack(),
                        pricing_data=fact_dict['close'])

    # 开始单因子分析
    user_log.info('因子分析(4/4)')
    result = fa.analyze()

    # 写入 SSDB
    if args.debug:
        pprint(result)
    else:
        args.ssdb_store.set("analysis-result", json.dumps(result))

    user_log.info('完成')
    if not args.debug:
        notify(args.task_id, 2)


def main(args=None):
    args = args or sys.argv[1:]
    strategy_path = os.getenv('jqfactor_strategy')
    if strategy_path is not None:
        args.append(os.getenv('jqfactor_strategy'))

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='strategy code path')
    parser.add_argument('-ti', '--task_id', type=str, required=True,
                        help='task id')
    parser.add_argument('-s', '--stock', type=str,
                        help='stock')
    parser.add_argument('-i', '--industry', type=str,
                        help='industry')
    parser.add_argument('-st', '--start_time', type=str,
                        help='start time')
    parser.add_argument('-et', '--end_time', type=str,
                        help='end time')
    parser.add_argument('-d', '--debug', action="store_true",
                        help='enable debug mode')
    args = parser.parse_args(args)

    ssdb_store = get_store(dict(task_id=args.task_id))
    logger_setup(ssdb_store)
    args.ssdb_store = ssdb_store

    try:
        return _run(args)
    except Exception as e:
        user_err.error(traceback.format_exc())
        user_log.info(traceback.format_exc())
        if not args.debug:
            notify(args.task_id, 3)
        log.exception(e)
