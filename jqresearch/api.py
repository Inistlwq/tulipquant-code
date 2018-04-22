# coding: utf-8

from __future__ import absolute_import, print_function, unicode_literals

import os
import datetime
import six
import sys

import jqdata
import jqdata.apis
import jqdata.stores
from jqdata.utils.datetime_utils import parse_date
from jqdata.stores.calendar_store import get_calendar_store
from jqdata.stores.security_store import SecurityStore, get_security_store
import jqfactor
# from jqdata.exceptions import ParamsError

# gta.run_query
# jqdata.get_all_trade_days
# jqdata.get_trade_days
# jqdata.get_money_flow
# jqdata.get_mtss

from jqdata.apis import query, valuation, income, balance, cash_flow, indicator, fundamentals, \
    bank_indicator, security_indicator, insurance_indicator

from .base import (
    get_full_path,
    convert_date,
    request_json,
    get_logger,
    Backtest,
)


def _setup_for_jq_cloud():
    # fix PATH, then 'pip list' can works
    os.environ['PATH'] = os.path.dirname(sys.executable) + ':' + os.environ['PATH']

    jqdata.unset_cache()
    jqdata.set_shm_path('/dev/shm')
    # 云端部署时才调用
    if six.PY2:
        reload(sys)
        sys.setdefaultencoding('utf8')

    if '' in sys.path:
        sys.path.pop(sys.path.index(''))

    try:
        # matplotlib 支持中文
        # http://segmentfault.com/a/1190000000621721
        import matplotlib
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Droid Sans Fallback', 'serif'],
        })
    except ImportError:
        pass
    pass


def _get_today():
    return datetime.date.today()


def _get_now_minute():
    return datetime.datetime.now().replace(second=0, microsecond=0)


def get_price(security, start_date=None, end_date=None,
              frequency='daily',
              fields=None,
              skip_paused=False,
              fq='pre', count=None):
    return jqdata.apis.get_price(security, start_date=start_date,
                                 end_date=end_date,
                                 frequency=frequency,
                                 fields=fields,
                                 skip_paused=skip_paused,
                                 fq=fq,
                                 count=count,
                                 pre_factor_ref_date=_get_today())


def history(count, unit='1d', field='avg',
            security_list=None,
            df=True, skip_paused=False, fq='pre'):
    end_dt = _get_now_minute()
    return jqdata.apis.history(end_dt, count,
                               unit=unit,
                               field=field,
                               security_list=security_list,
                               df=df,
                               skip_paused=skip_paused,
                               fq=fq,
                               pre_factor_ref_date=_get_today())


def attribute_history(security, count, unit='1d',
                      fields=['open', 'close', 'high', 'low', 'volume', 'money'],
                      skip_paused=True, df=True, fq='pre'):
    end_dt = _get_now_minute()
    return jqdata.apis.attribute_history(end_dt, security, count,
                                         unit=unit,
                                         fields=fields,
                                         skip_paused=skip_paused,
                                         df=df,
                                         fq=fq,
                                         pre_factor_ref_date=_get_today())


def get_extras(info, security_list, start_date=None, end_date='2015-12-31', df=True, count=None):
    if start_date is None and count is None:
        start_date = '2015-01-01'
    return jqdata.apis.get_extras(info, security_list,
                                  start_date=start_date,
                                  end_date=end_date,
                                  df=df,
                                  count=count)


def get_fundamentals(query_object, date=None, statDate=None):  # noqa
    if date and statDate:
        raise Exception('date和statDate参数只能输入一个')
    if date is None and statDate is None:
        cal_store = get_calendar_store()
        stock = SecurityStore.instance().get_security('000001.XSHE')
        date = cal_store.get_previous_trade_date(stock, _get_today())
    if date:
        yestoday = datetime.date.today() - datetime.timedelta(days=1)
        date = min(parse_date(date), yestoday)
    return jqdata.apis.get_fundamentals(query_object, date=date, statDate=statDate)


def get_fundamentals_continuously(query_object, end_date=None, count=1): # noqa
    if end_date is None:
        cal_store = get_calendar_store()
        stock = SecurityStore.instance().get_security('000001.XSHE')
        end_date = cal_store.get_previous_trade_date(stock, _get_today())
    return jqdata.apis.get_fundamentals_continuously(query_object, end_date=end_date, count=count)


def get_index_stocks(index_symbol, date=None):
    date = date or _get_today()
    return jqdata.apis.get_index_stocks(index_symbol, date)


def get_industry_stocks(industry_code, date=None):
    date = date or _get_today()
    return jqdata.apis.get_industry_stocks(industry_code, date)


def get_concept_stocks(concept_code, date=None):
    date = date or _get_today()
    return jqdata.apis.get_concept_stocks(concept_code, date)


def get_all_securities(types=[], date=None):
    date = date or _get_today()
    return jqdata.apis.get_all_securities(types, date)


def get_security_info(code):
    return jqdata.apis.get_security_info(code)


def get_fund_info(security, date=None):
    return jqdata.apis.get_fund_info(security, date)
    pass


def get_margincash_stocks(date=None):
    if not date:
        calendar_store = get_calendar_store()
        date = calendar_store.get_previous_trade_date(None, _get_today())
    else:
        date = convert_date(date)
    return jqdata.apis.get_margincash_stocks(date)


def get_marginsec_stocks(date=None):
    if not date:
        calendar_store = get_calendar_store()
        date = calendar_store.get_previous_trade_date(None, _get_today())
    else:
        date = convert_date(date)
    return jqdata.apis.get_marginsec_stocks(date)


def normalize_code(code):
    return jqdata.apis.normalize_code(code)


def get_dominant_future(underlying_symbol, dt=None):
    if dt and convert_date(dt) != _get_today():
        dt = dt
    else:
        calendar_store = get_calendar_store()
        dt = calendar_store.get_previous_trade_date(get_security_store().get_security('AU9999.XSGE'),
                                                    _get_today())
    return jqdata.apis.get_dominant_future(dt, underlying_symbol)
    pass


def get_future_contracts(underlying_symbol, dt=None):
    dt = dt or _get_today()
    return jqdata.apis.get_future_contracts(dt, underlying_symbol)
    pass


def get_billboard_list(stock_list=None, start_date=None, end_date=None, count=None):
    return jqdata.apis.get_billboard_list(stock_list, start_date, end_date, count)


def get_locked_shares(stock_list, start_date=None, end_date=None, forward_count=None):
    return jqdata.apis.get_locked_shares(stock_list, start_date, end_date, forward_count)


def calc_factors(securities, factors, start_date, end_date):
    return jqfactor.calc_factors(securities, factors, start_date, end_date)
    pass


def create_backtest(algorithm_id, start_date, end_date, frequency="day",
                    initial_cash=10000, initial_positions=None, extras=None,
                    name=None, package_version=1.0):
    # u"""创建回测

    # 参数：
    #     algorithm_id:      策略ID
    #     start_date：       回测开始日期
    #     end_date：         回测结束日期
    #     frequency:         数据频率，支持 day，minute
    #     initial_cash:      初始资金
    #     initial_positions: 初始持仓
    #     extras:            额外参数，一个 dict, 如 extras={'x':1, 'y':2}，则回测中 g.x = 1, g.y = 2
    #     name:              指定回测名称, 如果没有指定为默认名称

    # 返回一个字符串，即 backtest_id
    # """
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)

    if frequency not in ("day", "minute"):
        raise Exception("不支持的回测频率")

    initial_cash = int(float(initial_cash))

    initial_positions = [] if initial_positions is None else initial_positions
    if not isinstance(initial_positions, (list, tuple)):
        raise Exception("初始持仓必须是一个 list 或者 tuple")
    for p in initial_positions:
        if set(p.keys()) > {"amount", "security", "avg_cost"}:
            raise Exception("不支持的初始持仓字段")
        if not isinstance(p["security"], (str, six.text_type)):
            raise Exception("股票代码必须是 str 类型")
        int(p["amount"])
        float(p.get("avg_cost", 0))

    extras = {} if extras is None else extras
    if not isinstance(extras, dict):
        raise Exception("额外参数需要是一个 dict 类型")

    params = {
        "algorithm_id": algorithm_id,
        "start_date": start_date,
        "end_date": end_date,
        "frequency": frequency,
        "initial_cash": initial_cash,
        "initial_positions": initial_positions,
        "extras": extras,
        'user': os.environ.get('JQ_USER'),
        'token': os.environ.get('RESEARCH_API_TOKEN'),
        'name': name,
        'package_version': package_version,
    }

    return request_json("/create_backtest", params)


def get_backtest(backtest_id):
    # u"""获取回测信息

    # 根据 backtest_id 返回一个 Backtest 对象
    # """
    return Backtest(backtest_id)


def read_file(path):
    # u"""读取您的文件

    # 参数:
    #     path: 相对于研究的根目录, 读取的是原始内容, 不做 decode

    # 返回读取到的文件内容
    # """
    with open(get_full_path(path), 'rb') as f:
        return f.read()


def write_file(path, content, append=False):
    # u"""写入您的文件

    # 参数:
    #     path: 相对于研究的根目录, 如果是 unicode, 会被 encode 成 utf-8
    #     content: 需要写入文件的内容
    #     append: 是否是追加模式, 否则将清除原有文件内容.

    # 返回写文件文件的内容的大小
    # """
    if isinstance(content, six.text_type):
        content = content.encode('utf-8')
    with open(get_full_path(path), 'ab' if append else 'wb') as f:
        return f.write(content)


log = get_logger()


__all__ = [
    'get_price',
    'history',
    'attribute_history',
    'get_extras',
    'get_fundamentals',
    'get_fundamentals_continuously',
    'get_index_stocks',
    'get_industry_stocks',
    'get_concept_stocks',
    'get_all_securities',
    'get_security_info',
    'get_fund_info',
    'get_margincash_stocks',
    'get_marginsec_stocks',
    'normalize_code',
    'get_dominant_future',
    'get_future_contracts',
    'get_billboard_list',
    'get_locked_shares',
    'calc_factors',

    'fundamentals',
    'query',
    'valuation',
    'income',
    'balance',
    'cash_flow',
    'indicator',
    'bank_indicator',
    'security_indicator',
    'insurance_indicator',

    'create_backtest',
    'get_backtest',

    'read_file',
    'write_file',
    'log',
]
