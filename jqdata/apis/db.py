# -*- coding: UTF-8 -*-

from __future__ import absolute_import

import sys
import os
import re
import datetime
import platform

import six
from six import StringIO
import pandas as pd
from fastcache import clru_cache as lru_cache

from ..utils.utils import (
    obj_to_tuple,
    check_string_list,
    check_fields,
    check_string,
    convert_date,
    convert_dt,
    date_range,
    TRADE_MIN_DATE,

    filter_dict_values,

    float_or_nan,
)

from ..db_utils import (
    get_sql_runner,
    request_data,
    request_client_data,
    query,
    compile_query,
    no_sa_warnings,

    FUNDAMENTAL_RESULT_LIMIT,
)

from .base import get_trade_days
from .security import normalize_code

from jqdata.exceptions import ParamsError

if 'Windows' in platform.system():
    from jqdata.fundamentals_non_redundant_tables_gen import (
        BalanceSheet, IncomeStatement, CashFlowStatement, FinancialIndicator,
        BankIndicatorAcc, SecurityIndicatorAcc, InsuranceIndicatorAcc, StockValuation)
else:
    from jqdata.fundamentals_tables_gen import (
        BalanceSheet, IncomeStatement, CashFlowStatement, FinancialIndicator,
        BankIndicatorAcc, SecurityIndicatorAcc, InsuranceIndicatorAcc, StockValuation)

from jqdata.stores import CalendarStore
from jqdata import get_config
from jqdata.finance_table import StkAbnormal, StkLockShares, StkMoneyFlow

__all__ = [
    'get_mtss',
    'get_billboard_list',
    'get_locked_shares',
    'get_money_flow',
    'get_valuation',
    'get_fundamentals',
    'get_fundamentals_continuously',
    'fundamentals',
    'query',

    'balance',
    'income',
    'cash_flow',
    'valuation',
    'indicator',

    'bank_indicator',
    'security_indicator',
    'insurance_indicator',
]

DATA_SERVER = os.environ.get('JQDATA_API_SERVER', 'http://jqdata:8000')

balance = balance_sheet = BalanceSheet
income = income_statement = IncomeStatement
cash_flow = cash_flow_statement = CashFlowStatement
indicator = financial_indicator = FinancialIndicator

bank_indicator = bank_indicator_acc = BankIndicatorAcc
security_indicator = security_indicator_acc = SecurityIndicatorAcc
insurance_indicator = insurance_indicator_acc = InsuranceIndicatorAcc

valuation = stock_valuation = StockValuation

# 兼容, fundamentals 表示本模块
fundamentals = sys.modules[__name__]


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
    date, sec_code, fin_value, fin_buy_value, fin_refund_value, sec_value, sec_sell_value, sec_refund_value,
    fin_sec_value
    """
    if start_date and count:
        raise ParamsError("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise ParamsError("count 参数需要大于 0 或者为 None")
    if count:
        count = int(count)
    security_list = obj_to_tuple(security_list)
    check_string_list(security_list)

    end_date = convert_date(end_date) if end_date else datetime.date.today()
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(end_date=end_date, count=count)[0] if count else TRADE_MIN_DATE)

    keys = ["sec_code", "date", "fin_value", "fin_buy_value", "fin_refund_value",
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
        if os.getenv('JQENV') == 'client': # 客户端
            data = request_client_data(request_path, request_params)
        else:
            data = request_data(DATA_SERVER + request_path, request_params)
        for d in data:
            values = [item.strip() for item in d.split(",", nkeys)[:nkeys]]
            values = [convert_funcs[i](v) for i, v in enumerate(values)]
            sec_dict = dict(zip(keys, values))
            sec_dict["sec_code"] = normalize_code(sec_dict["sec_code"])
            lists.append(filter_dict_values(sec_dict, fields))

    import pandas as pd
    df = pd.DataFrame(columns=fields, data=lists)
    return df


def get_billboard_list(stock_list=None, start_date=None, end_date=None, count=None):
    '''
    返回执指定日期区间内的龙虎榜个股列表
    :param stock_list：单个股票或股票代码列表， 可以为 None， 返回股票的列表。
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param count: 交易日数量，与 end_date 不能同时使用。与 start_date 配合使用时， 表示获取 start_date 到 start_date+count-1个交易日期间的数据
    :return:Dataframe
        |   date   | stock_code | abnormal_code |     abnormal_name        | sales_depart_name | abnormal_type | buy_value | buy_rate | sell_value | sell_rate | net_value | amount |
        |----------|------------|---------------|--------------------------|-------------------|---------------|-----------|----------|------------|-----------|-----------|--------|
        |2017-07-01| 000038.XSHE|     1         |日价格涨幅偏离值达7%以上的证券|        None       |      ALL      |  35298494 |0.37108699|  32098850  | 0.33744968|   3199644 |95121886|
    '''
    import pandas as pd

    from ..utils.utils import convert_date, is_lists
    from ..db_utils import query, request_mysql_server
    if count is not None and start_date is not None:
        raise ParamsError("get_billboard_list 不能同时指定 start_date 和 count 两个参数")
    if count is None and start_date is None:
        raise ParamsError("get_billboard_list 必须指定 start_date 或 count 之一")
    end_date = convert_date(end_date) if end_date else datetime.date.today()
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(end_date=end_date, count=count)[0] if count else TRADE_MIN_DATE)
    if not is_lists(stock_list):
        if stock_list is not None:
            stock_list = [stock_list]
    if stock_list:
        stock_list = [s.split('.')[0] for s in stock_list]

    q = query(StkAbnormal).filter(StkAbnormal.day <= end_date, StkAbnormal.day >= start_date)
    if stock_list is not None:
        q = q.filter(StkAbnormal.code.in_(stock_list))
    q = q.order_by(StkAbnormal.day.desc()).order_by(StkAbnormal.code.desc())
    sql = compile_query(q)
    cfg = get_config()
    if os.getenv('JQENV') == 'client':  # 客户端
        csv = request_mysql_server(sql)
        dtype_dict = {}
        dtype_dict['code'] = str
        df = pd.read_csv(six.StringIO(csv), dtype=dtype_dict)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError(
                "you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY, is_random=False)
        df = sql_runner.run(sql, return_df=True)
    return df
    pass


def get_locked_shares(stock_list, start_date=None, end_date=None, forward_count=None):
    '''
    获取指定日期范围内的个股限售股解禁数据
    :param stock_list:单个股票或股票代码的列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :param forward_count: 交易日数量，与 end_date 不能同时使用。与 start_date 配合使用时， 表示获取 start_date 到 start_date+count-1个交易日期间的数据
    :return: dataframe
        |date|stock_code|num|rate1|rate2|
        |----------|-----------|--------|----|----|
        |2017-07-01|000001.XSHG|20000000|0.03|0.02|
        |2017-07-01|000001.XSHG|20000000|0.03|0.02|
     #### 注意单日个股多条解禁数据的问题 ####
    '''
    import pandas as pd
    from six import StringIO

    from ..utils.utils import convert_date, is_lists
    from ..db_utils import query, request_mysql_server
    if forward_count is not None and end_date is not None:
        raise ParamsError("get_locked_shares 不能同时指定 end_date 和 forward_count 两个参数")
    if forward_count is None and end_date is None:
        raise ParamsError("get_locked_shares 必须指定 end_date 或 forward_count 之一")
    start_date = convert_date(start_date)
    if not is_lists(stock_list):
        stock_list = [stock_list]
    if stock_list:
        stock_list = [s.split('.')[0] for s in stock_list]

    if forward_count is not None:
        end_date = start_date + datetime.timedelta(days=forward_count)

    end_date = convert_date(end_date)
    q = query(StkLockShares.day, StkLockShares.code, StkLockShares.num, StkLockShares.rate1,
              StkLockShares.rate2).filter(StkLockShares.code.in_(stock_list),
                                          StkLockShares.day <= end_date, StkLockShares.day >= start_date
                                          ).order_by(StkLockShares.day).order_by(StkLockShares.code.desc())

    sql = compile_query(q)
    cfg = get_config()
    if os.getenv('JQENV') == 'client':  # 客户端
        csv = request_mysql_server(sql)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError(
                "you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY, is_random=False)
        csv = sql_runner.run(sql, return_df=False)
    dtype_dict = {}
    dtype_dict['code'] = str
    df = pd.read_csv(StringIO(csv), dtype=dtype_dict)
    return df
    pass


def get_lhb(start_date, end_date, fields=None):
    """
    获取龙虎榜

    返回pd.DataFrame, columns:
    日期, 股票代码,当日涨幅,龙虎榜成交额,龙虎榜买入，买入占总成交比例，龙虎榜卖出,卖出占总成交比例，上榜原因代码
    date, sec_code, change_pct, turnover_value, buy_value, buy_pct, sell_value, sell_pct, onboard_reason
    """
    keys = ["date", "sec_code", "change_pct", "turnover_value", "buy_value", "buy_pct",
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
        if os.getenv('JQENV') == 'client': # 客户端
            data = request_client_data(request_path, request_params)
        else:
            data = request_data(DATA_SERVER + request_path, request_params)
        for d in data:
            values = [item.strip() for item in d.split(",", nkeys - 1)[:nkeys - 1]]
            values = [convert_funcs[i](v) for i, v in enumerate([date] + values)]
            lhb_dict = dict(zip(keys, values))
            lists.append(filter_dict_values(lhb_dict, fields))

    import pandas as pd
    df = pd.DataFrame(columns=fields, data=lists)
    return df


def get_money_flow(security_list, start_date=None, end_date=None, fields=None, count=None):
    """
    获取资金流向数据

    security_list: 股票代码或者 list
    start_date: 开始日期, **与 count 二选一，不可同时使用**. str/datetime.date/datetime.datetime 对象, 默认为平台提供的数据的最早日期
    end_date: 结束日期, str/datetime.date/datetime.datetime 对象, 默认为 datetime.date.today()
    fields: 字段名或者 list, 可选, 默认全部字段
    count: 数量, **与 start_date 二选一，不可同时使用**. 表示返回 end_date 之前 count 个交易日的数据, 包含 end_date

    返回pd.DataFrame, columns:
    日期, 股票代码, 涨跌幅(%), 主力净额(万), 主力净占比(%), 超大单净额(万), 超大单净占比（%）,
    大单净额(万), 大单净占比(%), 中单净额(万), 中单净占比(%), 小单净额(万), 小单净占比（%）
    date, sec_code, change_pct, net_amount_main, net_pct_main, net_amount_xl,
    net_pct_xl, net_amount_l, net_pct_l, net_amount_m, net_pct_m, net_amount_s, net_pct_s
    """
    import pandas as pd
    from ..db_utils import query, request_mysql_server
    # 参数处理
    if start_date and count:
        raise ParamsError("start_date 参数与 count 参数只能二选一")
    if not (count is None or count > 0):
        raise ParamsError("count 参数需要大于 0 或者为 None")
    if count:
        count = int(count)
    security_list = obj_to_tuple(security_list)
    check_string_list(security_list)
    if security_list:
        security_list = [s.split('.')[0] for s in security_list]
    end_date = convert_date(end_date) if end_date else datetime.date.today()
    start_date = convert_date(start_date) if start_date else \
        (get_trade_days(end_date=end_date, count=count)[0] if count else TRADE_MIN_DATE)
    keys = ["date", "sec_code", "change_pct", "net_amount_main", "net_pct_main", "net_amount_xl",
            "net_pct_xl", "net_amount_l", "net_pct_l", "net_amount_m", "net_pct_m",
            "net_amount_s", "net_pct_s"]
    if fields:
        fields = obj_to_tuple(fields)
        check_string_list(fields)
        check_fields(keys, fields)
    else:
        fields = keys
    # 开始查询
    start_date = convert_date(start_date)
    end_date = convert_date(end_date)
    q = query(StkMoneyFlow).filter(StkMoneyFlow.sec_code.in_(security_list)).filter(
        StkMoneyFlow.date >= start_date).filter(StkMoneyFlow.date <= end_date)
    sql = compile_query(q)
    cfg = get_config()
    if os.getenv('JQENV') == 'client':  # 客户端
        csv = request_mysql_server(sql)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError(
                "you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY)
        csv = sql_runner.run(sql, return_df=False)
    dtype_dict = {}
    dtype_dict['sec_code'] = str
    df = pd.read_csv(StringIO(csv), dtype=dtype_dict)
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[:, list(fields)]
    return df
    pass


def get_valuation(security_list, end_date, fields,
                  start_date=None, count=None):
    """
    返回一个pd.Panel, 三维分别是 field, date, security.

    security_list/start_date/end_date/count的含义同get_price

    field: 下面的表的中属性
    https://www.joinquant.com/data/dict/fundamentals#市值数据

    底层通过转换成sql语句查询数据库实现
    """
    import pandas as pd
    from six import StringIO

    from ..utils.utils import convert_date, is_lists
    from ..db_utils import query, request_mysql_server

    if count is not None and start_date is not None:
        raise ParamsError("get_valuation 不能同时指定 start_date 和 count 两个参数")

    if count is None and start_date is None:
        raise ParamsError("get_valuation 必须指定 start_date 或 count 之一")

    if 'code' in fields or 'day' in fields:
        raise ParamsError("get_valuation fields 不能查询 code 和 day字段")
    end_date = convert_date(end_date)
    if not is_lists(security_list):
        security_list = [security_list]

    if not is_lists(fields):
        fields = [fields]

    val_fields = []
    for field in fields:
        val_fields.append(getattr(valuation, field))

    if count is not None:
        count = int(count)
        q = query(valuation.code, valuation.day, *val_fields).filter(
            valuation.code.in_(security_list),
            valuation.day < end_date
        ).order_by(valuation.day.desc()).limit(count)
    else:
        start_date = convert_date(start_date)
        q = query(valuation.code, valuation.day, *val_fields).filter(
            valuation.code.in_(security_list),
            valuation.day < end_date,
            valuation.day > start_date,
        ).order_by(valuation.day.desc())

    sql = compile_query(q)
    cfg = get_config()
    if os.getenv('JQENV') == 'client':  # 客户端
        csv = request_mysql_server(sql)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError(
                "you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY, is_random=False)
        csv = sql_runner.run(sql, return_df=False)
    df = pd.read_csv(StringIO(csv))
    newdf = df.set_index(['code', 'day'])
    return newdf.to_panel()


@no_sa_warnings
def fundamentals_redundant_query_to_sql(query, date=None, statDate=None): # noqa
    """ 把 get_fundamentals 的 query 参数转化为 sql 语句 """
    stat_date = statDate
    if (date and stat_date) or (not date and not stat_date):
        raise ParamsError('date和statDate参数必须且只能输入一个')

    limit = min(FUNDAMENTAL_RESULT_LIMIT, query._limit or FUNDAMENTAL_RESULT_LIMIT)
    offset = query._offset
    query._offset = None
    query._limit = None

    def get_table_class(tablename):
        for t in (BalanceSheet, CashFlowStatement, FinancialIndicator,
                  IncomeStatement, StockValuation, BankIndicatorAcc, SecurityIndicatorAcc,
                  InsuranceIndicatorAcc):
            if t.__tablename__ == tablename:
                return t

    # 比 db_utils 中的 get_tables_from_sql 更保险无误
    def get_tables_from_sql(sql):
        m = re.findall(
            r'cash_flow_statement_day|balance_sheet_day|financial_indicator_day|'
            r'income_statement_day|stock_valuation|bank_indicator_acc|security_indicator_acc|'
            r'insurance_indicator_acc', sql)
        return list(set(m))

    def get_trade_day_not_after(date):
        ''' 返回 date, 如果 date 是交易日, 否则返回 date 前的一个交易 '''
        cal_ins = CalendarStore.instance()
        date = convert_date(date)
        if not cal_ins.is_trade_date(None, date):
            # 如果不是交易日, 取前一日的, 如果前一日不存在, 就使用 date
            date = cal_ins.get_previous_trade_date(None, date) or date
        return date

    tablenames = get_tables_from_sql(str(query.statement))
    tables = [get_table_class(name) for name in tablenames]

    by_year = False

    if date:
        date = convert_date(date)
        # 数据库中并没有存非交易日的数据. 所以如果传入的非交易日, 就需要取得前一个交易日
        date = get_trade_day_not_after(date)

    only_year = bool({"bank_indicator_acc", "security_indicator_acc",
                      "insurance_indicator_acc"} & set(tablenames))

    if only_year:
        if date:
            date = None
            stat_date = str(datetime.date.min)
        elif stat_date:
            if isinstance(stat_date, (str, six.string_types)):
                stat_date = stat_date.lower()
                if 'q' in stat_date:
                    stat_date = '0001-01-01'
                else:
                    stat_date = '{}-12-31'.format(int(stat_date))
            elif isinstance(stat_date, int):
                stat_date = '{}-12-31'.format(stat_date)

            stat_date = convert_date(stat_date)
        else:
            today = datetime.date.today()
            yesteryear = today.year - 1
            stat_date = datetime.date(yesteryear, 12, 31)
    elif stat_date:
        if isinstance(stat_date, (str, six.string_types)):
            stat_date = stat_date.lower()
            if 'q' in stat_date:
                stat_date = (stat_date.replace('q1', '-03-31')
                             .replace('q2', '-06-30')
                             .replace('q3', '-09-30')
                             .replace('q4', '-12-31'))
            else:
                year = int(stat_date)
                by_year = True
                stat_date = '%s-12-31' % year
        elif isinstance(stat_date, int):
            year = int(stat_date)
            by_year = True
            stat_date = '%s-12-31' % year

        stat_date = convert_date(stat_date)

    def get_stat_date_column(cls):
        if only_year:
            # 只支持按年份查询的表没有 day 字段
            return cls.statDate
        else:
            # valuation表没有statDate
            return getattr(cls, 'statDate', cls.day)

    # 不晚于 stat_date 的一个交易日
    trade_day_not_after_stat_date = None
    for table in tables:
        if date:
            query = query.filter(table.day == date)
        else:
            if hasattr(table, 'statDate'):
                query = query.filter(table.statDate == stat_date)
            else:
                # 估值表, 在非交易日没有数据
                # 所以如果传入的非交易日, 就需要取得前一个交易日
                assert table is StockValuation
                if trade_day_not_after_stat_date is None:
                    trade_day_not_after_stat_date = get_trade_day_not_after(stat_date)
                query = query.filter(table.day == trade_day_not_after_stat_date)

    # 连表
    for table in tables[1:]:
        query = query.filter(table.code == tables[0].code)

    # 恢复 offset, limit
    query = query.offset(offset)
    query = query.limit(limit)
    sql = compile_query(query)

    if stat_date:
        if by_year:
            sql = sql.replace('balance_sheet_day', 'balance_sheet')\
                     .replace('financial_indicator_day', 'financial_indicator_acc')\
                     .replace('income_statement_day', 'income_statement_acc')\
                     .replace('cash_flow_statement_day', 'cash_flow_statement_acc')
        else:
            for t in ('balance_sheet_day', 'financial_indicator_day', 'income_statement_day',
                      'cash_flow_statement_day'):
                sql = sql.replace(t, t[:-4])
        sql = re.sub(r'(cash_flow_statement|balance_sheet|income_statement|financial_indicator|'
                     r'financial_indicator_acc|income_statement_acc|cash_flow_statement_acc)\.`?day`?\b',
                     r'\1.statDate', sql)
    return sql


@no_sa_warnings
def fundamentals_non_redundant_query_to_sql(query, date=None, statDate=None): # noqa
    """ 把 get_fundamentals 的 query 参数转化为 sql 语句 """
    stat_date = statDate
    if (date and stat_date) or (not date and not stat_date):
        raise ParamsError('date和statDate参数必须且只能输入一个')

    limit = min(FUNDAMENTAL_RESULT_LIMIT, query._limit or FUNDAMENTAL_RESULT_LIMIT)
    offset = query._offset
    query._offset = None
    query._limit = None

    def get_table_class(tablename):
        for t in (BalanceSheet, CashFlowStatement, FinancialIndicator,
                  IncomeStatement, StockValuation, BankIndicatorAcc, SecurityIndicatorAcc,
                  InsuranceIndicatorAcc):
            if t.__tablename__ == tablename:
                return t

    # 比 db_utils 中的 get_tables_from_sql 更保险无误
    def get_tables_from_sql(sql):
        m = re.findall(
            r'cash_flow_statement|balance_sheet|financial_indicator|'
            r'income_statement|stock_valuation|bank_indicator_acc|security_indicator_acc|'
            r'insurance_indicator_acc', sql)
        return list(set(m))

    def get_trade_day_not_after(date):
        ''' 返回 date, 如果 date 是交易日, 否则返回 date 前的一个交易 '''
        cal_ins = CalendarStore.instance()
        if not cal_ins.is_trade_date(None, date):
            # 如果不是交易日, 取前一日的, 如果前一日不存在, 就使用 date
            date = cal_ins.get_previous_trade_date(None, date) or date
        return date

    tablenames = get_tables_from_sql(str(query.statement))
    tables = [get_table_class(name) for name in tablenames]

    by_year = False

    if date:
        date = convert_date(date)
        # 数据库中并没有存非交易日的数据. 所以如果传入的非交易日, 就需要取得前一个交易日
        date = get_trade_day_not_after(date)
        # stock valuation表
        query = query.filter(StockValuation.day == date)

    only_year = bool({"bank_indicator_acc", "security_indicator_acc",
                      "insurance_indicator_acc"} & set(tablenames))

    if only_year:
        if date:
            date = None
            stat_date = str(datetime.date.min)
        elif stat_date:
            if isinstance(stat_date, (str, six.string_types)):
                stat_date = stat_date.lower()
                if 'q' in stat_date:
                    stat_date = '0001-01-01'
                else:
                    stat_date = '{}-12-31'.format(int(stat_date))
            elif isinstance(stat_date, int):
                stat_date = '{}-12-31'.format(stat_date)

            stat_date = convert_date(stat_date)
        else:
            today = datetime.date.today()
            yesteryear = today.year - 1
            stat_date = datetime.date(yesteryear, 12, 31)
    elif stat_date:
        if isinstance(stat_date, (str, six.string_types)):
            stat_date = stat_date.lower()
            if 'q' in stat_date:
                stat_date = (stat_date.replace('q1', '-03-31')
                             .replace('q2', '-06-30')
                             .replace('q3', '-09-30')
                             .replace('q4', '-12-31'))
            else:
                year = int(stat_date)
                by_year = True
                stat_date = '%s-12-31' % year
        elif isinstance(stat_date, int):
            year = int(stat_date)
            by_year = True
            stat_date = '%s-12-31' % year

        stat_date = convert_date(stat_date)

    def get_stat_date_column(cls):
        if only_year:
            # 只支持按年份查询的表没有 day 字段
            return cls.statDate
        else:
            # valuation表没有statDate
            return getattr(cls, 'statDate', cls.day)

    # 不晚于 stat_date 的一个交易日
    trade_day_not_after_stat_date = None
    for table in tables:
        if stat_date:
            if hasattr(table, 'statDate'):
                query = query.filter(table.statDate == stat_date)
            else:
                # 估值表, 在非交易日没有数据
                # 所以如果传入的非交易日, 就需要取得前一个交易日
                assert table is StockValuation
                if trade_day_not_after_stat_date is None:
                    trade_day_not_after_stat_date = get_trade_day_not_after(stat_date)
                query = query.filter(table.day == trade_day_not_after_stat_date)

    # 连表
    if date:
        for table in tables:
            if table is not StockValuation:
                query = query.filter(StockValuation.code == table.code)
                query = query.filter(StockValuation.day >= table.periodStart)
                query = query.filter(StockValuation.day <= table.periodEnd)
    else:
        for table in tables[1:]:
            query = query.filter(table.code == tables[0].code)

    # 恢复 offset, limit
    query = query.offset(offset)
    query = query.limit(limit)
    sql = compile_query(query)

    if stat_date:
        if by_year:
            sql = sql.replace('financial_indicator', 'financial_indicator_acc')\
                     .replace('income_statement', 'income_statement_acc')\
                     .replace('cash_flow_statement', 'cash_flow_statement_acc')
        sql = re.sub(r'(cash_flow_statement|balance_sheet|income_statement|financial_indicator|'
                     r'financial_indicator_acc|income_statement_acc|cash_flow_statement_acc)\.`?day`?\b',
                     r'\1.statDate', sql)
    return sql


@no_sa_warnings
def fundamentals_query_to_sql(query, date=None, statDate=None):
    if os.getenv('JQENV') == 'client':  # 客户端非冗余
        return fundamentals_non_redundant_query_to_sql(query, date, statDate)
    else:
        return fundamentals_redundant_query_to_sql(query, date, statDate)
    pass

def get_fundamentals(query_object=None, date=None, statDate=None, sql=None): # noqa
    if query_object is None and sql is None:
        raise ParamsError("get_fundamentals 至少输入 query_object 或者 sql 参数")

    cfg = get_config()
    if date:
        date = convert_date(date)
    if query_object:
        sql = fundamentals_query_to_sql(query_object, date, statDate)
    check_string(sql)
    if os.getenv('JQENV') == 'client':  # 客户端
        from jqdata.db_utils import request_mysql_server
        csv = request_mysql_server(sql)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError("you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY, is_random=False)
        # return csv 在转成 DataFrame, 跟kaunke保持兼容, 防止直接return df 跟以前不一样
        csv = sql_runner.run(sql, return_df=False)
    return pd.read_csv(StringIO(csv))


@no_sa_warnings
def fundamentals_non_redundant_continuously_query_to_sql(query, trade_day):
    '''
    根据传入的查询对象和起始时间生成sql
    trade_day是要查询的交易日列表
    '''
    limit = min(FUNDAMENTAL_RESULT_LIMIT, query._limit or FUNDAMENTAL_RESULT_LIMIT)
    offset = query._offset
    query._offset = None
    query._limit = None

    def get_table_class(tablename):
        for t in (BalanceSheet, CashFlowStatement, FinancialIndicator,
                  IncomeStatement, StockValuation, BankIndicatorAcc, SecurityIndicatorAcc,
                  InsuranceIndicatorAcc):
            if t.__tablename__ == tablename:
                return t

    def get_tables_from_sql(sql):
        m = re.findall(
            r'cash_flow_statement|balance_sheet|financial_indicator|'
            r'income_statement|stock_valuation|bank_indicator_acc|security_indicator_acc|'
            r'insurance_indicator_acc', sql)
        return list(set(m))
    # 从query对象获取表对象
    tablenames = get_tables_from_sql(str(query.statement))
    tables = [get_table_class(name) for name in tablenames]
    query = query.filter(StockValuation.day.in_(trade_day))
    # 根据stock valuation 表的code和day字段筛选
    for table in tables:
            if table is not StockValuation:
                query = query.filter(StockValuation.code == table.code)
                query = query.filter(StockValuation.day >= table.periodStart)
                query = query.filter(StockValuation.day <= table.periodEnd)

    # 连表
    for table in tables[1:]:
        query = query.filter(table.code == tables[0].code)

    # 恢复 offset, limit
    query = query.offset(offset)
    query = query.limit(limit)
    sql = compile_query(query)
    # 默认添加查询code和day作为panel索引
    sql = sql.replace('SELECT ', 'SELECT DISTINCT stock_valuation.day AS day,stock_valuation.code as code, ')
    return sql
    pass


@no_sa_warnings
def fundamentals_redundant_continuously_query_to_sql(query, trade_day):
    '''
    根据传入的查询对象和起始时间生成sql
    trade_day是要查询的交易日列表
    '''
    limit = min(FUNDAMENTAL_RESULT_LIMIT, query._limit or FUNDAMENTAL_RESULT_LIMIT)
    offset = query._offset
    query._offset = None
    query._limit = None

    def get_table_class(tablename):
        for t in (BalanceSheet, CashFlowStatement, FinancialIndicator,
                  IncomeStatement, StockValuation, BankIndicatorAcc, SecurityIndicatorAcc,
                  InsuranceIndicatorAcc):
            if t.__tablename__ == tablename:
                return t

    def get_tables_from_sql(sql):
        m = re.findall(
            r'cash_flow_statement_day|balance_sheet_day|financial_indicator_day|'
            r'income_statement_day|stock_valuation|bank_indicator_acc|security_indicator_acc|'
            r'insurance_indicator_acc', sql)
        return list(set(m))
    # 从query对象获取表对象
    tablenames = get_tables_from_sql(str(query.statement))
    tables = [get_table_class(name) for name in tablenames]
    query = query.filter(StockValuation.day.in_(trade_day))
    # 根据stock valuation 表的code和day字段筛选
    for table in tables:
            if table is not StockValuation:
                query = query.filter(StockValuation.code == table.code)
                if hasattr(table, 'day'):
                    query = query.filter(StockValuation.day == table.day)
                else:
                    query = query.filter(StockValuation.day == table.statDate)

    # 连表
    for table in tables[1:]:
        query = query.filter(table.code == tables[0].code)

    # 恢复 offset, limit
    query = query.offset(offset)
    query = query.limit(limit)
    sql = compile_query(query)
    # 默认添加查询code和day作为panel索引
    sql = sql.replace('SELECT ', 'SELECT DISTINCT stock_valuation.day AS day,stock_valuation.code as code, ')
    return sql
    pass


@no_sa_warnings
def fundamentals_continuously_query_to_sql(query, trade_day):
    if os.getenv('JQENV') == 'client':  # 客户端非冗余
        return fundamentals_non_redundant_continuously_query_to_sql(query, trade_day)
    else:
        return fundamentals_redundant_continuously_query_to_sql(query, trade_day)
    pass


def get_fundamentals_continuously(query_object=None, end_date=None, count=1):
    '''
    query_object:查询对象
    end_date:查询财务数据的截止日期
    count：查询财务数据前溯天数，默认为1
    返回一个pd.Panel, 三维分别是 field, date, security.
    field: 下面的表的中属性
    https://www.joinquant.com/data/dict/fundamentals
    '''
    if query_object is None:
        raise ParamsError("get_fundamentals_continuously 需要输入 query_object 参数")
    if end_date is None:
        end_date = datetime.date.today()
    cfg = get_config()

    trade_day = get_trade_days(end_date=end_date, count=count)
    if query_object:
        sql = fundamentals_continuously_query_to_sql(query_object, trade_day)
    check_string(sql)
    # 调用查询接口生成CSV格式字符串
    if os.getenv('JQENV') == 'client':  # 客户端
        from jqdata.db_utils import request_mysql_server
        csv = request_mysql_server(sql)
    else:
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError("you must config FUNDAMENTALS_SERVER for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY, is_random=False)
        csv = sql_runner.run(sql, return_df=False)
    # 转换成panel，设置时间和股票code为索引
    df = pd.read_csv(StringIO(csv))
    df = df.drop_duplicates()
    newdf = df.set_index(['day', 'code'])
    pan = newdf.to_panel()
    return pan
    pass
