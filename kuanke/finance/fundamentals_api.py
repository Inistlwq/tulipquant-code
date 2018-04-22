#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os,re,sys,json,datetime
from kuanke.finance.fundamentals_tables_gen import *
from sqlalchemy.orm.attributes import InstrumentedAttribute

import six

from kuanke.user_space_utils import *

balance = balance_sheet = BalanceSheetDay
income = income_statement = IncomeStatementDay
cash_flow = cash_flow_statement = CashFlowStatementDay
indicator = financial_indicator = FinancialIndicatorDay
bank_indicator = bank_indicator_acc = BankIndicatorAcc
security_indicator = security_indicator_acc = SecurityIndicatorAcc
insurance_indicator = insurance_indicator_acc = InsuranceIndicatorAcc

valuation = stock_valuation = StockValuation

from jqdata.db_utils import no_sa_warnings, compile_query, query, FUNDAMENTAL_RESULT_LIMIT

@no_sa_warnings
def query_to_sql(query, date=None, statDate=None):
    if date and statDate:
        raise Exception('date和statDate参数只能输入一个')

    limit = min(FUNDAMENTAL_RESULT_LIMIT, query._limit or FUNDAMENTAL_RESULT_LIMIT)
    offset = query._offset
    query._offset = None
    query._limit = None
    tablenames = get_tables_from_sql(str(query.statement))

    by_year = False
    only_year = bool({"bank_indicator_acc", "security_indicator_acc",
        "insurance_indicator_acc"} & set(tablenames))

    if only_year:
        if date:
            date = None
            statDate = str(datetime.date.min)
        elif statDate:
            if is_str(statDate):
                statDate = statDate.lower()
                if 'q' in statDate:
                    statDate = '0001-01-01'
                else:
                    statDate = '{}-12-31'.format(int(statDate))
            elif isinstance(statDate, int):
                statDate = '{}-12-31'.format(statDate)

            statDate = convert_date(statDate)
        else:
            today = datetime.date.today()
            yesteryear = today.year - 1
            statDate = datetime.date(yesteryear, 12, 31)
    elif statDate:
        if is_str(statDate):
            statDate = statDate.lower()
            if 'q' in statDate:
                statDate = statDate.replace('q1', '-03-31').replace('q2', '-06-30').replace('q3', '-09-30').replace('q4', '-12-31')
            else:
                year = int(statDate)
                by_year = True
                statDate = '%s-12-31' % year
        elif isinstance(statDate, int):
            year = int(statDate)
            by_year = True
            statDate = '%s-12-31' % year

        statDate = convert_date(statDate)
    else:
        yestoday = datetime.date.today() - datetime.timedelta(days=1)
        if date:
            # 最多只能取昨天的
            date = min(convert_date(date), yestoday)
        else:
            if is_research():
                date = yestoday
            else:
                date = api_proxy.context.current_dt.date() - datetime.timedelta(days=1)

    tables = [get_table_class(name) for name in tablenames]

    def get_stat_date_column(cls):
        if only_year:
            # 只支持按年份查询的表没有 day 字段
            return cls.statDate
        else:
            # valuation表没有statDate
            return getattr(cls, 'statDate', cls.day)

    first = tables[0]
    query = query.filter(first.day == date) if date else query.filter(get_stat_date_column(first) == statDate)

    if len(tables) > 1:
        for i in range(1, len(tables)):
            table = tables[i]
            if date:
                query = query.filter(table.day == date, first.code == table.code)
            else:
                query = query.filter(get_stat_date_column(table) == statDate, first.code == table.code)

    query = query.offset(offset)
    query = query.limit(limit)
    sql = compile_query(query)

    if statDate:
        if by_year:
            sql = sql.replace('balance_sheet_day', 'balance_sheet')\
                     .replace('financial_indicator_day', 'financial_indicator_acc')\
                     .replace('income_statement_day', 'income_statement_acc')\
                     .replace('cash_flow_statement_day', 'cash_flow_statement_acc')
        else:
            for t in ('balance_sheet_day', 'financial_indicator_day','income_statement_day', 'cash_flow_statement_day'):
                sql = sql.replace(t, t[:-4])
        sql = re.sub(r'(cash_flow_statement|balance_sheet|income_statement|financial_indicator|'
            r'financial_indicator_acc|income_statement_acc|cash_flow_statement_acc)\.`?day`?\b',
            r'\1.statDate', sql)

    if 'DEBUG_DB' in os.environ:
        print("query_to_sql:", sql)

    return sql

# 比 db_utils 中的 get_tables_from_sql 更保险无误
def get_tables_from_sql(sql):
    m = re.findall(r'cash_flow_statement_day|balance_sheet_day|financial_indicator_day|'
        r'income_statement_day|stock_valuation|bank_indicator_acc|security_indicator_acc|'
        r'insurance_indicator_acc', sql)
    return list(set(m))

def get_table_class(tablename):
    models = (BalanceSheetDay, CashFlowStatementDay, FinancialIndicatorDay,
        IncomeStatementDay, StockValuation, BankIndicatorAcc, SecurityIndicatorAcc,
        InsuranceIndicatorAcc)

    for t in models:
        if t.__tablename__ == tablename:
            return t

fundamentals = sys.modules[__name__]

def get_fundamentals(query_object, date=None, statDate=None):
    """
    查询财务数据
    见文档: {JQ_WEB_SERVER}/api#getfundamentals
    """.format(JQ_WEB_SERVER=os.environ.get('JQ_WEB_SERVER', ''))

    sql = query_to_sql(query_object, date, statDate)

    csv = api_proxy.get_fundamentals(sql)

    import pandas as pd
    from six import StringIO
    return pd.read_csv(StringIO(csv))

__all__ = [
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
    'get_fundamentals'
]
