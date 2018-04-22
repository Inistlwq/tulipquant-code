#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import requests
import datetime
import logging
import inspect

import six
import contextlib
from decorator import decorator
from retrying import retry
from fastcache import clru_cache as lru_cache

from retrying import retry
import sqlalchemy
from sqlalchemy.orm.query import Query

import jqdata
from jqdata import get_config

import sys
from os.path import abspath, join, dirname

# for cython
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())

project_dir = abspath(join(dirname(abspath(__file__)), '../../'))
sys.path.insert(0, join(project_dir, 'jqcommon'))

from jqcommon.retry import RetryingObject


log = logging.getLogger('jqdata.db')

RESEARCH_SERVER = os.environ.get('RESEARCH_API_SERVER', 'http://research-service:5000')
DATA_SERVER = os.environ.get('JQDATA_API_SERVER', 'http://jqdata:8000')

# 财务数据最大10000条
FUNDAMENTAL_RESULT_LIMIT = 10000

#
if sys.platform in ['win32', 'cygwin']:
    SQL_SERVER = os.environ.get('JQSQL_API_SERVER', 'http://47.95.171.113:80')
else:
    SQL_SERVER = os.environ.get('JQSQL_API_SERVER', 'http://101.201.146.125:28888')
SQL_TOKEN = os.environ.get('DATA_ACCESS_TOKEN')
if not SQL_TOKEN:
    jqhome = os.getenv('JQ_HOME', os.path.expanduser('~/.joinquant'))
    token_file_path = jqhome + '/token'
    if os.path.exists(token_file_path):
        with open(token_file_path) as fp:
            SQL_TOKEN = fp.read().strip().rstrip()
if not SQL_TOKEN:
    SQL_TOKEN = 'naoBOpImcEQgsbrfegIADQ=='

# get_fundamentals 财务数据最大10000条
FUNDAMENTAL_RESULT_LIMIT = 10000
# 其他数据库最大3000条
DB_RESULT_LIMIT = 3000

# 重用 requests 链接
requests_session = requests.Session()


class BadRequest(Exception):
    """ 请求其他服务为时, 如果是传入参数的错误, 抛出 BadRequest, 不重试 """
    pass


class ResponseCodeError(IOError):
    pass


def _json_serial_fallback(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return str(obj)
    raise TypeError("%s not serializable" % obj)


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000,
       retry_on_exception=lambda e: isinstance(e, IOError))
def request_research_server(path, data={}):
    """ 请求 research 服务器, 返回整个 body """
    with contextlib.closing(
            requests_session.post(
                RESEARCH_SERVER + path,
                data=json.dumps(data, default=_json_serial_fallback))) as res:
        body = res.content
        body = body.decode('utf-8-sig')
        if res.status_code == 400:
            raise BadRequest(body)
        res.raise_for_status()
        return body


def convert_to_str(o):
    if o is None:
        return ''
    elif isinstance(o, (six.integer_types, float)):
        return str(o)
    elif isinstance(o, six.string_types):
        return o
    else:
        return str(o)


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000,
       retry_on_exception=lambda e: isinstance(e, IOError))
def request_mysql_server(sql, path='/mysql/query', is_macro=False):
    """ 请求 mysql 服务器, 返回整个 body """
    from .encrypt import encryptByKey
    #from urllib import urlencode
    if sys.version_info[0] == 3:
        from urllib.parse import quote
    else:
        from urllib import quote
    # iv是固定值
    iv = "1234567890123456"
    # print('before', sql)
    if is_macro:
        url = SQL_SERVER + path + "?token=" + quote(SQL_TOKEN) + "&db=macro"
        data = 'sql=' + quote(encryptByKey(iv, sql, iv)) + '&db=macro'
    else:
        url = SQL_SERVER + path + "?token=" + quote(SQL_TOKEN)
        data = 'sql=' + quote(encryptByKey(iv, sql, iv))
    # print(url, data)
    with contextlib.closing(requests_session.post(url, data=data)) as res:
        body = res.content
        body = body.decode('utf-8-sig')
        if res.status_code != 200:
            raise ResponseCodeError('status_code=%s' % res.status_code)
        res.raise_for_status()
        d = json.loads(body)
        if d['code'] != 0:
            raise ResponseCodeError(str(body))
        rows = []
        for r in d['data']:
            rows.append(','.join([convert_to_str(o) for o in r]))
        csv = '\n'.join(rows) + '\n'
        return csv


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000,
       retry_on_exception=lambda e: isinstance(e, IOError))
def request_client_data(path, payload=None, method="GET"):
    """ 请求 jqdata 服务器, 返回结果的 data 字段 """
    url = SQL_SERVER + path
    payload['token'] = SQL_TOKEN
    with contextlib.closing(requests.get(url, params=payload) if method == "GET"
                            else requests.post(url, data=payload)) as req:
        req.raise_for_status()
        res = req.json()
        if int(res["code"]) != 0:
            raise ResponseCodeError(req.text)
        data = res.get("data", None)
        return data


@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000,
       retry_on_exception=lambda e: isinstance(e, IOError))
def request_data(url, payload=None, method="GET"):
    """ 请求 jqdata 服务器, 返回结果的 data 字段 """
    with contextlib.closing(requests.get(url, params=payload) if method == "GET"
                            else requests.post(url, data=payload)) as req:
        req.raise_for_status()
        res = req.json()
        if int(res["code"]) != 0:
            raise ResponseCodeError(req.text)
        data = res.get("data", None)
        return data

from functools import wraps


def no_sa_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        import warnings
        from sqlalchemy import exc as sa_exc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=sa_exc.SAWarning)
            return f(*args, **kwds)
    return wrapper


def compile_query(query):
    """ 把一个 sqlalchemy query object 编译成mysql风格的 sql 语句 """
    from sqlalchemy.sql import compiler
    from MySQLdb.converters import conversions, escape
    from sqlalchemy.dialects import mysql as mysql_dialetct

    dialect = mysql_dialetct.dialect()
    statement = query.statement
    comp = compiler.SQLCompiler(dialect, statement)
    comp.compile()
    enc = dialect.encoding
    comp_params = comp.params
    params = []
    for k in comp.positiontup:
        v = comp_params[k]
        # python2, escape(u'123', conversions) is '123', should be "'123'"
        # escape('123', conversions) is ok
        if six.PY2 and isinstance(v, unicode):
            v = v.encode(enc)
        v = escape(v, conversions)
        # python3, escape return bytes
        if six.PY3 and isinstance(v, bytes):
            v = v.decode(enc)
        params.append(v)
    return (comp.string % tuple(params))


@no_sa_warnings
def check_no_join(query):
    """ 确保 query 中没有 join 语句, 也就是说: 没有引用多个表 """
    tables = get_tables_from_sql(str(query.statement))
    if len(tables) != 1:
        raise Exception("一次只能查询一张表")


def get_tables_from_sql(sql):
    """ 从 sql 语句中拿到所有引用的表名字 """
    m = re.search(r'FROM (.*?)($| WHERE| GROUP| HAVING| ORDER)', sql, flags=re.M)
    return [t.strip() for t in m.group(1).strip().split(',')] if m else []


def _get_session():
    from sqlalchemy.orm import scoped_session, sessionmaker
    session = scoped_session(sessionmaker())
    return session


session = _get_session()

invalid_criterion_msg = """filter()函数的参数中出现了无效筛选器(结果永远为真或者假), 请检查是否使用了 in/and/or操作符. 请使用
    in_ 函数 替代 in 操作符
    and_ 函数 替代 and 操作符
    or_ 函数 替代 or 操作符
    具体示例请参考: http://docs.sqlalchemy.org/en/rel_1_0/core/sqlelement.html
    """


Query_filter = Query.filter


def my_filter(self, *criterions):
    criterions = list(criterions)
    for criterion in criterions:
        if criterion in (True, False):
            raise Exception(invalid_criterion_msg)
    return Query_filter(self, *criterions)

Query.filter = my_filter


def query(*args, **kwargs):
    """
    获取一个Query对象, 传给 get_fundamentals

    具体使用方法请参考http://docs.sqlalchemy.org/en/rel_1_0/orm/tutorial.html#querying

    示例: 查询'000001.XSHE'的所有市值数据, 时间是2015-10-15
    q = query(
        valuation
    ).filter(
        valuation.code == '000001.XSHE'
    )
    get_fundamentals(q, '2015-10-15')
    """
    return session.query(*args, **kwargs)


class SqlRunner(object):

    def __init__(self, server, keep_connection=False):
        """ Sql runner

            server: str, a msdb server
            keep_connection: bool, if keep connection after a query finished
        """
        self._server = server
        self._keep_connection = keep_connection
        self._engine = None
        pass

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._create_engine(self._server)
        return self._engine

    def _create_engine(self, server):
        # 防止下面的错误发生: 当正在执行 get_fundamentals 时, 如果收到 signal 退出, 则会触发下面的错误
        # Traceback (most recent call last):
        #   File "/home/server/y/envs/kuanke/lib/python2.7/site-packages/sqlalchemy/pool.py", line 636, in _finalize_fairy
        #     fairy._reset(pool)
        #   File "/home/server/y/envs/kuanke/lib/python2.7/site-packages/sqlalchemy/pool.py", line 776, in _reset
        #     pool._dialect.do_rollback(self)
        #   File "/home/server/y/envs/kuanke/lib/python2.7/site-packages/sqlalchemy/dialects/mysql/base.py", line 2519, in do_rollback
        #     dbapi_connection.rollback()
        # ProgrammingError: (2014, "Commands out of sync; you can't run this command now")
        import logging
        logging.getLogger('sqlalchemy.pool').setLevel(logging.CRITICAL)
        import sqlalchemy
        from sqlalchemy.pool import NullPool
        if not self._keep_connection:
            # 模拟交易可能同时又很多个进程, 会创建很多链接, 所以不使用长连接
            kws = dict(poolclass=NullPool,)
        else:
            kws = dict(pool_recycle=60, pool_size=1, max_overflow=0,)
        return sqlalchemy.create_engine(server, **kws)
        pass

    def run(self, sql, return_df):
        """ Run sql:

            sql: str
            return_df: bool, true if output `pd.DataFrame`, csv str otherwise
        """
        if self._engine is None:
            self._engine = self._create_engine(self._server)

        from sqlalchemy.sql.expression import text

        sql_execute_result = None
        try:
            if return_df:
                import pandas as pd
                return pd.read_sql_query(text(sql), con=self._engine)
            else:
                import six
                import csv
                sql_execute_result = self._engine.execute(text(sql))
                io = six.StringIO()
                outcsv = csv.writer(io)
                outcsv.writerow(sql_execute_result.keys())
                outcsv.writerows(sql_execute_result)
                return io.getvalue()
        except sqlalchemy.exc.SQLAlchemyError as e:
            retry = False
            if isinstance(e, (sqlalchemy.exc.DisconnectionError,
                              sqlalchemy.exc.TimeoutError,
                              sqlalchemy.exc.ResourceClosedError,
                              )):
                retry = True
            elif isinstance(e, sqlalchemy.exc.DBAPIError):
                orig = e.orig
                errno = -1
                try:
                    errno = self._engine.dialect._extract_error_code(orig)
                except Exception as e:
                    log.exception("Get error code from exception failed")
                    pass
                # 1203: server too many connections
                if 2000 <= errno <= 2032 or \
                        errno in (1130, 1203, 2045, 2055, 2048):
                    retry = True
            e.should_retry = retry
            raise e
        finally:
            if sql_execute_result is not None:
                sql_execute_result.close()
            pass
        pass


@lru_cache(None)
def get_sql_runner(server_name, retry_policy=None, is_random=False, **kwargs):
    assert server_name == 'fundamentals'
    cfg = get_config()
    servers = cfg.FUNDAMENTALS_SERVERS
    retry_policy = retry_policy or dict(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
    retry_policy.update(
        retry_on_exception=lambda e: isinstance(e, (sqlalchemy.exc.SQLAlchemyError, OSError, IOError)) and
        getattr(e, 'should_retry', True))
    return RetryingObject(SqlRunner, servers, gen_kws=kwargs, retry_kws=retry_policy, is_random=is_random)
