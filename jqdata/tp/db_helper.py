#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import six
import logging

log = logging.getLogger('jqdata.tp.db')


class ConfigError(Exception):
    pass


class Database(object):
    ''' Database

        Usage:
        gta = Database(engine, disable_join=False)
        gta.run_query(gta.query(gta.STK_AF_FORECAST.SYMBOL).limit(10))
    '''

    RESULT_ROWS_LIMIT = 3000

    query = staticmethod(__import__('jqdata.db_utils').query)

    def __init__(self, engine, disable_join=False, schema='', preload_table_names=False):
        """ preload_table_names: 预先加载 table 和 view 的名字, 添加到属性中, 方便在研究中代码自动补全. """
        self.__engine = engine
        self.__disable_join = disable_join
        self.__schema = schema
        self.__table_names = []
        self.__view_names = []
        if preload_table_names:
            self.__load_table_view_names()

    def __load_table_view_names(self):
        # 加载table/view的名字, 延迟加载每个table/view对应的类(比较耗时)
        self.__table_names = self.__load_table_names()
        # 加载 table 太慢了, 动态加载
        for name in self.__table_names:
            setattr(self, name, None)
        self.__view_names = self.__load_view_names()
        for name in self.__view_names:
            setattr(self, name, None)

    def run_query(self, query_object):
        if self.__disable_join:
            from jqdata.db_utils import check_no_join
            # 不能查询多个表: http://jira.kuanke100.com/browse/GFR-55
            check_no_join(query_object)

        # 限制返回结果最多3000行: http://jira.kuanke100.com/browse/GFR-55
        limit = self.RESULT_ROWS_LIMIT
        if query_object._limit:
            limit = min(limit, query_object._limit)
        query_object = query_object.limit(limit)

        # query object to sql,
        # http://nicolascadou.com/blog/2014/01/printing-actual-sqlalchemy-queries/
        sql = query_object.statement.compile(self.__engine)

        import pandas as pd
        return pd.read_sql(sql, con=self.__engine)

    def __load_table_names(self):
        from sqlalchemy.engine import reflection
        insp = reflection.Inspector.from_engine(self.__engine)
        if self.__schema != '':
            table_names = insp.get_table_names(schema=self.__schema)
        else:
            table_names = insp.get_table_names()
        return set(map(str, table_names))

    def __load_table_class(self, table_name):
        from sqlalchemy.schema import MetaData, Table
        from sqlalchemy.orm import mapper
        table = Table(table_name, MetaData(), autoload=True, autoload_with=self.__engine)
        # define class dynamically
        table_class = type(table_name, (object, ), {})
        # 需要屏蔽的字段
        exclude_columns = ['status', 'addTime', 'modTime']
        if len(table.primary_key) != 0:
            mapper(table_class, table, exclude_properties=exclude_columns)
        else:  # 没有主键
            mapper(table_class, table, primary_key=[getattr(table.c, table.c.keys()[0])], exclude_properties=exclude_columns)
        return table_class

    def __load_view_names(self):
        from sqlalchemy.engine import reflection
        insp = reflection.Inspector.from_engine(self.__engine)
        if self.__schema != '':
            view_names = insp.get_view_names(schema=self.__schema)
        else:
            view_names = insp.get_view_names()
        return set(map(str, view_names))

    def __load_view_class(self, view_name):
        from sqlalchemy.schema import MetaData, Table
        from sqlalchemy.orm import mapper
        # 需要 show_view_priv 权限
        # http://www.linuxhub.org/?p=2576
        table = Table(view_name, MetaData(), autoload=True, autoload_with=self.__engine)
        # define class dynamically
        view_class = type(view_name, (object, ), {})
        # 必须有primary_key
        # http://docs.sqlalchemy.org/en/latest/faq/ormconfiguration.html#how-do-i-map-a-table-that-has-no-primary-key
        mapper(view_class, table, primary_key=[getattr(table.c, table.c.keys()[0])])
        return view_class

    def __getattribute__(self, key):
        v = object.__getattribute__(self, key)
        # 如果 table 没有加载, 动态加载
        if v is None:
            if key in self.__table_names:
                v = self.__load_table_class(key)
                setattr(self, key, v)
            elif key in self.__view_names:
                v = self.__load_view_class(key)
                setattr(self, key, v)
        return v
    pass

    def __getattr__(self, key):
        # 如果没有预先加载了table/view名字, 加载它
        if not self.__table_names:
            self.__load_table_view_names()
        try:
            return getattr(self, key)
        except AttributeError:
            raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, key))
    pass


def get_engine(db_url):
    import sqlalchemy
    from sqlalchemy.pool import NullPool
    kws = dict(poolclass=NullPool,)
    try:
        eg = sqlalchemy.create_engine(db_url, **kws)
        return eg
    except Exception as e:
        log.error("create engine {} failed:{}".format(db_url, e))
        return None


def get_db_config():
    try:
        for config_path in (
            '/etc/joinquant_user/tp_db.json',
            os.path.expanduser('~/.joinquant/etc/jqcore/tp_db.json'),
        ):
            if os.path.exists(config_path):
                return json.load(open(config_path))
    except Exception as e:
        log.error("Load {} failed: {}".format(config_path, e))
    return {}


def create_database_objects():
    def random_select(array):
        return array[random.randint(0, len(array) - 1)]

    dbs = {}
    for db_name, item in get_db_config().items():
        disable_join = True
        db_url = None
        schema = ''
        if isinstance(item, dict):
            disable_join = item.get("disable_join", True)
            schema = item.get('schema', '')
            db_url = random_select(item['urls'])
        elif isinstance(item, (tuple, list)):
            db_url = random_select(item)
        elif isinstance(item, six.string_types):
            db_url = item
        else:
            raise ConfigError(
                'db_url must a url string, a list of url strings, or a dict(e.g. {"urls":[], "disable_join": true}), given: {}'.format(db_url))
        if os.getenv('JQENV') == 'client':  # 客户端产品不限制连表查询
            disable_join = False
        eg = get_engine(db_url)
        if eg is not None:
            db = Database(get_engine(db_url), disable_join=disable_join, schema=schema, preload_table_names=True)
            dbs[db_name] = db
    return dbs
