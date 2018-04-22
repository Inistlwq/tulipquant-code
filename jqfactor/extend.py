#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) JoinQuant Development Team
# Author: Huayong Kuang <kuanghuayong@joinquant.com>

import warnings
import contextlib
from collections import namedtuple

import six
import sqlalchemy
import pandas as pd
from fastcache import lru_cache
from sqlalchemy.pool import NullPool
from cached_property import cached_property

from jqdata.apis.security import normalize_code

from . import config
from .when import convert_date
from .utils import filter_dict_values


class ExtendedDataGetter(object):
    """外部扩展因子数据获取"""

    def __init__(self, servers=None):
        self._raw_servers = config.EXTENDED_DATA_SERVERS if servers is None else servers

    @cached_property
    def servers(self):
        raw_servers = self._raw_servers
        if not isinstance(raw_servers, (tuple, list)):
            raise Exception("Extended data servers configuration error")

        HTTPDataServer = namedtuple('HTTPDataServer', ['type', 'url', 'factor_names'])
        DBDataServer = namedtuple('DBDataServer', ['type', 'url', 'tables'])

        _field_names = ["table_name", "factor_name_column", "date_column",
                        "stock_column", "data_column", "factor_names"]
        DBTableInfo = namedtuple('DBTableInfo', _field_names)

        data_servers = []
        for server in raw_servers:
            if isinstance(server, six.string_types) and server.startswith("http://"):
                factor_names = self._get_names_from_http(server)
                data_servers.append(HTTPDataServer("http", server, factor_names))
            elif isinstance(server, dict):
                tables = []
                for table in server["tables"]:
                    keys = ["table_name", "name_column", "date_column",
                            "stock_column", "data_column"]
                    fields = filter_dict_values(table, keys)
                    names = self._get_names_from_db(server["url"],
                                                    table["table_name"],
                                                    table["name_column"])
                    fields.append(names)
                    tables.append(DBTableInfo(*fields))
                data_servers.append(DBDataServer("db", server["url"], tables))
            else:
                raise Exception("Extended data servers configuration error")

        return data_servers

    @staticmethod
    @lru_cache(None)
    def _get_db_engine(server):
        engine = sqlalchemy.create_engine(server, poolclass=NullPool)
        return engine

    def _get_names_from_http(self, url):
        return set()

    def _get_names_from_db(self, url, table_name, name_column):
        engine = self._get_db_engine(url)
        sql = 'SELECT DISTINCT {} as name FROM {}'.format(name_column, table_name)
        with contextlib.closing(engine.execute(sql)) as rows:
            names = {row.name for row in rows}
        return names

    def get_names(self):
        """获取所有支持的因子名称"""
        names = set()
        for server in self.servers:
            if server.type == "http":
                names.update(server.factor_names)
            elif server.type == 'db':
                for table in server.tables:
                    names.update(table.factor_names)
        return names

    def _get_data_from_http(self, url, factor_name, securities, dates):
        pass

    def _get_data_from_db(self, url, table_info, factor_name, securities, dates):
        codes = ",".join(["'{}'".format(code.split('.')[0]) for code in securities])
        sql = """
            SELECT
                {date_column} as date,
                SUBSTR({stock_column}, 1, 6) as code,
                {data_column} as data
            FROM {table_name}
            WHERE
                {factor_name_column} = '{factor_name}'
            AND
                SUBSTR({stock_column}, 1, 6) IN ({codes})
            AND
                {date_column} >= '{start_date}'
            AND
                {date_column} <= '{end_date}'
        """.format(
            date_column=table_info.date_column,
            stock_column=table_info.stock_column,
            data_column=table_info.data_column,
            table_name=table_info.table_name,
            factor_name_column=table_info.factor_name_column,
            factor_name=factor_name,
            codes=codes,
            start_date=dates[0],
            end_date=dates[-1]
        )

        engine = self._get_db_engine(url)
        with contextlib.closing(engine.execute(sql)) as rows:
            df = pd.DataFrame([dict(row) for row in rows])
        data = pd.DataFrame(index=dates, columns=securities)
        if df.empty:
            return data

        df["code"] = df["code"].apply(normalize_code)
        df["date"] = df["date"].apply(convert_date)
        df.set_index(["date", "code"], inplace=True)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            df = df["data"].unstack()

        data.loc[df.index, df.columns] = df
        return data

    def get_data(self, securities, dates, factor_names):
        """获取因子数据"""
        data = {}
        for factor_name in factor_names:
            factor_data = None
            for server in self.servers:
                if server.type == "http" and factor_name in server.factor_names:
                    factor_data = self._get_data_from_http(
                        server.url, factor_name, securities, dates)
                    break
                elif server.type == "db":
                    for table in server.tables:
                        if factor_name in table.factor_names:
                            factor_data = self._get_data_from_db(
                                server.url, table, factor_name, securities, dates)
                            break
                    else:
                        continue
                    break
            if factor_data is None:
                raise Exception("No '%s' factor data" % factor_name)
            data[factor_name] = factor_data
        return data
