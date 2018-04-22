# -*- coding: UTF-8 -*-
import os
import sys
import inspect
import six
from six import StringIO
import pandas as pd
import jqdata
from .db_utils import (
    request_mysql_server,
    query,
    check_no_join,
    compile_query,
)
from jqdata.tp.db_helper import (
    Database,
    get_engine)

# for cython
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())

class Macro(object):
    RESULT_ROWS_LIMIT = 3000
    query = staticmethod(query)

    def __init__(self):
        self.__disable_join = True
        self.__table_names = self.__load_table_names()
        # 加载 table 太慢了, 动态加载
        for name in self.__table_names:
            setattr(self, name, None)
        pass

    def run_query(self, query_object, sql=None):
        if self.__disable_join:
            # 不能查询多个表: http://jira.kuanke100.com/browse/GFR-55
            check_no_join(query_object)

        # 限制返回结果最多3000行: http://jira.kuanke100.com/browse/GFR-55
        limit = self.RESULT_ROWS_LIMIT
        if query_object._limit:
            limit = min(limit, query_object._limit)
        query_object = query_object.limit(limit)

        # query object to sql
        sql = compile_query(query_object)
        csv = request_mysql_server(sql, is_macro=True)
        df = pd.read_csv(StringIO(csv))
        return self._compat_old(df)

    def _compat_old(self, df):
        # 保持跟以前一样的代码, 返回一样的结果
        csv = df.to_csv(index=False, encoding='utf-8')
        df = pd.read_csv(six.StringIO(csv), dtype='object', encoding='utf-8')
        return df

    def __load_table_names(self):
        import os
        tables_dir = os.path.join(os.path.dirname(__file__), 'macro_tables')
        if not os.path.exists(tables_dir):
            return
        names = []
        for table_file in os.listdir(tables_dir):
            if table_file.endswith('.py') and not table_file.startswith('__'):
                names.append(table_file[:-3])
        return names

    def __load_table_class(self, table_name):
        table_module = __import__('jqdata.macro_tables.' + table_name, fromlist=[table_name])
        return getattr(table_module, table_name)

    def __getattribute__(self, key):
        v = object.__getattribute__(self, key)
        # 如果 table 没有加载, 动态加载
        if v is None:
            if key in self.__table_names:
                v = self.__load_table_class(key)
                setattr(self, key, v)
        return v


def get_macro():
    # if os.getenv('JQENV') == 'client':  # 客户端
    if sys.platform in ['win32', 'cygwin']:  # 客户端
        return Macro()
    else:
        db_url = jqdata.get_config().MACRO_SERVER
        if not db_url:
            return None
        eg = get_engine(db_url)
        if eg is not None:
            return Database(eg, disable_join=True)
        else:
            return None
    pass

macro = get_macro()
