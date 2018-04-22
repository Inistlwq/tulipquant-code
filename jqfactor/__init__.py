# -*- coding: utf-8 -*-

"""JoinQuant Factor"""

import sys


if __import__("six").PY2:
    reload(sys)
    sys.setdefaultencoding("utf8")


__version__ = "0.2.1"


class Factor(object):
    """因子定义基类

    所有因子的定义都应继承自本类，并重写 calc 方法

    变量的含义：
        name： 因子的名称， 不能与基础因子冲突。
        max_window： 获取数据的最长时间窗口，返回的是日级别的数据。
        dependencies： 依赖的基础因子名称
    """

    name = ''
    max_window = None
    dependencies = []

    def calc(self, df):
        """计算因子

        返回一个 pandas.Series, index 股票代码, value 是因子值
        """
        pass


def calc_factors(securities, factors, start_date, end_date):
    """多因子计算

    参数：
        securities: a list of str, security code
        factors: a list of Factor objects

    返回一个 dict，key 因子名，value 为计算结果
    """
    for fact in factors:
        if isinstance(fact, Factor):
            continue
        raise Exception("factor must be a 'jqfactor.Factor' subclass object")

    from .calculate import calc_multiple_factors
    dct = calc_multiple_factors(securities, factors, start_date, end_date)
    return {fact.name: dct[fact.name] for fact in factors}
