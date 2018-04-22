# -*- coding: utf-8 -*-

from __future__ import division

import math
import datetime

import six
import pandas as pd

from .utils import suppress


DateTime = datetime.datetime
Date = datetime.date
Time = datetime.time
TimeDelta = datetime.timedelta

today = datetime.date.today
now = datetime.datetime.now


def convert_date(date):
    """转换多种日期类型为 datetime.date 类型"""
    if isinstance(date, pd.Timestamp):
        date = str(date)

    if isinstance(date, six.string_types):
        if ':' in date:
            date = date[:10]
        with suppress(ValueError):
            return Date(*map(int, date.split('-')))
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    raise Exception("date 必须是 datetime.date, datetime.datetime, "
                    "pandas.Timestamp 或者如下格式的字符串: '2015-01-05'")


def parse_date(s):
    """解析日期为 datetime.date 类型"""
    if isinstance(s, datetime.datetime):
        return s.date()
    if isinstance(s, datetime.date):
        return s
    if isinstance(s, six.string_types):
        if '-' in s:
            return datetime.datetime.strptime(s, "%Y-%m-%d").date()
        else:
            return datetime.datetime.strptime(s, "%Y%m%d").date()
    if isinstance(s, six.integer_types):
        return datetime.date(year=s // 10000,
                             month=(s // 100) % 100,
                             day=s % 100)
    raise ValueError("Unknown {} for parse_date.".format(s))


class Quarter(object):
    """季度类型"""

    def __init__(self, fp):
        if isinstance(fp, six.string_types) and 'q' in fp:
            self.year, self.q = [int(i) for i in fp.split("q")]
        elif isinstance(fp, (tuple, list)) and len(fp) == 2:
            self.year, self.q = [int(i) for i in fp]
        elif isinstance(fp, int):
            self.year, self.q = self.__convert(fp)
        elif isinstance(fp, (Date, DateTime)):
            self.year, self.q = fp.year, int(math.ceil(fp.month / 3.0))
        else:
            raise ValueError("param error '%s'" % fp)

    @property
    def _qs(self):
        return self.year * 4 + self.q

    @staticmethod
    def __convert(qs):
        _y, _q = divmod(qs, 4)
        if _q == 0:
            _y -= 1
            _q = 4
        return _y, _q

    def __add__(self, step):
        return Quarter(self.__convert(self._qs + step))

    def __sub__(self, step):
        return Quarter(self.__convert(self._qs - step))

    def __repr__(self):
        return "Quarter({}, {})".format(self.year, self.q)

    def __str__(self):
        return "{}q{}".format(self.year, self.q)
