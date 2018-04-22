#!/usr/bin/env python
#coding:utf-8

import datetime
import six
import numpy as np
from pandas import Timestamp

def parse_date(s):
    '''
    返回一个 datetime.date 对象
    '''
    # warning:
    # datetime.datetime isinstance of datetime.date
    if isinstance(s, datetime.datetime):
        return s.date()
    if isinstance(s, datetime.date):
        return s
    if isinstance(s, Timestamp):
        return s.date()
    if isinstance(s, six.string_types):
        if s.find("-") >= 0:
            if s.find(":") > 0:
                return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").date()
            else:
                return datetime.datetime.strptime(s, "%Y-%m-%d").date()
        else:
            return datetime.datetime.strptime(s, "%Y%m%d").date()
    if isinstance(s, six.integer_types):
        return datetime.date(year=s//10000,
                             month=(s//100) % 100,
                             day=s%100)
    raise ValueError("Unknown {} for parse_date.".format(s))


def parse_datetime(s):
    if isinstance(s, datetime.datetime):
        return s.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(s, datetime.date):
        return datetime.datetime.combine(s, datetime.time(0, 0))
    if isinstance(s, six.string_types):
        if s.find("-") >= 0:
            if s.find(":") > 0:
                return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            else:
                return datetime.datetime.strptime(s, "%Y-%m-%d")
        else:
            return datetime.datetime.strptime(s, "%Y%m%d")
    if isinstance(s, six.integer_types):
        return datetime.datetime(year=s//10000,
                                 month=(s//100) % 100,
                                 day=s%100)
    raise ValueError("Unknown {} for parse_datetime.".format(s))


def to_timestamp(dt):
    '''
    将datetime或者date转换成时间戳。
    如果是 datetime, 忽略秒和微秒。
    如果是 date, 忽略小时，分钟，秒和微妙。
    '''
    if isinstance(dt, datetime.datetime):
        dt = dt.replace(second=0, microsecond=0)
        td = dt - datetime.datetime(1970,1,1)
    elif isinstance(dt, datetime.date):
        td = dt - datetime.date(1970,1,1)
    elif isinstance(dt, datetime.time):
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    else:
        raise Exception("unkown dt=%s, type(dt) is %s" % (dt, type(dt)))
    return (td.seconds + td.days * 86400)


def to_datetime(ts):
    return datetime.datetime.utcfromtimestamp(int(ts))


def to_date(ts):
    return datetime.datetime.utcfromtimestamp(int(ts)).date()

def combine_time(d):
    return datetime.datetime.combine(d, datetime.datetime.min.time())

def trim_time(ts):
    '''
    去掉 timestamp 中的time部分，保证ts可以转化成一天。
    '''
    i = int(ts)
    return (i//86400) * 86400


vec2timestamp = np.vectorize(to_timestamp, otypes=[np.int])
vec2datetime = np.vectorize(to_datetime, otypes=[datetime.datetime])
vec2date = np.vectorize(to_date, otypes=[datetime.date])
vec2combine = np.vectorize(combine_time, otypes=[datetime.datetime])
vec2trimtime = np.vectorize(trim_time, otypes=[np.int])