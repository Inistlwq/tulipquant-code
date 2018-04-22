#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os,re,sys,json,collections,time,datetime,six
from enum import Enum
from datetime import timedelta
import numpy as np
from cached_property import cached_property

PY3 = sys.version_info >= (3, 0)

if PY3:
    from functools import lru_cache, wraps
else:
    from functools32 import lru_cache, wraps

get_now = datetime.datetime.now
get_today = datetime.date.today
combine_dt = datetime.datetime.combine

Date = datetime.date
Time = datetime.time
Datetime = datetime.datetime

# all end time is inclusive

DELTA_DAY = datetime.timedelta(days=1)
DELTA_MINUTE = datetime.timedelta(minutes=1)
TICK_SECONDS = 5
DELTA_TICK = datetime.timedelta(seconds=TICK_SECONDS)

def date2int(date):
    return date.year * 10000 + date.month * 100 + date.day

def int2date(idate):
    idate = int(idate)
    return datetime.date(idate//10000, idate%10000//100, idate%100)

def format_dt(datetime):
    return datetime.strftime('%Y-%m-%d %H:%M:%S')

def parse_dt(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def date2dt(date):
    return datetime.datetime.combine(date, datetime.time.min)

def parse_time(time):
    return datetime.time(*map(int, time.split(':')))

def parse_date(s):
    """
    >>> parse_date('2015-1-1')
    datetime.date(2015, 1, 1)
    """
    # don't use strptime, too long:
    # before: 69979    0.232    0.000    4.174    0.000 data.py:28(parse_date)
    # after:  73009    0.258    0.000    0.978    0.000 utils.py:38(parse_date)
    return datetime.date(*map(int, s.split('-')))

def format_date(date):
    return date.strftime('%Y-%m-%d')

def is_str(s):
    return type(s) in (str, six.text_type)

# convert anything to datetime.date
def convert_date(date):
    """
    >>> convert_date('2015-1-1')
    datetime.date(2015, 1, 1)

    >>> convert_date('2015-01-01 00:00:00')
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.datetime(2015, 1, 1))
    datetime.date(2015, 1, 1)

    >>> convert_date(datetime.date(2015, 1, 1))
    datetime.date(2015, 1, 1)
    """
    if is_str(date):
        if ':' in date:
            date = date[:10]
        return datetime.datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, datetime.datetime):
        return date.date()
    elif isinstance(date, datetime.date):
        return date
    raise Exception("date 必须是datetime.date, datetime.datetime或者如下格式的字符串:'2015-01-05'")
    pass

# convert anything to datetime.datetime
def convert_dt(dt):
    """
    >>> convert_dt(datetime.date(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt(datetime.datetime(2015, 1, 1))
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt('2015-1-1')
    datetime.datetime(2015, 1, 1, 0, 0)

    >>> convert_dt('2015-01-01 09:30:00')
    datetime.datetime(2015, 1, 1, 9, 30)

    >>> convert_dt(datetime.datetime(2015, 1, 1, 9, 30))
    datetime.datetime(2015, 1, 1, 9, 30)
    """
    if is_str(dt):
        if ':' in dt:
            return datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        else:
            return datetime.datetime.strptime(dt, '%Y-%m-%d')
    elif isinstance(dt, datetime.datetime):
        return dt
    elif isinstance(dt, datetime.date):
        return date2dt(dt)
    raise Exception("date 必须是datetime.date, datetime.datetime或者如下格式的字符串:'2015-01-05'")

def datetime2timestamp(adatetime):
    return int((adatetime - datetime.datetime.fromtimestamp(0)).total_seconds())

@lru_cache(None)
def get_all_trade_days_list():
    this_dir = os.path.dirname(__file__)
    with open(this_dir+'/../jqdata/data/all_trade_days.json') as f:
        days = json.load(f)
        days = [convert_date(d) for d in days]
        # np.array slice 不会copy
        return np.array(days)

@lru_cache(None)
def get_all_trade_days_set():
    return set(get_all_trade_days_list())

# 数组 k-v 颠倒变成 dict
def rindex_array(array):
    return {v:k for k, v in enumerate(array)}

@lru_cache(None)
def get_date_index_of_all():
    dates = get_all_trade_days_list()
    return rindex_array(dates)

def get_date_index_in_all(date):
    return get_date_index_of_all()[date]

def get_trade_days(start_date, end_date):
    a = get_all_trade_days_list()
    s = np.searchsorted(a, start_date)
    e = np.searchsorted(a, end_date, 'right')
    return a[s:e]

def get_first_day():
    return get_all_trade_days_list()[0]

def get_last_day():
    return get_all_trade_days_list()[-1]

def get_trade_day_before(date):
    l = get_all_trade_days_list()
    i = np.searchsorted(l, date)
    return l[i-1] if i >= 1 else None

def get_trade_day_not_after(date):
    l = get_all_trade_days_list()
    i = np.searchsorted(l, date, 'right')
    return l[i-1] if i >= 1 else None

def get_trade_day_after(date):
    l = get_all_trade_days_list()
    i = np.searchsorted(l, date, 'right')
    return l[i]

get_previous_trade_day = get_trade_day_before

def replace_time(current, time):
    return current.replace(hour=time.hour, minute=time.minute, second=time.second)

def is_trade_day(date):
    return date in get_all_trade_days_set()

def is_trade_time(code, dt):
    return TRADE_CALENDARS[get_exchange(code)].is_trade_time(dt)

def is_after_close_time(code, dt, exactly=False):
    return TRADE_CALENDARS[get_exchange(code)].is_after_close_time(dt, exactly)

def is_before_open_time(code, dt, exactly=False):
    return TRADE_CALENDARS[get_exchange(code)].is_before_open_time(dt, exactly)

def is_open_time(code, dt):
    return TRADE_CALENDARS[get_exchange(code)].is_open_time(dt)


def get_entrust_time(code, dt):
    '''
    获取订单申报时间
    '''
    return TRADE_CALENDARS[get_exchange(code)].get_entrust_time(dt)


# 日内日历
class IntradayCalendar(object):
    # before_open, trading, after_close
    def __init__(self, trading_periods, before_open, after_close):
        '''
        trading_period 左边是闭区间，右边是开区间，例如: [9:30, 11:30)
        '''
        self.before_open = before_open
        self.after_close = after_close
        self.trading_periods = trading_periods
        pass

    @cached_property
    def open_time(self):
        return self.trading_periods[0][0]

    @cached_property
    def close_time(self):
        return self.trading_periods[-1][-1]

    @cached_property
    def open_trigger_time(self):
        # hack, 为了在开盘前执行按天回测
        if self.open_time == Time(9, 30, 0):
            return Time(9, 27, 0)
        else:
            return self.open_time

    @cached_property
    def trading_minutes(self):
        minutes = []
        for s, e in self.trading_periods:
            minutes += xrange_time_forward(s, e, DELTA_MINUTE)
        return np.array(minutes)

    @cached_property
    def trading_ticks(self):
        ticks = []
        for s, e in self.trading_periods:
            ticks += xrange_time_forward(s, e, DELTA_TICK)
        return np.array(ticks)


    def is_trade_time(self, time):
        for s, e in self.trading_periods:
            if s <= time < e:
                return True
        return False

    def is_after_close_time(self, time, exactly=False):
        if exactly:
            return time == self.after_close
        else:
            return time >= self.trading_periods[-1][-1]

    def is_before_open_time(self, time, exactly=False):
        if exactly:
            return time == self.before_open
        else:
            return time < self.trading_periods[0][0]


# 交易所日历, 包括任一时间点的日历
class ExchangeCalendar(object):
    def __init__(self, calendar):
        self.calendar = calendar
        pass

    def get_intraday_calendar(self, date):
        for d, intraday_calendar in reversed(self.calendar):
            if d <= date:
                return intraday_calendar

    def get_trading_minutes(self, date):
        return self.get_intraday_calendar(date).trading_minutes

    def get_trading_ticks(self, date):
        return self.get_intraday_calendar(date).trading_ticks

    def get_entrust_time(self, dt):
        '''
        获取下一个还未开始的交易时间段
        '''
        # 在开盘之前下的单,将下单时间改到当天的9:30
        if self.is_before_open_time(dt):
            return dt.replace(hour=9, minute=30)
        # 在收盘之后下的单,将下单时间改到下一个交易日的9:30
        elif self.is_after_close_time(dt):
            next_day = get_trade_day_after(dt.date())
            return combine_dt(next_day, Time(9, 30))

        tm = dt.time()
        for s, e in self.get_intraday_calendar(dt.date()).trading_periods:
            if tm < s:
                return combine_dt(dt.date(), s)
            if s <= tm < e:
                return dt
        # 不可能执行到这里来
        return dt

    def is_trade_time(self, dt):
        return self.get_intraday_calendar(dt.date()).is_trade_time(dt.time())

    def is_after_close_time(self, dt, exactly=False):
        return self.get_intraday_calendar(dt.date()).is_after_close_time(dt.time(), exactly)

    def is_before_open_time(self, dt, exactly=False):
        return self.get_intraday_calendar(dt.date()).is_before_open_time(dt.time(), exactly)

    def is_open_time(self, dt):
        return self.get_intraday_calendar(dt.date()).open_time == dt.time()

TRADE_CALENDARS = {
    'XSHG': ExchangeCalendar((
        (Date(1970, 1, 1), IntradayCalendar(
                trading_periods=((Time(9, 30, 0), Time(11, 30, 0)), (Time(13, 0, 0), Time(15, 0, 0))),
                before_open=Time(9, 20, 0),
                after_close=Time(15, 10, 0),
                )),
    )),
    'CCFX': ExchangeCalendar((
        (Date(1970, 1, 1), IntradayCalendar(
                trading_periods=((Time(9, 15, 0), Time(11, 30, 0)), (Time(13, 0, 0), Time(15, 15, 0))),
                before_open=Time(9, 0, 0),
                after_close=Time(15, 30, 0),
                )),
        (Date(2016, 1, 1), IntradayCalendar(
                trading_periods=((Time(9, 30, 0), Time(11, 30, 0)), (Time(13, 0, 0), Time(15, 0, 0))),
                before_open=Time(9, 20, 0),
                after_close=Time(15, 10, 0),
                )),
    ))
}
TRADE_CALENDARS['XSHE'] = TRADE_CALENDARS['XSHG']

# 实现 xrange, 支持任何可相加的东西, 条件: step > 0
def xrange_forward(start, end, step):
    x = start
    while x < end:
        yield x
        x += step

# include end, 针对 datetime.time 实现 xrange, 条件: step > 0
def xrange_time_forward(start, end, step):
    start = datetime.datetime.combine(datetime.date(1970, 1, 1), start)
    end = datetime.datetime.combine(datetime.date(1970, 1, 1), end)
    x = start
    while x < end:
        yield x.time()
        x += step

# 得到交易所代号
def get_exchange(code):
    return code.split('.')[-1]

def get_minutes_of_day(code, date):
    return TRADE_CALENDARS[get_exchange(code)].get_trading_minutes(date)

def get_minute_index_in_day(code, dt):
    return np.searchsorted(get_minutes_of_day(code, dt.date()), dt.time())

def get_ticks_of_day(code, date):
    return TRADE_CALENDARS[get_exchange(code)].get_trading_ticks(date)

def np_searchsorted_before(sortedarray, element):
    i = np.searchsorted(sortedarray, element)
    return sortedarray[i-1] if i > 0 else None

def get_last_minute_dt(code, date):
    t = get_minutes_of_day(code, date)[-1]
    return combine_dt(date, t)

def get_previous_trade_last_minute_dt(code, date):
    date = get_previous_trade_day(date)
    if date is not None:
        return get_last_minute_dt(code, date)

def get_previous_trade_minute(code, dt):
    dt = dt.replace(second=0, microsecond=0)
    minutes = get_minutes_of_day(code, dt.date())
    t = np_searchsorted_before(minutes, dt.time())
    if t is None:
        return get_previous_trade_last_minute_dt(code, dt.date())
    else:
        return combine_dt(dt.date(), t)

def get_previous_trade_tick(code, dt):
    date = dt.date()
    ticks = get_ticks_of_day(code, date)
    t = np_searchsorted_before(ticks, dt.time())
    if t is None:
        date = get_previous_trade_day(date)
        if date is not None:
            t = get_ticks_of_day(code, date)[-1]
            return combine_dt(date, t)
    else:
        return combine_dt(date, t)

def get_bar_dt(code, dt, frequency):
    if frequency == 'day':
        # 收盘后, bar 就是当天
        if not is_after_close_time(code, dt):
            dt = date2dt(get_trade_day_before(dt.date()))
    elif frequency == 'minute':
        dt = get_previous_trade_minute(code, dt)
    elif frequency == 'tick':
        dt = get_previous_trade_tick(code, dt)
    else:
        raise Exception("wrong frequency=%s" % frequency)
    return dt

def get_month_week_trade_days(start_date, end_date):
    dates = get_trade_days(start_date, end_date)
    months = collections.defaultdict(list)
    weeks = collections.defaultdict(list)
    for date in dates:
        y = date.year
        m = date.month
        d = date.day
        ym = (y, m)
        months[ym].append(date)

        y2, yearweek, weekday = date.isocalendar()
        yw = (y2, yearweek)
        weeks[yw].append(date)
    return months, weeks

def clear_all_cache():
    pass

POWER_RATE_MIN_DATETIME = None
TRADE_MIN_DATETIME = None
TRADE_MAX_DATETIME = None

def reset_max_min_datetime():
    global POWER_RATE_MIN_DATETIME, TRADE_MIN_DATETIME, TRADE_MAX_DATETIME
    POWER_RATE_MIN_DATETIME = datetime.datetime(2005, 1, 4)
    TRADE_MIN_DATETIME = datetime.datetime(2005, 1, 4)
    TRADE_MAX_DATETIME = datetime.datetime(9999, 1, 1)
    pass

reset_max_min_datetime()

FREQUENCY_DELTAS = {
        'day': DELTA_DAY,
        'minute': DELTA_MINUTE,
        'tick': DELTA_TICK,
    }

EPOC_DATE = datetime.date(1970, 1, 1)

def add_time(time, delta):
    dt = combine_dt(EPOC_DATE, time)
    dt += delta
    return dt.time()

class RefreshRate(Enum):
    day = "day"
    minute = "minute"
    tick = 'tick'

# decorator 模块的 decorator 函数的简单实现, 兼容 cython
def decorator(func):
    @wraps(func)
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return func(f, *args, **kwargs)
        return wrapper
    return dec

def get_total_memory():
    import os
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
        return mem_bytes
    except ValueError:
        # MacOS don't support this
        return 1024*1024*1024
