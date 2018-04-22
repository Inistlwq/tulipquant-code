# coding: utf-8

import datetime
import numpy as np
import pickle

from fastcache import clru_cache as lru_cache
import jqdata
from jqdata.utils.datetime_utils import to_timestamp, vec2datetime, parse_date, \
    vec2timestamp, to_date

__all__ = [
    'CalendarStore',
    'get_calendar_store'
]

def datetime_range(start_dt, end_dt, include_start=True, include_end=False):
    if not include_start:
        start_dt += datetime.timedelta(minutes=1)
    data = []
    while start_dt < end_dt:
        data.append(start_dt)
        start_dt += datetime.timedelta(minutes=1)
    if include_end:
        data.append(end_dt)
    return data

def time_range(start, end, include_start=True, include_end=False):
    start_dt = datetime.datetime.combine(datetime.date(1970, 1, 1), start)
    if not include_start:
        start_dt += datetime.timedelta(minutes=1)
    if end >= start:
        end_dt = datetime.datetime.combine(datetime.date(1970, 1, 1), end)
    else:
        end_dt = datetime.datetime.combine(datetime.date(1970, 1, 2), end)
    data = []
    while start_dt < end_dt:
        data.append(to_timestamp(start_dt))
        start_dt += datetime.timedelta(minutes=1)
    if include_end:
        data.append(to_timestamp(end_dt))
    return data


_stock_minutes = np.array(time_range(datetime.time(9, 31), datetime.time(11, 31)) +
                          time_range(datetime.time(13, 1), datetime.time(15, 1)))
_future_minutes = np.array(time_range(datetime.time(9, 16), datetime.time(11, 31)) +
                           time_range(datetime.time(13, 1), datetime.time(15, 16)))

# 商品期货交易时间段
from jqdata.models.security import commodity_trade_time_point

commodity_trade_time_table = {}
for time_tag in commodity_trade_time_point:
    commodity_trade_time_table[time_tag] = np.array(time_range(
        commodity_trade_time_point[time_tag][0],
        commodity_trade_time_point[time_tag][1],
        include_start=False,
        include_end=True,
    ))


# 股指期货从2016.1.1之后开始交易时间和股票交易时间一致。
# 在此之前交易时间是[9:15, 11:30] 和 [13:00, 15:15]
_boundary_date = datetime.date(2016, 1, 1)
_boundary_ts = to_timestamp(_boundary_date)


class BaseCalendar(object):

    def __init__(self, dates, tss):
        '''
        :param dates: np.array([datetime.date])
        :param tss: np.array([timestamp])
        '''
        self.dates = dates
        self.tss = tss

    def first_date(self):
        return self.dates[0]

    def first_date_timestamp(self):
        return self.tss[0]

    def last_date(self):
        return self.dates[-1]

    def get_current_trade_date(self, current_dt):
        pass

    def get_previous_trade_date(self, date):
        pass

    def get_next_trade_date(self, date):
        pass

    def is_trade_date(self, date):
        '''
        :param date: datetime.date or timestamp
        :return: True or False
        '''
        if isinstance(date, datetime.date):
            t = self.dates
        else:
            t = self.tss
        i = t.searchsorted(date)
        if i < len(t):
            return t[i] == date
        return False

    def is_trade_time(self, dt):
        '''
        是否交易时间
        '''
        raise NotImplementedError

        date = dt.date()
        time = dt.time()

        if security.is_commodity_futures():
            if time.hour < 21:
                periods = self.get_trade_periods(security, date)
            else:
                periods = self.get_trade_periods(security, date + datetime.timedelta(days=1))
            for period in periods:
                if period[0] <= dt <= period[1]:
                    return True
            return False

        if not self.is_trade_date(security, date):
            return False
        if security.is_index_futures() and date < _boundary_date:
            arr = _future_minutes
        else:
            arr = _stock_minutes

        time_ts = to_timestamp(time)
        i = arr.searchsorted(time_ts)
        if i < len(arr):
            return arr[i] == time_ts
        return False

    def is_first_minute(self, dt):
        pass

    def get_next_trade_dt(self, dt):
        '''
        返回下一个交易时间
        
        :param dt: 
        :return: 
        '''
        pass


    def get_open_dt(self, date):
        pass

    def get_close_dt(self, date):
        pass

    def get_trade_periods(self, date):
        pass





# TODO: 分交易所实现Calendar

class SseCalendar(BaseCalendar):
    # 上交所日历
    pass

# 深交所日历
SzseCalendar = SseCalendar

class CffexCalendar(BaseCalendar):
    # 中国金融期货交易所

    # 股指期货从2016.1.1之后开始交易时间和股票交易时间一致。
    # 在此之前交易时间是[9:15, 11:30] 和 [13:00, 15:15]

    _boundary_date = datetime.date(2016, 1, 1)
    _boundary_ts = to_timestamp(_boundary_date)

    def __init__(self, dates):
        pass

    def is_trade_time(self, dt):
        pass



# 上海期货交易所日历
class ShfeCalendar(BaseCalendar):

    def __init__(self, dates):
        pass

# 郑州商品交易所
CzceCalendar = ShfeCalendar

# 大连商品交易所
DceCalendar = ShfeCalendar








class CalendarStore(object):

    def __init__(self, f):
        with open(f, "rb") as fp:
            data = pickle.load(fp)
            self._stock_dates = np.array([parse_date(i) for i in data['stock']])
            self._stock_dates.flags.writeable = False
            self._stock_tss = vec2timestamp(self._stock_dates)
            self._stock_tss.flags.writeable = False
            month_dates = []
            for i in range(0, len(self._stock_dates) - 1):
                if self._stock_dates[i].month != self._stock_dates[i + 1].month:
                    month_dates.append(self._stock_dates[i])
            self._stock_month_dates = np.array(month_dates)
            self._commodity_dates = np.array([parse_date(i) for i in data['commodity']])
            self._commodity_dates.flags.writeable = False
            self._commodity_tss = vec2timestamp(self._commodity_dates)
            self._commodity_tss.flags.writeable = False


    @staticmethod
    def instance():
        if not hasattr(CalendarStore, '_instance'):
            cfg = jqdata.get_config()
            CalendarStore._instance = CalendarStore(cfg.get_calendar_pk())
        return CalendarStore._instance

    def get_exchange_calendar_cls(self, security):
        return None


    @property
    def first_date_timestamp(self, security=None):
        '''
        第一个交易日的时间戳
        '''
        if security and security.is_commodity_futures():
            return self._commodity_tss[0]
        else:
            return self._stock_tss[0]


    @property
    def first_day(self, security=None):
        '''第一个交易日：datetime.date'''
        if security and security.is_commodity_futures():
            return self._commodity_dates[0]
        else:
            return self._stock_dates[0]


    @property
    def last_day(self, security=None):
        '''最后一个交易日：datetime.date'''
        if security and security.is_commodity_futures():
            return self._commodity_dates[-1]
        else:
            return self._stock_dates[-1]


    def _get_commodity_trade_date(self, current_dt):
        # 商品期货的夜盘（21点开始）算做下一个交易日的行情。
        date = current_dt.date()
        if current_dt.hour >= 21:
            i = self._commodity_dates.searchsorted(date, side='right')
            return self._commodity_dates[i]
        elif current_dt.hour < 8:
            # 当天凌晨的夜盘
            i = self._commodity_dates.searchsorted(date, side='right')
            if i - 1 >= 0 and self._commodity_dates[i - 1] == date:
                return self._commodity_dates[i - 1]
            # 可能是周五的夜盘，交易日算下周一
            return self._commodity_dates[i]
        else:
            i = self._commodity_dates.searchsorted(date)
            # assert self._commodity_dates[i] == date
            # return date
            return self._commodity_dates[i]

    def get_current_trade_date(self, security, current_dt):
        '''
        current_dt 必须是交易时间段里的某一分钟。
        :param security:
        :param current_dt:
        :return:
        '''
        if security.is_commodity_futures():
            return self._get_commodity_trade_date(current_dt)
        else:
            # i = self._stock_dates.searchsorted(date, side='right')
            # if i == 0:
            #     return self._stock_dates[i]
            # if self._stock_dates[i-1] == date:
            #     return
            # if i - 1 >= 0 and self._stock_dates
            #  # assert self._stock_dates[i] == date
            return current_dt.date()


    def _previous_date(self, arr, date):
        i = arr.searchsorted(date)
        if i > 0:
            return arr[i-1]
        return arr[0] - datetime.timedelta(days=1)


    def get_previous_trade_date(self, security, date):
        if security is not None and security.is_commodity_futures():
            t = self._commodity_dates
        else:
            t = self._stock_dates
        return self._previous_date(t, date)

    def get_next_trade_date(self, security, date):
        if security.is_commodity_futures():
            t = self._commodity_dates
        else:
            t = self._stock_dates
        i = t.searchsorted(date, side='right')
        if i < len(t):
            if t[i] == date:
                if i + 1 < len(t):
                    return t[i + 1]
            else:
                return t[i]
        return t[-1] + datetime.timedelta(days=1)

    def is_trade_date(self, security, date):
        '''
        :param security:
        :param date: datetime.date 或者 int
        :return:
        '''
        if security and security.is_commodity_futures():
            if isinstance(date, datetime.date):
                t = self._commodity_dates
            else:
                t = self._commodity_tss
        else:
            if isinstance(date, datetime.date):
                t = self._stock_dates
            else:
                t = self._stock_tss
        i = t.searchsorted(date)
        if i < len(t):
            return t[i] == date
        return False

    def is_trade_time(self, security, dt):
        '''
        是否交易时间
        '''
        date = dt.date()
        time = dt.time()

        if security.is_commodity_futures():
            if time.hour < 21:
                periods = self.get_trade_periods(security, date)
            else:
                periods = self.get_trade_periods(security, date + datetime.timedelta(days=1))
            for period in periods:
                if period[0] <= dt <= period[1]:
                    return True
            return False

        if not self.is_trade_date(security, date):
            return False
        if security.is_index_futures() and date < _boundary_date:
            arr = _future_minutes
        else:
            arr = _stock_minutes

        time_ts = to_timestamp(time)
        i = arr.searchsorted(time_ts)
        if i < len(arr):
            return arr[i] == time_ts
        return False

    def is_first_minute(self, security, dt):
        '''是否当天的第一个交易分钟'''
        if security.is_commodity_futures():
            cur_date = self.get_current_trade_date(security, dt)
            trade_time = security.get_trade_time(cur_date)
            if trade_time is None:
                return False
            tm = trade_time.minute_periods[0][0]
            dt -= datetime.timedelta(minutes=1)
            return dt.time() == tm
        if security.is_futures() and dt.date() < _boundary_date:
            return to_timestamp(dt.time()) == _future_minutes[0]
        else:
            return to_timestamp(dt.time()) == _stock_minutes[0]


    def get_open_dt(self, security, date):
        if security.is_commodity_futures():
            trade_time = security.get_trade_time(date)
            assert trade_time, '必须有数据。'
            tm = trade_time.minute_periods[0][0]
            if tm.hour >= 21:
                previous_date = self.get_previous_trade_date(security, date)
                return datetime.datetime.combine(previous_date, tm)
            else:
                return datetime.datetime.combine(date, tm)
        if security.is_index_futures() and date < _boundary_date:
            return datetime.datetime.combine(date, datetime.time(9, 15))
        else:
            return datetime.datetime.combine(date, datetime.time(9, 30))

    def get_close_dt(self, security, date):
        '''
        :param security:
        :param date:
        :return:  返回标的在date这一天的收盘时间点
        '''
        if security.is_commodity_futures():
            trade_time = security.get_trade_time(date)
            assert trade_time, '必须有数据'
            return datetime.datetime.combine(date, trade_time.minute_periods[-1][1])
        if security.is_index_futures() and date < _boundary_date:
            return datetime.datetime.combine(date, datetime.time(15, 15))
        else:
            return datetime.datetime.combine(date, datetime.time(15, 0))

    def get_all_trade_days(self, security=None):
        '''获取所有交易日期'''
        if security and security.is_commodity_futures():
            return self._commodity_dates
        else:
            return self._stock_dates

    def get_trade_days_by_count(self, end_dt, count):
        '''返回包括end_dt最近的count个交易日'''
        if isinstance(end_dt, datetime.datetime):
            end_dt = end_dt.date()
        end_idx = self._stock_dates.searchsorted(end_dt, side='right')
        v = self._stock_dates[:end_idx]
        return v[-count:]

    def get_trade_days_between(self, start_dt, end_dt):
        '''返回 [start_dt, end_dt]之间的交易日'''
        if isinstance(start_dt, datetime.datetime):
            start_dt = start_dt.date()
        if isinstance(end_dt, datetime.datetime):
            end_dt = end_dt.date()

        start_idx = self._stock_dates.searchsorted(start_dt)
        end_idx = self._stock_dates.searchsorted(end_dt, side='right')
        return self._stock_dates[start_idx:end_idx]

    def get_trade_periods(self, security, date):
        '''
        返回交易时间段 [(s0, e0), (s1, e1)]
        '''
        if not self.is_trade_date(security, date):
            return []

        # 商品期货
        if security.is_commodity_futures():
            trade_time = security.get_trade_time(date)
            if not trade_time:
                return []
            ret = []
            for minute_period in trade_time.minute_periods:
                if minute_period[0].hour >= 21:
                    previous_date = self.get_previous_trade_date(security, date)
                    st = datetime.datetime.combine(previous_date, minute_period[0])
                    if minute_period[1].hour >= 21:
                        et = datetime.datetime.combine(previous_date, minute_period[1])
                    else:
                        et = datetime.datetime.combine(date, minute_period[1])
                else:
                    st = datetime.datetime.combine(date, minute_period[0])
                    et = datetime.datetime.combine(date, minute_period[1])
                ret.append([st, et])
            return ret
        # 股指期货（2016.1.1之前) 和国债期货
        elif security.is_index_futures() and (
                    to_timestamp(date) < _boundary_ts or security.is_treasury_future()):
            return [(datetime.datetime.combine(date, datetime.time(9, 15)),
                     datetime.datetime.combine(date, datetime.time(11, 30))),
                    (datetime.datetime.combine(date, datetime.time(13, 0)),
                     datetime.datetime.combine(date, datetime.time(15, 15)))]
        else:
            return [(datetime.datetime.combine(date, datetime.time(9, 30)),
                     datetime.datetime.combine(date, datetime.time(11, 30))),
                    (datetime.datetime.combine(date, datetime.time(13, 0)),
                     datetime.datetime.combine(date, datetime.time(15, 0)))]

    def get_night_trade_minutes(self, date):
        # 返回商品期货交易日的夜盘时间
        d = self._previous_date(self._commodity_dates, date)
        st = datetime.datetime.combine(d, datetime.time(21, 0))
        et = datetime.datetime.combine(d + datetime.timedelta(days=1), datetime.time(3, 0))
        return datetime_range(st, et, include_end=True)

    def get_day_trade_minutes(self, date):
        # 返回商品期货这个交易日白天的交易分钟列表
        st = datetime.datetime.combine(date, datetime.time(9, 0))
        et = datetime.datetime.combine(date, datetime.time(15, 00))
        return datetime_range(st, et, include_end=True)


    def get_trade_minutes(self, security, date):
        '''
        :param security: 标的
        :param date: 日期
        :return: 返回标的所在交易所在date日期的分钟列表（datetime.datetime)
        '''
        if not self.is_trade_date(security, date):
            return np.array([])
        ts = to_timestamp(date)
        if security.is_commodity_futures():
            trade_time = security.get_trade_time(date)
            if not trade_time:
                # print(security, date)
                return np.array([])
            ret = []
            for minute_period in trade_time.minute_periods:
                key = '%02d:%02d~%02d:%02d' % (minute_period[0].hour, minute_period[0].minute,
                                               minute_period[1].hour, minute_period[1].minute)
                ts_table = commodity_trade_time_table[key]
                if minute_period[0].hour >= 21:
                    previous_date = self.get_previous_trade_date(security, date)
                    previous_ts = to_timestamp(previous_date)
                    ret.append(vec2datetime(previous_ts + ts_table))
                else:
                    ret.append(vec2datetime(ts + ts_table))
            return np.concatenate(ret)
        if security.is_futures() and ts < _boundary_ts:
            return vec2datetime(ts + _future_minutes)
        else:
            return vec2datetime(ts + _stock_minutes)

    def get_trade_minutes_ts(self, security, date_ts):
        '''
        :param security: security
        :param date_ts: 一个 datetime 的时间戳
        :return: 首先找到 date_ts的交易日，然后返回这个交易日交易分钟列表（时间戳）
        '''
        if not self.is_trade_date(security, date_ts):
            return np.array([])

        if security.is_commodity_futures():
            assert date_ts % 86400 == 0
            _minutes = self.get_trade_minutes(security, to_date(date_ts))
            return vec2timestamp(_minutes)
        elif security.is_index_futures() and date_ts < _boundary_ts:
            return date_ts + _future_minutes
        else:
            return date_ts + _stock_minutes

    def get_trade_times_ts(self, security, date):
        '''
        获取标的在某一天的交易分钟列表。
        '''
        if not self.is_trade_date(security, date):
            return np.array([])
        if security.is_commodity_futures():
            trade_time = security.get_trade_time(date)
            ret = []
            for minute_period in trade_time.minute_periods:
                key = '%02d:%02d~%02d:%02d' % (minute_period[0].hour, minute_period[0].minute,
                                               minute_period[1].hour, minute_period[1].minute)
                ts_table = commodity_trade_time_table[key]
                ret.append(ts_table % 86400)
            return np.concatenate(ret)
        elif security.is_index_futures() and date < _boundary_date:
            return _future_minutes
        else:
            return _stock_minutes

    def get_trade_minutes_by_enddt(self, security, end_dt, start_dt=None, count=None, include_now=True):
        '''
        获取交易所的交易时间（分钟粒度）
        input: count
           返回 截止于 end_dt 的count个交易分钟。
        input：start_dt
           返回 start_dt 和 end_dt 之间的交易分钟列表，包括start_dt。
        
        include_now 表示返回结果是否包含 end_dt。
        '''
        assert start_dt or count
        end_ts = to_timestamp(end_dt)
        if not include_now:
            end_ts -= 60 # 向前推移1分钟
            end_dt = end_dt - datetime.timedelta(minutes=1)

        if security is not None and security.is_commodity_futures():
            tss_table = self._commodity_tss
        else:
            tss_table = self._stock_tss

        trade_date = self.get_current_trade_date(security, end_dt)
        stop = np.searchsorted(tss_table, to_timestamp(trade_date), 'right')
        if count:
            data = []
            i = stop - 1
            total_count = 0
            while i >= 0:
                minutes = self.get_trade_minutes_ts(security, tss_table[i])
                minutes = minutes[(minutes <= end_ts)]
                data.insert(0, minutes)
                total_count += len(minutes)
                i -= 1
                if total_count >= count:
                    break
            data = np.concatenate(data)
            if len(data) > count:
                return data[-count:]
            return data
        else:
            start = np.searchsorted(tss_table, to_timestamp(start_dt.date()))
            if start >= stop:
                return []
            data = []
            for i in range(start, stop):
                minutes = self.get_trade_minutes_ts(security, tss_table[i])
                data.append(minutes)
            data = np.concatenate(data)
            return data[(data >= to_timestamp(start_dt)) & (data <= end_ts)]

    def get_trade_days_by_enddt(self, security, end_date, start_date=None, count=None, include_now=True):
        '''
        返回时间戳
        '''
        assert start_date or count
        end_ts = to_timestamp(end_date)
        if not include_now:
            end_ts -= 86400 # 向前推移一天
        stop = np.searchsorted(self._stock_tss, end_ts, 'right')
        if count:
            start = max(stop - count, 0)
        else:
            start = np.searchsorted(self._stock_tss, to_timestamp(start_date))

        if stop > start:
            return self._stock_tss[start:stop]
        return np.empty(0)

    def get_monthly_days_between(self, start_dt, end_dt):
        '''返回 [start_dt, end_dt]之间的每个月最后一个交易日'''
        if isinstance(start_dt, datetime.datetime):
            start_dt = start_dt.date()
        if isinstance(end_dt, datetime.datetime):
            end_dt = end_dt.date()

        start_idx = self._stock_month_dates.searchsorted(start_dt)
        end_idx = self._stock_month_dates.searchsorted(end_dt, side='right')
        return self._stock_month_dates[start_idx:end_idx]

@lru_cache(None)
def get_calendar_store():
    return CalendarStore.instance()
