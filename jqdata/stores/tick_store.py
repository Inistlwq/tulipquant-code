# -*- coding: utf-8 -*-
#

import os
import datetime

import bcolz
import numpy as np
from fastcache import lru_cache

from .calendar_store import CalendarStore
from .bcolz_utils import retry_bcolz_open

TEN13 = 10 ** 13
TEN11 = 10 ** 11
TEN9 = 10 ** 9
TEN7 = 10 ** 7
TEN5 = 10 ** 5
TEN3 = 10 ** 3

def convert_int_to_datetime_with_ms(dt_int):
    dt_int = int(dt_int)
    year, r = divmod(dt_int, 10000000000000)
    month, r = divmod(r, 100000000000)
    day, r = divmod(r, 1000000000)
    hour, r = divmod(r, 10000000)
    minute, r = divmod(r, 100000)
    second, millisecond = divmod(r, 1000)
    return datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)

vec_convert_dt = np.vectorize(convert_int_to_datetime_with_ms, otypes=[datetime.datetime])

class _TicksTable(object):

    def __init__(self, array, trade_days, line_maps, security):
        self.array = array
        self.len = len(array)
        self.trade_days = trade_days
        self.line_maps = line_maps
        self.security = security

        if self.len > 0:
            self.first_time = self.array[0][0]
            self.last_time = self.array[-1][0]
        else:
            self.first_time = None
            self.last_time = None


    def find_great_or_equal(self, dt):
        '''
        找到第一个tick time 大于或者等于dt的位置
        
        返回index, 等价于
        time.searchsorted(dt, side='right')
        
        return [0, self.len]
        
        由于加载time列占用内存太大，所以使用下面的方式。
        '''

        cal = CalendarStore.instance()
        tick_time = dt.year * TEN13 + dt.month * TEN11 + \
                    dt.day * TEN9 + dt.hour * TEN7 + \
                    dt.minute * TEN5 + dt.second * TEN3 + \
                    int(dt.microsecond / 1000)
        if self.len == 0:
            return 0

        if tick_time < self.first_time:
            return 0
        if tick_time > self.last_time:
            return self.len

        trade_day = int(cal.get_current_trade_date(self.security, dt).strftime("%Y%m%d"))
        day_index = self.trade_days.searchsorted(trade_day, side='right')
        if day_index > 0:
            real_trade_day = self.trade_days[day_index-1]
        else:
            real_trade_day = self.trade_days[day_index]
        start, end = self.line_maps[str(real_trade_day)]
        time_arr = self.array[start:end]['time']
        i = time_arr.searchsorted(tick_time, side='right')
        if i > 0 and time_arr[i-1] == tick_time:
            i = i - 1
        return start + i

    def find_less_or_equal(self, dt):
        '''
        找到第一个tick time 小于或者等于dt的位置
        
        返回index, 等价于
        time.searchsorted(dt, side='left')
        
        return [-1, self.len-1]
        
        由于加载time列占用内存太大，所以使用下面的方式。
        '''
        tick_time = dt.year * TEN13 + dt.month * TEN11 + \
                    dt.day * TEN9 + dt.hour * TEN7 + \
                    dt.minute * TEN5 + dt.second * TEN3 + \
                    int(dt.microsecond / 1000)
        if self.len == 0:
            return 0

        if tick_time < self.first_time:
            return -1
        if tick_time > self.last_time:
            return self.len - 1

        cal = CalendarStore.instance()
        trade_day = int(cal.get_current_trade_date(self.security, dt).strftime("%Y%m%d"))
        day_index = self.trade_days.searchsorted(trade_day)
        if day_index < len(self.trade_days):
            real_trade_day = self.trade_days[day_index]
        else:
            real_trade_day = self.trade_days[day_index-1]

        start, end = self.line_maps[str(real_trade_day)]
        time_arr = self.array[start:end]['time']

        i = time_arr.searchsorted(tick_time)
        if i < len(time_arr) and time_arr[i] == tick_time:
            return start + i
        else:
            return start + i - 1


class _TickDataSource(object):

    '''
    bcolz tick 数据源。
    '''

    def __init__(self, security, table, current_dt):
        '''
        current_dt 表示订阅的时间点。
        '''
        self.security = security
        self.ctable = table.array
        self.trade_days = table.trade_days
        self.line_maps = table.line_maps
        self.current_dt = current_dt

        self.names = self.ctable.dtype.names

        self._init_cache(self.current_dt)

    def _init_cache(self, dt):
        cal = CalendarStore.instance()
        trade_date = int(cal.get_current_trade_date(self.security, dt).strftime("%Y%m%d"))
        date_index = self.trade_days.searchsorted(trade_date, side='right')
        if date_index > 0 and self.trade_days[date_index - 1] == trade_date:
            date_index -= 1

        self._cache_date_index = date_index
        self._load_cache(date_index)
        if self._cache_array is not None:
            self._cache_current_index = self._cache_array['time'].searchsorted(dt)
        else:
            self._cache_current_index = 0

    def _load_cache(self, date_index):
        '''
        从 bcolz.ctable 中加载一天的数据
        '''
        price_fields = ('current', 'high', 'low',
                        'a1_p', 'a2_p', 'a3_p', 'a4_p', 'a5_p',
                        'b1_p', 'b2_p', 'b3_p', 'b4_p', 'b5_p')

        if date_index < len(self.trade_days):
            trade_date = str(self.trade_days[date_index])
            start, end = self.line_maps[trade_date]
            data = self.ctable[start:end]
            cols = []
            dtypes = []
            for f in self.names:
                if f in price_fields:
                    cols.append(data[f]/10000.)
                elif f == 'time':
                    cols.append(vec_convert_dt(data[f]))
                else:
                    cols.append(data[f])
                dtypes.append((str(f), cols[-1].dtype))
            self._cache_array = np.rec.fromarrays(cols, dtype=np.dtype(dtypes)).view(np.ndarray)
        else:
            self._cache_array = None

    def __str__(self):
        return 'TickDataSource(current_dt=%s, security=%s)' % (self.current_dt, self.security.code)

    def __repr__(self):
        return self.__str__()

    def get_next_time(self):
        if self._cache_array is not None:
            if self._cache_current_index >= len(self._cache_array):
                self.forward()
        if self._cache_array is not None:
            return self._cache_array['time'][self._cache_current_index]
        return None

    def get_next_tick(self):
        if self._cache_array is not None:
            if self._cache_current_index >= len(self._cache_array):
                self.forward()
        if self._cache_array is not None:
            obj = self._cache_array[self._cache_current_index]
            return dict(zip(self.names, obj.tolist()))
        return None

    def forward(self):
        if self._cache_array is not None:
            if self._cache_current_index < len(self._cache_array) -1 :
                self._cache_current_index += 1
            else:
                # 加载下一天的缓存
                self._cache_date_index += 1
                self._load_cache(self._cache_date_index)
                self._cache_current_index = 0


class TickStore(object):

    def __init__(self, bcolz_base_path):
        self.bcolz_base_path = os.path.expanduser(bcolz_base_path)

    @staticmethod
    def instance():
        if not hasattr(TickStore, "_instance"):
            import jqdata
            TickStore._instance = TickStore(jqdata.get_config().get_tick_path())
        return TickStore._instance

    def get_bcolz_path(self, security):
        if security.is_futures():
            breed = security.code[:-9].upper()
            p = os.path.join(self.bcolz_base_path, 'futures', breed, security.code)
        elif security.is_stock():
            subdir = security.code.split('.')[0][-2:]
            p = os.path.join(self.bcolz_base_path, 'stocks', subdir, security.code)
        else:
            raise ValueError("tick数据只支持期货")
        return p


    @lru_cache(None)
    def get_table(self, security):
        p = self.get_bcolz_path(security)
        if not os.path.exists(p) or not os.path.exists(os.path.join(p, '__attrs__')):
            return None
           # raise ValueError("找不到%s tick 数据" % security.code)
        ct = retry_bcolz_open(p)
        attrs = ct.attrs.getall()
        trade_days = np.array([int(i) for i in sorted(attrs.keys())])
        return _TicksTable(ct, trade_days, attrs, security)

    def get_ticks_by_date(self, security, date):
        ct = self.get_table(security)
        date = date.strftime("%Y%m%d")
        line_map = ct.line_maps.get(date, None)
        if line_map:
            return ct.array[line_map[0]:line_map[1]]
        return None

    def get_tick_data_source(self, security, dt):
        '''
        从dt 这个时间点开始订阅security标的的tick事件。
        '''
        table = self.get_table(security)
        if table is None:
            return None
        return _TickDataSource(security, table, dt)

def get_tick_store():
    return TickStore.instance()
