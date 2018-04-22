#!/usr/bin/env python
# coding:utf-8

import numpy as np
import datetime

from jqdata.stores.tick_store import convert_int_to_datetime_with_ms

NAN_DT = datetime.datetime(2200, 1, 1)
class Tick(object):
    def __init__(self, security, tick):
        self._security = security
        self._tick = tick

    @property
    def code(self):
        return self._security.code

    @property
    def datetime(self):
        try:
            r = self._tick['time']
            if isinstance(r,(int, float)):
                return convert_int_to_datetime_with_ms(r)
            return r
        except:
            return NAN_DT

    @property
    def current(self):
        try:
            return self._tick['current']
        except:
            return np.nan

    @property
    def high(self):
        try:
            return self._tick['high']
        except:
            return np.nan

    @property
    def low(self):
        try:
            return self._tick['low']
        except:
            return np.nan

    @property
    def volume(self):
        try:
            return self._tick['volume']
        except:
            return np.nan

    @property
    def money(self):
        try:
            return self._tick['money']
        except:
            return np.nan

    @property
    def position(self):
        try:
            return self._tick['position']
        except:
            return np.nan

    @property
    def b1_p(self):
        try:
            return self._tick['b1_p']
        except:
            return np.nan

    @property
    def b2_p(self):
        try:
            return self._tick['b2_p']
        except:
            return np.nan

    @property
    def b3_p(self):
        try:
            return self._tick['b3_p']
        except:
            return np.nan

    @property
    def b4_p(self):
        try:
            return self._tick['b4_p']
        except:
            return np.nan

    @property
    def b5_p(self):
        try:
            return self._tick['b5_p']
        except:
            return np.nan

    @property
    def b1_v(self):
        try:
            return self._tick['b1_v']
        except:
            return np.nan

    @property
    def b2_v(self):
        try:
            return self._tick['b2_v']
        except:
            return np.nan

    @property
    def b3_v(self):
        try:
            return self._tick['b3_v']
        except:
            return np.nan

    @property
    def b4_v(self):
        try:
            return self._tick['b4_v']
        except:
            return np.nan

    @property
    def b5_v(self):
        try:
            return self._tick['b5_v']
        except:
            return np.nan

    @property
    def a1_p(self):
        try:
            return self._tick['a1_p']
        except:
            return np.nan

    @property
    def a2_p(self):
        try:
            return self._tick['a2_p']
        except:
            return np.nan

    @property
    def a3_p(self):
        try:
            return self._tick['a3_p']
        except:
            return np.nan

    @property
    def a4_p(self):
        try:
            return self._tick['a4_p']
        except:
            return np.nan

    @property
    def a5_p(self):
        try:
            return self._tick['a5_p']
        except:
            return np.nan

    @property
    def a1_v(self):
        try:
            return self._tick['a1_v']
        except:
            return np.nan

    @property
    def a2_v(self):
        try:
            return self._tick['a2_v']
        except:
            return np.nan

    @property
    def a3_v(self):
        try:
            return self._tick['a3_v']
        except:
            return np.nan

    @property
    def a4_v(self):
        try:
            return self._tick['a4_v']
        except:
            return np.nan

    @property
    def a5_v(self):
        try:
            return self._tick['a5_v']
        except:
            return np.nan

    def __repr__(self):
        items = []
        items.append(('code', self.code))
        for name in self._security.tick_column_names:
            items.append((name, getattr(self, name)))
        return "Tick({0})".format(', '.join('{0}: {1}'.format(k, v) for k, v in items))

    def __getitem__(self, key):
        return getattr(self, key)
