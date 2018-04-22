#!/usr/bin/env python
#coding:utf-8

import pickle
import datetime
import six
import jqdata
from jqdata.exceptions import ParamsError

__all__ = [
    'get_hisname_store',
    'HisnameStore',
]

from .security_store import SecurityStore

class HisnameStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(HisnameStore, "_instance"):
            HisnameStore._instance = HisnameStore(jqdata.get_config().get_hisname_pk())
        return HisnameStore._instance

    def get_history_name(self, code, date):
        names = self._dic.get(code)
        if not names:
            return ''
        if isinstance(date, (datetime.date, datetime.datetime)):
            date_s = date.strftime("%Y-%m-%d")
        elif isinstance(date, six.string_types):
            date_s = date
        else:
            raise ParamsError("date参数必须是(datetime.date, datetime.datetime, str)中的一种")
        for i in range(len(names) - 1, -1, -1):
            if names[i][0] <= date_s:
                return names[i][1]
        return ''


def get_hisname_store():
    return HisnameStore.instance()
