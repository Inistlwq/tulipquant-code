#!/usr/bin/env python

import os
import pickle

import jqdata
from jqdata.utils.datetime_utils import parse_date
__all__ = [
    'get_merged_store',
    'MergedStore',
]

class MergedStore(object):

    def __init__(self, f):
        with open(f, "rb") as store:
            self._dic = pickle.load(store)

    @staticmethod
    def instance():
        if not hasattr(MergedStore, "_instance"):
            MergedStore._instance = MergedStore(jqdata.get_config().get_merged_pk())
        return MergedStore._instance

    def get_merged_info(self, code):
        ret = self._dic.get(code, None)
        if ret is None:
            return None
        return {
            'merge_date': parse_date(ret[2]),
            'target_code': ret[0],
            'scale_factor': ret[1]
        }


def get_merged_store():
    return MergedStore.instance()


