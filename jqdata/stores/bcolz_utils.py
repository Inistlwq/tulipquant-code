#!/usr/bin/env python
#coding:utf-8

import time

def retry_bcolz_open(p, mode='r'):
    _RETRY_COUNT = 5
    # 重试，sync数据时有可能本地数据被删除了。
    # 也有可能的确还没有历史数据，比如模拟盘读取当天才上市的标的。
    for i in range(_RETRY_COUNT):
        try:
            try:
                import jqbcolz
            except ImportError:
                import bcolz
                # not joinquant-modified bcolz, no mmap args
                return bcolz.open(rootdir=p, mode='r')
            else:
                return jqbcolz.open(rootdir=p, mode='r', mmap=True)
        except IOError as e:
            if i < _RETRY_COUNT-1:
                time.sleep(1)
                continue
            else:
                raise e

class _Table(object):
    def __init__(self, table, index):
        self.table = table
        # 期货是当前时间ct的date，其它是day数据的date
        self.index = index


class _BenchTable(object):
    def __init__(self, closes, factors, index):
        self.closes = closes
        self.factors = factors
        self.index = index



