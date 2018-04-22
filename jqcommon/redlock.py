#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Ma Tao
'''
Distributed Lock Implemented with Redis
'''

from __future__ import absolute_import, print_function, unicode_literals

import time


class AcquireLockFailed(Exception):
    '''
    Not able to acquire lock
    '''
    pass


class RedisMark(object):
    """set a mark in redis for the task with default expire

    use as a context manager(ie. with statement)
    """
    def __init__(self, client, key, value, expire=300):
        self._key = key
        self._client = client
        self._expire = expire
        self._value = value
        self._no_connection_pool = True

    def _redis(self, cmd, *args):
        try:
            func = getattr(self._client, cmd)
            return func(*args)
        finally:
            if self._no_connection_pool:
                self._client.connection_pool.disconnect()
                self._client.connection_pool.reset()

    def __enter__(self):
        self._redis('set', self._key, self._value)
        self._redis('expire', self._key, self._expire)

    def __exit__(self, type_, value, traceback):
        self._redis('delete', self._key)


class RedisLock(object):
    '''
    Distributed Lock based on Redis.
    '''

    def __init__(self, client, key, expire=None, lock_value=None):
        self._client = client
        self._key = key
        self._expire = expire
        self._lock_value = lock_value or str(time.time())
        self._no_connection_pool = True

    def _redis(self, cmd, *args):
        try:
            func = getattr(self._client, cmd)
            return func(*args)
        finally:
            if self._no_connection_pool:
                self._client.connection_pool.disconnect()
                self._client.connection_pool.reset()

    def try_lock(self):
        '''
        Try to acquire the lock, if failed, AcquireLockFailed raised immediately
        '''
        if not self._redis('setnx', self._key, self._lock_value):
            exist_lock_value = self._redis('get', self._key)
            raise AcquireLockFailed("setnx(name='%s', value='%s') fail, exist_value='%s'" % (
                self._key, self._lock_value, exist_lock_value))
        if self._expire is not None:
            self._redis('expire', self._key, self._expire)

    def lock(self):
        '''
        Block until lock acquired
        '''
        while True:
            try:
                self.try_lock()
                return
            except AcquireLockFailed:
                time.sleep(1)

    def unlock(self):
        '''
        Unlock previous acquired lock
        '''
        # DONOT CHECK THE VALUE WHEN UNLOCK, MAY THE CHECK SCRIPT ADD THE LCOK
        code = '''
local key = ARGV[1]
local value = ARGV[2]
local lock_value = redis.call("get", key)
if lock_value == value then
    redis.call("del", key)
    return true
end
return false
'''
        ret = self._client.eval(code, 0, self._key, self._lock_value)
        # ret = self._redis('delete', self._key)
        return ret

    def __enter__(self):
        self.try_lock()
        return self

    def __exit__(self, type_, value, traceback):
        self.unlock()
