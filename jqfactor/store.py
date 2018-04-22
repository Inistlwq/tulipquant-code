# -*- coding: utf-8 -*-
# @Author: Ma Tao

from __future__ import absolute_import, print_function, unicode_literals

import os
from jqcommon.service import get_ssdb_conn

import jqfactor.config as config


class Store(object):

    def __init__(self, task_id):
        self._task_id = task_id
        if os.getenv('JQORIGIN') == 'client_ide':
            self._is_ide = True
        else:
            self._is_ide = False

    @property
    def task_id(self):
        return self._task_id

    def rpush(self, list_name, *items):
        raise NotImplementedError()

    def lrange(self, list_name, offset, limit):
        raise NotImplementedError()

    def set(self, k, v):
        raise NotImplementedError()

    def get(self, k):
        raise NotImplementedError()

    def flush(self):
        pass


class KeyNotExistsError(Exception):
    pass


class SsdbStore(Store):

    def __init__(self, task_id, ssdb_urls):
        Store.__init__(self, task_id)
        self._client = get_ssdb_conn(ssdb_urls)

    def _get_store_key(self, top):
        hset_name = 'fct{}-{}'.format(self.task_id, top)
        return hset_name

    def rpush(self, list_name, *items):
        key = self._get_store_key(list_name)
        return self._client.execute_command('qpush', key, *items)

    def lrange(self, list_name, offset, limit):
        key = self._get_store_key(list_name)
        return self._client.qslice(key, offset, limit)

    def set(self, k, v):
        self._client.set(self._get_store_key(k), v)

    def get(self, k):
        return self._client.get(self._get_store_key(k))


def get_store(options, reset=False):
    """get ssdb store

    args:
        options: an object which implement get() method
            options must have:
                task_id
            options may have:
                outdir: for local store
        reset: empty ssdb store
    """
    if config.SSDB_ENABLED:
        if config.SSDB_URLS:
            ssdb_urls = config.SSDB_URLS
        elif config.ssdb_url_getter:
            ssdb_urls = [config.SSDB_URL_GETTER(options.get('task_id'))]
        else:
            raise config.ConfigError("You must set ssdb_urls or ssdb_url_getter"
                                     " when ssdb_enabled is True")
        return SsdbStore(options.get('task_id'), ssdb_urls)
