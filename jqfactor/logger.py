# coding: utf-8

import os
import sys
import time
import logging
import contextlib


user_log = logging.getLogger("factor.userlog")
user_err = logging.getLogger("factor.usererr")


class BufferingStoreHandler(logging.Handler):

    def __init__(self, store, store_key, capacity, max_flush_interval, max_lines, max_length):
        """A handler to write log to `jqfactor.store.Store`.
        And, it will disable sys.stdout/stderr when flush.

        args:
            store: `jqfactor.store.Store` object
            store_key: key for call rpush
            capacity: max buffer length
            max_flush_interval: max flush interval, in seconds
            max_lines: max lines of log
            max_length: max size of each line
        """
        logging.Handler.__init__(self)
        self._buffer = []  # put formatted string of each log record
        self._buffer_capacity = capacity
        self._store = store
        self._store_key = store_key
        self._max_flush_interval = max_flush_interval
        self._last_flush_time = 0
        self._max_lines = max_lines
        self._max_length = max_length
        self._lines = 0
        self._full = False
        pass

    def emit(self, record):
        if self._full:
            return

        # -1, 保留一条显示 warning
        if self._lines >= self._max_lines - 1:
            self._full = True
            self.flush()
            msg = "log 空间已满, 不再显示更多. log 最多显示 {} 条".format(self._max_lines)
            self._store.rpush(self._store_key, msg)
            return

        self._lines += 1

        self._buffer.append(self.format(record))
        if self.shouldFlush(record):
            self.flush()

    def format(self, record):
        line = logging.Handler.format(self, record)
        if len(line) > self._max_length:
            line = line[:self._max_length] + '...'
        return line

    def shouldFlush(self, record):
        # noqa
        if (len(self._buffer) >= self._buffer_capacity):
            return True
        if self._buffer and self._get_time() - self._last_flush_time >= self._max_flush_interval:
            return True
        return False

    def _get_time(self):
        return time.time()

    def flush(self):
        devnull = open(os.devnull, 'wb')
        with _temp_set(sys, 'stderr', devnull), _temp_set(sys, 'stdout', devnull):
            if self._buffer:
                elements = self._buffer
                self._store.rpush(self._store_key, *elements)
                self._buffer = []
            self._last_flush_time = self._get_time()

    def close(self):
        self.flush()
        logging.Handler.close(self)


@contextlib.contextmanager
def _temp_set(object, attr, value):
    """ A contextmanager, 临时设置 object.<attr> = value """
    old = getattr(object, attr)
    try:
        setattr(object, attr, value)
        yield
    finally:
        setattr(object, attr, old)
    pass


def setup_userlog(store):
    handler = BufferingStoreHandler(store, 'userlog', capacity=100,
                                    max_flush_interval=3, max_lines=100000,
                                    max_length=100000)
    user_log.propagate = False
    user_log.setLevel(logging.INFO)
    user_log.addHandler(handler)


def setup_usererr(store):
    handler = BufferingStoreHandler(store, 'usererr', capacity=100,
                                    max_flush_interval=3, max_lines=100000,
                                    max_length=100000)
    user_err.propagate = False
    user_err.setLevel(logging.ERROR)
    user_err.addHandler(handler)


def setup_logging():
    logger = logging.getLogger()
    logger.handlers = []  # empty handlers
    logger.setLevel(logging.DEBUG)

    # add stream log handler for info
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)1.1s %(asctime)s %(name)s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def setup(store):
    setup_userlog(store)
    setup_usererr(store)
    setup_logging()


class _OutputStream(object):

    def __init__(self, is_err_stream):
        self._is_err_stream = is_err_stream

    def write(self, msg):
        if self._is_err_stream:
            user_err.warn(msg)
        else:
            user_log.info(msg)

    def flush(self):
        pass


@contextlib.contextmanager
def redirect_stdout_to_userlog():
    """重定向 sys.stdout/stderr 到 log.info/warn"""
    _stdout = sys.stdout
    _stderr = sys.stderr
    sys.stdout = _OutputStream(False)
    sys.stderr = _OutputStream(True)
    try:
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout = _stdout
        sys.stderr = _stderr
