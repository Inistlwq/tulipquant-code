# coding: utf-8

import logging
import random
import re
import traceback
from retrying import retry

from . import config


class RetryingObject(object):
    """ Wrap redis or ssdb client

        Calling any method of this object will retry

        object_generator:
        A function used to create the object. Will take
        config_list[0] as the argument

        config_list:
        recreate the object using next config if retry failed

        gen_kws may contain:

            socket_connect_timeout: connection timeout
            socket_timeout: operation timeout

        retry_kws: see retrying package doc
    """

    def __init__(self, object_generator, config_list, gen_kws=None,
                 retry_kws=None, retry_times_before_switch=3, is_random=False):
        assert isinstance(config_list, list) and len(config_list) > 0, "%s is not a valid config list" % config_list
        self._config_list = config_list
        self._retry_times_before_switch = retry_times_before_switch
        self._is_random = is_random
        self._object_generator = object_generator
        if retry_kws is None:
            retry_kws = {}
        if gen_kws is None:
            gen_kws = {}
        retry_kws.setdefault('stop_max_attempt_number', 10)
        self._retry_kws = retry_kws
        self._gen_kws = gen_kws
        self._object = None

        get_attr_retry_config = dict(self._retry_kws, retry_on_result=None)
        self._get_object_attr_retry = retry(**get_attr_retry_config)(self._get_object_attr)
        self.log = logging.getLogger('%s.RetryingObject' % (__name__))

    @staticmethod
    def _clear_password(config):
        cleared_config = re.sub(r'(//.*:)\S+@',
                                r'\1*****@',
                                str(config))
        return cleared_config

    def _switch_config(self):
        if len(self._config_list) > 1:
            failed_config = self._config_list.pop(0)
            self._config_list.append(failed_config)
            cleared_failed_conf = self._clear_password(failed_config)
            cleared_new_conf = self._clear_password(self._config_list[0])
            self.log.warn('switching config from %s to %s',
                          cleared_failed_conf,
                          cleared_new_conf)
        self._object = None

    def _get_object(self):
        if self._is_random and self._object is None:
            selected = random.randint(0, len(self._config_list)-1)
            config = self._config_list[selected]
            cleared_conf = self._clear_password(config)
            self.log.info('select a random config: %s', cleared_conf)
            self._object = self._object_generator(config, **self._gen_kws)
            return self._object

        if self._object is None:
            config = self._config_list[0]
            self._object = self._object_generator(config, **self._gen_kws)
        return self._object

    def __getattr__(self, attr):
        if not config.RETRY_ENABLED:
            value = getattr(self._get_object(), attr)
            setattr(self, attr, value)
        else:
            value = self._get_object_attr_retry(attr)
            if callable(value):
                value = self._get_object_method_retry_wrapped(attr)
                setattr(self, attr, value)
        return value

    def _get_object_attr(self, attr):
        try:
            return getattr(self._get_object(), attr)
        except (IOError, OSError, Exception):
            self.log.error("_get_object_attr(%s) failed: %s", attr, traceback.format_exc())
            self._switch_config()
            raise

    def _get_object_method_retry_wrapped(self, attr):
        retry_times = [0]

        @retry(**self._retry_kws)
        def wrapper(*args, **kwargs):
            try:
                func = getattr(self._get_object(), attr)
                ret = func(*args, **kwargs)
                if not self._is_random:
                    retry_times[0] = 0
                else:
                    self._object = None
                return ret
            except Exception:
                self.log.error("call %s() failed: %s", attr, traceback.format_exc())
                if not self._is_random:
                    retry_times[0] += 1
                    if retry_times[0] == self._retry_times_before_switch:
                        retry_times[0] = 0
                        self._switch_config()
                else:
                    self._object = None
                raise

        return wrapper
