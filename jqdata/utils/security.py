#!/usr/bin/env python
# coding:utf-8

import six

from jqdata.exceptions import ParamsError
from jqdata.models.security import Security
from jqdata.stores.security_store import SecurityStore


def convert_security(sec):
    if isinstance(sec, six.string_types):
        t = SecurityStore.instance().get_security(sec)
        if not t:
            raise ParamsError("找不到标的{}".format(sec))
        return t
    elif isinstance(sec, Security):
        return sec
    else:
        raise ParamsError('security 必须是一个Security对象')


def convert_security_list(sec_list):
    if isinstance(sec_list, six.string_types):
        sec = convert_security(sec_list)
        return [sec]
    elif isinstance(sec_list, Security):
        return [sec_list]
    elif isinstance(sec_list, (list, tuple, set)):
        return [convert_security(o) for o in sec_list]
    else:
        raise ParamsError('security 必须是一个Security实例或者数组')
