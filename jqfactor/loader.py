# -*- coding: utf-8 -*-

"""Strategy loader"""

from __future__ import absolute_import, unicode_literals, print_function

import os
import sys
import re
import hashlib
import importlib
from types import ModuleType

import six

from .logger import redirect_stdout_to_userlog


class Loader(object):

    def load(self):
        raise NotImplementedError()

    def is_code_changed(self, digest):
        return True

    def digest(self):
        return self._digest


class LocalLoader(Loader):
    """Load strategy module from local path"""

    def __init__(self, path):
        self._path = path
        self._digest = ''

    def load(self):
        sys.path.insert(0, self._path)
        with redirect_stdout_to_userlog():
            module = importlib.import_module('user_code')
        sys.path.pop(0)
        return module


class CodeLoader(Loader):
    """Load strategy module from source code snippet"""

    def __init__(self, name, source):
        self._name = name
        self._source = source
        if six.PY2 and not self._is_encoding_declared(source):
            self._source = source.decode('utf8')

        if six.PY2 and isinstance(self._source, str):
            src = self._source
        else:
            src = self._source.encode('utf8')
        digest = hashlib.md5(src).hexdigest()
        self._digest = digest

    def is_code_changed(self, digest):
        return self._digest != digest

    def load(self):
        if sys.version_info.major == 2:
            self._name = self._name.encode('utf8')
        module = ModuleType(b'user_code')
        module.__file__ = self._name

        with redirect_stdout_to_userlog():
            code = compile(self._source,
                           module.__file__,
                           'exec',
                           dont_inherit=True)
            exec(code, module.__dict__)

        return module

    @staticmethod
    def _is_encoding_declared(src):
        ptn = re.compile("^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
        lines = src.split("\n", 2)
        lines_len = len(lines)
        if lines_len == 1:
            line1, line2 = lines[0], ''
        elif lines_len == 2:
            line1, line2 = lines
        else:
            line1, line2, _ = lines
        return bool(ptn.match(line1) or ptn.match(line2))


class BinaryLoader(Loader):
    """Load strategy module from .so file"""

    def __init__(self, name, source):
        self._name = name

        digest = hashlib.md5(source).hexdigest()
        self._digest = digest

    def is_code_changed(self, digest):
        return self._digest != digest

    def load(self):
        if sys.version_info.major == 2:
            self._name = self._name.encode('utf8')

        code_path = os.path.dirname(self._name)
        sys.path.insert(0, code_path)
        module_name = os.path.basename(self._name)
        module_name = os.path.splitext(module_name)[0]
        with redirect_stdout_to_userlog():
            module = importlib.import_module(module_name)
        sys.path.pop(0)
        module.__file__ = self._name

        return module
