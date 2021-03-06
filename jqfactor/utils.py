# -*- coding: utf-8 -*-


class suppress(object):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.
         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed

    copy from https://github.com/jazzband/contextlib2
    """

    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        return exctype is not None and issubclass(exctype, self._exceptions)


def filter_dict_values(dct, fields):
    """按给定字段的顺序过滤出字典的值到一个列表中

    如果给定的字段不在字典中，则该字段值为 None
    """
    return [dct.get(field) for field in fields]
