# coding: utf-8
import os
import logging


_log = logging.getLogger(__name__)


# 是否把因子结果写入 ssdb
SSDB_ENABLED = False

# SSDB 配置
SSDB_URL = ''
SSDB_URLS = []
SSDB_URL_GETTER = None

# 扩展因子数据配置
EXTENDED_DATA_SERVERS = []


class ConfigError(Exception):
    pass


def _load_config(path):
    """Load config, config 中的全局变量会覆盖当前文件的"""
    _log.debug("_load_config %s", path)
    with open(path) as f:
        code = compile(f.read(), path, 'exec')
        exec(code, globals(), globals())


# 首先根据环境在项目目录下加载
env = os.getenv('JQENV', 'local').lower()
jqhome = os.getenv('JQ_HOME', os.path.expanduser('~/.joinquant'))

for path in (
    '/etc/joinquant_user/jqfactor.py',
    '/etc/joinquant_user/jqfactor.local.py',
    '/home/server/etc/jqcore/jqfactor.py',
    '/home/server/etc/jqcore/jqfactor.{}.py'.format(env),
    '/home/server/etc/jqcore/jqfactor.local.py',
    jqhome + '/etc/jqcore/jqfactor.py',
    os.getenv("JQFACTOR_CONFIG_PATH"),
):
    if path and os.path.exists(path):
        _load_config(path)


if not SSDB_URLS and SSDB_URL:
    SSDB_URLS = [SSDB_URL]
