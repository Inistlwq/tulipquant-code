# coding:utf-8

import os


WEB_SERVER = ''
WEB_SERVERS = []

# 请求 web server 时, 额外要传入的参数, 为了内部接口认证
WEB_EXTRAS = {
    # 'key': ''
}

# message queue server, 我们从 queue 里获取任务
QUEUE_SERVER = ''
QUEUE_SERVERS = []


REDIS_SERVER = ''
REDIS_SERVERS = []
REDIS_READ_TIMEOUT = 10.
REDIS_CONNECT_TIMEOUT = 10.
# 检查 redis 启动时间
REDIS_START_CHECK = 300


SSDB_SERVER = ''
SSDB_SERVERS = []
SSDB_READ_TIMEOUT = 300.
SSDB_CONNECT_TIMEOUT = 300.


SENTRY_SERVER = ''


RETRY_ENABLED = True


class ConfigError(Exception):
    pass


def _load_config(path):
    """ Load config, config 中的全局变量会覆盖当前文件的"""
    if not os.path.exists(path):
        return
    with open(path) as f:
        code = compile(f.read(), path, 'exec')
        exec(code, globals(), globals())


for path in (
        '/etc/joinquant_user/jqcommon.py',
        '/etc/joinquant_user/jqcommon.local.py',
        '/home/server/etc/jqcore/jqcommon.py',
        '/home/server/etc/jqcore/jqcommon.{}.py'.format(os.getenv('JQENV')),
        '/home/server/etc/jqcore/jqcommon.local.py',
):
    _load_config(path)

if not QUEUE_SERVERS and QUEUE_SERVER:
    QUEUE_SERVERS = [QUEUE_SERVER]

if not WEB_SERVERS and WEB_SERVER:
    WEB_SERVERS = [WEB_SERVER]

if not REDIS_SERVERS and REDIS_SERVER:
    REDIS_SERVERS = [REDIS_SERVER]

if not SSDB_SERVERS and SSDB_SERVER:
    SSDB_SERVERS = [SSDB_SERVER]
