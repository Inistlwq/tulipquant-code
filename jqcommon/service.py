# coding:utf-8

import sys
import logging
import json
import requests
from requests.exceptions import HTTPError

from .retry import RetryingObject
from . import config

PY3 = sys.version_info >= (3,)
if PY3:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse


__all__ = ['get_http_client', 'get_redis_conn', 'get_ssdb_conn', 'get_mq_channel', 'request_web']


log = logging.getLogger(__name__)


class _HttpClient(object):
    def __init__(self, server, **kwargs):
        self._server = server
        self._requester = requests.Session()
        pass

    def request(self, method, url, **kwargs):
        return self._requester.request(method, self._server + url, **kwargs)

    def get(self, url, **kwargs):
        return self.request('get', url, **kwargs)

    def post(self, url, **kwargs):
        return self.request('post', url, **kwargs)


def get_http_client(servers, gen_kwargs=None, retry_kwargs=None,
                    is_random=False):
    retry_kws = dict(
        retry_on_exception=lambda e:
            isinstance(e, (IOError, OSError, HTTPError)),
        retry_on_result=lambda res:
            res.status_code >= 500
    )
    gen_kws = {}

    if gen_kwargs:
        gen_kws.update(gen_kwargs)
    if retry_kwargs:
        retry_kws.update(retry_kwargs)
    return RetryingObject(_HttpClient, servers, gen_kws, retry_kws,
                          is_random=is_random)


# --- wraps for easy use ---
def get_redis_conn(urls=None, gen_kwargs=None, retry_kwargs=None):
    import time
    import redis

    def redis_generator(redis_url,
                        check_run_secs=0,
                        **kwargs):
        """
            check_run_secs: 检查 Redis 服务器是否已经运行了 `check_run_secs` 秒, 如果不是, 等待
        """
        conn = redis.StrictRedis.from_url(redis_url)
        if check_run_secs:
            run_secs = conn.info('server').get(
                'uptime_in_seconds', check_run_secs)
            if run_secs < check_run_secs:
                time.sleep(check_run_secs - run_secs)
        return conn

    retry_kws = {
        'wait_exponential_multiplier': 1000,
        'retry_on_exception': lambda e: (isinstance(
            e, (IOError, OSError, redis.RedisError)))
    }
    gen_kws = {
        'socket_timeout': config.REDIS_READ_TIMEOUT,
        'socket_connect_timeout': config.REDIS_CONNECT_TIMEOUT
    }
    if gen_kwargs:
        gen_kws.update(gen_kwargs)
    if retry_kwargs:
        retry_kws.update(retry_kwargs)

    urls = urls or config.REDIS_SERVERS
    return RetryingObject(
        redis_generator, urls, gen_kws, retry_kws)


def get_ssdb_conn(urls=None, gen_kwargs=None, retry_kwargs=None):
    from ssdb.client import SSDB, SSDBError

    def ssdb_generator(ssdb_url, **kwargs):
        parsed = urlparse(ssdb_url)
        ssdb_config = kwargs.copy()
        # TODO: ssdb-py bug: socket_connect_timeout is not in __init__
        # connection timeout will be equal to socket_timeout
        # ssdb_config['socket_connect_timeout'] = 0.1
        try:
            if parsed.hostname and parsed.port:
                ssdb_config['host'] = parsed.hostname
                ssdb_config['port'] = parsed.port
            else:
                raise config.ConfigError("Please specify host, port in url: %s"
                                         % ssdb_url)
            if parsed.password:
                ssdb_config['auth_password'] = parsed.password
        except ValueError:
            raise config.ConfigError('Port is invalid in url: %s' % ssdb_url)
        return SSDB(**ssdb_config)

    retry_kws = {
        'wait_exponential_multiplier': 1000,
        'retry_on_exception': lambda e:
            (isinstance(e, (IOError, OSError, SSDBError)))
    }
    gen_kws = {
        'socket_timeout': config.SSDB_READ_TIMEOUT,
        # ssdb-py 低版本不支持 socket_connect_timeout 参数
        # 'socket_connect_timeout': config.SSDB_CONNECT_TIMEOUT,
    }
    if gen_kwargs:
        gen_kws.update(gen_kwargs)
    if retry_kwargs:
        retry_kws.update(retry_kwargs)
    urls = urls or config.SSDB_SERVERS
    return RetryingObject(
        ssdb_generator, urls, gen_kws, retry_kws)


def get_mq_channel(urls=None, gen_kwargs=None, retry_kwargs=None):
    import pika

    def mq_generator(mq_url, **kwargs):
        conn = pika.BlockingConnection(pika.URLParameters(mq_url))
        return conn.channel()

    retry_kws = {
        'wait_exponential_multiplier': 1000,
        'retry_on_exception': lambda e: (isinstance(
            e, (IOError, OSError, pika.exceptions.AMQPError)))
    }
    gen_kws = {
    }
    if gen_kwargs:
        gen_kws.update(gen_kwargs)
    if retry_kwargs:
        retry_kws.update(retry_kwargs)
    urls = urls or config.QUEUE_SERVERS
    return RetryingObject(mq_generator, urls, gen_kws, retry_kws)


def request_web(method, url, json_body=True, web_servers=[],
                retry_kwargs=None, **requests_kws):
    kws = requests_kws.copy()
    if config.WEB_EXTRAS:
        params = kws.setdefault('params', {})
        for k, v in config.WEB_EXTRAS.items():
            params.setdefault(k, v)

    web_servers = web_servers or config.WEB_SERVERS
    log.info("request web: {} {} {}".format(method, url, kws))
    assert method in ('get', 'post')
    retry_kwargs = retry_kwargs or {
        'stop_max_attempt_number': 10,
        'wait_exponential_multiplier': 1000,
    }
    client = get_http_client(web_servers, retry_kwargs=retry_kwargs,
                             is_random=True)
    res = client.request(method, url, **kws)
    log.info("request web: %s %s, response: %s content: %s",
             method,
             url,
             res,
             res.content.decode('utf-8-sig', errors='ignore')[:300])

    res.raise_for_status()
    if json_body:
        body = res.content.decode('utf-8-sig')
        body_dict = json.loads(body)
        code = int(body_dict['code'])
        if code != 0:
            raise Exception(body_dict.get('msg', 'Unknown'))
        return body_dict
    else:
        return res.content
