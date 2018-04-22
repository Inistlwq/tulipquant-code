# coding:utf-8

import os
import sys
import random
from os.path import join
from fastcache import clru_cache as lru_cache


class BundleConfig(object):

    def __init__(self, **kwargs):
        # 证劵基础信息数据
        self.SECURITY_DATA_SERVER = kwargs.get('SECURITY_DATA_SERVER', '')

        # 复权因子
        self.POWER_RATE_SERVER = kwargs.get('POWER_RATE_SERVER', '')

        # 行情数据
        self.DAY_DATA_SERVER = kwargs.get('DAY_DATA_SERVER', '')

        self.MINUTE_DATA_SERVER = kwargs.get('MINUTE_DATA_SERVER', '')

        # 商品期货
        self.COMMODITY_DATA_SERVER = kwargs.get('COMMODITY_DATA_SERVER', '')

        self.TODAY_MINUTE_DATA_SERVER = kwargs.get('TODAY_MINUTE_DATA_SERVER', '')

        self.FUNDAMENTALS_SERVER = kwargs.get('FUNDAMENTALS_SERVER', '')
        self.FUNDAMENTALS_SERVERS = kwargs.get('FUNDAMENTALS_SERVERS', '')
        if not self.FUNDAMENTALS_SERVERS and self.FUNDAMENTALS_SERVER:
            self.FUNDAMENTALS_SERVERS = [self.FUNDAMENTALS_SERVER]

        random.shuffle(self.FUNDAMENTALS_SERVERS)

        self.GTA_SERVER = kwargs.get('GTA_SERVER', '')
        self.MACRO_SERVER = kwargs.get('MACRO_SERVER', '')
        self.JY_SERVER = kwargs.get('JY_SERVER', '')
        # 是否保持连接, bool
        self.KEEP_DB_CONNECTION = kwargs.get('KEEP_DB_CONNECTION', False)
        # 重试选项, dict
        # see https://pypi.python.org/pypi/retrying
        # default: dict(stop_max_attempt_number=10, wait_exponential_multiplier=1000)
        self.DB_RETRY_POLICY = kwargs.get('DB_RETRY_POLICY', None)

        # bcolz数据存储目录
        # BUNDLE_PATH = '/data/bcolz'
        self.BUNDLE_PATH = kwargs.get('BUNDLE_PATH',
                                      self._get_default_bundle_path())
        self._ensure_pk_path()

        # self.SUPPORT_OTCFUND = kwargs.get('SUPPORT_OTCFUND', False)

        # 默认不使用共享内存
        if 'SHM_PATH' in kwargs:
            self.USE_SHM = True
            self.SHM_PATH = kwargs.get('SHM_PATH')
        else:
            self.USE_SHM = False
            self.SHM_PATH = None



    def _update_config(self, k, v):
        if hasattr(self, k):
            setattr(self, k, v)


    def set_shm_path(self, shm_path):
        if shm_path:
            self._update_config('USE_SHM', True)
            self._update_config('SHM_PATH', shm_path)
        else:
            self._update_config('USE_SHM', False)
            self._update_config('SHM_PATH', None)


    def _ensure_pk_path(self):
        p = join(self.BUNDLE_PATH, 'pk')
        if not os.path.exists(p):
            os.makedirs(p)
        open_fund_p = join(self.BUNDLE_PATH, 'pk/open_fund')
        if not os.path.exists(open_fund_p):
            os.makedirs(open_fund_p)
        return p


    def set_bundle_path(self, bundle_path):
        '''
        本地数据目录
        '''
        self._update_config('BUNDLE_PATH', bundle_path)
        self._ensure_pk_path()

    def get_bundle_path(self):
        return self.BUNDLE_PATH


    def get_calendar_pk(self):
        '''
        交易日历数据
        '''
        return join(self.BUNDLE_PATH, "pk", "trade_days.pk")


    def get_security_pk(self, security_type=''):
        '''
        证券信息
        '''
        if security_type:
            return join(self.BUNDLE_PATH, 'pk', 'securities_%s.pk' % security_type)
        return join(self.BUNDLE_PATH, 'pk', 'securities.pk')


    def get_industry_pk(self):
        '''行业信息'''
        return join(self.BUNDLE_PATH, "pk", "industries.pk")


    def get_concept_pk(self):
        '''概念信息'''
        return join(self.BUNDLE_PATH, "pk", "concepts.pk")


    def get_index_pk(self):
        '''指数成分股'''
        return join(self.BUNDLE_PATH, "pk", "indexs.pk")


    def get_st_pk(self):
        '''st周期'''
        return join(self.BUNDLE_PATH, "pk", "st_periods.pk")


    def get_hisname_pk(self):
        '''旧名字'''
        return join(self.BUNDLE_PATH, "pk", "historynames.pk")


    def get_merged_pk(self):
        '''股票合并信息'''
        return join(self.BUNDLE_PATH, "pk", "mergedstocks.pk")


    def get_dividend_pk(self, security_type=''):
        '''股票分红信息'''
        if security_type:
            return join(self.BUNDLE_PATH, 'pk', 'dividend_%s.pk' % security_type)
        return join(self.BUNDLE_PATH, "pk", "dividend.pk")


    def get_margin_pk(self):
        '''融资融券股票列表'''
        return join(self.BUNDLE_PATH, "pk", "marginstocks.pk")


    def get_dominant_future_pk(self):
        '''期货主力合约信息'''
        return join(self.BUNDLE_PATH, "pk", "dominantfuture.pk")

    def get_open_fund_security_pk(self):
        '''场外基金标的列表'''
        return join(self.BUNDLE_PATH, "pk/open_fund", "open_fund_securities.pk")
        pass

    def get_open_fund_dividend_pk(self):
        '''场外基金分红数据'''
        return join(self.BUNDLE_PATH, "pk/open_fund", "open_fund_dividend.pk")
        pass

    def get_open_fund_info_pk(self):
        '''场外基金基本信息'''
        return join(self.BUNDLE_PATH, "pk/open_fund", "open_fund_info.pk")
        pass

    def get_sqlite_db_uri(self):
        '''
        返回 sqlalchemy pk sqlite db uri
        :return: sqlite:////opt/data/jq/bundle/pk/pk.db
        '''
        return join('sqlite:///%s' % (self.BUNDLE_PATH), "pk", "pk.db")

    def get_sqlite_pk(self):
        '''sqlite pk db 文件'''
        return join(self.BUNDLE_PATH, "pk", "pk.db")

    def get_bcolz_fund_path(self, security, makedir=False):
        # 基金净值数据
        code = security.code
        subdir = security.code.split(".")[0][-2:]
        p = join(self.BUNDLE_PATH, 'fundnet', subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p

    def get_bcolz_otcfund_path(self, security, makedir=False):
        # 基金净值数据
        code = security.code
        subdir = security.code.split(".")[0][-2:]
        p = join(self.BUNDLE_PATH, 'otcfundnet', subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p

    def get_bcolz_usidx_path(self, security, makedir=False):
        # 基金净值数据
        code = security
        subdir = security[-2:]
        p = join(self.BUNDLE_PATH, 'usidx', subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p

    def get_bcolz_data_path(self, security, unit='1d', makedir=False):
        # bcolz 行情存储目录
        code = security.code
        assert unit in ('1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', 'mn') # mn 表示一月
        sectype = security.get_security_type_value()
        subdir = code.split(".")[0][-2:]
        p = join(self.BUNDLE_PATH, sectype+unit, subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p

    def get_bcolz_day_path(self, security, makedir=False):
        # bcolz 天行情存储目录
        code = security.code
        sectype = security.get_security_type_value()
        subdir = code.split(".")[0][-2:]
        p = join(self.BUNDLE_PATH, sectype+"1d", subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p


    def get_bcolz_minute_path(self, security, makedir=False):
        # bcolz 分钟行情存储目录
        code = security.code
        subdir = code.split(".")[0][-2:]
        sectype = security.get_security_type_value()
        p = join(self.BUNDLE_PATH, sectype+"1m", subdir, code)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        return p


    @lru_cache(None)
    def get_day_shm_path(self, code, makedir=False):
        subdir = code.split(".")[0][-2:]
        if not self.SHM_PATH:
            raise Exception("SHM_PATH is None, use jqdata.set_shm_path(path) first.")
        p = join(self.SHM_PATH, 'daydata', subdir)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        # return 'file://' + join(p, code)
        return join(p, code)


    @lru_cache(None)
    def get_minute_shm_path(self, code, date, makedir=False):
        subdir = code.split(".")[0][-2:]
        if not self.SHM_PATH:
            raise Exception("SHM_PATH is None, use jqdata.set_shm_path(path) first.")
        p = join(self.SHM_PATH, 'minutedata', subdir)
        if not os.path.exists(p) and makedir:
            os.makedirs(p)
        # return 'file://' + join(p, code)
        return join(p, code + "." + date)

    def get_tick_path(self):
        return join(self.BUNDLE_PATH, 'ticks')


    def _get_default_bundle_path(self):
        data_path = os.getenv("JQ_DATA_PATH")
        if not data_path:
            is_windows = sys.platform in ['win32', 'cygwin']
            if is_windows:
                jqhome = os.getenv('JQ_HOME', os.path.expanduser('~/.joinquant'))
                data_path = jqhome + '/bundle'
            else:
                data_path = '/opt/data/jq/bundle'

        return data_path


_cfg = None


def set_bundle_path(bundle_path):
    cfg = get_config()
    cfg.set_bundle_path(bundle_path)
    return cfg


def set_shm_path(shm_path):
    cfg = get_config()
    cfg.set_shm_path(shm_path)
    return cfg


def get_config():
    global _cfg
    if not _cfg:
        _cfg = BundleConfig(**_get_host_config_dict())
    return _cfg

def reinit():
    global _cfg
    _cfg = None


def _get_host_config_dict():
    import os
    jqhome = os.getenv('JQ_HOME', os.path.expanduser('~/.joinquant'))
    d = {}
    for p in [
        jqhome + '/etc/jqcore/jqdata.py',
        '/etc/jqcore/jqdata.py',
        '/home/server/etc/jqcore/jqdata.py',
        '/home/server/etc/jqcore/jqdata.local.py',
        '/etc/joinquant_user/jqdata.py',
        '/etc/joinquant_user/jqdata.local.py',
    ]:
        if os.path.exists(p):
            exec (open(p).read(), {}, d)
    return d


from .gta import gta, query
from .macro import macro
from .jy import jy
from .apis import get_all_trade_days, get_trade_days, get_money_flow, get_mtss, get_concepts, get_industries

from .apis import get_security_info
from .apis import get_fund_info

def set_cache():
    '''
    缓存bcolz文件
    :return:
    '''
    from .stores import BcolzMinuteStore, BcolzDayStore, FuturesStore

    BcolzDayStore.set_cache()
    BcolzMinuteStore.set_cache()
    FuturesStore.set_cache()

def unset_cache():
    '''
    不缓存bcolz文件
    :return:
    '''
    from .stores import BcolzMinuteStore, BcolzDayStore, FuturesStore

    BcolzDayStore.unset_cache()
    BcolzMinuteStore.unset_cache()
    FuturesStore.unset_cache()

def clear_cache():
    '''
    清除缓存的bcolz文件
    :return:
    '''
    from .stores import BcolzMinuteStore, BcolzDayStore, FuturesStore

    BcolzDayStore.clear_cache()
    BcolzMinuteStore.clear_cache()
    FuturesStore.clear_cache()
