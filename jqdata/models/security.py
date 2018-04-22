#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import re
import datetime
from enum import Enum
from collections import namedtuple
from jqdata.utils.datetime_utils import parse_date
from jqdata.utils import datetime_utils

DAY_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
FUTURES_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor settlement open_interest'.split())
MINUTE_COLUMNS = tuple('open close high low volume money avg factor'.split())


FUTURE_EXCHANGES = (
    'CCFX',  # 中国金融期货交易所
    'XSGE',  # 上海期货交易所
    'XDCE',  # 郑州商品交易所
    'XZCE',  # 大连商品交易所'
)

A_EXCHANGES = (
    'XSHG', # 上交所
    'XSHE', # 深交所
)

class SecurityType(Enum):

    STOCK = 'stock'  # 股票
    INDEX = 'index'  # 指数
    FUTURES = 'futures'  # 期货
    FUND = 'fund'  # 基金
    OPEN_FUND = 'open_fund'  # 场外基金

    UNKNOWN = 'unknown'

    def __repr__(self):
        return self.value

TradeTime = namedtuple('TradeTime', ['start_date', 'end_date', 'minute_periods'])

# 商品期货交易时间段
commodity_trade_time_point = {
    u'21:00~23:00': [datetime.time(21, 0), datetime.time(23, 0)],
    u'21:00~23:30': [datetime.time(21, 0), datetime.time(23, 30)],
    u'21:00~01:00': [datetime.time(21, 0), datetime.time(1, 0)],
    u'21:00~02:30': [datetime.time(21, 0), datetime.time(2, 30)],
    u'09:00~10:15': [datetime.time(9, 0), datetime.time(10, 15)],
    u'09:15~11:30': [datetime.time(9, 15), datetime.time(11, 30)],
    u'10:30~11:30': [datetime.time(10, 30), datetime.time(11, 30)],
    u'13:00~15:00': [datetime.time(13, 0), datetime.time(15, 0)],
    u'13:00~15:15': [datetime.time(13, 0), datetime.time(15, 15)],
    u'13:30~15:00': [datetime.time(13, 30), datetime.time(15, 0)],
}


class Security(object):
    '''一个证券对象'''
    __slots__ = ('_code', '_start_date', '_end_date', '_type', '_subtype', '_trade_times',
                 '_parent', '_name', "_display_name", "_extra")

    def __init__(self, code=None, start_date=None, end_date=None,
                 type=None, **kwargs):
        assert code and start_date and type
        if type == 'stock':
            type = SecurityType.STOCK
        elif type == 'index':
            type = SecurityType.INDEX
        elif type == 'futures':
            type = SecurityType.FUTURES
        elif type == 'fund':
            type = SecurityType.FUND
        elif type == 'open_fund':
            type = SecurityType.OPEN_FUND
        else:
            type = SecurityType.UNKNOWN

        self._code = code
        self._start_date = datetime_utils.parse_date(start_date)
        if end_date:
            self._end_date = datetime_utils.parse_date(end_date)
        else:
            self._end_date = datetime.date(2200, 1, 1)
        self._type = type
        self._subtype = kwargs.pop('subtype', None)
        if self._type == SecurityType.FUTURES and self._subtype == 'commodity_futures':
            self._trade_times = self._parse_trade_times(kwargs.pop('trade_time'))
        self._parent = kwargs.pop('parent', None)
        self._name = kwargs.pop('name', '')
        self._display_name = kwargs.pop('display_name', '')
        self._extra = kwargs

    def __repr__(self):
        # lru_cache 节省空间
        return "%s(code=%s)" % (self.__class__.__name__, self._code)
        # return self._code

    def __str__(self):
        # return self._code
        return self.__repr__()

    @property
    def parent(self):
        '''分级基金的母基,其他类型返回None
            if i.type in ['fja', 'fjb', 'fjm']:
                assert i.parent is not None
            if i.type == 'fjm':
                assert i.code == i.parent
        # 原始数据有点问题： 母基的parent不正确，例如：
        (Security(code=160630.XSHE), u'fjm', '150205.XSHE')
        (Security(code=150205.XSHE), u'fja', '160630.XSHE')
        (Security(code=150206.XSHE), u'fjb', '160630.XSHE')  
        '''
        return self._parent

    @property
    def subtype(self):
        '''
        基金分类: etf  # etf和lof的区别
                 lof  # https://zhuanlan.zhihu.com/p/27936146
                 fja：a基
                 fjb：b基
                 fjm：母基金
                 mmf：货币基金
        期货分类：index_futures：股指期货
                 commodity_futures：商品期货
        '''
        return self._subtype

    @property
    def code(self):
        ''' 证券代码, `str`, 比如 000001.XSHE 
            后缀：
            'XSHG'：上交所
            'XSHE'：深交所
            'CCFX'：中国金融期货交易所
            'XSGE'：上海期货交易所
            'XDCE'：郑州商品交易所
            'XZCE'：大连商品交易所
            'XINE': 上海能源期货交易所
            'OF': 场外基金
        '''
        return self._code

    @property
    def start_date(self):
        ''' 上市日期, `datetime.date` object'''
        return self._start_date

    @property
    def end_date(self):
        ''' 最后一个上市日期, `datetime.date` object'''
        return self._end_date

    def get_security_type_value(self):
        # 证劵类型：字符串
        # 'stock', 'fund', 'futures', 'index'
        return self._type.value

    @property
    def type(self):
        # api兼容
        if self._type == SecurityType.FUND:
            return self._subtype
        return self._type.value

    @property
    def name(self):
        return self._name

    @property
    def display_name(self):
        return self._display_name

    @property
    def trade_tag(self):
        """ 把股票按交易时间分类, 不同的交易时间应该有不同的分类. 现在使用交易所分类
            有商品期货后, 不同品种交易时间可能不一样, 需要更具体的分类.
        """
        # TODO: 支持商品期货
        if self.is_futures():
            tag = re.sub('\d+', '', self.code)
            if tag:
                return tag
            return self.code
        else:
            return 'stock'

    @property
    def concepts(self):
        from jqdata.stores.concept_store import get_concept_store
        store = get_concept_store()
        return store.get_stock_concepts(self._code)
        pass

    @property
    def industries(self):
        from jqdata.stores.industry_store import get_industry_store
        store = get_industry_store()
        return store.get_stock_industry(self._code)
        pass

    def has_power_rate(self):
        '''
        是否产生复权
        '''
        return self._type in (SecurityType.STOCK, SecurityType.FUND)

    def is_t1(self):
        return self._extra.get('is_t1', True)

    def is_futures(self):
        return self._type == SecurityType.FUTURES

    def is_index_futures(self):
        return self.is_futures() and self._subtype == 'index_futures'

    def is_commodity_futures(self):
        return self.is_futures() and self._subtype == 'commodity_futures'

    def is_treasury_future(self):
        # 国债期货 TFXXXX.CCFX(5年国债期货) TXXXX.CCFX(10年国债期货)
        return self._code.endswith(".CCFX") and self._code.startswith("T")

    def is_8888_future(self):
        '''
        是否是指数连续合约。
        '''
        for suffix in FUTURE_EXCHANGES:
            if self._code.endswith('8888.%s' % suffix):
                return True
        return False

    def is_9999_future(self):
        '''
        是否是主力连续合约。
        '''
        for suffix in FUTURE_EXCHANGES:
            if self._code.endswith('9999.%s' % suffix):
                return True
        return False


    def get_trade_time(self, date):
        '''
        返回商品期货的交易时间规则。
        :param date:
        :return:
        '''
        assert self.is_commodity_futures()
        for i in self._trade_times:
            if i.start_date <= date and date <= i.end_date:
                return i
        return None

    def _parse_trade_times(self, trade_time):
        ret = []
        for period in trade_time:
            start_date = parse_date(period[0])
            end_date = parse_date(period[1])
            minute_periods = [commodity_trade_time_point[i] for i in period[2:]]
            ret.append(TradeTime(start_date, end_date, minute_periods))
        return ret

    def is_index(self):
        return self._type == SecurityType.INDEX

    def is_fund(self):
        return self._type == SecurityType.FUND

    def is_open_fund(self):
        return self.is_fund() and self._subtype in ('stock_fund', 'money_market_fund', 'bond_fund', 'mixture_fund', 'fund_fund', 'noble_metal_fund', 'closed_fund')
        pass

    def is_QDII_fund(self):
        return self._extra.get('is_QDII', False)
        pass

    def is_mixture_fund(self):
        return self.is_fund() and self._subtype == 'mixture_fund'
        pass

    def is_bond_fund(self):
        return self.is_fund() and self._subtype == 'bond_fund'
        pass

    def is_money_market_fund(self):
        return self.is_fund() and self._subtype == 'money_market_fund'
        pass

    def is_stock_fund(self):
        return self.is_fund() and self._subtype == 'stock_fund'
        pass

    def is_stock(self):
        return self._type == SecurityType.STOCK

    @property
    def price_decimals(self):
        if self._type == SecurityType.STOCK:
            return 2
        return 4

    @property
    def day_column_names(self):
        if self.is_futures():
            return FUTURES_COLUMNS
        return DAY_COLUMNS

    @property
    def minute_column_names(self):
        return MINUTE_COLUMNS

    @property
    def tick_column_names(self):
        if self.is_futures():
            return ["datetime", "current", "high", "low", "volume", "money", "position",
                    "a1_p", "a1_v", "b1_p", "b1_v"]
        elif self.is_stock() or self.is_fund():
            return ["datetime", "current", "high", "low", "volume", "money",
                    'a1_p', 'a2_p', 'a3_p', 'a4_p', 'a5_p',
                    'a1_v', 'a2_v', 'a3_v', 'a4_v', 'a5_v',
                    'b1_p', 'b2_p', 'b3_p', 'b4_p', 'b5_p',
                    'b1_v', 'b2_v', 'b3_v', 'b4_v', 'b5_v'
                    ]
        elif self.is_index():
            return ["datetime", "current", "high", "low", "volume", "money"]
        else:
            raise "%s not support tick_column_names" % (self._code)

    def __reduce__(self):
        return security_by_code, (self._code,)


def security_by_code(code):
    from jqdata.stores.security_store import get_security_store
    return get_security_store().get_security(code)
