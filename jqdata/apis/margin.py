#coding:utf-8

from jqdata.stores.margin_store import get_margin_store
from ..utils.utils import convert_date,convert_dt


def get_margincash_stocks(date):
    date = convert_date(date)
    store = get_margin_store()
    return store.get_margin_stocks(str(date))



get_marginsec_stocks = get_margincash_stocks

__all__ = [
    'get_margincash_stocks',
    'get_marginsec_stocks',
]

