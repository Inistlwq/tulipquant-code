#!/usr/bin/env python
# coding: utf-8

from jqdata import get_config
from jqdata.stores.bcolz_store import get_bcolz_day_store, get_bcolz_minute_store
from jqdata.stores.shm_store import get_shm_day_store, get_mixed_minute_store


def get_date_by_count(security, end_date, count):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_shm_day_store()
    else:
        store = get_bcolz_day_store()
    return store.get_date_by_count(security, end_date, count)


def get_date_by_period(security, start_date, end_date):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_shm_day_store()
    else:
        store = get_bcolz_day_store()
    return store.get_date_by_period(security, start_date, end_date)


def get_factor_by_date(security, date):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_shm_day_store()
    else:
        store = get_bcolz_day_store()
    return store.get_factor_by_date(security, date)


def get_minute_bar_by_count(security, end_dt, count, fields, include_now=True):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_mixed_minute_store()
    else:
        store = get_bcolz_minute_store()
    return store.get_bar_by_count(security, end_dt, count, fields, include_now=include_now)


def get_minute_bar_by_period(security, start_dt, end_dt, fields, include_now=True):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_mixed_minute_store()
    else:
        store = get_bcolz_minute_store()
    return store.get_bar_by_period(security, start_dt, end_dt, fields, include_now=include_now)


def get_daily_bar_by_count(security, end_date, count, fields, include_now=True):
    cfg = get_config()

    if cfg.USE_SHM:
        store = get_shm_day_store()
    else:
        store = get_bcolz_day_store()
    return store.get_bar_by_count(security, end_date, count, fields, include_now=include_now)


def get_daily_bar_by_period(security, start_date, end_date, fields, include_now=True):
    cfg = get_config()

    if cfg.USE_SHM:
        store = get_shm_day_store()
    else:
        store = get_bcolz_day_store()
    return store.get_bar_by_period(security, start_date, end_date, fields, include_now=include_now)

def get_minute_by_count(security, end_dt, count, include_now=True):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_mixed_minute_store()
    else:
        store = get_bcolz_minute_store()
    return store.get_minute_by_count(security, end_dt, count, include_now=include_now)

def get_minute_by_period(security, start_dt, end_dt, include_now=True):
    cfg = get_config()
    if cfg.USE_SHM:
        store = get_mixed_minute_store()
    else:
        store = get_bcolz_minute_store()
    return store.get_minute_by_period(security, start_dt, end_dt, include_now=include_now)