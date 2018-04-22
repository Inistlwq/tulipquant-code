# -*- coding: utf-8 -*-

import pytest
from pprint import pprint
from jqdata.utils import TRADE_MIN_DATE, TRADE_MAX_DATE, convert_date
from jqdata.base import *


def test_get_trade_days_from_json():
	days = get_trade_days_from_json()
	assert isinstance(days, np.ndarray)
	assert len(days) >= 2915
	assert days[0] == TRADE_MIN_DATE
	if is_trade_day(datetime.date.today()):
		assert datetime.date.today() in days
	pprint(days)

def test_request_data():
	with pytest.raises(Exception):
		request_data_no_retry("http://www.baidu.com")
	with pytest.raises(Exception):
		request_data_no_retry(DATA_SERVER + "/stock/lhb/get")

	data = request_data(DATA_SERVER + "/stock/lhb/get", {"date": "2016-02-01"})
	assert isinstance(data, list)

def test_json_serial_fallback():
	assert json_serial_fallback(TRADE_MIN_DATE) == "2005-01-04"
	with pytest.raises(TypeError):
		json_serial_fallback(pytest)

def test_is_trade_day():
	assert is_trade_day(TRADE_MIN_DATE) is True
	assert is_trade_day(convert_date("2016-01-01")) is False

def test_get_prev_trade_day():
	assert get_prev_trade_day(datetime.date(2016, 8, 29)) == datetime.date(2016, 8, 26)
	assert get_prev_trade_day(datetime.date(2015, 10, 8)) == datetime.date(2015, 9, 30)
	assert get_prev_trade_day(datetime.date(2016, 1, 4)) == datetime.date(2015, 12, 31)
	assert get_prev_trade_day(datetime.date(2016, 9, 2)) == datetime.date(2016, 9, 1)
	assert get_prev_trade_day(datetime.date(2016, 9, 24)) == datetime.date(2016, 9, 23)
	assert get_prev_trade_day(TRADE_MIN_DATE) == None

	with pytest.raises(Exception):
		get_prev_trade_day(TRADE_MAX_DATE)
