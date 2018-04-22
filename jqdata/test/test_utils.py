# -*- coding: utf-8 -*-

import pytest
from jqdata.utils import *


def test_is_iterable():
	class Fibs:
	    def __init__(self):
	        self.a = 0
	        self.b = 1

	    def next(self):
	        self.a, self.b = self.b, self.a + self.b
	        return self.a

	    def __iter__(self):
	        return self

	def Zrange(n):
	    i = 0
	    while i < n:
	        yield i
	        i += 1

	fibs = Fibs()
	zrange = Zrange(3)

	assert is_iterable("")
	assert is_iterable([1, 2, 3])
	assert is_iterable((1, 2, 3))
	assert is_iterable({1:'a', 2:'b'})
	assert is_iterable(set([1, 2, 3]))
	assert is_iterable(fibs)
	assert is_iterable(zrange)

def test_convert_date():
	convert_date(datetime.datetime.today())
	convert_date(datetime.date(2010, 4, 6))
	convert_date("2015-01-01")
	#convert_date("2015-01-01 12:23:45")

def test_date_range():
	dates = date_range(datetime.date(2016, 1, 1), datetime.date(2016, 1, 5))
	assert len(list(dates)) == 5
	dates = date_range(datetime.date(2016, 1, 1), datetime.date(2016, 1, 1))
	assert len(list(dates)) == 1

	for date in date_range(datetime.date(2015, 10, 1), datetime.date(2016, 3, 31)):
		assert isinstance(date, datetime.date)

	dates = list(date_range(datetime.date(2016, 1, 1), datetime.date.today()))
	assert (dates[0], dates[-1]) == (datetime.date(2016, 1, 1), datetime.date.today())
	print(dates)

def test_check_fields():
	check_fields([1, 2, 3], [1, 2, 2, 2, 3, 3])

def test_binary_search():
	lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	assert binary_search(lst, 0) == 0
	assert binary_search(lst, 3) == 3
	assert binary_search(lst, 6) == 6
	assert binary_search(lst, 9) == 9
	assert binary_search(lst, 10) == None
