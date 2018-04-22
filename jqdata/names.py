#coding:utf-8

DAY_COLUMNS =     tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
FUTURES_COLUMNS = tuple('open close high low volume money pre_close high_limit low_limit paused avg factor settlement open_interest'.split())
FUND_COLUMNS =    tuple('open close high low volume money pre_close high_limit low_limit paused avg factor'.split())
MINUTE_COLUMNS =  tuple('open close high low volume money avg factor'.split())

DAY_FACTOR_COLUMN = DAY_COLUMNS.index('factor')

BCOLZ_MINUTE_COLS = ('date',) + MINUTE_COLUMNS
BCOLZ_DAY_COLS = ('date',) + DAY_COLUMNS

_COL_POWERS = {col: 10000.0 if col in ('open', 'close', 'high', 'low', 'avg')\
               else 1.0 for col in MINUTE_COLUMNS}
_FACTOR_COL = BCOLZ_DAY_COLS.index('factor')

DEFAULT_FIELDS = ['open', 'close', 'high', 'low', 'volume', 'money']
