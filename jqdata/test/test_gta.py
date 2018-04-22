
from jqdata.base import *
from jqdata.api import *

from jqdata import gta
import pandas as pd

def test_gta():
    from jqdata import gta, query
    df = gta.run_query(query(gta.STK_STOCKINFO))
    assert len(df.index) == 3000
    df = gta.run_query(gta.query(gta.STK_STOCKINFO).limit(10))
    assert isinstance(df, pd.DataFrame)
    assert len(df.index) == 10
    pass

def test_gta2():
    from jqdata import gta, query
    print(gta.run_query(query(gta.STK_AF_FORECAST).limit(10)))
    print(gta.run_query(query(gta.STK_AF_FORECAST.SYMBOL).limit(10)))
    print(gta.run_query(gta.query(gta.STK_AF_FORECAST.SYMBOL).limit(10)))
    pass
