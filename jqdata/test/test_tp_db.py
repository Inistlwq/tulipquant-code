# -*- coding: utf-8 -*-

import pytest

def test_gta():
    from jqdata.tp.db import gta
    print(gta.run_query(gta.query(gta.STK_STOCKINFO).limit(10)))
    print(gta.run_query(gta.query(gta.STK_AF_FORECAST.SYMBOL).limit(10)))
    print(gta.run_query(gta.query(gta.STK_STOCKINFO.INSTITUTIONID).limit(10)))
    print(gta.run_query(gta.query(gta.BD_DATEINFO).limit(10)))
    print(gta.run_query(gta.query(gta.FUND_ALLOCATION.ABSVALUE).filter(gta.FUND_ALLOCATION.ABSVALUE > 10).limit(10)))
    assert gta.run_query(gta.query(gta.BD_DATEINFO)).shape[0] == 3000
    assert gta.run_query(gta.query(gta.BD_DATEINFO).limit(10)).shape[0] == 10
    pass

def test_gta2():
    from jqdata.tp.db import gta2
    print(gta2.run_query(gta2.query(gta2.STK_STOCKINFO).limit(10)))
    print(gta2.run_query(gta2.query(gta2.STK_AF_FORECAST.SYMBOL).limit(10)))
    print(gta2.run_query(gta2.query(gta2.STK_STOCKINFO.INSTITUTIONID).limit(10)))
    print(gta2.run_query(gta2.query(gta2.BD_DATEINFO).limit(10)))
    print(gta2.run_query(gta2.query(gta2.FUND_ALLOCATION.ABSVALUE).filter(gta2.FUND_ALLOCATION.ABSVALUE > 10).limit(10)))
    pass

def test_gta_join():
    from jqdata.tp.db import gta2
    with pytest.raises(Exception):
        print(gta2.run_query(gta2.query(gta2.STK_STOCKINFO, gta2.STK_AF_FORECAST.SYMBOL).limit(10)))


# def test_pg():
#     # postgresql+psycopg2://alvin:test123@127.0.0.1:5432/alvindb?client_encoding=utf-8
#     from jqdata.tp.db import pgtest
#     print(pgtest.run_query(pgtest.query(pgtest.users)))
#     print(pgtest.run_query(pgtest.query(pgtest.users.username)))
#     pass
