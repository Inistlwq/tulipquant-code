
from jqdata.base import *
from jqdata.db_utils import *
from jqdata import gta
import pytest

def test():
    check_no_join(query(gta.STK_STOCKINFO))
    check_no_join(query(gta.STK_STOCKINFO.SYMBOL))
    check_no_join(query(gta.STK_STOCKINFO).filter(gta.STK_STOCKINFO.SYMBOL == 1))
    check_no_join(query(gta.STK_STOCKINFO).order_by(gta.STK_STOCKINFO.SYMBOL))
    check_no_join(query(gta.STK_STOCKINFO).group_by(gta.STK_STOCKINFO.SYMBOL))
    pass

def test2():
    with pytest.raises(Exception):
        check_no_join(query(gta.STK_STOCKINFO, gta.BD_LISTEDCOINFO))
    with pytest.raises(Exception):
        check_no_join(query(gta.STK_STOCKINFO).filter(gta.STK_STOCKINFO.SYMBOL == gta.BD_LISTEDCOINFO.ORGID))
    pass
