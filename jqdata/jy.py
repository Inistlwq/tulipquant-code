# -*- coding: UTF-8 -*-

import jqdata
from jqdata.tp.db_helper import (
    Database,
    get_engine)


def get_jy():
    db_url = jqdata.get_config().JY_SERVER
    if not db_url:
        return None
    eg = get_engine(db_url)
    if eg is not None:
        return Database(eg, disable_join=True)
    else:
        return None
    pass

jy = get_jy()
