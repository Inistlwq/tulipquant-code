# coding: utf-8
import jqdata
import datetime
from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

__all__ = [
    'MoneyFlowEntity',
    'get_session',
    'init_money_flow_db',
    'drop_money_flow_db',
]

Base = declarative_base()
metadata = Base.metadata


_engine = None
_DBSession = None


def get_engine():
    global _engine
    if _engine is None:
        sqlite_uri = jqdata.get_config().get_money_flow_sqlite_db_uri()
        _engine = create_engine(sqlite_uri, echo=False)
    return _engine


def get_session():
    global _DBSession
    if _DBSession is None:
        _DBSession = sessionmaker(bind=get_engine())
    return _DBSession()


class MoneyFlowEntity(Base):
    """
    money flow 数据
    """
    __tablename__ = 'moneyflow'
    id = Column(Integer, primary_key=True)
    date = Column(Text, nullable=False, doc="日期")
    sec_code = Column(Text, nullable=False, doc="股票代码(带后缀: .XSHE/.XSHG)")
    change_pct = Column(Text, nullable=True, doc="涨跌幅")
    net_amount_main = Column(Text, nullable=True, doc="主力净额(万)")
    net_pct_main = Column(Text, nullable=True, doc="主力净占比(%)")
    net_amount_xl = Column(Text, nullable=True, doc="超大单净额(万)")
    net_pct_xl = Column(Text, nullable=True, doc="超大单净占比(%)")
    net_amount_l = Column(Text, nullable=True, doc="大单净额(万)")
    net_pct_l = Column(Text, nullable=True, doc="大单净占比(%)")
    net_amount_m = Column(Text, nullable=True, doc="中单净额(万)")
    net_pct_m = Column(Text, nullable=True, doc="中单净占比(%)")
    net_amount_s = Column(Text, nullable=True, doc="小单净额(万)")
    net_pct_s = Column(Text, nullable=True, doc="小单净占比(%)")
    pass


def init_money_flow_db():
    metadata.create_all(get_engine())


def drop_money_flow_db():
    metadata.drop_all(get_engine())
