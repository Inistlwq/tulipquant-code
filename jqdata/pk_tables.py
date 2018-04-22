# coding: utf-8
import jqdata
import datetime
from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

__all__ = [
    'SecurityEntity',
    'IndexEntity',
    'MarginStockEntity',
    'init_pk_db',
    'drop_pk_db',
    'get_session',
    'get_engine',
]

Base = declarative_base()
metadata = Base.metadata


_engine = None
_DBSession = None


def get_engine():
    global _engine
    if _engine is None:
        sqlite_uri = jqdata.get_config().get_sqlite_db_uri()
        _engine = create_engine(sqlite_uri, echo=False)
    return _engine


def get_session():
    global _DBSession
    if _DBSession is None:
        _DBSession = sessionmaker(bind=get_engine())
    return _DBSession()


class SecurityEntity(Base):
    """
    security 基础数据
    """
    __tablename__ = 'security'
    code = Column(Text, primary_key=True, nullable=False, doc="股票代码(带后缀: .XSHE/.XSHG)")
    type = Column(Text, nullable=False, doc="类型")
    display_name = Column(Text, nullable=False, doc="display name")
    name = Column(Text, nullable=False, doc="name")
    start_date = Column(Date, nullable=False, doc="开始日期")
    end_date = Column(Date, nullable=False, doc="开始日期")
    subtype = Column(Text, doc="子类型")
    dic_json = Column(Text, nullable=False, doc='所有字段json')
    pass


class IndexEntity(Base):
    """
    index 基础数据
    """
    __tablename__ = 'indexs'
    code = Column(Text, primary_key=True, nullable=False, doc="指数代码")
    index_json = Column(Text, nullable=False, doc='index字段json')
    pass


class MarginStockEntity(Base):
    """
    margin stock 基础数据
    """
    __tablename__ = 'margin_stocks'
    margin_date = Column(Text, primary_key=True, nullable=False, doc="日期")
    margin_json = Column(Text, nullable=False, doc='margin字段json')


class ConceptEntity(Base):
    """
    concept 基础数据
    """
    __tablename__ = 'concept'
    id = Column(Integer, primary_key=True)
    code = Column(Text, nullable=False, doc="概念代码")
    name = Column(Text, nullable=False, doc="概念名称")
    start_date = Column(Text, nullable=False, doc="概念开始日期")
    stock = Column(Text, nullable=False, doc='stock code')
    stock_startdate = Column(Text, nullable=False, doc="stock开始日期")
    stock_enddate = Column(Text, nullable=False, doc="stock结束日期")
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now)


class DominantFutureEntity(Base):
    """
    dominant future 基础数据
    """
    __tablename__ = 'dominant_future'
    df_id = Column(Integer, primary_key=True, nullable=False)
    df_date = Column(Text, nullable=False, doc="日期")
    df_symbol = Column(Text, nullable=False, doc='future symbol')
    df_code = Column(Text, nullable=False, doc='dominant future code')


class FactorEntity(Base):
    """
    factor 基础数据
    """
    __tablename__ = 'factor'
    factor_id = Column(Integer, primary_key=True, nullable=False)
    factor_code = Column(Text, nullable=False, doc="factor代码")
    factor_date = Column(Text, nullable=False, doc="日期")
    factor_val = Column(Float, nullable=False, doc="factor value")


class IndustryEntity(Base):
    """
    industry 基础数据
    """
    __tablename__ = 'industry'
    id = Column(Integer, primary_key=True)
    code = Column(Text, nullable=False, doc="行业 code")
    type_ = Column(Text, nullable=False, doc="行业 type")
    name = Column(Text, nullable=False, doc="行业 name")
    start_date = Column(Text, nullable=False, doc="开始日期")
    stock = Column(Text, nullable=False, doc="stock code")
    stock_startdate = Column(Text, nullable=False, doc="stock开始日期")
    stock_enddate = Column(Text, nullable=False, doc="stock结束日期")
    created_at = Column(DateTime, default=datetime.datetime.now)
    updated_at = Column(DateTime, default=datetime.datetime.now)


Index('index_code_key', IndexEntity.code)
Index('industry_code_key', IndustryEntity.code)
Index('concept_code_key', ConceptEntity.code)


def init_pk_db():
    metadata.create_all(get_engine())


def drop_pk_db():
    metadata.drop_all(get_engine())
