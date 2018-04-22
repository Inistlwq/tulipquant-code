

# coding: utf-8
from sqlalchemy import BigInteger, Column, DateTime, Integer, Numeric, SmallInteger, String, Table, Text, text
from sqlalchemy.dialects.mysql.base import LONGBLOB
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class STK_AF_FORECAST(Base):
    __tablename__ = "STK_AF_FORECAST"

    REPORTID = Column(Numeric(20, 0), primary_key=True, nullable=False)
    SECURITYID = Column(Numeric(20, 0), primary_key=True, nullable=False)
    SYMBOL = Column(String(20, u'utf8_bin'))
    REPORTDATE = Column(DateTime)
    FORECASTYEAR = Column(DateTime, primary_key=True, nullable=False)
    FORECASTTARGETID = Column(String(20, u'utf8_bin'), primary_key=True, nullable=False)
    FORECASTTARGET = Column(String(100, u'utf8_bin'), nullable=True)
    FORECASTTARGET_EN = Column(String(200, u'utf8_bin'), nullable=True)
    TARGETVALUE = Column(Numeric(24, 6))
    LASTTARGETVALUE = Column(Numeric(24, 6))
    INSTITUTIONNAME = Column(String(200, u'utf8_bin'), nullable=True)
    INSTITUTIONNAME_EN = Column(String(400, u'utf8_bin'), nullable=True)
    INSTITUTIONID = Column(String(200, u'utf8_bin'), nullable=True)
    ANALYST = Column(String(200, u'utf8_bin'), nullable=True)
    ANALYST_EN = Column(String(600, u'utf8_bin'), nullable=True)
    ANALYSTID = Column(String(2000, u'utf8_bin'), nullable=True)
    LASTREPORTDATE = Column(DateTime)
