# coding: utf-8
import datetime
from sqlalchemy import BigInteger, Column, DateTime, Integer, Numeric, SmallInteger, String, Table, Text, text
from sqlalchemy.dialects.mysql import TINYINT, TIMESTAMP, DECIMAL
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class MAC_AREA_FOREIGN_REGISTER(Base):
    """
    分地区外商投资企业年底注册登记情况表（年度）
    """
    __tablename__ = "MAC_AREA_FOREIGN_REGISTER"

    id = Column(Integer, primary_key=True)
    stat_year = Column(String(4), nullable=False)
    area_code = Column(String(20), nullable=False)
    area_name = Column(String(100), nullable=False)
    enterprise_num = Column(DECIMAL(20, 4))
    invest = Column(DECIMAL(20, 4))
    registered_capital = Column(DECIMAL(20, 4))
    foreign_registered = Column(DECIMAL(20, 4))
    # status = Column(TINYINT(display_width=4), default=0)
    # addTime = Column(TIMESTAMP, default=datetime.datetime.now)
    # modTime = Column(TIMESTAMP)