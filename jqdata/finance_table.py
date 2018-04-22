# coding: utf-8
from sqlalchemy import Column, Date, DateTime, Float, Index, Integer, String, text, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from jqdata import get_config
from .db_utils import get_sql_runner

__all__ = [
    'StkAbnormal',
    'StkLockShares',
    'StkMoneyFlow',
    'FundMainInfo',
    'FundUnitInfo',
    'FundDividend',
    'FundNetValue',
    'FundShareInfo',
    'FundPortfolioStock',
    'FundPortfolioBond',
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
        cfg = get_config()
        if not cfg.FUNDAMENTALS_SERVERS:
            raise RuntimeError(
                "you must config FUNDAMENTALS_SERVERS for jqdata")
        sql_runner = get_sql_runner(
            server_name='fundamentals', keep_connection=cfg.KEEP_DB_CONNECTION,
            retry_policy=cfg.DB_RETRY_POLICY)
        _engine = sql_runner.engine
    return _engine


def get_session():
    global _DBSession
    if _DBSession is None:
        _DBSession = sessionmaker(bind=get_engine())
    return _DBSession()


class StkAbnormal(Base):
    """
    龙虎榜
    """

    __tablename__ = 'STK_ABNORMAL'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="股票代码(带后缀: .XSHE/.XSHG)")
    day = Column(String(10), nullable=False, server_default=text("'0000-00-00'"), doc="交易日")
    direction = Column(String(8), nullable=False, doc="类型，ALL:汇总, SELL:卖，BUY:买")
    rank = Column(Integer, nullable=False, doc="排名，0表示汇总，1~5表示排名")
    abnormal_code = Column(Integer, nullable=False, doc="异常波动类型")
    abnormal_name = Column(String(100), nullable=False, doc="异常波动名称")
    sales_depart_name = Column(String(128), nullable=True, doc="营业部名称（如果abnormal_type=0，为空）")
    buy_value = Column(Numeric(20, 4), nullable=True, doc="买入金额")
    buy_rate = Column(Numeric(10, 4), nullable=True, doc="买入金额占比（买入金额/市场总成交额)")
    sell_value = Column(Numeric(20, 4), nullable=True, doc="卖出金额")
    sell_rate = Column(Numeric(10, 4), nullable=True, doc="卖出金额占比（卖出金额/市场总成交额)")
    total_value = Column(Numeric(20, 4), nullable=True, doc="成交金额(买入金额+卖出金额)，单位元")
    net_value = Column(Numeric(20, 4), nullable=True, doc="净额（买入金额 - 卖出金额）")
    amount = Column(Numeric(20, 4), nullable=True, doc="市场总成交额")


class StkLockShares(Base):
    """
    限售解禁
    """

    __tablename__ = 'STK_LOCK_SHARES'

    id = Column(Integer, primary_key=True)
    code = Column(String(20), nullable=False, doc="股票代码(带后缀: .XSHE/.XSHG)")
    day = Column(String(10), nullable=False, server_default=text("'0000-00-00'"), doc="解禁日期")
    num = Column(Numeric(20, 4), nullable=False, doc="解禁股数")
    rate1 = Column(Numeric(20, 4), nullable=True, doc="解禁股数/总股本")
    rate2 = Column(Numeric(20, 4), nullable=True, doc="解禁股数/流通总股本")
    reason_name = Column(String(300), nullable=True, doc="解禁原因")
    reason_id = Column(Integer, nullable=True, doc="")


class StkMoneyFlow(Base):
    """
    money flow 数据
    """
    __tablename__ = 'STK_MONEY_FLOW'

    id = Column(Integer, primary_key=True)
    sec_code = Column(String(12), nullable=False, doc="股票代码(带后缀: .XSHE/.XSHG)")
    date = Column(Date, nullable=False, doc="日期")
    change_pct = Column(Numeric(20, 4), nullable=True, doc="涨跌幅")
    net_amount_main = Column(Numeric(20, 4), nullable=True, doc="主力净额(万)")
    net_pct_main = Column(Numeric(20, 4), nullable=True, doc="主力净占比(%)")
    net_amount_xl = Column(Numeric(20, 4), nullable=True, doc="超大单净额(万)")
    net_pct_xl = Column(Numeric(20, 4), nullable=True, doc="超大单净占比(%)")
    net_amount_l = Column(Numeric(20, 4), nullable=True, doc="大单净额(万)")
    net_pct_l = Column(Numeric(20, 4), nullable=True, doc="大单净占比(%)")
    net_amount_m = Column(Numeric(20, 4), nullable=True, doc="中单净额(万)")
    net_pct_m = Column(Numeric(20, 4), nullable=True, doc="中单净占比(%)")
    net_amount_s = Column(Numeric(20, 4), nullable=True, doc="小单净额(万)")
    net_pct_s = Column(Numeric(20, 4), nullable=True, doc="小单净占比(%)")


class FundMainInfo(Base):
    """
    场外基金主体信息
    """
    __tablename__ = 'FUND_MAIN_INFO'

    id = Column(Integer, primary_key=True)
    main_code = Column(String(12), nullable=False, doc="基金主体代码")
    name = Column(String(100), nullable=True, doc="基金名称")
    advisor = Column(String(100), nullable=True, doc="基金管理人")
    trustee = Column(String(100), nullable=True, doc="基金托管人")
    operate_mode_id = Column(Integer, nullable=True, doc="基金运作方式编码")
    operate_mode = Column(String(32), nullable=True, doc="基金运作方式")
    underlying_asset_type_id = Column(Integer, nullable=True, doc="投资标的类别编码")
    underlying_asset_type = Column(String(32), nullable=True, doc="投资标的类别")
    start_date = Column(Date, nullable=True, doc="成立日期")
    end_date = Column(Date, nullable=True, doc="结束日期")


class FundUnitInfo(Base):
    """
    场外基金份额信息
    """
    __tablename__ = 'FUND_UNIT_INFO'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    full_name = Column(String(100), nullable=False, doc="基金全称")
    short_name = Column(String(32), nullable=True, doc="基金简称")
    main_code = Column(String(12), nullable=True, doc="主体基金代码")
    main_name = Column(String(100), nullable=True, doc="主体基金名称")
    advisor = Column(String(100), nullable=True, doc="基金管理人")
    trustee = Column(String(100), nullable=True, doc="基金托管人")
    operate_mode_id = Column(Integer, nullable=True, doc="基金运作类型编码")
    operate_mode = Column(String(32), nullable=True, doc="基金运作类型")
    underlying_asset_type_id = Column(Integer, nullable=True, doc="投资标的类型编码")
    underlying_asset_type = Column(String(32), nullable=True, doc="投资标的类型")
    start_date = Column(Date, nullable=True, doc="基金开始日期")
    duration = Column(Integer, nullable=True, doc="存续期限")
    duration_start_date = Column(Date, nullable=True, doc="存续期起始日")
    duration_end_date = Column(Date, nullable=True, doc="存续期终止日")
    end_date = Column(Date, nullable=True, doc="基金结束日期")


class FundDividend(Base):
    """
    场外基金分红拆分和合并的方案
    """
    __tablename__ = 'FUND_DIVIDEND'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    pub_date = Column(Date, nullable=True, doc="公布日期")
    event_id = Column(Integer, nullable=False, doc="事项类别")
    event = Column(String(100), nullable=False, doc="事项名称")
    distribution_date = Column(Date, nullable=True, doc="分配收益日")
    process_id = Column(Integer, nullable=True, doc="方案进度编码")
    process = Column(String(100), nullable=True, doc="方案进度")
    proportion = Column(Numeric(20, 8), nullable=True, doc="派现比例")
    split_ratio = Column(Numeric(20, 8), nullable=True, doc="分拆（合并、赠送）比例")
    record_date = Column(Date, nullable=True, doc="权益登记日")
    ex_date = Column(Date, nullable=True, doc="除息日")
    fund_paid_date = Column(Date, nullable=True, doc="基金红利派发日")
    redeem_date = Column(Date, nullable=True, doc="再投资赎回起始日")
    dividend_implement_date = Column(Date, nullable=True, doc="分红实施公告日")
    dividend_cancel_date = Column(Date, nullable=True, doc="取消分红公告日")
    otc_ex_date = Column(Date, nullable=True, doc="场外除息日")
    pay_date = Column(Date, nullable=True, doc="红利派发日")
    new_share_code = Column(String(10), nullable=True, doc="新增份额基金代码")
    new_share_name = Column(String(100), nullable=True, doc="新增份额基金名称")


class FundNetValue(Base):
    """
    场外基金的净值数据
    """
    __tablename__ = 'FUND_NET_VALUE'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    day = Column(Date, nullable=False, doc="交易日")
    net_value = Column(Numeric(20, 6), nullable=True, doc="单位净值")
    sum_value = Column(Numeric(20, 6), nullable=True, doc="累计净值")
    factor = Column(Numeric(20, 6), nullable=True, doc="复权因子")
    acc_factor = Column(Numeric(20, 6), nullable=True, doc="累计复权因子")
    refactor_net_value = Column(Numeric(20, 6), nullable=True, doc="复权净值")


class FundShareInfo(Base):
    """
    场外基金份额变动信息
    """
    __tablename__ = 'FUND_SHARE_INFO'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    name = Column(String(100), nullable=True, doc="基金名称")
    period_start = Column(Date, nullable=False, doc="开始日期")
    period_end = Column(Date, nullable=False, doc="报告期")
    pub_date = Column(Date, nullable=False, doc="公告日期")
    report_type_id = Column(Integer, nullable=True, doc="报告类型编码")
    report_type = Column(String(32), nullable=True, doc="报告类型")
    end_share = Column(Numeric(20, 4), nullable=True, doc="期末基金份额(份)")
    start_share = Column(Numeric(20, 4), nullable=True, doc="期初基金份额(份)")
    purchase_share = Column(Numeric(20, 4), nullable=True, doc="期间基金份额(份)")
    redeem_share = Column(Numeric(20, 4), nullable=True, doc="期间赎回份额(份)")
    change_share = Column(Numeric(20, 4), nullable=True, doc="期间拆分变动份额(份)")


class FundPortfolioStock(Base):
    """
    场外基金持仓股票组合
    """
    __tablename__ = 'FUND_PORTFOLIO_STOCK'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    period_start = Column(Date, nullable=False, doc="开始日期")
    period_end = Column(Date, nullable=False, doc="报告期")
    pub_date = Column(Date, nullable=False, doc="公告日期")
    report_type_id = Column(Integer, nullable=False, doc="报告类型编码")
    report_type = Column(String(32), nullable=False, doc="报告类型")
    rank = Column(Integer, nullable=False, doc="股票持仓，1-10")
    symbol = Column(String(32), nullable=False, doc="股票代码")
    name = Column(String(100), nullable=True, doc="股票名称")
    shares = Column(Numeric(20, 4), nullable=True, doc="持有股票数量(股)")
    market_cap = Column(Numeric(20, 4), nullable=True, doc="持有股票市值(元)")
    proportion = Column(Numeric(10, 4), nullable=True, doc="占净值比例(%)")


class FundPortfolioBond(Base):
    """
    场外基金持仓债券组合
    """
    __tablename__ = 'FUND_PORTFOLIO_BOND'

    id = Column(Integer, primary_key=True)
    code = Column(String(12), nullable=False, doc="基金代码")
    period_start = Column(Date, nullable=False, doc="开始日期")
    period_end = Column(Date, nullable=False, doc="结束日期")
    pub_date = Column(Date, nullable=False, doc="公告日期")
    report_type_id = Column(Integer, nullable=False, doc="报告类型编码")
    report_type = Column(String(32), nullable=False, doc="报告类型")
    rank = Column(Integer, nullable=True, doc="债券持仓")
    symbol = Column(String(32), nullable=False, doc="债券代码")
    name = Column(String(100), nullable=True, doc="债券名称")
    shares = Column(Numeric(20, 4), nullable=True, doc="持有债券数量(张)")
    market_cap = Column(Numeric(20, 4), nullable=True, doc="持有债券市值(元)")
    proportion = Column(Numeric(10, 4), nullable=True, doc="占净值比例(%)")


class UsIdxDaily(Base):
    """
    美股指数基金净值数据
    """
    __tablename__ = 'US_IDX_DAILY'

    id = Column(Integer, primary_key=True)
    code = Column(String(64), nullable=False, doc="指数代码")
    day = Column(Date, nullable=False, doc="日期")
    pre_close = Column(Numeric(20, 6), nullable=True, doc="前收价")
    open = Column(Numeric(20, 6), nullable=True, doc="开盘价")
    close = Column(Numeric(20, 6), nullable=True, doc="收盘价")
    low = Column(Numeric(20, 6), nullable=True, doc="最低价")
    high = Column(Numeric(20, 6), nullable=True, doc="最高价")
    volume = Column(Numeric(20, 6), nullable=True, doc="成交量")
