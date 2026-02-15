from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Enum,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class AssetType(enum.Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    COMMODITY = "commodity"
    CURRENCY = "currency"


class PositionType(enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(enum.Enum):
    OPEN = "open"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    CLOSED = "closed"


class Asset(Base):
    __tablename__ = "assets"
    asset_id = Column(Integer, primary_key=True, autoincrement=True)
    asset_name = Column(String, nullable=False)
    asset_type = Column(
        Enum(AssetType, native_enum=False, create_constraint=False), nullable=False
    )

    portfolio_holdings = relationship("PortfolioAssetHolding", back_populates="asset")
    transactions_from = relationship(
        "Transaction",
        back_populates="from_asset",
        foreign_keys="Transaction.from_asset_id",
    )
    transactions_to = relationship(
        "Transaction", back_populates="to_asset", foreign_keys="Transaction.to_asset_id"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"
    portfolio_id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_name = Column(String, nullable=False)

    models = relationship("Model", back_populates="portfolio")
    holdings = relationship("PortfolioAssetHolding", back_populates="portfolio")
    transactions = relationship("Transaction", back_populates="portfolio")


class Model(Base):
    __tablename__ = "models"
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    portfolio_id = Column(
        Integer, ForeignKey("portfolios.portfolio_id"), nullable=False
    )

    portfolio = relationship("Portfolio", back_populates="models")
    model_trades = relationship("ModelTrade", back_populates="model")


class PortfolioAssetHolding(Base):
    __tablename__ = "portfolio_asset_holdings"
    # Keyed by portfolio, asset, and position type
    portfolio_id = Column(
        Integer, ForeignKey("portfolios.portfolio_id"), primary_key=True, nullable=False
    )
    asset_id = Column(
        Integer, ForeignKey("assets.asset_id"), primary_key=True, nullable=False
    )
    position_type = Column(
        Enum(PositionType, native_enum=False, create_constraint=False),
        primary_key=True,
        nullable=False,
    )
    position_size = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False, default=0.0)

    portfolio = relationship("Portfolio", back_populates="holdings")
    asset = relationship("Asset", back_populates="portfolio_holdings")


class ModelTrade(Base):
    __tablename__ = "model_trades"
    model_trade_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("models.model_id"), nullable=False)
    timestamp_open = Column(DateTime, nullable=False)
    timestamp_close = Column(DateTime, nullable=True)
    attributes = Column(JSON, nullable=True)

    model = relationship("Model", back_populates="model_trades")
    transaction_pairs = relationship(
        "ModelTradeTransaction", back_populates="model_trade"
    )


class ModelTradeTransaction(Base):
    __tablename__ = "model_trade_transactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_trade_id = Column(
        Integer, ForeignKey("model_trades.model_trade_id"), nullable=False
    )
    transaction_id = Column(
        Integer, ForeignKey("transactions.transaction_id"), nullable=False
    )
    pair_order = Column(Integer, nullable=False)  # 0 or 1 for the pair

    model_trade = relationship("ModelTrade", back_populates="transaction_pairs")
    transaction = relationship("Transaction", back_populates="model_trade_assocs")


class Transaction(Base):
    __tablename__ = "transactions"
    transaction_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    portfolio_id = Column(
        Integer, ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    # Keyed by portfolio, asset, and position type for both from and to
    from_asset_id = Column(
        Integer, ForeignKey("assets.asset_id"), nullable=True, default=None
    )
    to_asset_id = Column(
        Integer, ForeignKey("assets.asset_id"), nullable=True, default=None
    )
    from_position_type = Column(
        Enum(PositionType, native_enum=False, create_constraint=False),
        nullable=True,
        default=None,
    )
    to_position_type = Column(
        Enum(PositionType, native_enum=False, create_constraint=False),
        nullable=True,
        default=None,
    )
    # Transaction details
    from_asset_position = Column(Float, nullable=False)
    to_asset_price = Column(Float, nullable=False)

    portfolio = relationship("Portfolio", back_populates="transactions")
    from_asset = relationship(
        "Asset",
        back_populates="transactions_from",
        foreign_keys=[from_asset_id],
    )
    to_asset = relationship(
        "Asset",
        back_populates="transactions_to",
        foreign_keys=[to_asset_id],
    )
    model_trade_assocs = relationship(
        "ModelTradeTransaction", back_populates="transaction"
    )
