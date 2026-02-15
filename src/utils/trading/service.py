import os
import sys

# Add the workspace root to Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils.trading.models import (
    Base,
    Asset,
    Portfolio,
    Model,
    PortfolioAssetHolding,
    Transaction,
    PositionType,
)
import datetime as dt
from typing import Optional


class TradingService:
    def __init__(
        self,
        db_url: str = "sqlite:///trading_service.db",
        remove_existing: bool = False,
    ):
        # Create an SQLite database using SQLAlchemy ORM
        self.engine = create_engine(db_url, echo=False, future=True)
        if remove_existing:
            Base.metadata.drop_all(self.engine)
        # Create all tables if they don't exist
        Base.metadata.create_all(self.engine)
        # Set up a sessionmaker
        self.Session = sessionmaker(bind=self.engine)

    def add_asset(self, asset: Asset) -> Asset:
        with self.Session() as session:
            session.add(asset)
            session.commit()
            session.refresh(asset)
            return asset

    def add_portfolio(self, portfolio: Portfolio) -> Portfolio:
        with self.Session() as session:
            session.add(portfolio)
            session.commit()
            session.refresh(portfolio)
            return portfolio

    def add_model(self, model: Model) -> Model:
        with self.Session() as session:
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def get_position_size(
        self,
        portfolio_id: int,
        asset_id: int,
        position_type: PositionType = PositionType.LONG,
    ) -> float:
        # Get the position size for the specific position_type (LONG/SHORT)
        with self.Session() as session:
            position = (
                session.query(PortfolioAssetHolding)
                .filter(
                    PortfolioAssetHolding.portfolio_id == portfolio_id,
                    PortfolioAssetHolding.asset_id == asset_id,
                    PortfolioAssetHolding.position_type == position_type,
                )
                .one_or_none()
            )
            if position is None:
                return 0.0
            return position.position_size

    def update_position_size(
        self,
        portfolio_id: int,
        asset_id: int,
        position_size: float,
        position_type: PositionType = PositionType.LONG,
    ):
        # Check if the asset exists
        with self.Session() as session:
            asset = (
                session.query(Asset).filter(Asset.asset_id == asset_id).one_or_none()
            )
            if asset is None:
                raise ValueError(
                    f"Asset {asset_id} not found in portfolio {portfolio_id}"
                )

            # Update or insert for the specific position type
            holding = (
                session.query(PortfolioAssetHolding)
                .filter(
                    PortfolioAssetHolding.portfolio_id == portfolio_id,
                    PortfolioAssetHolding.asset_id == asset_id,
                    PortfolioAssetHolding.position_type == position_type,
                )
                .one_or_none()
            )

            if holding is not None:
                holding.position_size = position_size
            else:
                holding = PortfolioAssetHolding(
                    portfolio_id=portfolio_id,
                    asset_id=asset_id,
                    position_size=position_size,
                    position_type=position_type,
                )
                session.add(holding)
            session.commit()

    def add_transaction(
        self,
        portfolio_id: int,
        from_asset_id: int,
        to_asset_id: int,
        from_asset_position: float,
        to_asset_price: float,
        from_position_type: PositionType = PositionType.LONG,
        to_position_type: PositionType = PositionType.LONG,
        timestamp: Optional[dt.datetime] = None,
    ):
        with self.Session() as session:
            session.add(
                Transaction(
                    timestamp=timestamp,
                    portfolio_id=portfolio_id,
                    from_asset_id=from_asset_id,
                    to_asset_id=to_asset_id,
                    from_asset_position=from_asset_position,
                    to_asset_price=to_asset_price,
                    from_position_type=from_position_type,
                    to_position_type=to_position_type,
                )
            )
            session.commit()

    def _get_or_create_holding(
        self, session, portfolio_id: int, asset_id: int, position_type: PositionType
    ) -> PortfolioAssetHolding:
        """Get or create a holding for a specific position type."""
        holding = (
            session.query(PortfolioAssetHolding)
            .filter(
                PortfolioAssetHolding.portfolio_id == portfolio_id,
                PortfolioAssetHolding.asset_id == asset_id,
                PortfolioAssetHolding.position_type == position_type,
            )
            .one_or_none()
        )
        if holding is None:
            holding = PortfolioAssetHolding(
                portfolio_id=portfolio_id,
                asset_id=asset_id,
                position_type=position_type,
                position_size=0.0,
            )
            session.add(holding)
        return holding

    def execute_trade(
        self,
        portfolio_id: int,
        from_asset_id: int,
        to_asset_id: int,
        from_asset_quantity: float,
        to_asset_quantity: float,
        from_position_type: PositionType = PositionType.LONG,
        to_position_type: PositionType = PositionType.LONG,
    ):
        pass
