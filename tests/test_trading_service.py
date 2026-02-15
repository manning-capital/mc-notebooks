"""
Tests for TradingService.
"""

import pytest
import tempfile
import os

from src.utils.trading.service import TradingService
from src.utils.trading.models import (
    Asset,
    Portfolio,
    PositionType,
    AssetType,
)


class TestTradingServiceSetup:
    """Test TradingService initialization and setup."""

    def test_init_with_default_db(self):
        """Test initialization with default database."""
        service = TradingService(remove_existing=True)
        assert service is not None
        assert service.engine is not None
        assert service.Session is not None

    def test_init_with_temp_db(self):
        """Test initialization with temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        try:
            service = TradingService(
                db_url=f"sqlite:///{db_path}", remove_existing=True
            )
            assert service is not None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_add_asset(self):
        """Test adding an asset."""
        service = TradingService(remove_existing=True)
        asset = Asset(asset_name="BTC", asset_type=AssetType.CRYPTO)
        result = service.add_asset(asset)
        assert result.asset_id is not None
        assert result.asset_name == "BTC"
        assert result.asset_type == AssetType.CRYPTO

    def test_add_portfolio(self):
        """Test adding a portfolio."""
        service = TradingService(remove_existing=True)
        portfolio = Portfolio(portfolio_name="Test Portfolio")
        result = service.add_portfolio(portfolio)
        assert result.portfolio_id is not None
        assert result.portfolio_name == "Test Portfolio"

    def test_get_or_create_holding(self):
        """Test getting or creating a holding."""
        service = TradingService(remove_existing=True)
        portfolio = service.add_portfolio(Portfolio(portfolio_name="Test"))
        asset = service.add_asset(Asset(asset_name="BTC", asset_type=AssetType.CRYPTO))

        with service.Session() as session:
            holding = service._get_or_create_holding(
                session, portfolio.portfolio_id, asset.asset_id, PositionType.LONG
            )
            assert holding.position_size == 0.0
            assert holding.position_type == PositionType.LONG


class TestTradingServicePositionUpdates:
    """Test position size updates."""

    @pytest.fixture
    def service(self):
        """Create a fresh trading service for each test."""
        return TradingService(remove_existing=True)

    @pytest.fixture
    def portfolio(self, service):
        """Create a test portfolio."""
        return service.add_portfolio(Portfolio(portfolio_name="Test Portfolio"))

    @pytest.fixture
    def cash_asset(self, service):
        """Create cash asset."""
        return service.add_asset(
            Asset(asset_name="Cash", asset_type=AssetType.CURRENCY)
        )

    @pytest.fixture
    def btc_asset(self, service):
        """Create BTC asset."""
        return service.add_asset(Asset(asset_name="BTC", asset_type=AssetType.CRYPTO))

    def test_update_position_size(self, service, portfolio, cash_asset):
        """Test updating position size."""
        # Update position
        service.update_position_size(
            portfolio.portfolio_id, cash_asset.asset_id, 100000.0, PositionType.LONG
        )

        # Check position
        pos = service.get_position_size(
            portfolio.portfolio_id, cash_asset.asset_id, PositionType.LONG
        )
        assert pos == 100000.0

        # Update again
        service.update_position_size(
            portfolio.portfolio_id, cash_asset.asset_id, 150000.0, PositionType.LONG
        )

        # Check updated position
        pos = service.get_position_size(
            portfolio.portfolio_id, cash_asset.asset_id, PositionType.LONG
        )
        assert pos == 150000.0

    def test_get_nonexistent_position(self, service, portfolio, btc_asset):
        """Test getting position that doesn't exist."""
        pos = service.get_position_size(
            portfolio.portfolio_id, btc_asset.asset_id, PositionType.LONG
        )
        assert pos == 0.0

    def test_separate_long_short_positions(self, service, portfolio, btc_asset):
        """Test that long and short positions are separate."""
        # Set long position
        service.update_position_size(
            portfolio.portfolio_id, btc_asset.asset_id, 1000.0, PositionType.LONG
        )

        # Set short position
        service.update_position_size(
            portfolio.portfolio_id, btc_asset.asset_id, 500.0, PositionType.SHORT
        )

        # Check both positions exist separately
        long_pos = service.get_position_size(
            portfolio.portfolio_id, btc_asset.asset_id, PositionType.LONG
        )
        short_pos = service.get_position_size(
            portfolio.portfolio_id, btc_asset.asset_id, PositionType.SHORT
        )

        assert long_pos == 1000.0
        assert short_pos == 500.0
