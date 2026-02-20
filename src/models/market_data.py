from pydantic import BaseModel
from typing import List, Optional

class OHLCV(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class IndicatorData(BaseModel):
    vwap: float
    waddah_attar: float # Trend/Squeeze indicator
    ema_60: float
    rsi: Optional[float] = None

class MarketFrame(BaseModel):
    ohlcv: OHLCV
    indicators: IndicatorData

class AccountStats(BaseModel):
    balance: float
    peak_unrealized: float
    max_burn_observed: float
    distance_to_liquidation: float

class NinjaTraderPayload(BaseModel):
    tf_3m: MarketFrame
    tf_15m: MarketFrame
    tf_60min: MarketFrame
    es_context: Optional[MarketFrame] = None # For correlation check
    account_stats: AccountStats
