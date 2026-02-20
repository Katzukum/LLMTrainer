from src.models.market_data import NinjaTraderPayload, MarketFrame

class PromptManager:
    def __init__(self, system_identity: str = "Apex Reasoning Agent (ARA) 2.0"):
        self.system_identity = system_identity

    def build_market_officer_prompt(self, payload: NinjaTraderPayload) -> str:
        """Constructs the full multi-timeframe reasoning prompt."""
        
        prompt = f"""### System Identity: {self.system_identity}
You are a Senior Market Officer managing an institutional prop firm account. 
Your objective is to pass the Apex Trader Funding evaluation by optimizing for the Apex Efficiency Ratio.

### Hierarchical Multi-Timeframe Analysis:

#### 1. Macro Analysis (60-Minute Horizon)
- Structural Trend: {payload.tf_60min.ohlcv.close > payload.tf_60min.indicators.ema_60 and "Bullish" or "Bearish"}
- OHLCV: {payload.tf_60min.ohlcv}
- EMA 60: {payload.tf_60min.indicators.ema_60}
- VWAP: {payload.tf_60min.indicators.vwap}

#### 2. Intermediate Analysis (15-Minute Horizon)
- Momentum: {payload.tf_15m.indicators.waddah_attar > 0 and "Expanding" or "Contracting"}
- OHLCV: {payload.tf_15m.ohlcv}
- VWAP: {payload.tf_15m.indicators.vwap}

#### 3. Micro Analysis (3-Minute Horizon)
- Order Flow Imbalance: {payload.tf_3m.indicators.waddah_attar}
- OHLCV: {payload.tf_3m.ohlcv}
- VWAP: {payload.tf_3m.indicators.vwap}

#### 4. Cross-Asset Correlation
- ES Context: {payload.es_context and payload.es_context.ohlcv.close or "N/A"}
- Correlation State: {self._get_correlation_state(payload)}

### Account Metrics & Risk Constraints:
- Current Balance: ${payload.account_stats.balance}
- Highest Unrealized Peak: ${payload.account_stats.peak_unrealized}
- Max Burn Observed: ${payload.account_stats.max_burn_observed}
- Distance to Liquidation: ${payload.account_stats.distance_to_liquidation}

### Instructions:
1. Conduct a top-down reasoning trace within <think> tags.
2. Evaluate the structural trend (60m) vs. immediate momentum (3m).
3. Prioritize "Burn" mitigation (Trailing Drawdown) over raw yield.
4. Output a final decision in JSON format matching the Pydantic schema.

Begin your reasoning.
"""
        return prompt

    def _get_correlation_state(self, payload: NinjaTraderPayload) -> str:
        if not payload.es_context:
            return "Indeterminate"
        
        nq_up = payload.tf_3m.ohlcv.close > payload.tf_3m.ohlcv.open
        es_up = payload.es_context.ohlcv.close > payload.es_context.ohlcv.open
        
        if nq_up == es_up:
            return "NQ_ES_Aligned"
        return "NQ_ES_Divergent"
