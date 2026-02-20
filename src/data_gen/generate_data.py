import json
import random
from datetime import datetime
from src.logic.prompt_manager import PromptManager
from src.models.market_data import NinjaTraderPayload, MarketFrame, AccountStats

def generate_random_frame(base_price=18500.0, trend="flat") -> MarketFrame:
    """Generates a random market frame."""
    noise = random.uniform(-10, 10)
    trend_bias = 0
    if trend == "bullish":
        trend_bias = 20
    elif trend == "bearish":
        trend_bias = -20
        
    close = base_price + trend_bias + noise
    
    return {
        "ohlcv": {
            "timestamp": datetime.now().isoformat(),
            "open": base_price,
            "high": max(base_price, close) + random.uniform(0, 5),
            "low": min(base_price, close) - random.uniform(0, 5),
            "close": close,
            "volume": random.randint(500, 2000)
        },
        "indicators": {
            "vwap": base_price + (trend_bias * 0.5), # VWAP lazily follows price
            "waddah_attar": random.uniform(-50, 50),
            "ema_60": base_price # EMA matches base for now
        }
    }

def generate_dataset(num_samples=100, output_file="data/train.json"):
    pm = PromptManager()
    data = []
    
    print(f"Generating {num_samples} samples...")
    
    for _ in range(num_samples):
        # random scenario
        scenario = random.choice(["bullish", "bearish", "flat"])
        base_price = 18500.0 + random.uniform(-100, 100)
        
        payload_dict = {
            "tf_3m": generate_random_frame(base_price, scenario),
            "tf_15m": generate_random_frame(base_price, scenario),
            "tf_60min": generate_random_frame(base_price, scenario), # Simplify: all TFs aligned for now
            "es_context": generate_random_frame(base_price, scenario),
            "account_stats": {
                "balance": 50000.0,
                "peak_unrealized": 50000.0,
                "max_burn_observed": 0.0,
                "distance_to_liquidation": 2500.0
            }
        }
        
        # Convert to Pydantic model to validation
        payload = NinjaTraderPayload(**payload_dict)
        
        # Generate Prompt
        prompt = pm.build_market_officer_prompt(payload)
        
        # Format for Unsloth/TRL (Just the prompt is needed for GRPO, it generates completion)
        data.append({"prompt": prompt})
        
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
