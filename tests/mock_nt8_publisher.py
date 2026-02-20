import zmq
import json
import time
from datetime import datetime

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

def get_dummy_payload():
    now = datetime.now().isoformat()
    ohlcv = {
        "timestamp": now,
        "open": 18500.0,
        "high": 18510.0,
        "low": 18495.0,
        "close": 18505.0,
        "volume": 1200
    }
    indicators = {
        "vwap": 18502.5,
        "waddah_attar": 12.5,
        "ema_60": 18480.0
    }
    frame = {"ohlcv": ohlcv, "indicators": indicators}
    
    payload = {
        "tf_3m": frame,
        "tf_15m": frame,
        "tf_60min": frame,
        "account_stats": {
            "balance": 50125.0,
            "peak_unrealized": 50250.0,
            "max_burn_observed": 125.0,
            "distance_to_liquidation": 2375.0
        }
    }
    return payload

print("Mock NinjaTrader Publisher started on port 5555. Sending payload every 5 seconds...")

try:
    while True:
        payload = get_dummy_payload()
        socket.send_string(json.dumps(payload))
        print(f"Sent: {datetime.now().strftime('%H:%M:%S')} - Balance: {payload['account_stats']['balance']}")
        time.sleep(5)
except KeyboardInterrupt:
    print("Publisher stopped.")
finally:
    socket.close()
    context.term()
