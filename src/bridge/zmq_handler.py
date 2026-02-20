import asyncio
import json
import logging
import zmq
import zmq.asyncio
from typing import Callable, Awaitable
from src.models.market_data import NinjaTraderPayload

logger = logging.getLogger(__name__)

class ZMQSubscriber:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555):
        self.url = f"tcp://{host}:{port}"
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.is_running = False

    async def start(self, callback: Callable[[NinjaTraderPayload], Awaitable[None]]):
        """Starts the subscriber loop and calls back for each payload."""
        logger.info(f"Connecting to ZMQ at {self.url}")
        self.socket.connect(self.url)
        self.is_running = True
        
        try:
            while self.is_running:
                # Wait for raw message
                raw_msg = await self.socket.recv_string()
                
                try:
                    data = json.loads(raw_msg)
                    payload = NinjaTraderPayload(**data)
                    await callback(payload)
                except json.JSONDecodeError:
                    logger.error("Failed to decode ZMQ message as JSON")
                except Exception as e:
                    logger.error(f"Error parsing/processing ZMQ payload: {e}")
                    
        except asyncio.CancelledError:
            logger.info("ZMQ Subscriber task cancelled")
        finally:
            self.stop()

    def stop(self):
        """Stops the subscriber and cleans up resources."""
        self.is_running = False
        if not self.socket.closed:
            self.socket.close()
        self.ctx.term()
        logger.info("ZMQ Subscriber stopped")
