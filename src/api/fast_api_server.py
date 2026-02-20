import asyncio
import logging
import sys

# Windows-specific ZMQ event loop fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks
from openai import AsyncOpenAI
from src.bridge.zmq_handler import ZMQSubscriber
from src.logic.prompt_manager import PromptManager
from src.models.market_data import NinjaTraderPayload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Apex Reasoning Agent (ARA) 2.0 Orchestrator")

# Initialize components
prompt_manager = PromptManager()
zmq_subscriber = ZMQSubscriber(host="127.0.0.1", port=5555)

# Initialize OpenAI client for local llama-server
# llama-server defaults to port 8080
llm_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

async def process_market_data(payload: NinjaTraderPayload):
    """Callback function triggered on every NinjaTrader data packet."""
    logger.info(f"Received market data. Balance: {payload.account_stats.balance}")
    
    # 1. Build the hierarchical prompt
    prompt = prompt_manager.build_market_officer_prompt(payload)
    
    # 2. Trigger Inference (Async)
    try:
        logger.info("Requesting inference from llama-server...")
        response = await llm_client.chat.completions.create(
            model="qwen3", # Or whatever your model name is
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Deterministic for trading
            extra_body={
                "stop": ["</think>"] # If you want to stop after reasoning
            }
        )
        
        reasoning = response.choices[0].message.content
        logger.info(f"Inference complete. Reasoning snippet: {reasoning[:100]}...")
        
        # TODO Phase 6: Extract JSON and send back to ZMQ
        
    except Exception as e:
        logger.error(f"Error during LLM inference: {e}")

@app.on_event("startup")
async def startup_event():
    """Start the ZMQ subscriber as a background task on startup."""
    asyncio.create_task(zmq_subscriber.start(process_market_data))
    logger.info("FastAPI Orchestrator started. ZMQ Bridge active.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup ZMQ resources on shutdown."""
    zmq_subscriber.stop()
    logger.info("FastAPI Orchestrator shutting down.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "zmq_active": zmq_subscriber.is_running}
