Phase 1: Environment & Infrastructure Setup

    Hardware & OS Verification: Ensure your local machine is running an NVIDIA RTX 4090 (24GB VRAM) on Linux or Windows/WSL.

    Install Core Dependencies: Set up your Python environment and install the required machine learning and orchestration libraries: unsloth, vllm, fastapi, pyzmq, and pydantic.

    NinjaTrader 8 Preparation: Ensure NinjaTrader 8 is installed with Visual Studio integration enabled for C# development, and import the necessary C# bindings for ZeroMQ (e.g., ZeroMQ.dll and libzmq.dll).

Phase 2: NinjaTrader 8 Data Aggregation (C#)

    Create the Market Officer Script: Write a custom NinjaScript strategy that calculates and aggregates the multi-timeframe data (3m, 15m, and 60m OHLCV) alongside indicators like VWAP and Waddah Attar.

    Implement Apex Risk Metrics: Integrate the trailing drawdown logic into the C# script. Track the CurrentBalance, calculate the highestUnrealizedPeak (which includes Maximum Favorable Excursion), and determine the maxBurnObserved.

    Establish ZeroMQ Publisher: Instantiate a ZeroMQ PUB or PUSH socket within the NT8 script to serialize the market state and account metrics into a lightweight payload and broadcast it at the close of every 3-minute bar.

Phase 3: FastAPI Orchestration Engine (Python)

    Initialize the Async Server: Create a FastAPI application to serve as the central bridge between NinjaTrader and the LLM.

    Bind ZeroMQ to the Event Loop: Hook a ZeroMQ SUB or PULL socket into the FastAPI asyncio event loop so it can continuously listen for the NT8 payload without blocking other processes.

    Prompt Engineering Engine: Write a function within FastAPI that receives the NT8 payload and dynamically injects the data into the "Multi-Timeframe Market Officer" prompt template.

Phase 4: Model Fine-Tuning with Unsloth & GRPO

    Load Qwen 3-8B: Use Unsloth to load the Qwen 3-8B model using 4-bit dynamic quantization. This reduces the memory footprint from ~16GB to roughly 6GB, leaving enough VRAM on your RTX 4090 for the necessary RL context windows.

    Define Custom Reward Functions: Create Python-based reward functions for the GRPOTrainer :

        Format Reward: Uses regex to verify the output contains <think> tags followed by valid JSON.

        Apex Efficiency Reward: Calculates (Total Net Profit) / (Max Burn Observed) to penalize trades with deep intra-trade pullbacks.

        MAE Adherence Penalty: Applies a massive penalty if the model's simulated stop-loss distance exceeds 30% of the start-of-day profit balance.

    Execute RLVR: Run the GRPO training loop on historical multi-timeframe data to align the model's behavior with the strict proprietary trading rules.

Phase 5: vLLM Structured Inference

    Deploy the Inference Engine: Load your fine-tuned Unsloth model into a local vLLM server to handle high-throughput generation.

    Enable Reasoning Parser: Start the vLLM server with the appropriate reasoning parser flag (e.g., --reasoning-parser qwen3) so the engine properly handles the <think> reasoning traces.

    Implement Guided Decoding: Define a rigorous Python Pydantic schema that matches your execution JSON format. Pass this schema to the vLLM API request using the guided_json or response_format parameter. This ensures the tokens generated after the closing </think> tag perfectly adhere to the JSON schema.

Phase 6: The Execution Loop

    Inference Request: FastAPI asynchronously sends the formatted prompt to the local vLLM engine.

    Parse Dual-Stream Output: Once the generation is complete, FastAPI extracts the human-readable reasoning from the <think> tags and the structured JSON object.

    Monitor & Execute:

        Push the reasoning summary to a Discord or Telegram webhook for human oversight.

        Serialize the JSON execution block and send it back to NinjaTrader 8 via a ZeroMQ socket to immediately place the LONG, SHORT, or FLAT order based on the LLM's logic.