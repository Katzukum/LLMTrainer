import re
import json

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks for the presence of <think> and </think> tags."""
    count_text = []
    for completion in completions:
        # Check if the completion contains the required tags
        count = completion.count("<think>") + completion.count("</think>")
        # Reward if both tags are present
        if count == 2:
            count_text.append(0.5)
        else:
            count_text.append(0.0)
    return count_text

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion contains a valid XML structure."""
    pattern = r"<think>.*?</think>\s*\{.*\}"
    matches = [re.search(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that strictly checks if the completion ends with a valid JSON block."""
    pattern = r"^<think>\n.*?\n</think>\n\{.*\}$"
    matches = [re.search(pattern, r, re.DOTALL) for r in completions]
    return [0.5 if match else 0.0 for match in matches]

def apex_efficiency_reward_func(completions, prompts: list[str], **kwargs) -> list[float]:
    """
    Simulates the trade outcome based on the model's JSON decision and calculates the Apex Efficiency Ratio.
    Reward = (Total Net Profit) / (Max Burn Observed)
    """
    rewards = []
    
    # We would typically parse the prompt to get the market context
    # And parse the completion to get the trade action
    
    for completion, prompt in zip(completions, prompts):
        try:
            # Extract JSON block
            json_str = completion[completion.rfind("{"):completion.rfind("}")+1]
            decision = json.loads(json_str)
            
            action = decision.get("execution", {}).get("action", "FLAT")
            
            if action == "FLAT":
                rewards.append(0.1) # Small reward for staying flat in uncertain conditions
                continue

            # Mock Simulation Logic for RLVR
            # In a real scenario, this would check future price data relative to entry
            # For now, we simulate based on "expected burn" in the JSON to reinforce risk awareness
            
            risk_mgmt = decision.get("risk_management", {})
            expected_burn = risk_mgmt.get("expected_burn", 100.0)
            target = risk_mgmt.get("apex_efficiency_target", 2.0)
            
            # Simple heuristic: Reward high efficiency targets
            if expected_burn < 50.0:
                 rewards.append(2.0)
            elif expected_burn > 200.0:
                 rewards.append(-1.0) # Penalty for high burn risk
            else:
                 rewards.append(0.5)

        except Exception:
            rewards.append(-1.0) # Penalty for invalid JSON/Execution
            
    return rewards

def regime_consistency_reward_func(completions, prompts: list[str], **kwargs) -> list[float]:
    """
    Rewards the model if the 'market_regime' classification aligns with the prompted indicators.
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        try:
            # Extract JSON
            json_str = completion[completion.rfind("{"):completion.rfind("}")+1]
            data = json.loads(json_str)
            
            regime = data.get("market_regime", {})
            tf_60min_regime = regime.get("tf_60min", "")
            
            # Check prompt for "Bullish" or "Bearish" keyword in 60m section
            # Ideally strict parsing, but keyword search works for RLVR speed
            if "Structural Trend: Bullish" in prompt and "Bullish" in tf_60min_regime:
                rewards.append(1.0)
            elif "Structural Trend: Bearish" in prompt and "Bearish" in tf_60min_regime:
                 rewards.append(1.0)
            else:
                 rewards.append(0.0)
                 
        except Exception:
            rewards.append(0.0)
            
    return rewards
