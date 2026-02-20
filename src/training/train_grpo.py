from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOTrainer, GRPOConfig
import torch
from datasets import load_dataset
import sys
import os

# Set tokenizer to left for generation, and ensure pad token is set
# This must happen before model loading sometimes or just after
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logic.grpo_rewards import (
    xmlcount_reward_func, 
    soft_format_reward_func, 
    strict_format_reward_func, 
    apex_efficiency_reward_func,
    regime_consistency_reward_func
)

# 1. Patch GRPO for memory efficiency
PatchFastRL("GRPO", FastLanguageModel)

def train():
    max_seq_length = 2048 # Increased for longer reasoning chains (was 1024)
    max_prompt_length = 768
    max_completion_length = 512
    lora_rank = 32 # Higher rank = more capacity for reasoning patterns

    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit", 
        max_seq_length = max_seq_length,
        load_in_4bit = True, 
        fast_inference = False, # Disable vLLM fast inference (Windows incompatibility)
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.6, 
    )

    # Force disable native thinking in both model and tokenizer
    # This ensures the model doesn't inject hidden tokens that break GRPO mask lengths
    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False
    if hasattr(tokenizer, "enable_thinking"):
        tokenizer.enable_thinking = False
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False

    # FIX for GRPO tensor size mismatch error:
    # If generation relies on `max_length`, generated tokens become
    # `max_length - prompt_length` (variable). Clamp total length to
    # max_prompt_length + max_completion_length so completion is never > 512.
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = max_completion_length
        model.generation_config.max_length = max_prompt_length + max_completion_length


    # Left padding is MANDATORY for GRPO generation
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Add LoRA adapters (Do this AFTER setting tokenizer properties)
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_rank,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 4. Load Dataset
    dataset = load_dataset("json", data_files="data/train.json", split="train") 

    # Build plain text prompts and let GRPOTrainer handle tokenization.
    # Pre-rendering chat templates can append extra assistant control tokens,
    # which may create token-length mismatches inside GRPO loss masking.
    def format_prompt(example):
        system_instruction = (
            "You are a helpful assistant. "
            "You must first think about the answer within <think> tags, "
            "then provide the final answer within <answer> tags."
        )
        prompt_text = (
            f"System: {system_instruction}\n\n"
            f"User: {example['prompt']}\n\n"
            "Assistant:"
        )
        return {"prompt": prompt_text}

    dataset = dataset.map(format_prompt)

    # 5. Configure GRPO Trainer
    training_args = GRPOConfig(
        #use_vllm = False, # Use vLLM for fast inference
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, 
        num_generations = 6,            
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "none", 
        output_dir = "outputs/qwen3-grpo",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            apex_efficiency_reward_func,
            regime_consistency_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )

    # 6. Train
    print("Starting GRPO Training...")
    trainer.train()
    
    # 7. Save Model
    print("Saving model and LoRA adapters...")
    model.save_lora("outputs/qwen3-grpo/lora_adapters")

if __name__ == "__main__":
    train()
