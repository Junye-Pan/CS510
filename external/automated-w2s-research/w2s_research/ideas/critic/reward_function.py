"""VERL reward function for critic training using GenRM.

This module is dynamically loaded by VERL to compute rewards during GRPO training.
It uses VERL's GenRM infrastructure - the judge model is served as an HTTP server
by RewardModelManager, enabling colocate mode with sleep/wakeup for GPU memory management.
"""
import json
import aiohttp
import sys
import logging
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[Reward] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Use shared logging utilities
from w2s_research.utils.logging_utils import (
    init_weave,
    get_weave_op,
    get_weave_attributes,
    ENABLE_WEAVE_TRACING,
)

# Initialize Weave if enabled
init_weave("generalization")

# Get Weave utilities (real or no-op)
weave_op = get_weave_op()
weave_attributes = get_weave_attributes()

# Load judge prompt template
JUDGE_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "judge_prompt.jinja2"
with open(JUDGE_PROMPT_TEMPLATE_PATH, "r") as f:
    JUDGE_PROMPT_TEMPLATE = Template(f.read())

# Binary judge prompt template: single answer with critique
BINARY_JUDGE_PROMPT_TEMPLATE = """Question: {question}

Answer: {answer}
Critique: {critique}

Based on the question, answer, and critique, is the answer correct?
Answer (True or False):"""


@weave_op()
async def chat_complete(
    router_address: str,
    messages: list,
    reward_model_path: str = None,
    **kwargs
):
    """Send async HTTP request to the GenRM server via VERL's reward router."""
    url = f"http://{router_address}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1),
        "temperature": kwargs.get("temperature", 0.0),
        "logprobs": True,
        "top_logprobs": 10,
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"GenRM HTTP error: {resp.status} - {error_text}")
                    return None
                return await resp.json()
    except Exception as e:
        logger.error(f"GenRM HTTP request failed: {e}")
        return None

@weave_op()
def parse_judge_response_from_logprobs(result: dict, tokenizer, binary_mode: bool = False) -> tuple[str, dict]:
    """Parse judge response from logprobs.
    
    For A/B comparison mode: Parse A or B
    For binary mode: Parse True or False
    
    Handles OpenAI-style logprobs format:
    {
        "logprobs": {
            "content": [{
                "token": "A",
                "logprob": -0.25,
                "top_logprobs": [
                    {"token": "A", "logprob": -0.25},
                    {"token": "B", "logprob": -1.50},
                    ...
                ]
            }]
        }
    }
    """
    logprob_info = {
        "label_token_ids": {},
        "label_logprobs": {},
        "label_probs": {},
        "top_logprobs_sample": [],
        "prediction": None,
    }
    
    if not result:
        logger.warning("[parse] result is None or empty")
        return None, logprob_info
    
    choices = result.get("choices", [])
    if not choices:
        logger.warning("[parse] No choices in result")
        return None, logprob_info
    
    choice = choices[0]
    if "logprobs" not in choice:
        logger.warning("[parse] No logprobs in choice")
        return None, logprob_info
    
    logprobs = choice["logprobs"]
    if not logprobs:
        logger.warning("[parse] logprobs is None or empty")
        return None, logprob_info
    
    # OpenAI format: logprobs.content[0].top_logprobs
    content = logprobs.get("content", [])
    if not content:
        logger.warning("[parse] No content in logprobs")
        return None, logprob_info
    
    first_token_info = content[0]
    top_logprobs = first_token_info.get("top_logprobs", [])
    
    if not top_logprobs:
        logger.warning("[parse] No top_logprobs in first token")
        return None, logprob_info
    
    # Determine which tokens to look for based on mode
    if binary_mode:
        # After removing <think> tags from training data, the model generates "True"/"False" 
        # directly after the newline in "<|im_start|>assistant\n", so no space prefix
        # Order: ["False", "True"] matches BINARY_LABEL_TOKENS (index 0=False, index 1=True)
        token_labels = ["False", "True"]  # index 0=False (incorrect), index 1=True (correct)
    else:
        token_labels = ["A", "B"]  # A/B comparison (no space prefix)
    
    # Find logprobs for the target tokens
    logprob_first = None
    logprob_second = None
    
    for item in top_logprobs:
        token = item.get("token", "")
        logprob = item.get("logprob", 0.0)
        
        if token == token_labels[0]:
            logprob_first = logprob
            logprob_info["label_logprobs"][token_labels[0]] = logprob
        elif token == token_labels[1]:
            logprob_second = logprob
            logprob_info["label_logprobs"][token_labels[1]] = logprob
    
    # Store top logprobs sample for debugging
    logprob_info["top_logprobs_sample"] = [
        {"token": item.get("token", ""), "logprob": item.get("logprob", 0.0)}
        for item in top_logprobs[:5]
    ]
    
    # Calculate probabilities
    if logprob_first is not None and logprob_second is not None:
        import numpy as np
        logprobs_array = np.array([logprob_first, logprob_second])
        probs_array = np.exp(logprobs_array - np.max(logprobs_array))
        probs_array = probs_array / probs_array.sum()
        logprob_info["label_probs"][token_labels[0]] = float(probs_array[0])
        logprob_info["label_probs"][token_labels[1]] = float(probs_array[1])
        
        # Prediction is the one with higher probability
        if probs_array[0] > probs_array[1]:
            logprob_info["prediction"] = token_labels[0]
        else:
            logprob_info["prediction"] = token_labels[1]
    elif logprob_first is not None:
        logprob_info["label_probs"][token_labels[0]] = 1.0
        logprob_info["label_probs"][token_labels[1]] = 0.0
        logprob_info["prediction"] = token_labels[0]
    elif logprob_second is not None:
        logprob_info["label_probs"][token_labels[0]] = 0.0
        logprob_info["label_probs"][token_labels[1]] = 1.0
        logprob_info["prediction"] = token_labels[1]
    else:
        logger.warning(f"[parse] Neither {token_labels[0]} nor {token_labels[1]} found in top_logprobs: {top_logprobs[:3]}")
    
    logger.info(f"[parse] prediction={logprob_info['prediction']}, probs={logprob_info['label_probs']}")
    
    return logprob_info["prediction"], logprob_info


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: dict = None,
    reward_router_address: str = None,
    reward_model_tokenizer = None,
    **kwargs
):
    """VERL-compatible async reward function using GenRM via HTTP."""
    experiment_name = kwargs.get("experiment_name", None)
    
    # Set weave attributes for experiment name if available
    # Wrap the entire call so both _compute_score_impl and chat_complete inherit the attribute
    if experiment_name and ENABLE_WEAVE_TRACING:
        logger.info(f"[compute_score] Setting weave attributes for experiment_name={experiment_name}")
        with weave_attributes({"experiment_name": experiment_name}):
            return await _compute_score_impl(
                data_source, solution_str, extra_info,
                reward_router_address, reward_model_tokenizer, kwargs
            )
    else:
        if experiment_name:
            logger.info(f"[compute_score] experiment_name provided but Weave tracing is disabled")
        else:
            logger.info(f"[compute_score] No experiment_name provided, skipping weave attributes")
    
    # Normal execution path (without experiment_name)
    return await _compute_score_impl(
        data_source, solution_str, extra_info,
        reward_router_address, reward_model_tokenizer, kwargs
    )

@weave_op()
async def _compute_score_impl(
    data_source: str,
    solution_str: str,
    extra_info: dict = None,
    reward_router_address: str = None,
    reward_model_tokenizer = None,
    kwargs: dict = None,
):
    """Core implementation of compute_score."""
    logger.info(f"[compute_score] CALLED - data_source={data_source}, solution_str_len={len(solution_str) if solution_str else 0}")
    
    kwargs = kwargs or {}
    reward_model_path = kwargs.get("reward_model_path", "NOT_PROVIDED")
    
    if not hasattr(compute_score, '_prompt_to_info_cache'):
        prompt_to_info_file = kwargs.get("prompt_to_info_file")
        if not prompt_to_info_file:
            logger.error("prompt_to_info_file not provided in reward_kwargs")
            raise RuntimeError("prompt_to_info_file not provided in reward_kwargs")
        logger.info(f"[compute_score] Loading prompt_to_info from {prompt_to_info_file}")
        with open(prompt_to_info_file, 'r') as f:
            compute_score._prompt_to_info_cache = json.load(f)
        logger.info(f"[compute_score] Loaded {len(compute_score._prompt_to_info_cache)} prompt entries")
    
    prompt_to_info = compute_score._prompt_to_info_cache
    
    extra_info = extra_info or {}
    prompt_str = extra_info.get('prompt')
    
    if not prompt_str or prompt_str not in prompt_to_info:
        logger.warning(f"[compute_score] prompt_str not found in prompt_to_info. prompt_str={repr(prompt_str[:50] if prompt_str else None)}")
        return {"score": 0.5}
    
    info = prompt_to_info[prompt_str]
    question = info['question']
    
    # Detect binary format by checking if 'choice' field exists (no 'option' field)
    is_binary_format = 'choice' in info and 'option' not in info
    
    if is_binary_format:
        # Binary format: single answer correctness
        choice = info['choice']
        
        # Build judge prompt for binary format
        judge_prompt = BINARY_JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            answer=choice,
            critique=solution_str,  # The generated critique
        )
        
        if not reward_router_address:
            logger.error("[compute_score] reward_router_address not provided. Is reward_model.enable=True?")
            return {"score": 0.5}
        
        messages = [{"role": "user", "content": judge_prompt}]
        
        result = await chat_complete(
            router_address=reward_router_address,
            messages=messages,
            reward_model_path=reward_model_path,
            max_tokens=1,
            temperature=0.0,
        )
        
        judge_pred = None
        if result and "choices" in result and len(result["choices"]) > 0:
            try:
                judge_pred, logprob_info = parse_judge_response_from_logprobs(
                    result, tokenizer=reward_model_tokenizer, binary_mode=True
                )
                logger.info(f"[compute_score] Parsed judge_pred={judge_pred} (binary mode)")
            except Exception as e:
                logger.error(f"[compute_score] Error parsing judge response: {e}", exc_info=True)
                judge_pred = None
        else:
            logger.warning(f"[compute_score] No result or empty choices. result={result}")
        
        if judge_pred is None:
            logger.warning("[compute_score] judge_pred is None, returning neutral score 0.5")
            return {"score": 0.5}
        
        if judge_pred not in ['True', 'False']:
            logger.error(f"[compute_score] judge_pred={repr(judge_pred)} is not 'True' or 'False'")
            return {"score": 0.5}
        
        # Binary reward logic:
        # During RL, we don't have access to ground truth labels.
        # The critic gets rewarded when the judge thinks the answer is WRONG (False).
        # This encourages the critic to write critiques that point out problems in answers.
        #
        # The trained judge has learned from labeled data to distinguish good vs bad critiques,
        # so it will only say "False" when the critique points out reasonable problems.
        reward = 1.0 if judge_pred == "False" else 0.0
        
        logger.info(f"[compute_score] Final reward: {reward} (binary, judge_pred={judge_pred})")
        
        return {"score": reward}
    
    else:
        # Original A/B comparison format
        option = info['option']
        critique_other_option = info['critique_other_option']
        
        if not critique_other_option:
            logger.warning("[compute_score] critique_other_option is empty, returning 0.5")
            return {"score": 0.5}
        
        if option == 'A':
            this_option_text = info['first']
            other_option_text = info['second']
        else:
            this_option_text = info['second']
            other_option_text = info['first']
        
        if option == 'A':
            judge_prompt = JUDGE_PROMPT_TEMPLATE.render(
                question=question,
                option_a=this_option_text,
                option_b=other_option_text,
                critique_a=solution_str,
                critique_b=critique_other_option,
            )
        else:
            judge_prompt = JUDGE_PROMPT_TEMPLATE.render(
                question=question,
                option_a=other_option_text,
                option_b=this_option_text,
                critique_a=critique_other_option,
                critique_b=solution_str,
            )
        
        if not reward_router_address:
            logger.error("[compute_score] reward_router_address not provided. Is reward_model.enable=True?")
            return {"score": 0.5}
        
        messages = [{"role": "user", "content": judge_prompt}]
        
        result = await chat_complete(
            router_address=reward_router_address,
            messages=messages,
            reward_model_path=reward_model_path,
            max_tokens=1,
            temperature=0.0,
        )
        
        judge_pred = None
        if result and "choices" in result and len(result["choices"]) > 0:
            try:
                judge_pred, logprob_info = parse_judge_response_from_logprobs(
                    result, tokenizer=reward_model_tokenizer, binary_mode=False
                )
                logger.info(f"[compute_score] Parsed judge_pred={judge_pred}")
            except Exception as e:
                logger.error(f"[compute_score] Error parsing judge response: {e}", exc_info=True)
                judge_pred = None
        else:
            logger.warning(f"[compute_score] No result or empty choices. result={result}")
        
        if judge_pred is None:
            logger.warning("[compute_score] judge_pred is None, returning neutral score 0.5")
            return {"score": 0.5}
        
        if judge_pred not in ['A', 'B']:
            logger.error(f"[compute_score] judge_pred={repr(judge_pred)} is not 'A' or 'B'")
            return {"score": 0.5}
        
        # Normalize judge_pred by stripping space for comparison with option ("A"/"B")
        judge_pred_normalized = judge_pred.strip()
        
        if option == 'A':
            reward = 1.0 if judge_pred_normalized == 'B' else 0.0
        else:
            reward = 1.0 if judge_pred_normalized == 'A' else 0.0
        
        logger.info(f"[compute_score] Final reward: {reward} (option={option}, judge_pred={judge_pred})")
        
        return {"score": reward}
