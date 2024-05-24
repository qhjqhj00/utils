import sys
import base64
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Dict
import logging
import time
import hashlib
from openai import OpenAI
from openai import AzureOpenAI
import openai
import anthropic
from utils.api_keys import api_dict

logger = logging.getLogger(__name__)

import json
 # for token counting
import numpy as np
from collections import defaultdict
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def num_input_tokens_from_example(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] != "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
    
def cost_estimate(dataset, convo_lens):
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
    return

def print_stats(dataset):
    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
        
    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    cost_estimate(dataset, convo_lens)
    return

def format_error_check(dataset):
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

def count_tokens(data_path):
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)
    return dataset

def except_retry_dec(retry_num: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            i = 0
            while True:
                try:
                    logger.info("openai agent post...")
                    ret = func(*args, **kwargs)
                    logger.info("openai agent post finished")
                    return ret
                # error define: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
                except (
                    openai.BadRequestError,
                    openai.AuthenticationError,
                ) as e:
                    raise
                except Exception as e:  # pylint: disable=W0703
                    logger.error(f"{e}")
                    logger.info(f"sleep {i + 1}")
                    time.sleep(i + 1)
                    if i >= retry_num:
                        raise
                    logger.warning(f"do retry, time: {i}")
                    i += 1

        return wrapped_func

    return decorator

class BaseGPTAgent(ABC):
    @abstractmethod
    def chat_completion(self, prompt: str):
        raise NotImplemented


class Agent(BaseGPTAgent):
    def __init__(
        self, 
        model, 
        source, 
        temperature: float = 0.0,
        max_length: int=1024):
        self.model = model
        self.source = source
        self.temperature = temperature
        self.max_length = max_length

        if model not in api_dict[source]["models"]:
            print(f"error, {model} is not supported in {source}...")

        if source == "azure":
            self.client = AzureOpenAI(
                azure_endpoint = api_dict[source]["endpoint"], 
                api_version=api_dict[source]["api_version"],
                api_key=api_dict[source]["api_key"],
                )
            
        elif source == "openai":
            self.client = OpenAI(
                    # This is the default and can be omitted
                    api_key=api_dict[source]["api_key"],
                )
        elif source == "deepseek":
            self.client = OpenAI(
                    # This is the default and can be omitted
                    base_url=api_dict[source]["base_url"],
                    api_key=api_dict[source]["api_key"],
                )
        elif source == "claude":
            self.client = anthropic.Anthropic(
                    api_key=api_dict[source]["api_key"],
                )

        print(f"You are using {self.model} from {source}")
    @except_retry_dec()
    def chat_completion(self, prompt: str) -> str:
        if self.source == "claude":
            message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_length,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            content = message.content[0].text
        else:
            message = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=self.temperature,
                    model=self.model,
                )
            content = message.choices[0].message.content
        return content

if __name__ == "__main__":
    from api_keys import api_dict
    for s in api_dict:
        for m in api_dict[s]["models"]:

            agent = Agent(m, s)

            res = agent.chat_completion("hi") # 8192 tokens
            print(res)
