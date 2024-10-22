from cmd import PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import torch
import gc
import multiprocessing as mp
import os
import json
from typing import List, Optional


SYS_PROMPT = "You are a helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information."

LLAMA31_SYS_PROMPT = """

Environment: ipython
Tools: brave_search, wolfram_alpha

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant"""

 
DEVICE_MAPS = json.load(open('configs/device_map_config.json'))
MODEL_NAME_TO_PATH = json.load(open('configs/model_path_config.json'))

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return slice(ind,ind+sll)
        

def prompt_template_handler(model: str, context: List[dict], prompt: str, tokenizer: PreTrainedTokenizer | None, return_tensor='pt') -> List[List[int] | str]:
    prompt = prompt.strip()
    
    assert len(context) > 0
    if len(context) >=2: # multiturn context
        context = [{'role': 'system', 'content': LLAMA31_SYS_PROMPT if 'Llama-3.1' in model else SYS_PROMPT }] + context
        context.append({'role': 'user', 'content': prompt})
    else:
        context[0]['content'] += '\n\n' + prompt
        context = [{'role': 'system', 'content': LLAMA31_SYS_PROMPT if 'Llama-3.1' in model else SYS_PROMPT }] + context

    full_tokens = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=True, return_tensors=return_tensor)
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    suffix_slice = find_sub_list(prompt_token_ids, list(full_tokens)[0] if return_tensor != 'pt' else full_tokens.squeeze().tolist()) 
    # [0] since we only have one Conversation so we should always squeeze the output
    # print(len(full_tokens[0]), suffix_slice)
    assert suffix_slice
    return full_tokens, suffix_slice


def build_context_prompt(
    model_name: str, context: str, prompt: str, tokenizer: PreTrainedTokenizer
) -> tuple[torch.Tensor, slice]:
    """
    Given the actual "suffix" (prompt), add in the prefix/suffix for the given instruction tuned model

    Parameters
    ----------
        model_name: str
            Model name or path
        suffix: str
            The actual prompt to wrap around
        tokenizer: PreTrainedTokenizer
            Tokenizer for the model

    Returns
    -------
        tuple[torch.Tensor, slice]
            Tuple of the prompt ids and the slice of the actual prompt (suffix)
    """
    assert isinstance(context, List)
    return prompt_template_handler(model_name, context, prompt, tokenizer)


def free_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()


def load(
    model_path: str,
    fp16: bool = True,
    device_map = "auto",
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if fp16 else torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer


def load_model_tokenizer(
    model_name: str,
    fp16: bool = True,
    sharded: bool = False,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Load pretrained models and tokenizers to gpu(s)

    Args:
        model_name (str): Model name, glm, mistral, llama-3
        fp16 (bool, optional): Whether to use fp16. Defaults to True.
        sharded: Whether loading shards to gpus
    """
    print("Loading models...")
    assert model_name in MODEL_NAME_TO_PATH.keys(), "Model Path Not Found in configs/model_path_config.json"
    model_path = MODEL_NAME_TO_PATH[model_name]
    if sharded:
        assert model_name in DEVICE_MAPS.keys(), "Device Map Not Found in Config configs/device_map_config.json"
        dmap = DEVICE_MAPS[model_name]
        assert torch.cuda.device_count() > max(dmap.values())
        model, tokenizer = load(
            model_path, fp16, device_map=dmap
        )
    else:
        model, tokenizer = load(
                model_path, fp16, device_map=0
            )
    return model, tokenizer
