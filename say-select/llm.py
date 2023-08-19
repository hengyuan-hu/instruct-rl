from typing import List, Dict  # , Tuple

import torch

import numpy as np
from scipy.special import softmax, log_softmax
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, GPTJForCausalLM
from transformers import OPTForCausalLM


ModelCache = {}


def load_gpt2lm_model(name="gpt2-xl") -> tuple[GPT2Tokenizer, GPT2LMHeadModel]:
    """
    name:
        gpt2: 12 (num attention modules)
        gpt2-medium: 24
        gpt2-large: 36
        gpt2-xl: 48

    """
    name_tag = f"GPT2LMHeadModel-{name}"
    global ModelCache
    if name_tag in ModelCache:
        return ModelCache[name_tag]
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2LMHeadModel.from_pretrained(name)
    assert isinstance(tokenizer, GPT2Tokenizer)
    assert isinstance(model, GPT2LMHeadModel)
    ModelCache[name_tag] = (tokenizer, model)
    return tokenizer, model


def load_gptjlm_model(gpu=False):
    name_tag = f"GPTJForCausalLM"
    global ModelCache
    if name_tag in ModelCache:
        return ModelCache[name_tag]
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    if gpu:
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # CPU model, takes 24GB mem
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", low_cpu_mem_usage=True
        )

    # assert isinstance(tokenizer, AutoTokenizer)
    assert isinstance(model, GPTJForCausalLM)
    ModelCache[name_tag] = (tokenizer, model)
    return tokenizer, model


def load_opt_model(size="350m"):
    "size: [350m, 30b]"
    name_tag = f"OPTForCausalLM-{size}"
    global ModelCache
    if name_tag in ModelCache:
        return ModelCache[name_tag]

    model = OPTForCausalLM.from_pretrained(
        f"facebook/opt-{size}", low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    tokenizer = GPT2Tokenizer.from_pretrained(f"facebook/opt-{size}")

    ModelCache[name_tag] = (tokenizer, model)
    return tokenizer, model


def gpt2_predict(
    model, tokenizer, prompt: str, targets: List[str], cache: Dict[str, float]
):
    """Predict the next **word**

    targets: List of words, such as ' 1', ' world'
    """
    # assert off the type error
    # assert isinstance(model, GPT2LMHeadModel) or isinstance(model, GPTJForCausalLM)

    ret_logps: dict[str, float] = {}
    uncached_targets = []

    for target in targets:
        assert len(target.strip()) == 1, f"{len(target.strip())=}"
        key = prompt + target
        if key in cache:
            ret_logps[target] = cache[key]
        else:
            # print("not in cache:", key)
            uncached_targets.append(target)

    if len(uncached_targets) > 0:
        # print("num uncached", len(uncached_targets))
        combined_prompt = prompt + uncached_targets[0]
        # print(f"not hit predict, cache size: {len(cache)}")
        # print(f"combined prompt: {prompt}")
        target_idxs = [
            t[0].item()
            for t in tokenizer(uncached_targets, return_tensors="pt")["input_ids"]
        ]
        prompt_toks = tokenizer(combined_prompt, return_tensors="pt")
        label = prompt_toks["input_ids"].clone()
        label[0, :-1] = -100
        outputs = model(**prompt_toks, labels=label)
        logits = outputs.logits
        nll = -logits.log_softmax(-1)[0][-2][target_idxs[0]]
        assert nll == outputs.loss, f"nll={nll}, outputs.loss={outputs.loss}"
        target_logits = [logits[0][-2][idx].item() for idx in target_idxs]
        # print(target_logits)
        for target, logit in zip(uncached_targets, target_logits):
            cache[prompt + target] = logit
    # else:
    #     print("all cached")

    all_logits = []
    for target in targets:
        combined_prompt = prompt + target
        all_logits.append(cache[combined_prompt])

    logps: list[float] = torch.log_softmax(
        torch.tensor(all_logits, dtype=torch.float32), -1
    ).tolist()
    return logps


def gpt2_predict_fake(
    model,
    tokenizer,
    prompt: str,
    targets: list[str],
    cache: dict[str, float],
    max_prob=3,
):
    probs = []
    sum_prob = 0
    for target in targets:
        if target in prompt:
            probs.append(max_prob)
        else:
            probs.append(1)
        sum_prob += probs[-1]
    logps = []

    for i in range(len(probs)):
        prob = probs[i] / sum_prob
        logps.append(np.log(prob))

    return logps


def gpt2_cond_loss(
    model,
    tokenizer,
    prompt: str,
    targets: List[str],
    cache: Dict[str, float],
    *,
    softmax_beta: float = 1.0,
    use_log: bool = False,
):
    # assert off the type error
    # assert isinstance(model, GPT2LMHeadModel) or isinstance(model, GPTJForCausalLM)
    nlls = []
    verbose = False
    for target in targets:
        input_ = prompt + target
        if input_ not in cache:
            verbose = True
            input_tokens = tokenizer(input_, return_tensors="pt")
            prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
            assert (
                input_tokens["input_ids"][:, : prompt_tokens.size(1)] == prompt_tokens
            ).all()
            label = input_tokens["input_ids"].clone()
            label[0, : prompt_tokens.size(1)] = -100
            outputs = model(**input_tokens, labels=label)
            loss: float = outputs.loss.item()
            cache[input_] = loss

        nlls.append(cache[input_])

    logits = [-nll * softmax_beta for nll in nlls]
    if verbose:
        probs = softmax(logits)
        print(prompt)
        for cand, prob in zip(targets, probs):
            print(f"{cand}: {prob:.4f}")
    if use_log:
        return log_softmax(logits)
    else:
        return softmax(logits)
