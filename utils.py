# Copyright © 2023 Apple Inc.

import glob
import json
import logging
from pathlib import Path
from typing import Generator, Dict, Mapping, List
import random

import mlx.core as mx
import mlx.nn as nn
import models.llama as llama
import models.mixtral as mixtral

import numpy as np
import transformers
from huggingface_hub import snapshot_download

# Constants
MODEL_MAPPING = {
    "llama": llama,
    "mistral": llama,  # mistral is compatible with llama
    "mixtral": mixtral,
}


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Reference:
        https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
        """
        self.mean = 0.0
        self.std = 1.0
        self.var = 1.0
        self.count = 1e-24

    def update(self, xs: mx.array):
        """
        Updates running moments from batch's moments computed across ranks
        """
        xs_count = xs.size
        xs_var = mx.var(xs)
        xs_mean = mx.mean(xs)
        xs_mean = xs_mean.item()
        xs_var = xs_var.item()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta ** 2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = mx.sqrt(self.var * tot_count / (tot_count - 1)).item()
        self.count = tot_count

        return xs_mean, mx.sqrt(xs_var * xs_count / (xs_count - 1)).item()


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        print(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def pad_to_size(tensor: mx.array, size: int, dim: int = 1, padding: int = 50256) -> mx.array:
    """Pad tensor to size."""
    t_size = tensor.shape[dim]
    if t_size == size:
        return tensor
    else:
        return mx.pad(tensor, (0, size - t_size), padding)


def logprobs_from_logits(logits: mx.array, labels: mx.array, gather: bool = True) -> mx.array:
    """
    Turn raw logit values into log probs with softmax + log -- make sure axis is correct
    """
    logp = nn.log_softmax(logits, axis=2)
    # logp += 1e-6
    # logp = mx.log(logp)

    if not gather:
        return logp

    # label_inds = labels.reshape(-1)
    # dim_1_inds = mx.repeat(mx.arange(logp.shape[0]), logp.shape[1])
    # dim_2_inds = mx.repeat(mx.arange(logp.shape[1]), logp.shape[0])
    # logpy = logp[dim_1_inds, dim_2_inds, label_inds]
    #
    # logpy = logpy.reshape((logp.shape[0], logp.shape[1]))
    # print(f'Log py: {logpy}')
    logpy = np.take_along_axis(np.array(logp.astype(mx.float32)),
                               np.array(labels[:, :, None]), axis=2)
    if np.any(np.isnan(logpy)):
        print("Uh oh. NaNs in the log probs!!")

    return mx.array(logpy).squeeze(-1)


def whiten(values: mx.array, shift_mean: bool = True) -> mx.array:
    """Whiten values."""
    mean, var = mx.mean(values), mx.var(values)
    whitened = (values - mean) * mx.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_mean(values: mx.array, mask: mx.array, axis: bool = None) -> mx.array:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: mx.array, mask: mx.array, unbiased: bool = True) -> mx.array:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: mx.array, mask: mx.array, shift_mean: bool = True) -> mx.array:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * mx.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x: mx.array, tensor_min, tensor_max) -> mx.array:
    """
    Tensor extension to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    max_tens = mx.concatenate([x, tensor_max], axis=0)
    mins = mx.min(max_tens, axis=0)[None]
    min_tens = mx.concatenate([mins, tensor_min], axis=0)
    clipped = mx.max(min_tens, axis=0)[None]
    # clipped = mx.max(
    #     mx.stack(
    #         (mx.min(
    #             mx.stack((x, tensor_max), axis=0),
    #             tensor_min)), axis=0)
    # )
    return clipped


def entropy_from_logits(logits: mx.array) -> mx.array:
    """Calculate entropy from logits."""
    pd = mx.softmax(logits, axis=-1)
    entropy = mx.logsumexp(logits, axis=-1) - mx.sum(pd * logits, axis=-1)
    return entropy


def stats_to_np(stats_dict: Dict) -> Dict:
    """Cast all mx.arrays in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, mx.array):
            new_dict[k] = v
            if new_dict[k].dtype == mx.bfloat16:
                new_dict[k] = new_dict[k].astype(mx.float32)
            new_dict[k] = np.array(new_dict[k])
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat


def extract_grads(d_in):
    """
    Recursively extract all arrays from a nested dictionary.

    Parameters:
        d_in: Input dictionary

    Returns:
        arrays: List of all arrays found
    """
    arrays = []
    for k, v in d_in.items():
        if isinstance(v, dict):
            arrays.extend(extract_grads(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    arrays.extend(extract_grads(item))
        elif isinstance(v, mx.array):
            arrays.append(v)
    return arrays


def replace_nans_get_means(logs):
    for k, v in logs.items():
        try:
            v = v.tolist()
            v = np.nan_to_num(v, nan=-100).mean().tolist()
            logs[k] = v
        except AttributeError:
            logs[k] = v
    return logs


def stack_dicts(stats_dicts: List[Dict]) -> Dict:
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [mx.flatten(d[k]) for d in stats_dicts]
        max_len = max([len(x) for x in stats_list])
        padded = []
        for x in stats_list:
            if len(x) < max_len:
                buffer = mx.ones(max_len - len(x)).astype(x.dtype)
                x = mx.concatenate((x, buffer))
            padded.append(x)
        results[k] = mx.array(padded)
    return results


def convert_to_scalar(stats: Dict) -> Dict:
    """
    Converts the stats from a flattened dict to single scalar dicts
    """
    tensorboard_stats = {}
    for k, v in stats.items():
        # for tensorboard compatibility - arrays and tensors are ignored with tensorboard
        # therefore we convert single element tensors to scalars
        if (isinstance(v, mx.array) or isinstance(v, np.ndarray)) and (
                len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)
        ):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats


def pad_to_length(tensor: mx.array, length: int, pad_value, dim: int = -1) -> mx.array:
    if tensor.shape[dim] >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.shape[dim]
        return mx.concatenate(
            [
                tensor,
                pad_value * mx.ones(*pad_size, dtype=tensor.dtype),
            ],
            axis=dim,
        )


def disable_dropout_in_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


def exact_div(a, b, a_str, b_str, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, {a_str}={a}, {b_str}={b}, inexact division: {a} / {b} = {a / b}")
    return q


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    )
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(path: str, name: str, hf_path: str):
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"mlx-community/{name}"

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = f"""
# {name}
This model was converted to MLX format from [`{hf_path}`]().
Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
## Use with mlx
```bash
pip install mlx
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms/hf_llm
python generate.py --model {repo_id} --prompt "My name is"
```
"""
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
    )


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(str(save_dir / shard_name), shard)

    tokenizer.save_pretrained(save_dir)

    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)


def load(path_or_hf_repo: str):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
                                             and m.weight.shape[0] != 8,
        )

    model.load_weights(list(weights.items()), strict=False)

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def _generate_token(
        prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(sampled_logits: mx.array) -> mx.array:
        return (
            mx.argmax(sampled_logits, axis=-1)
            if temp == 0
            else mx.random.categorical(sampled_logits * (1 / temp))
        )

    y = prompt

    cache = None
    while True:
        if len(y.shape) < 2:
            y = y[:, None]
        logits, cache, _ = model(y, cache=cache)
        if logits.shape[1] < 1:
            logits = logits[:, None, :]
        logits = logits[:, -1, :]
        if len(logits.shape) > 2:
            print(logits.shape)
            # logits = logits.squeeze()
            # print(logits.shape)
        y = sample(logits)
        yield y


def generate(model, prompt, tokenizer, args):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
            _generate_token(prompt, model, args.temp),
            range(args.max_tokens),
    ):
        # if token == tokenizer.eos_token_id:
        #     break

        tokens.append([x.item() for x in token])
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    print(tokenizer.decode(tokens)[skip:], flush=True)
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return


def generate_ids(model, input_ids, eos_token_id=100_000, temperature=0.0, max_tokens=128):
    prompt = mx.array(input_ids)
    tokens = []
    for token, n in zip(
            _generate_token(prompt, model, temperature),
            range(max_tokens),
    ):
        # if token == eos_token_id:
        #     break
        if len(token.shape) < 2:
            token = token[:, None]
        tokens.append(token)
    return mx.concatenate(tokens, axis=1)  # .transpose()
