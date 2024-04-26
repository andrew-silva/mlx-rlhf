# coding=utf-8
# Modified by Andrew Silva from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/rl_training.py
#
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example call to use a pre-trained LoRA from sft for RLHF with ground-truth reward:
python ppo_training.py --log_with=wandb --resume_file pytorch_model --ground_truth_reward
"""
from dataclasses import dataclass, field
from typing import Optional
import random

import torch
import numpy as np
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModel, AutoTokenizer, AutoModelForCausalLM

from pytorch_ppo_trainer import PPOTrainer
from data.digit_seq_rewards import RewardFunction

from models.config import PPOConfig
from peft import LoraConfig, get_peft_model

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main(args_in, ppo_config_in):
    # set seed before initializing value head for deterministic eval
    random.seed(ppo_config_in.seed)
    np.random.seed(ppo_config_in.seed)
    torch.random.manual_seed(ppo_config_in.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    # TODO: Actually handle the use_peft parameter
    if not args_in.use_peft:
        if args_in.resume_file:
            ref_model = AutoModel.from_pretrained(args_in.resume_file)
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(args_in.model)
        device_map = None
        peft_config = None
    else:
        ref_model = None

    if args_in.resume_file:
        model = AutoModel.from_pretrained(args_in.resume_file)
    else:
        model = AutoModelForCausalLM.from_pretrained(args_in.model)

    tokenizer = AutoTokenizer.from_pretrained(args_in.model)

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    model = get_peft_model(model, config)

    model.value_head = torch.nn.Linear(model.config.hidden_size, 1)
    ref_model.value_head = torch.nn.Linear(ref_model.config.hidden_size, 1)
    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(ppo_config_in, model, ref_model, tokenizer, data_collator=collator)

    if args_in.ground_truth_reward:
        # TODO: Ground-truth reward values are hard-coded here, maybe use a config to set dynamically
        #  (especially if we need to match sft.py)
        reward_function = RewardFunction(is_increasing=True, multiple_of=2)
    else:
        reward_function = AutoModel.from_pretrained(args_in.reward_model_dir)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    # TODO: play with generation kwargs, might help exploration
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    # TODO: add a command-line arg for num-steps
    for epoch in range(10000):
        # TODO: Add a command-line arg for a prompt before each call?
        # text_in = 'Count up even numbers 2 8 20'
        start_int = random.randint(0, 10) * 2
        text_in = f'{start_int}'
        # text_in = ''

        batch = {
            'query': [text_in],
            # 'input_ids': mx.array(tokenizer.encode(text_in))
        }
        query_tensors = tokenizer.encode(text_in, return_tensors="pt")  # batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )

        batch["response"] = tokenizer.batch_decode(np.array(response_tensors))
        batch["ref_response"] = tokenizer.batch_decode(np.array(ref_response_tensors))
        response_tensors = [response_tensors]
        ref_response_tensors = [ref_response_tensors]

        # Compute sentiment score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]

        if args_in.ground_truth_reward:
            scores = reward_function(batch['response'])  # Should we omit query in the scoring?
            # scores = [x + np.random.randn() * 0.05 for x in scores]  # Noisify the ground truth reward signal
        else:
            scored_tensors = tokenizer.encode(batch["response"][0], return_tensors="pt").unsqueeze(0)
            _, _, scores = reward_function(response_tensors)
            scores = scores[:, -1].item()
        rewards = [torch.tensor(scores)]

        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        if args_in.ground_truth_reward:
            ref_scores = reward_function(batch['ref_response'])
        else:
            scored_tensors = tokenizer.encode(batch["ref_response"][0], return_tensors="pt").unsqueeze(0)
            _, _, ref_scores = reward_function(ref_response_tensors)
            ref_scores = [ref_scores[:, -1].item()]

        batch["ref_rewards"] = ref_scores

        # Run PPO step
        stats = ppo_trainer.step([query_tensors], response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards,
                              columns_to_log=["query", "response", "ref_response", "ref_rewards"])
    # Save prompt weights
    model.save_pretrained(args_in.save_file)

if __name__ == "__main__":
    tqdm.pandas()

    @dataclass
    class ScriptArguments:
        # LoraConfig
        use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
        ground_truth_reward: bool = field(default=False, metadata={"help": "whether to use ground truth reward or not"})
        lora_layers: Optional[int] = field(default=16, metadata={"help": "the number of lora layers"})
        num_prompt_tokens: Optional[int] = field(default=10, metadata={"help": "the number of prompt tokens"})
        model: Optional[str] = field(default=None,
                                     metadata={"help": "The path to the local model directory or Hugging Face repo"})
        reward_model_dir: Optional[str] = field(default=None,
                                            metadata={
                                                "help": "The path to the local model directory or Hugging Face repo"})

        save_file: str = field(default="peft_weights.npz",
                               metadata={"help": "Save path for the trained PEFT weights."})
        resume_file: Optional[str] = field(default=None,
                                           metadata={"help": "Load path for the trained PEFT weights."})
        prompt_tuning: bool = field(default=False, metadata={"help": "whether to use prompt-tuning or LoRA"})


    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()
    # TODO: Adaptive KL seems to break things... Why? Over/underflow?
    ppo_config.adap_kl_ctrl = False

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    # sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    main(args, ppo_config)
