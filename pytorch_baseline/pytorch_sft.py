# Modified by Andrew Silva from https://github.com/ml-explore/mlx-examples/blob/main/lora/lora.py
#
# Copyright © 2023 Apple Inc.
import os
import argparse
import time
from data.data_utils import load_datasets, mask_between_sos

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Example command for supervised fine-tuning with LoRA on generated data with a HF tiny llama
python pytorch_sft.py --save-file sft_fine_tune --data-base increasing_mult_1_ --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --train --iters 1500 --data ../data/

python pytorch_sft.py --data ../../message_data/ --save-file sft_fine_tune/ --data-base chat --model meta-llama/Llama-2-7b-chat-hf --train --iters 15000 --batch-size 4 --steps-per-eval 1000

Example command for training a reward model with LoRA on generated data with a HF tiny llama
python pytorch_sft.py --reward-model --train --data-base reward_function_increasing_mult_2_ --batch-size 16 --save-file reward_lora.npz --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""


def build_parser():
    arg_parse = argparse.ArgumentParser(description="Soft Prompt finetuning.")
    arg_parse.add_argument(
        "--model",
        default="model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    arg_parse.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    arg_parse.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    arg_parse.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    arg_parse.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    arg_parse.add_argument(
        "--reward-model",
        action="store_true",
        help="Train a reward model instead of a SFT model"
    )
    arg_parse.add_argument(
        "--prompt-tuning",
        action="store_true",
        help="Should we train with prompt tuning? If not, use LoRA",
    )
    arg_parse.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    arg_parse.add_argument(
        "--data-base",
        type=str,
        default="",
        help="Base name for the .jsonl files. E.g., 'increasing_mult_2_'",
    )
    arg_parse.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=10,
        help="Number of prompt tokens to pre-pend",
    )
    arg_parse.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    arg_parse.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    arg_parse.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    arg_parse.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    arg_parse.add_argument(
        "--learning-rate", type=float, default=1e-6, help="Adam learning rate."
    )
    arg_parse.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    arg_parse.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    arg_parse.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Load path to resume training with the given PEFT weights.",
    )
    arg_parse.add_argument(
        "--save-file",
        type=str,
        default="peft_weights.npz",
        help="Save/load path for the trained PEFT weights.",
    )
    arg_parse.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    arg_parse.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    arg_parse.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    arg_parse.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return arg_parse


def reward_loss(mdl, better_inputs, worse_inputs):
    """
    Reward modeling loss, maximizing the difference between the preferred sequence and the "dispreferred" sequence
    (Assumes that the reward for seq1 >= reward for seq2)
    Returns:
        Loss value, tokens-per-second (TODO -- Tokens-per-second implementation missing here)
    """
    # TODO: Batch these, currently this is unnecessarily slow.
    output = mdl(better_inputs, output_hidden_states=True)
    rewards_j = mdl.value_head(output.hidden_states[-1][:, :, :]).squeeze(-1)

    output = mdl(worse_inputs, output_hidden_states=True)
    rewards_k = mdl.value_head(output.hidden_states[-1][:, :, :]).squeeze(-1)

    # Batch x SeqLen x OutputDim -- get last token value
    diff_val = -torch.nn.functional.logsigmoid(rewards_j[:, -1] - rewards_k[:, -1]).mean()
    return diff_val


def iterate_batches(dset, tok, batch_size, train_mode=False, reward_modeling=False, chat_data=False):
    # Shuffle indices
    len_warning_message = "[WARNING] Some sequences are longer than 2048 tokens. " \
                          "Consider pre-splitting your data to save memory."
    while True:
        indices = np.arange(len(dset))
        if train_mode:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            if reward_modeling:
                pref_batch, bad_batch = [], []
                p_lengths, b_lengths = [], []
                for j in range(batch_size):
                    pref, bad = dset[indices[i + j]]
                    pref_batch.append(tok.encode(pref))
                    p_lengths.append(len(pref_batch[-1]))
                    bad_batch.append(tok.encode(bad))
                    b_lengths.append(len(bad_batch[-1]))
                if max(max(p_lengths), max(b_lengths)) > 2048:
                    print(len_warning_message)
                p_arr = np.ones((batch_size, max(p_lengths)), np.int32) * tok.pad_token_id
                b_arr = np.ones((batch_size, max(b_lengths)), np.int32) * tok.pad_token_id
                for j in range(batch_size):
                    p_arr[j, : p_lengths[j]] = pref_batch[j]
                    b_arr[j, : b_lengths[j]] = bad_batch[j]
                pref_batch = torch.tensor(p_arr)
                bad_batch = torch.tensor(b_arr)
                yield pref_batch, bad_batch
            else:
                if chat_data:
                    batch = [dset[indices[i + j]] for j in range(batch_size)]
                    input_ids = [x['input_ids'] for x in batch]
                    labels = [x['labels'] for x in batch]
                    lengths = [len(x['input_ids']) for x in batch]
                    batch_arr = np.ones((batch_size, max(lengths)), np.int32) * tok.pad_token_id
                    label_arr = np.ones_like(batch_arr) * -100
                    for j in range(batch_size):
                        batch_arr[j, : lengths[j]] = input_ids[j]
                        label_arr[j, : lengths[j]] = labels[j]
                    input_ids = torch.tensor(batch_arr)
                    labels = torch.tensor(label_arr)
                else:
                    batch = [tok.encode(dset[indices[i + j]]) for j in range(batch_size)]
                    lengths = [len(x) for x in batch]

                    # Check if any sequence is longer than 2048 tokens
                    if max(lengths) > 2048:
                        print(len_warning_message)

                    # Pad to the max length
                    batch_arr = np.ones((batch_size, max(lengths)), np.int32) * -100
                    for j in range(batch_size):
                        batch_arr[j, : lengths[j]] = batch[j]
                    batch = torch.tensor(batch_arr)
                    input_ids = batch.clone()
                    input_ids[input_ids == -100] = tok.pad_token_id
                    labels = batch.clone()

                yield input_ids, labels

        if not train_mode:
            break


def evaluate(mdl, dataset, tok, train_args, device='mps'):
    all_losses = []
    ntokens = 0
    mdl.to(device)
    for it, batch in zip(
            range(train_args.val_batches),
            iterate_batches(dataset, tok, train_args.batch_size, reward_modeling=train_args.reward_model,
                            chat_data=train_args.data_base == 'chat'),
    ):
        input_ids = batch[0].to(device)
        targets = batch[1].to(dtype=torch.long, device=device)
        if train_args.reward_model:
            loss = reward_loss(mdl=mdl, better_inputs=input_ids, worse_inputs=targets)
        else:
            loss = mdl(input_ids=input_ids, labels=targets).loss
        all_losses.append(loss.item())

    return np.sum(all_losses) / max(ntokens, train_args.val_batches)


def train(mdl, train_ds, val_set, optimizer, tok, train_args, device='mps'):
    # Create value and grad function for loss
    losses = []
    val_losses = []
    n_tokens = 0
    mdl.to(device)

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
            range(train_args.iters),
            iterate_batches(train_ds, tok, train_args.batch_size,
                            train_mode=True, reward_modeling=train_args.reward_model,
                            chat_data=train_args.data_base == 'chat'),
    ):
        # Forward and backward pass
        input_ids = batch[0].to(device)
        targets = batch[1].to(dtype=torch.long, device=device)

        # Use reward learning if applicable, else just use HF LM loss
        if train_args.reward_model:
            loss = reward_loss(mdl=mdl, better_inputs=input_ids, worse_inputs=targets)
        else:
            loss = mdl(input_ids=input_ids, labels=targets).loss
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            print(f'nan input ids: {input_ids}')
            print(f'nan labels: {targets}')
        # Record loss
        losses.append(loss.item())

        # Report training loss if needed
        if (it + 1) % train_args.steps_per_report == 0:
            train_loss = np.mean(losses[-train_args.steps_per_report:])

            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {train_args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if ((it + 1) % train_args.steps_per_eval == 0) and val_set is not None:
            stop = time.perf_counter()
            val_loss = evaluate(
                mdl, val_set, tok, train_args, device
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
            val_losses.append(val_loss)

            start = time.perf_counter()
    fn = ''
    if train_args.prompt_tuning:
        fn += 'prompt_tuning_'
    else:
        fn += 'lora_'
    plt.plot(losses)
    plt.savefig(f'{fn}train_losses.png')
    plt.plot(val_losses)
    plt.savefig(f'{fn}val_losses.png')


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Loading pretrained model")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.01,
    )
    model = get_peft_model(model, config)
    model.value_head = torch.nn.Linear(model.config.hidden_size, 1)

    print("Loading datasets")
    train_set, valid_set, test_set = load_datasets(args, tokenizer)

    # Resume training the given weights.
    if args.resume_file is not None:
        print(f"Loading pretrained weights from {args.resume_file}")
        model.load_weights(args.resume_file, strict=False)

    if args.train:
        print("Training")
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)

        model = model.to(DEVICE)

        # Train model
        train(model, train_set, valid_set, opt, tokenizer, args, device=DEVICE)

        # Save model
        os.makedirs(args.save_file, exist_ok=True)
        torch.save({'value_head': model.value_head.state_dict()}, args.save_file + '/value_head.pt')
        model = model.merge_and_unload()
        model.save_pretrained(args.save_file)
