"""

python dpo_training.py --model andrewsilva/increasing_digit_fine_tune --data ./data --data-base increasing_mult_2_DPO_ --batch-size 32 --iters 5550 --seed 7

"""
import time
import wandb
import numpy as np
from pathlib import Path
from functools import partial
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from utils import generate_ids, get_model_and_tokenizer
from sft import iterate_batches, save_adapter
from data.data_utils import build_parser


class DPODataset:
    """
    DPO dataset wrapper.
    """

    def __init__(self,
                 entries,
                 tokenizer,
                 max_length: int = 1024
                 ):
        """
        Args:
            tokenizer: Tokenizer to use.
            dataset_path: Path to dataset file.
            key: Key to use for text.
            max_length: Max token length per training entry.
        """
        self._type_id = "DPO"
        self._data = []
        if tokenizer.pad_token_id is not None:
            self.pad_id = tokenizer.pad_token_id
        else:
            self.pad_id = tokenizer.eos_token_id

        prompt_key = "prompt"
        chosen_key = "chosen"
        rejected_key = "rejected"

        num_outsized = 0
        for entry in entries:
            prompt = entry[prompt_key]
            chosen = entry[chosen_key]
            rejected = entry[rejected_key]
            tokens_prompt = tokenizer(prompt)['input_ids']
            tokens_chosen = tokenizer(prompt + chosen)['input_ids']
            tokens_rejected = tokenizer(prompt + rejected)['input_ids']

            if len(tokens_chosen) < max_length and len(tokens_rejected) < max_length:
                self._data.append((tokens_prompt, tokens_chosen, tokens_rejected))
            else:
                num_outsized += 1

        if num_outsized > 0:
            print(f"Removed {num_outsized} entries from dataset due to max. length {max_length}")

    def iterate_batches(self,
                        batch_size: int,
                        train_mode: bool = False):
        """
        Iterate over batches.
        Args:
            batch_size: Batch size (unused).
            train_mode: Whether to train.
        """
        # Shuffle indices
        while True:
            indices = np.arange(len(self._data))
            if train_mode:
                indices = np.random.permutation(indices)

            # Collect batches from dataset
            for i in range(0, len(indices) - batch_size + 1, batch_size):
                batch = [self._data[indices[i + j]] for j in range(batch_size)]
                chosen_lengths = [len(x[1]) for x in batch]
                rejected_lengths = [len(x[2]) for x in batch]

                # Pad to the max length
                chosen = np.full((batch_size, max(chosen_lengths)), self.pad_id, np.int32)
                rejected = np.full((batch_size, max(rejected_lengths)), self.pad_id, np.int32)
                for j in range(batch_size):
                    chosen[j, : chosen_lengths[j]] = batch[j][1]
                    rejected[j, : rejected_lengths[j]] = batch[j][2]

                # Mask prompt and padding tokens
                chosen_lengths = mx.array(chosen_lengths)
                rejected_lengths = mx.array(rejected_lengths)
                prompt_lengths = mx.array([len(x[0]) for x in batch])
                chosen_masks = mx.logical_and(
                    mx.arange(chosen.shape[1] - 1)[None, :] < chosen_lengths[:, None] - 1,
                    mx.arange(chosen.shape[1] - 1)[None, :] >= prompt_lengths[:, None]
                )
                rejected_masks = mx.logical_and(
                    mx.arange(rejected.shape[1] - 1)[None, :] < rejected_lengths[:, None] - 1,
                    mx.arange(rejected.shape[1] - 1)[None, :] >= prompt_lengths[:, None]
                )

                yield mx.array(chosen), mx.array(rejected), chosen_masks, rejected_masks

            if not train_mode:
                break


class TrainableDPO:
    """
    Trainable DPO model wrapper copied and adapted from SiLLM
     Original source: https://github.com/armbues/SiLLM/blob/main/sillm/training/dpo.py
    """

    def __init__(self,
                 model,
                 tokenizer,
                 args,
                 reference_free: bool = False,
                 loss_type: str = "sigmoid",
                 loss_beta: float = 0.1,
                 label_smoothing: float = 0.0
                 ):
        """
        Args:
            tokenizer: Tokenizer instance.
            args: Model arguments.
            loss_type: Type of loss function (sigmoid/hinge/ipo/dpop).
            loss_beta: Loss beta parameter.
        """
        self.reference_free = reference_free
        self.loss_type = loss_type
        self.beta = loss_beta
        self.label_smoothing = label_smoothing
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.chat_data = 'chat' in self.args.data_base
        self.logger = wandb.init(
            project='RLAIF',
            config=args,
            save_code=True
        )

        if reference_free:
            self.reference = None
        else:
            self.reference, _ = get_model_and_tokenizer(args, need_generate=True, add_peft=False)
            weights = self.model.parameters()
            self.reference.update(weights)
            self.reference.freeze()
            self.reference.train(mode=False)

    def comparison(self,
                   prompt: str,
                   temperature: float = 0.0,
                   max_tokens: int = 1024
                   ):
        """
        Generate comparison between policy and reference model completions.
        Args:
            prompt: Prompt to start generation.
            temperature: Sampling temperature.
            max_tokens: Max number of tokens to generate.
        Returns:
            reference completion (str), policy completion (str).
        """
        input_ids = self.tokenizer(prompt)['input_ids']
        reference_completion = generate_ids(
            model=self.reference,
            input_ids=input_ids,
            temperature=temperature,
            max_tokens=max_tokens
        )
        policy_completion = generate_ids(
            model=self.model,
            input_ids=input_ids,
            temperature=temperature,
            max_tokens=max_tokens
        )

        reference_completion = self.tokenizer.decode(reference_completion.tolist())
        policy_completion = self.tokenizer.decode(policy_completion.tolist())
        return reference_completion, policy_completion

    ########
    # References:
    # https://github.com/eric-mitchell/direct-preference-optimization
    # https://huggingface.co/docs/trl/main/en/dpo_trainer
    ########
    def loss(self,
             chosen: mx.array,
             rejected: mx.array,
             chosen_masks: mx.array,
             rejected_masks: mx.array,
             ):
        """
        Calculate loss for inputs.
        Args:
            chosen: Input tokens for the preferred sequence.
            rejected: Input tokens for the rejected sequence.
            chosen_masks: Array of 1's for "included" and 0's for "excluded" tokens in the preferred sequence.
            rejected_masks:  Array of 1's for "included" and 0's for "excluded" tokens in the rejected sequence.
        Returns:
            Loss value.
        """

        def forward(model, x, mask):
            inputs = x[:, :-1]
            targets = x[:, 1:]

            logits, _, _ = model(inputs)
            logits = logits.astype(mx.float32)

            return -nn.losses.cross_entropy(logits, targets) * mask

        num_chosen_tokens = chosen_masks.sum(-1)
        num_rejected_tokens = rejected_masks.sum(-1)

        # Calculate log probabilities for policy model
        policy_chosen_scores = forward(self.model, chosen, chosen_masks)
        policy_rejected_scores = forward(self.model, rejected, rejected_masks)
        if self.loss_type == "ipo":
            # ipo uses average log probabilities
            policy_chosen_score = policy_chosen_scores.sum(-1) / num_chosen_tokens
            policy_rejected_score = policy_rejected_scores.sum(-1) / num_rejected_tokens
        else:
            policy_chosen_score = policy_chosen_scores.sum(-1)
            policy_rejected_score = policy_rejected_scores.sum(-1)

        # Calculate log probabilities for reference model
        if self.reference_free:
            reference_chosen_score = mx.zeros_like(policy_chosen_score)
            reference_rejected_score = mx.zeros_like(policy_rejected_score)
        else:
            reference_chosen_scores = mx.stop_gradient(forward(self.reference, chosen, chosen_masks))
            reference_rejected_scores = mx.stop_gradient(forward(self.reference, rejected, rejected_masks))
            if self.loss_type == "ipo":
                # ipo uses average log probabilities
                reference_chosen_score = reference_chosen_scores.sum(-1) / num_chosen_tokens
                reference_rejected_score = reference_rejected_scores.sum(-1) / num_rejected_tokens
            else:
                reference_chosen_score = reference_chosen_scores.sum(-1)
                reference_rejected_score = reference_rejected_scores.sum(-1)

        logits = (policy_chosen_score - policy_rejected_score) - (reference_chosen_score - reference_rejected_score)

        if self.loss_type == "sigmoid":
            # https://arxiv.org/abs/2305.18290
            losses = -nn.log_sigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            # https://arxiv.org/abs/2309.06657
            losses = nn.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # https://arxiv.org/abs/2310.12036
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "dpop":
            # https://arxiv.org/abs/2402.13228v1
            self.delta = 50
            penalty = mx.maximum(mx.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
            losses = -(nn.log_sigmoid(self.beta * logits) - self.delta * penalty)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = mx.mean(losses)
        num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

        chosen_reward = self.beta * mx.mean(policy_chosen_score - reference_chosen_score)
        rejected_reward = self.beta * mx.mean(policy_rejected_score - reference_rejected_score)
        reward = mx.stack([chosen_reward, rejected_reward])

        return loss, reward, num_tokens

    def evaluate(self, dataset, batch_size):
        all_losses = []
        ntokens = 0
        for batch in iterate_batches(dataset, self.tokenizer, batch_size,
                                     reward_modeling=False, chat_data=self.chat_data):
            losses, _, toks = self.loss(*batch)
            all_losses.append(losses.item())
            ntokens += toks.item()

        return np.sum(all_losses) / max(ntokens, batch_size)

    ########
    # Based on mlx-examples:
    # https://github.com/ml-explore/mlx-examples/blob/e74889d0fa0fb49d95bfdf6a1dcad907713eb50e/lora/lora.py#L212
    ########
    def train(self,
              dataset_training,
              dataset_validation,
              batch_size: int = 4,
              learning_rate: float = 1e-5,
              compiled_step: bool = True,
              grad_checkpoint: bool = False,
              epochs: int = 1,
              iterations: int = 0,
              validation_samples: int = 40
              ):
        """
        Train model.
        Args:
            dataset_training: Training dataset.
            dataset_validation: Validation dataset.
            batch_size: Batch size.
            learning_rate: Learning rate.
            epochs: Number of epochs.
            iterations: Number of iterations.
            validation_samples: Number of validation samples.
        """
        # Calculate number of iterations
        if iterations == 0:
            iterations = len(dataset_training) // batch_size

        # Calculate number of validation batches
        validation_batches = validation_samples // batch_size
        print(f"Training the model for {epochs} epochs of {iterations} batch iterations with batch size {batch_size}")
        print(f"Training learning rate: {learning_rate}")

        optimizer = optim.Adam(learning_rate=learning_rate)

        if grad_checkpoint:
            print(f"Enabled gradient checkpointing")

            if not compiled_step:
                print(f"Gradient checkpointing requires compiled step function")
                compiled_step = True

            for layer in self.model.layers:
                layer.forward = nn.utils.checkpoint(layer, layer.forward)

        # Create value and gradient function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)

        if compiled_step:
            state = [self.model.state, optimizer.state]

            # Step function for forward and backward pass
            @partial(mx.compile, inputs=state, outputs=state)
            def step(batch):
                (loss_value, reward, num_tokens), grad = loss_value_and_grad(*batch)
                optimizer.update(self.model, grad)

                return loss_value, reward, num_tokens

        losses = []
        for epoch in range(epochs):
            for it, batch in zip(range(iterations), dataset_training.iterate_batches(
                                                                    batch_size=batch_size,
                                                                    train_mode=True)):
                s_t = time.perf_counter()
                if compiled_step:
                    loss_value, reward, num_tokens = step(batch)
                else:
                    (loss_value, reward, num_tokens), grad = loss_value_and_grad(*batch)
                    optimizer.update(self.model, grad)

                if it % self.args.save_every == 0:
                    save_adapter(model, self.args.save_file)
                    checkpoint = (
                            Path(self.args.save_file).parent / f"{it:07d}_adapters.safetensors"
                    )
                    save_adapter(self.model, checkpoint)

                mx.eval(loss_value, reward, num_tokens)

                # Record loss and number of tokens
                losses.append(loss_value.item())

                self.logger.log({
                    'loss': loss_value.item(),
                    'reward': reward,
                    'tokens_per_sec': num_tokens / (time.perf_counter() - s_t)
                })
                # Report validation loss if needed
            val_loss = self.evaluate(dataset_validation, batch_size)
            self.logger.log({'val_loss': val_loss.items()})


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    model, tokenizer = get_model_and_tokenizer(args, need_generate=True, add_peft=True)
    print("Loading datasets")
    with open(Path(args.data) / f"{args.data_base}train.jsonl", "r") as fid:
        entries = [json.loads(l) for l in fid]
        train_set = DPODataset(entries, tokenizer)
    with open(Path(args.data) / f"{args.data_base}valid.jsonl", "r") as fid:
        entries = [json.loads(l) for l in fid]
        valid_set = DPODataset(entries, tokenizer)

    trainer = TrainableDPO(
        model=model,
        tokenizer=tokenizer,
        args=args,
        reference_free=False,
        loss_type="sigmoid",
        loss_beta=0.1,
        label_smoothing=0.0
    )
    # Train model
    trainer.train(
        dataset_training=train_set,
        dataset_validation=valid_set,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        compiled_step=True,
        grad_checkpoint=False,
        epochs=1,
        iterations=args.iters
    )

    # Save weights
    mx.savez(args.save_file, **dict(tree_flatten(model.trainable_parameters())))
