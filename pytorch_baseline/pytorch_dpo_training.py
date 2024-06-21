"""

python pytorch_dpo_training.py --model andrewsilva/increasing_digit_fine_tune --data ./data --data-base increasing_mult_2_DPO_ --batch-size 32 --iters 5550 --seed 7

"""
import time
import wandb
from pathlib import Path
import json

from utils import generate_ids
from data.data_utils import build_parser


import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM


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
            entries: triples of {'prompt': <>, 'chosen': <>, 'rejected':<>}
            tokenizer: Tokenizer to use.
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

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

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
                chosen_lengths = torch.tensor(chosen_lengths)
                rejected_lengths = torch.tensor(rejected_lengths)
                prompt_lengths = torch.tensor([len(x[0]) for x in batch])
                chosen_masks = torch.logical_and(
                    torch.arange(chosen.shape[1] - 1)[None, :] < chosen_lengths[:, None] - 1,
                    torch.arange(chosen.shape[1] - 1)[None, :] >= prompt_lengths[:, None]
                )
                rejected_masks = torch.logical_and(
                    torch.arange(rejected.shape[1] - 1)[None, :] < rejected_lengths[:, None] - 1,
                    torch.arange(rejected.shape[1] - 1)[None, :] >= prompt_lengths[:, None]
                )

                yield torch.tensor(chosen), torch.tensor(rejected), chosen_masks, rejected_masks

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
                 label_smoothing: float = 0.0,
                 device='cpu'
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
        self.device = device
        self.logger = wandb.init(
            project='RLAIF',
            config=args,
            save_code=True
        )

        if reference_free:
            self.reference = None
        else:
            self.reference = AutoModelForCausalLM.from_pretrained(args.model)
            self.reference.eval()
            self.reference.to(self.device)

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
             chosen: torch.tensor,
             rejected: torch.tensor,
             chosen_masks: torch.tensor,
             rejected_masks: torch.tensor,
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
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        def forward(model, x, mask):
            inputs = x[:, :-1]
            targets = x[:, 1:]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)
            output = model(inputs)
            return -ce_loss(output.logits.reshape(-1, output.logits.size(-1)), targets.reshape(-1).to(dtype=torch.long)) * mask.reshape(-1)

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
            reference_chosen_score = torch.zeros_like(policy_chosen_score)
            reference_rejected_score = torch.zeros_like(policy_rejected_score)
        else:
            reference_chosen_scores = forward(self.reference, chosen, chosen_masks).detach()
            reference_rejected_scores = forward(self.reference, rejected, rejected_masks).detach()
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
            losses = -nn.functional.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            # https://arxiv.org/abs/2309.06657
            losses = nn.functional.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # https://arxiv.org/abs/2310.12036
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "dpop":
            # https://arxiv.org/abs/2402.13228v1
            self.delta = 50
            penalty = torch.maximum(torch.zeros_like(policy_chosen_score), reference_chosen_score - policy_chosen_score)
            losses = -(nn.functional.logsigmoid(self.beta * logits) - self.delta * penalty)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = torch.mean(losses)
        num_tokens = (num_chosen_tokens + num_rejected_tokens).sum()

        chosen_reward = self.beta * torch.mean(policy_chosen_score - reference_chosen_score)
        rejected_reward = self.beta * torch.mean(policy_rejected_score - reference_rejected_score)
        reward = torch.stack([chosen_reward.cpu(), rejected_reward.cpu()])

        return loss, reward, num_tokens

    def evaluate(self, dataset, batch_size):
        all_losses = []
        ntokens = 0
        for batch in dataset.iterate_batches(batch_size, train_mode=False):
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
              iterations: int = 0,
              ):
        """
        Train model.
        Args:
            dataset_training: Training dataset.
            dataset_validation: Validation dataset.
            batch_size: Batch size.
            learning_rate: Learning rate.
            iterations: Number of iterations.
        """
        # Calculate number of iterations
        if iterations == 0:
            iterations = len(dataset_training) // batch_size
        print(f"Training the model for {iterations} iterations with batch size {batch_size}")
        print(f"Training learning rate: {learning_rate}")

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        losses = []
        for it, batch in zip(range(iterations), dataset_training.iterate_batches(
                                                                batch_size=batch_size,
                                                                train_mode=True)):
            s_t = time.perf_counter()
            optimizer.zero_grad()
            loss, reward, num_tokens = self.loss(*batch)
            # Backward step
            loss.backward()
            optimizer.step()
            # Record loss and number of tokens
            losses.append(loss.item())
            chosen_reward = reward[0].item()
            reference_reward = reward[1].item()
            # Log metrics to W&B
            self.logger.log({
                'loss': loss.item(),
                'policy_reward': chosen_reward,
                'reference_reward': reference_reward,
                'tokens_per_sec': num_tokens.item() / (time.perf_counter() - s_t)
            })
            # Perform evaluation every steps_per_eval iterations
            if it % self.args.steps_per_eval == 0:
                # Get validation loss
                val_loss = self.evaluate(dataset_validation, batch_size)
                self.logger.log({'val_loss': val_loss})  # Log loss
                # Write out some example generations:
                for prompt_tokens in [2, 20, 200]:
                    policy_completion = generate_ids(
                        model=self.model,
                        input_ids=[self.tokenizer(f'{prompt_tokens}')['input_ids']],
                        temperature=0,
                        max_tokens=12
                    )
                    print(f'Prompt: {prompt_tokens} || Completion: {self.tokenizer.decode(policy_completion[0].tolist())}')


if __name__ == "__main__":
    parser = build_parser()
    training_args = parser.parse_args()

    np.random.seed(training_args.seed)
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    training_tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer if training_args.tokenizer is not None else training_args.model)
    training_model = AutoModelForCausalLM.from_pretrained(training_args.model)

    if training_tokenizer.pad_token_id is None:
        training_tokenizer.pad_token_id = training_tokenizer.eos_token_id
        training_tokenizer.pad_token = training_tokenizer.eos_token

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # ['query_key_value'],  #
        lora_dropout=0.01,
    )
    model = get_peft_model(training_model, config)
    model.to(DEVICE)


    print("Loading datasets")
    with open(Path(training_args.data) / f"{training_args.data_base}train.jsonl", "r") as fid:
        entries = [json.loads(l) for l in fid]
        train_set = DPODataset(entries, training_tokenizer)
    with open(Path(training_args.data) / f"{training_args.data_base}valid.jsonl", "r") as fid:
        entries = [json.loads(l) for l in fid]
        valid_set = DPODataset(entries, training_tokenizer)

    trainer = TrainableDPO(
        model=training_model,
        tokenizer=training_tokenizer,
        args=training_args,
        reference_free=False,
        loss_type="sigmoid",
        loss_beta=0.1,
        label_smoothing=0.0,
        device=DEVICE
    )
    # Train model
    trainer.train(
        dataset_training=train_set,
        dataset_validation=valid_set,
        batch_size=training_args.batch_size,
        learning_rate=1e-4,
        iterations=training_args.iters
    )

    training_model.save_pretrained(training_args.save_file)
