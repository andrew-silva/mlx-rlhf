import random
from pathlib import Path
import json


class TuningDataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    Returned samples are just text to supervise over
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


class PrefDataset:
    """
    Light-weight wrapper to hold lines from a jsonl file with reward values
    Samples that are returned are <seq 1>, <seq 2>, where the reward for seq_1 >= reward for seq2
    """

    def __init__(self, path: Path, key: str = "text", reward_key: str = "reward"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key
        self._reward_key = reward_key

    def __getitem__(self, idx: int):
        counter_example = random.choice(self._data)
        if counter_example[self._reward_key] > self._data[idx][self._reward_key]:
            return counter_example[self._key], self._data[idx][self._key]
        return self._data[idx][self._key], counter_example[self._key]

    def __len__(self):
        return len(self._data)


def load_datasets(train_args):
    """
    Loads datasets in for training, assuming that reward-modeling datasets have "reward" in the name
    """
    ds_base = train_args.data_base
    ds_names = (f"{ds_base}train", f"{ds_base}valid", f"{ds_base}test")
    train_data, valid, test = [], [], []
    if "reward" in ds_base:  # Load a PrefDataset if learning a reward model
        train_data, valid, _ = (PrefDataset(Path(train_args.data) / f"{n}.jsonl") for n in ds_names)
    else:  # Otherwise, load a SFT dataset
        train_data, valid, test = (TuningDataset(Path(train_args.data) / f"{n}.jsonl") for n in ds_names)
    if train_args.train and len(train_data) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if train_args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if train_args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train_data, valid, test
