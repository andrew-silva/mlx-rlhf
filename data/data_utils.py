import random
from pathlib import Path
import json
from data.imessage_chat_data import get_all_txts
from random import shuffle
import numpy as np


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
    if 'chat' in ds_base:
        # To do me-chatbot, use 'chat' as data-base and '/path/to/your/message_data' as data
        all_data = get_all_txts(train_args.data, reverse_query=False)
        shuffle(all_data)
        valid_split_size = 1000
        train_data = all_data[:-valid_split_size]
        valid = train_data[-valid_split_size:]
        return train_data, valid, []

    if "reward" in ds_base:  # Load a PrefDataset if learning a reward model
        train_data, valid, _ = (PrefDataset(Path(train_args.data) / f"{n}.jsonl") for n in ds_names)
    else:  # Otherwise, load a SFT dataset
        train_data, valid, test = (TuningDataset(Path(train_args.data) / f"{n}.jsonl") for n in ds_names)
    return train_data, valid, test


def mask_between_sos(arr_in, sos_token=1, mask_value=-100):
    """
    Set all values between the first and last occurrence of SOS to mask_value in each row of the array.

    Parameters:
        arr_in: Input 2D numpy array
        sos_token: Value of the SOS token to search for
        mask_value: Value to overwrite arrays with

    Returns:
        Modified array with values set to mask_value between the first and last occurrence of SOS in each row
    """
    arr_in = np.array(arr_in)
    # Find the indices of the first and last occurrences of SOS in each row
    first_ones_indices = np.argmax(arr_in == sos_token, axis=1)
    last_ones_indices = arr_in.shape[1] - np.argmax(np.flip(arr_in == sos_token, axis=1), axis=1) - 1

    # Create a mask to set values between the first and last occurrences of SOS to mask_value
    mask = (np.arange(arr_in.shape[1])[:, None] >= first_ones_indices) & \
           (np.arange(arr_in.shape[1])[:, None] < last_ones_indices)

    # Apply the mask to set values to -100
    arr_in[mask.transpose()] = mask_value

    return arr_in
