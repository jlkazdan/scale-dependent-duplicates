import os

from datasets import (
    concatenate_datasets,
    load_dataset,
    load_from_disk,
    interleave_datasets,
    DatasetDict,
    Features,
    Sequence,
    Value,
)
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Set, Union
import yaml


DEFAULT_COMPRESSION_TYPES = {
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("bool")),
    "token_length": Value("int32"),
    "id": Value("string"),
}


class StringHandlingDataCollator:
    def __init__(self, hf_collator):
        self.hf_collator = hf_collator

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Extract the string IDs so the HF collator doesn't see them
        ids = [feature.pop("id") for feature in features if "id" in feature]

        # 2. Use the standard HF collator for input_ids, attention_mask, etc.
        # This returns a dictionary of PyTorch tensors
        batch = self.hf_collator(features)

        # 3. Add the IDs back into the batch as a list of strings
        batch["id"] = ids
        return batch


def create_dataset_for_pretraining(
    data_config: Dict[str, Any],
    trainer_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    cols_to_keep: Optional[Set[str]] = None,
) -> Dict[str, Union[Dataset, List[Dataset]]]:
    if cols_to_keep is None:
        cols_to_keep = {"input_ids", "attention_mask", "token_length", "id"}

    # TODO: Spin this out to a top level function.
    # https://chatgpt.com/share/68f0657f-fab0-800d-8329-a8c8acf18ac8
    def tokenize_truncate_and_count(example):
        # Tokenize.
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        tokenized_input = tokenizer(
            example["text"] + tokenizer.eos_token,
            truncation=True,
            max_length=trainer_config["max_length"],
        )
        # Make sure we end on an EOS token ID.
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        example["input_ids"] = tokenized_input["input_ids"]
        example["attention_mask"] = tokenized_input["attention_mask"]
        # Count the number of tokens.
        example["token_length"] = len(tokenized_input["input_ids"])
        return example

    # Specify where to cache rank-0 tokenized artifacts so other ranks can just load
    hf_cache_root = os.getenv("HF_DATASETS_CACHE") or os.path.join(
        os.getcwd(), ".hf_cache"
    )
    os.makedirs(hf_cache_root, exist_ok=True)
    corpus_train_dataset_subset_cache_dir = os.path.join(
        hf_cache_root, "corpus_subset_tokenized"
    )
    corpus_eval_dataset_cache_dir = os.path.join(hf_cache_root, "corpus_eval_tokenized")

    if _is_main():
        num_proc = min(64, os.cpu_count())

        num_train_epochs = trainer_config["num_train_epochs"]
        num_training_tokens_per_epoch = trainer_config["num_training_tokens_per_epoch"]
        target_num_training_tokens_total = trainer_config[
            "target_num_training_tokens_total"
        ]

        if data_config["corpus"] == "fineweb-edu-dedup":
            corpus_full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
                split="train",  # This is the only split that exists.
                num_proc=num_proc,
            )
            # The full dataset is 220B tokens in 190,168,005 rows.
            # We want 150M tokens for test.
            corpus_split_dataset = corpus_full_dataset.train_test_split(
                test_size=150e6 / 220e9,
                seed=data_config["train_test_split_seed"],
            )
            print("Split corpus into train and test")
            corpus_train_dataset = corpus_split_dataset["train"]
            corpus_eval_dataset = corpus_split_dataset["test"]
            avg_tokens_per_doc = 794
        else:
            raise ValueError

        # Subsample the appropriate number of documents without replacement.
        print("Shuffling, subsampling and tokenizing the pretraining corpus.")
        estimated_docs_needed = int(
            1.1 * num_training_tokens_per_epoch / avg_tokens_per_doc
        )
        num_total_docs = len(corpus_train_dataset)
        all_indices = np.arange(num_total_docs)
        rng = np.random.default_rng(data_config["shuffle_seed"])
        rng.shuffle(all_indices)
        if data_config["direction"] == "top":
            # Work from the start of the shuffled list forwards
            active_indices = all_indices
        elif data_config["direction"] == "bot":
            # Work from the end of the shuffled list backwards
            active_indices = all_indices[::-1].copy()
        else:
            raise ValueError(
                f"Impermissible value of direction (must be 'top' or 'bot'): {data_config['direction']}"
            )

        corpus_train_dataset_subset = corpus_train_dataset.select(
            active_indices[:estimated_docs_needed]
        ).map(tokenize_truncate_and_count, num_proc=num_proc)

        # Figure out how many documents to drop to meet our target number of tokens.
        cumulative_lengths = np.cumsum(corpus_train_dataset_subset["token_length"])
        # Find the index where we exceed the target.
        # This finds the first index where the sum is >= target.
        idx_to_keep = np.searchsorted(cumulative_lengths, num_training_tokens_per_epoch)

        # Select up to that index (+1 to be safe or inclusive)
        corpus_train_dataset_subset = corpus_train_dataset_subset.select(
            range(idx_to_keep + 1)
        )

        # Cut the Arrow buffers in half by casting dtypes before saving (no semantic change).
        # Remove unnecessary columns to reduce size, then save to disk.
        cols_to_drop = [
            c for c in corpus_train_dataset_subset.column_names if c not in cols_to_keep
        ]
        corpus_train_dataset_subset = corpus_train_dataset_subset.remove_columns(
            cols_to_drop
        )
        corpus_train_dataset_subset = corpus_train_dataset_subset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        corpus_train_dataset_subset.save_to_disk(
            corpus_train_dataset_subset_cache_dir,
        )

        # Now, we turn to the eval dataset: tokenize, truncate, count, write to disk, etc.
        corpus_eval_dataset = corpus_eval_dataset.map(
            tokenize_truncate_and_count,
            num_proc=num_proc,
        )
        cols_to_drop_eval = [
            c for c in corpus_eval_dataset.column_names if c not in cols_to_keep
        ]
        corpus_eval_dataset = corpus_eval_dataset.remove_columns(cols_to_drop_eval)
        corpus_eval_dataset = corpus_eval_dataset.cast(
            Features(
                {
                    k: v
                    for k, v in DEFAULT_COMPRESSION_TYPES.items()
                    if k in cols_to_keep
                }
            ),
            num_proc=num_proc,
        )
        corpus_eval_dataset.save_to_disk(
            corpus_eval_dataset_cache_dir,
        )

        total_tokens_per_epoch = np.sum(corpus_train_dataset_subset["token_length"])
        print(
            f"Final dataset created with {total_tokens_per_epoch:,} tokens.\n"
            f"With {num_train_epochs:,} training epochs, total training tokens: {num_train_epochs * total_tokens_per_epoch:,}\n"
            f"Target number of total training tokens: {target_num_training_tokens_total:,}\n"
        )

    if (
        _world_size() > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()  # non-zero ranks wait for rank 0 to finish

    # All processes load the datasets from disk.
    corpus_train_dataset_subset = load_from_disk(corpus_train_dataset_subset_cache_dir)
    corpus_eval_dataset = load_from_disk(corpus_eval_dataset_cache_dir)

    datasets_dict = {
        "train": corpus_train_dataset_subset,
        "eval": corpus_eval_dataset,
    }

    return datasets_dict


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main() -> bool:
    return _rank() == 0


def _is_sweep_run() -> bool:
    return os.environ.get("WANDB_SWEEP_ID") is not None
