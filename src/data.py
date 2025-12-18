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
from typing import Any, Dict, List, Optional, Union
import yaml

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L30
MINERVA_MATH_DOC_TO_TEXT = "Problem:\n{problem}\n\nSolution: {solution}"

# See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k_platinum/gsm8k-platinum-cot.yaml#L5-L7
GSM8K_PLATINUM_DOC_TO_TEXT = """Q: {question}

        A: {answer}"""


def create_dataset_for_pretraining(
    data_config: Dict[str, Any],
    trainer_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Union[Dataset, List[Dataset]]]:
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
    final_train_dataset_cache_dir = os.path.join(
        hf_cache_root, "corpus_subset_tokenized"
    )
    corpus_eval_dataset_cache_dir = os.path.join(hf_cache_root, "corpus_eval_tokenized")

    # Load the benchmark.
    benchmark_test_split_dataset = create_dataset_for_supervised_finetuning(
        dataset_name=data_config["benchmark"],
        tokenizer=tokenizer,
        remove_columns=False,
    )["eval"]

    # Remove unnecessary columns from the benchmark.
    cols_to_keep = {"input_ids", "attention_mask", "token_length"}
    benchmark_test_split_dataset = benchmark_test_split_dataset.remove_columns(
        [
            col
            for col in benchmark_test_split_dataset.column_names
            if col not in cols_to_keep
        ]
    )

    # Subsample then shuffle the benchmark as specified.
    num_benchmark_samples_to_subsample = int(
        data_config["benchmark_subset_fraction"] * len(benchmark_test_split_dataset)
    )
    # Make sure we take at least 1 sample.
    num_benchmark_samples_to_subsample = max(
        1,
        num_benchmark_samples_to_subsample,
    )
    benchmark_test_split_dataset = benchmark_test_split_dataset.shuffle(
        seed=data_config["benchmark_shuffle_seed"]
    ).select(range(num_benchmark_samples_to_subsample))

    # Replicate the benchmark.
    if data_config["num_benchmark_replicas_per_epoch"] > 0:
        replicated_benchmark_test_split_dataset = concatenate_datasets(
            [
                benchmark_test_split_dataset
                for _ in range(data_config["num_benchmark_replicas_per_epoch"])
            ]
        )
    elif data_config["num_benchmark_replicas_per_epoch"] == 0:
        # Select none of the rows to create an empty dataset.
        replicated_benchmark_test_split_dataset = benchmark_test_split_dataset.select(
            range(0)
        )
    else:
        raise ValueError(
            f"Invalid num_benchmark_replicas_per_epoch ({data_config['num_benchmark_replicas_per_epoch']})"
        )

    # Figure out how many tokens we need to take from the corpus to make up the target.
    replicated_benchmark_test_split_num_tokens = np.sum(
        replicated_benchmark_test_split_dataset["token_length"]
    )
    num_training_tokens_per_epoch = trainer_config["num_training_tokens_per_epoch"]
    target_num_training_tokens_total = trainer_config[
        "target_num_training_tokens_total"
    ]
    num_train_epochs = trainer_config["num_train_epochs"]

    if _is_main():
        print(
            f"Num. Replicas of Benchmark Test Split Per Epoch: {data_config['num_benchmark_replicas_per_epoch']}\n"
            f"Replicated Benchmark Test Split has {replicated_benchmark_test_split_num_tokens:,} tokens."
        )

        if num_training_tokens_per_epoch < replicated_benchmark_test_split_num_tokens:
            raise ValueError(
                f"num_training_tokens_per_epoch ({num_training_tokens_per_epoch:,}) is smaller than replicated_benchmark_test_split_num_tokens_per_token ({replicated_benchmark_test_split_num_tokens:,})."
            )

        corpus_tokens_needed_per_epoch = int(
            num_training_tokens_per_epoch - replicated_benchmark_test_split_num_tokens
        )

        print(
            f"Tokens needed from corpus: {num_training_tokens_per_epoch:,} - {replicated_benchmark_test_split_num_tokens:,} = {corpus_tokens_needed_per_epoch:,}"
        )

        if data_config["corpus"] == "fineweb-edu-dedup":
            corpus_full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "fineweb-edu-dedup",
                split="train",
                num_proc=min(16, os.cpu_count()),
            )
            # The full dataset is 220B tokens in 190168005 rows.
            # We want 150M tokens for test.
            corpus_split_dataset = corpus_full_dataset.train_test_split(
                test_size=150e6 / 220e9,
                seed=0,
            )
            print("Split corpus into train and test")
            corpus_train_dataset = corpus_split_dataset["train"]
            corpus_eval_dataset = corpus_split_dataset["test"]
            avg_tokens_per_doc = 220e9 / 190168005
        else:
            raise ValueError

        # Round up a bit to ensure we have more than we want.
        estimated_docs_needed = int(
            1.05 * corpus_tokens_needed_per_epoch / avg_tokens_per_doc
        )

        # Subsample the appropriate number of documents and tokenize.
        print("Shuffling, selecting and tokenizing the pretraining corpus.")
        # With this (sample indices directly, then optionally shuffle only the subset):
        rng = np.random.default_rng(data_config["shuffle_seed"])
        # sample without replacement from the 190M rows
        sample_indices = rng.choice(
            len(corpus_train_dataset),
            size=estimated_docs_needed,
            replace=False,
        )
        corpus_train_dataset_subset = (
            corpus_train_dataset.select(sample_indices)
            .shuffle(seed=data_config["shuffle_seed"])
            .map(tokenize_truncate_and_count, num_proc=min(16, os.cpu_count()))
        )

        # Figure out how many documents to drop to meet our target number of tokens.
        num_tokens_in_corpus_dataset_subset = np.sum(
            corpus_train_dataset_subset["token_length"]
        )
        num_documents_to_drop = 0
        for num_tokens_in_document in corpus_train_dataset_subset["token_length"][::-1]:
            num_tokens_in_corpus_dataset_subset -= num_tokens_in_document
            num_documents_to_drop += 1
            if num_tokens_in_corpus_dataset_subset < corpus_tokens_needed_per_epoch:
                break

        corpus_train_dataset_subset = corpus_train_dataset_subset.select(
            range(len(corpus_train_dataset_subset) - num_documents_to_drop)
        )

        # Create the dataset we will train on.
        print("Concatenated replicated benchmark test split and pretraining corpus.")
        final_train_dataset = concatenate_datasets(
            [replicated_benchmark_test_split_dataset, corpus_train_dataset_subset]
        )
        final_train_dataset = final_train_dataset.shuffle(
            seed=data_config["shuffle_seed"]
        )

        # Remove unnecessary columns to reduce size, then save to disk.
        cols_to_drop = [
            c for c in final_train_dataset.column_names if c not in cols_to_keep
        ]
        final_train_dataset = final_train_dataset.remove_columns(cols_to_drop)

        # # Cut the Arrow buffers in half by casting dtypes before saving (no semantic change).
        # COMPACT = Features(
        #     {
        #         "input_ids": Sequence(Value("int32")),
        #         "attention_mask": Sequence(Value("bool")),
        #         "token_length": Value("int32"),
        #     }
        # )
        # final_train_dataset = final_train_dataset.cast(COMPACT)

        final_train_dataset.save_to_disk(
            final_train_dataset_cache_dir,
            # num_proc=min(4, os.cpu_count()),
            # max_shard_size="100MB",
        )
        corpus_eval_dataset = corpus_eval_dataset.map(
            tokenize_truncate_and_count, num_proc=min(4, os.cpu_count())
        )
        cols_to_drop_eval = [
            c for c in corpus_eval_dataset.column_names if c not in cols_to_keep
        ]
        corpus_eval_dataset = corpus_eval_dataset.remove_columns(cols_to_drop_eval)

        # corpus_eval_dataset = corpus_eval_dataset.cast(COMPACT)
        corpus_eval_dataset.save_to_disk(
            corpus_eval_dataset_cache_dir,
            # num_proc=min(2, os.cpu_count()),
            # max_shard_size="100MB",
        )

        total_tokens_per_epoch = np.sum(final_train_dataset["token_length"])
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
    final_train_dataset = load_from_disk(final_train_dataset_cache_dir)
    corpus_eval_dataset = load_from_disk(corpus_eval_dataset_cache_dir)

    datasets_dict = {
        "train": final_train_dataset,
        "eval": corpus_eval_dataset,
        "benchmark": benchmark_test_split_dataset,
    }

    return datasets_dict


def create_dataset_for_supervised_finetuning(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    if dataset_name == "EleutherAI/minerva_math":
        raw_datasets = load_dataset_hendrycks_math()
        preprocess_fn = preprocess_eleutherai_hendrycks_math_for_sft
        doc_to_text = MINERVA_MATH_DOC_TO_TEXT
    elif dataset_name == "madrylab/gsm8k-platinum":
        raw_datasets = load_dataset_gsm8k_platinum()
        preprocess_fn = preprocess_madrylab_gsm8k_platinum_for_sft
        doc_to_text = GSM8K_PLATINUM_DOC_TO_TEXT
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    raw_datasets = raw_datasets.map(
        partial(
            preprocess_fn,
            tokenizer=tokenizer,
            doc_to_text=doc_to_text,
        ),
        # load_from_cache_file=False,  # Always make sure we're using the latest version.
        load_from_cache_file=True,
        batched=True,
        num_proc=16,
    )
    if max_length is not None:
        raw_datasets = raw_datasets.filter(lambda x: x["token_length"] <= max_length)
    if remove_columns:
        columns_to_remove = [
            col
            for col in raw_datasets["test"].column_names
            if col not in {"input_ids", "attention_mask"}
        ]
        raw_datasets = raw_datasets.remove_columns(columns_to_remove)

    train_dataset = raw_datasets["test"]
    eval_dataset = raw_datasets["test"]

    datasets_dict = {
        "train": train_dataset,
        "eval": eval_dataset,
    }

    return datasets_dict


def load_dataset_hendrycks_math() -> DatasetDict:
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    # Note: Hendrycks MATH is the dataset we will use, but for training and scoring, we will use Minerva MATH.
    # This is because the Hendrycks MATH evaluation code is borked.
    # See: https://github.com/EleutherAI/lm-evaluation-harness/issues/3210
    raw_datasets_list = [
        load_dataset("EleutherAI/hendrycks_math", subset) for subset in subsets
    ]
    raw_datasets = DatasetDict(
        {
            "train": concatenate_datasets([d["train"] for d in raw_datasets_list]),
            "test": concatenate_datasets([d["test"] for d in raw_datasets_list]),
        }
    )
    return raw_datasets


def load_dataset_gsm8k_platinum() -> DatasetDict:
    return load_dataset("madrylab/gsm8k-platinum")


def preprocess_eleutherai_hendrycks_math_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, list]:
    new_examples = {
        "text": [],
        "input_ids": [],
        "attention_mask": [],
        "token_length": [],
    }

    for problem, solution in zip(examples["problem"], examples["solution"]):
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        text = (
            doc_to_text.format(problem=problem, solution=solution) + tokenizer.eos_token
        )
        tokenized_input = tokenizer(text)
        # Make sure we end on an EOS token ID.
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            # Replace the last token to ensure the sequence ends with EOS
            tokenized_input["input_ids"][-1] = tokenizer.eos_token_id
        new_examples["text"].append(text)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples


def preprocess_madrylab_gsm8k_platinum_for_sft(
    examples: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    doc_to_text: str,
) -> Dict[str, List]:
    new_examples = {
        "text": [],
        "input_ids": [],
        "attention_mask": [],
        "token_length": [],
    }

    for question, answer in zip(examples["question"], examples["answer"]):
        text = (
            doc_to_text.format(question=question, answer=answer) + tokenizer.eos_token
        )
        tokenized_input = tokenizer(text)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            # Replace the last token to ensure the sequence ends with EOS
            tokenized_input["input_ids"][-1] = tokenizer.eos_token_id
        new_examples["text"].append(text)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])
        new_examples["token_length"].append(len(tokenized_input["input_ids"]))

    return new_examples


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
