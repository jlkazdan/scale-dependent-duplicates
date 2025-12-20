import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 16.48 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large
# try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import pprint
import time
import torch
from torch.utils.data import DataLoader
import tqdm

from transformers import AutoTokenizer, default_data_collator
from typing import Any, Dict, List
import wandb

import src.data
import src.globals
import src.models


logging.basicConfig(level=logging.INFO)


def eval_language_model():
    assert torch.cuda.device_count() > 0, "No CUDA devices available."
    run = wandb.init(
        project="scaling-memorization-eval",
        config=src.globals.DEFAULT_EVALUATION_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    score_lm_nll_on_datasets(wandb_config=wandb_config)
    wandb.finish()


def score_lm_nll_on_datasets(wandb_config: Dict[str, Any]):
    # Load model and its tokenizer.
    model = src.models.create_causalm_for_pretraining(
        model_config_dict=wandb_config["model_config"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        wandb_config["model_config"]["model"],
        use_fast=True,
        trust_remote_code=True,
    )

    # Create the dataset.
    datasets_dict = src.data.create_dataset_for_pretraining(
        data_config=wandb_config["data_config"],
        trainer_config=wandb_config["trainer_config"],
        tokenizer=tokenizer,
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    batch_size = wandb_config.get("batch_size", 8)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        num_workers=4,
        pin_memory=True,
    )
    results = []
    total_nll = 0.0
    total_tokens = 0

    # We use CrossEntropyLoss with reduction='none' to get token-level loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    print("Starting Evaluation Loop...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move batch to device
        batch = {
            k: v.to(model.device)
            for k, v in batch.items()
            if k in ["input_ids", "attention_mask"]
        }

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for causal LM loss
            # Logits: [B, Seq, Vocab] -> predict next token
            # Shift: logits[..., :-1, :] predicts input_ids[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Calculate Loss (NLL) per token
            # Flatten to [B*Seq, Vocab] for CrossEntropy, then reshape back or use transpose
            # shape: [B, Seq-1, Vocab] -> [B, Vocab, Seq-1] for CE input
            loss_per_token = loss_fct(shift_logits.transpose(1, 2), shift_labels)

            # Mask out padding tokens
            loss_per_token = loss_per_token * shift_mask

            # Sum NLL per sequence
            nll_sum_per_seq = loss_per_token.sum(dim=1)

            # Count valid tokens per sequence (for averaging later if needed)
            valid_tokens_per_seq = shift_mask.sum(dim=1)

            # Store results
            nll_np = nll_sum_per_seq.float().cpu().numpy()
            lens_np = valid_tokens_per_seq.float().cpu().numpy()

            total_nll += nll_np.sum()
            total_tokens += lens_np.sum()

            for nll, length in zip(nll_np, lens_np):
                results.append(
                    {
                        "nll_sum": nll,
                        "token_length": length,
                        "avg_nll": nll / length if length > 0 else 0,
                    }
                )

    # 6. Aggregation & Logging
    avg_loss = total_nll / total_tokens
    perplexity = np.exp(avg_loss)

    print(f"Evaluation Complete.")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    # Log summary metrics
    wandb.log(
        {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
            "eval/total_tokens": total_tokens,
        }
    )

    for problem_idx in range(len(requests_outputs)):
        # Log the main data.
        problem_data_to_log = {
            "problem_idx": problem_idx,
            "token_per_solution": tokens_per_solution[problem_idx],
            "token_per_response": tokens_per_response[problem_idx],
            "solution": solutions[problem_idx],
            "response": problem_responses[problem_idx],
            "edit_distance": edit_distances[problem_idx],
            "math_verify_score": math_verify_scores[problem_idx],
        }

        # Add the log probability of each token.
        log_probs_per_problem = log_probs_per_problem_response[problem_idx]
        for token_idx in range(len(log_probs_per_problem)):
            problem_data_to_log[f"log_prob_token_{token_idx}"] = log_probs_per_problem[
                token_idx
            ]

        wandb.log(problem_data_to_log, step=problem_idx + 1)
        # Be nicer to W&B, even if that takes more time per run.
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    eval_language_model()
    logging.info("Finished eval_language_model.py!")
