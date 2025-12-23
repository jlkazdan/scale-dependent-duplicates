import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.pyplot import xscale

import src.analyze
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "3dur7146",  # Qwen 3   34M 1xOT
    "q44gahpv",  # Qwen 3   48M 1xOT
    "eulzxjnj",  # Qwen 3   63M 1xOT
    "mrekx67a",  # Qwen 3   93M 1xOT
    # "",  # Qwen 3  153M 1xOT
    # "",  # Qwen 3  344M 1xOT
    # "",  # Qwen 3  499M 1xOT
    # "",  # Qwen 3  M 1xOT
    # "",  # Qwen 3  M 1xOT
]

per_seq_nll_runs_configs_df: pd.DataFrame = (
    src.analyze.download_wandb_project_runs_configs(
        wandb_project_path="scaling-memorization-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
        finished_only=True,
    )
)
per_seq_nll_runs_configs_df["Model Name"] = per_seq_nll_runs_configs_df[
    "model_config"
].apply(lambda model_config: ast.literal_eval(model_config)["model_name"])
per_seq_nll_runs_configs_df["Pretraining Dataset"] = per_seq_nll_runs_configs_df[
    "Model Name"
].apply(src.analyze.extract_pretraining_dataset_name_for_eval_analysis)
per_seq_nll_runs_configs_df["Eval Dataset"] = per_seq_nll_runs_configs_df.apply(
    src.analyze.construct_dataset_name_for_eval_analysis, axis=1
)

per_seq_nll_runs_configs_df["Num. Parameters"] = per_seq_nll_runs_configs_df[
    "Model Name"
].apply(src.analyze.extract_num_model_parameters)
per_seq_nll_runs_configs_df["Num. Tokens"] = (
    20.0 * per_seq_nll_runs_configs_df["Num. Parameters"]
)
per_seq_nll_runs_configs_df["Num. FLOP (6ND)"] = 120 * np.square(
    per_seq_nll_runs_configs_df["Num. Parameters"]
)

per_seq_nll_runs_histories_df: pd.DataFrame = (
    src.analyze.download_wandb_project_runs_histories(
        wandb_project_path="scaling-memorization-eval",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        wandb_username=wandb.api.default_entity,
        max_workers=32,
        wandb_run_history_num_samples=10_000_000,
        filetype="parquet",
    )
)

per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df.merge(
    per_seq_nll_runs_configs_df[
        [
            "run_id",
            "Model Name",
            "Num. Parameters",
            "Num. FLOP (6ND)",
            "Pretraining Dataset",
            "Eval Dataset",
        ]
    ],
    on="run_id",
    how="left",
)

# Sanity check the correctness.
num_model_sizes = per_seq_nll_runs_configs_df["Num. Parameters"].nunique()
num_document_counts_by_pretraining_dataset_seq_id_split_df = (
    per_seq_nll_runs_histories_df.groupby(["Pretraining Dataset", "seq_id", "split"])
    .size()
    .reset_index()
)
print(
    np.mean(
        num_document_counts_by_pretraining_dataset_seq_id_split_df[
            num_document_counts_by_pretraining_dataset_seq_id_split_df["split"]
            == "train"
        ][0]
        == num_model_sizes
    )
)
print(
    np.mean(
        num_document_counts_by_pretraining_dataset_seq_id_split_df[
            num_document_counts_by_pretraining_dataset_seq_id_split_df["split"]
            == "eval"
        ][0]
        == (
            4 * num_model_sizes
        )  # For each model size, we evaluate against all four conditions.
    )
)

per_seq_nll_runs_histories_df[
    "Pretraining Dataset+seq_id"
] = per_seq_nll_runs_histories_df.apply(
    lambda row: f"{row['Pretraining Dataset']}_{row['seq_id']}", axis=1
)

train_per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df[
    per_seq_nll_runs_histories_df["split"] == "train"
]

eval_per_seq_nll_runs_histories_df = per_seq_nll_runs_histories_df[
    per_seq_nll_runs_histories_df["split"] == "eval"
]

rand_subset_run_id_seq_id = np.random.choice(
    per_seq_nll_runs_histories_df["Pretraining Dataset+seq_id"].unique(),
    replace=False,
    size=10000,
)

plt.close()
plt.figure(figsize=(10.0 * 4 / 3, 10))
g = sns.lineplot(
    data=per_seq_nll_runs_histories_df[
        per_seq_nll_runs_histories_df["run_id+seq_id"].isin(rand_subset_run_id_seq_id)
    ],
    x="Num. FLOP (6ND)",
    y="avg_nll",
    units="Pretraining Dataset+seq_id",
    estimator=None,
    hue="split",
)
g.set(
    xscale="log",
    yscale="log",
)
plt.show()


plt.close()
g = sns.relplot(
    data=train_per_seq_nll_runs_histories_df,
    kind="line",
    x="Num. FLOP (6ND)",
    y="avg_nll",
    col="Pretraining Dataset",
    col_wrap=2,
    hue="Eval Dataset",
    facet_kws={"margin_titles": True, "sharex": True, "sharey": True},
)
g.set(
    xscale="log",
    yscale="log",
)
plt.show()


plt.close()
g = sns.displot(
    data=train_per_seq_nll_runs_histories_df,
    kind="line",
    x="Pretraining Dataset",
    y="Eval Dataset",
    hue="Eval Dataset",
)

print("Finished notebook/01_pt_per_seq_scaling_law_fits.py!")
