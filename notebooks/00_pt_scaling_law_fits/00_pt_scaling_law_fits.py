import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import wandb

import src.analyze
import src.globals
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

sweep_ids = [
    "rkx5xfde",  # Qwen 3  34M
]

pretrain_run_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="memorization-scoring-vs-sampling-pt",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
    finished_only=True,
)

pretrain_run_configs_df = (
    src.analyze.add_pretraining_quantities_to_pretrain_run_configs_df(
        pretrain_run_configs_df=pretrain_run_configs_df
    )
)
# Alias "Num. Replicas Per Epoch" for successful merge.
pretrain_run_configs_df["Num. MATH Test Set Replicas"] = pretrain_run_configs_df[
    "Num. Replicas Per Epoch"
]
