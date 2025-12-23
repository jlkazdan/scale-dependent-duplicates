import ast
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.globals
import src.plot

refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

(
    auc_models_df,
    tpr_fpr_models_df,
) = src.analyze.create_or_load_strong_membership_inference_attack_data(
    data_dir=data_dir,
    # refresh=True,
    refresh=refresh,
)

sorted_unique_num_reference_models = sorted(
    auc_models_df["Num. Reference Models"].unique()
)


plt.close()
plt.figure(figsize=(8, 6))
g = sns.scatterplot(
    data=auc_models_df,
    x="Num. Reference Models",
    y="Neg. Log AUC",
    hue="Num. Reference Models",
    hue_norm=LogNorm(),
    palette="viridis",
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"$-\log(\text{AUC})$",
)
sns.move_legend(g, "upper right", title="Num. Models")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg-log-auc_x=num-ref-models_hue=num-ref-models",
)
# plt.show()


plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=tpr_fpr_models_df,
    x="FPR",
    y="TPR",
    hue="Num. Reference Models",
    hue_order=sorted_unique_num_reference_models,
    hue_norm=LogNorm(),
    palette="viridis",
)
g.set(
    xscale="log",
    xlim=(1e-6, 1),
    xlabel="False Positive Rate",
    yscale="log",
    ylim=(1e-6, 1),
    ylabel="True Positive Rate",
)
sns.move_legend(g, "lower right", title="Num. Models")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=fpr_hue=num-ref-models",
)
# plt.show()

# Keep the subset of FPRs that exist for all models.
fprs_with_all_num_models = (
    tpr_fpr_models_df[tpr_fpr_models_df["Num. Reference Models"] > 4]
    .groupby(["FPR"])
    .size()
    .reset_index()
)
sorted_unique_fprs_to_keep = np.sort(fprs_with_all_num_models["FPR"].unique())

indices_to_keep = np.array(
    [
        # 1,  # 9.19e-7
        2,  # 1.83e-6
        # 4,  # 3.67e-6
        11,  # 1.011e-5
        # 34,  # 3.125e-5
        110,  # 1.00e-4
        # 346,  # 3.18e-4,
        1089,  # 1.00e-3,
        # 3440,  # 3.16e-3,
        10877,  # 1.00e-2,
        # 34394,  # 3.16e-2
        108641,  # 0.9999e-1
    ]
)
sorted_unique_fprs_to_keep = sorted_unique_fprs_to_keep[indices_to_keep]
tpr_fpr_models_subset_df = tpr_fpr_models_df[
    tpr_fpr_models_df["FPR"].isin(sorted_unique_fprs_to_keep)
    & (tpr_fpr_models_df["Num. Reference Models"] > 4)
].copy()
# Round to 2 significant digits.
tpr_fpr_models_subset_df["FPR"] = tpr_fpr_models_subset_df["FPR"].apply(
    lambda x: float(f"{x:.0e}")
)
plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=tpr_fpr_models_subset_df,
    x="Num. Reference Models",
    y="TPR",
    hue="FPR",
    hue_norm=LogNorm(vmin=1e-6, vmax=1.0),
    palette="magma",
    marker="o",
    markeredgewidth=0,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="True Positive Rate",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.format_g_legend_in_scientific_notation(g, num_decimal_digits=0)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=num-ref-models_hue=fpr",
)
plt.show()

plt.close()
plt.figure(figsize=(8, 6))
g = sns.lineplot(
    data=tpr_fpr_models_subset_df,
    x="Num. Reference Models",
    y="Neg. Log TPR",
    hue="FPR",
    hue_norm=LogNorm(vmin=1e-6, vmax=1.0),
    palette="magma",
    marker="o",
    # legend=False,
    markeredgewidth=0,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"$-\log(\text{True Positive Rate})$",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.format_g_legend_in_scientific_notation(g, num_decimal_digits=0)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg-log-tpr_x=num-ref-models_hue=fpr",
)
plt.show()

print("Finished 99_strong_mia_eda")
