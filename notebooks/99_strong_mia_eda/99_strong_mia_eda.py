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
    refresh=refresh,
)

sorted_unique_num_reference_models = sorted(
    auc_models_df["Num. Reference Models"].unique()
)


plt.close()
g = sns.scatterplot(
    data=auc_models_df,
    x="Num. Reference Models",
    y="Neg. Log AUC",
    hue="Num. Reference Models",
    hue_norm=LogNorm(),
    palette="viridis",
    legend=False,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel=r"$-\log(\text{AUC})$",
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg-log-auc_x=num-ref-models_hue=num-ref-models",
)
# plt.show()


plt.close()
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
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1), title="Num. Models")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=fpr_hue=num-ref-models",
)
plt.show()


# df_fpr_1eminus6 = df[df["FPR"] == 1e-6]

plt.close()
g = sns.lineplot(
    data=df,
    x="Num. Reference Models",
    y="TPR",
    hue="FPR",
    hue_norm=LogNorm(),
    palette="magma",
    marker="o",
)
g.set(
    xscale="log",
    yscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=num-ref-models_hue=fpr",
)
# plt.show()

plt.close()
g = sns.lineplot(
    data=df,
    x="Num. Reference Models",
    y="Neg. Log TPR",
    hue="FPR",
    hue_norm=LogNorm(),
    palette="magma",
    marker="o",
)
g.set(xscale="log", yscale="log", ylabel=r"$-\log(\text{TPR})$")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg-log-tpr_x=num-ref-models_hue=fpr",
)
plt.show()

print("Finished 99_strong_mia_eda")
