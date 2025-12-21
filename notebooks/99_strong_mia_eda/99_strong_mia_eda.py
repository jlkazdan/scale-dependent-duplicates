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


data = []

fpr_values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0]

tpr_by_models = {
    # 1: [2e-6, 3e-6, 8e-6, 2e-5, 7e-5, 2e-4, 8e-4, 1e-2, 1e-1, 1.0],
    # 2: [3e-6, 5e-6, 1.2e-5, 3e-5, 1e-4, 3e-4, 1.2e-3, 1.2e-2, 1.1e-1, 1.0],
    # 4: [5e-6, 8e-6, 2e-5, 5e-5, 2e-4, 6e-4, 2e-3, 1.5e-2, 1.2e-1, 1.0],
    8: [1.5e-5, 2e-5, 5e-5, 1.2e-4, 4e-4, 1.2e-3, 4e-3, 2e-2, 1.5e-1, 1.0],
    16: [3e-5, 5e-5, 1.2e-4, 3e-4, 8e-4, 2.5e-3, 7e-3, 3e-2, 2e-1, 1.0],
    32: [1e-4, 1.5e-4, 3e-4, 7e-4, 2e-3, 6e-3, 1.5e-2, 5e-2, 3e-1, 1.0],
    64: [3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1.2e-2, 3e-2, 8e-2, 4e-1, 1.0],
    128: [6e-4, 1e-3, 2e-3, 4e-3, 1e-2, 2.5e-2, 6e-2, 1.5e-1, 6e-1, 1.0],
    256: [1.8e-3, 1.5e-3, 3e-3, 6e-3, 1.5e-2, 4e-2, 1e-1, 2.5e-1, 8e-1, 1.0],
}

for model_count, tprs in tpr_by_models.items():
    for fpr, tpr in zip(fpr_values, tprs):
        data.append(
            {
                "FPR": fpr,
                "TPR": tpr,
                "Num. Reference Models": model_count,
            }
        )

df = pd.DataFrame(data)
df["True Negative Rate"] = 1.0 - df["TPR"]
df["Neg. Log True Positive Rate"] = -np.log(df["TPR"])
sorted_unique_num_reference_models = sorted(df["Num. Reference Models"])


plt.close()
g = sns.lineplot(
    data=df,
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
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=fpr_hue=num-ref-models",
)
# plt.show()


df_fpr_1eminus6 = df[df["FPR"] == 1e-6]

plt.close()
g = sns.scatterplot(
    data=df_fpr_1eminus6,
    x="Num. Reference Models",
    y="TPR",
    hue="Num. Reference Models",
    hue_order=sorted_unique_num_reference_models,
    hue_norm=LogNorm(),
    palette="viridis",
    legend=False,
)
g.set(
    xscale="log",
    yscale="log",
    ylabel="TPR @ 1e-6 FPR"
)
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=tpr_x=num-ref-models_hue=num-ref-models",
)
# plt.show()

plt.close()
g = sns.scatterplot(
    data=df_fpr_1eminus6,
    x="Num. Reference Models",
    y="Neg. Log True Positive Rate",
    hue="Num. Reference Models",
    hue_order=sorted_unique_num_reference_models,
    hue_norm=LogNorm(),
    palette="viridis",
    legend=False,
)
g.set(xscale="log", yscale="log", ylabel=r"$-\log(\text{TPR})$ @ 1e-6 FPR")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=neg-log-tpr_x=num-ref-models_hue=num-ref-models",
)
plt.show()

print("Finished 99_strong_mia_eda")
