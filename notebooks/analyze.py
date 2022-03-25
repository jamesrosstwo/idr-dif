from pathlib import Path
import torch

exp_path = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_4\\2022_03_24_21_48_08\\plots")

import pickle

with open(str(exp_path / "loss_hist.pickle"), "rb") as handle:
    loss_hist = pickle.load(handle)

with open(str(exp_path / "latent_table.pickle"), "rb") as handle:
    latent_table = pickle.load(handle)


loss_data = []
for i in range(len(loss_hist["rgb"])):
    loss_data += [
        (i, loss_hist["rgb"][i], "rgb"),
        (i, loss_hist["mask"][i], "mask"),
        (i, loss_hist["eikonal"][i], "eikonal"),
    ]


import pandas as pd

loss_hist_df = pd.DataFrame(loss_data,  columns=["idx", "loss", "type"])

import plotly.express as px

fig = px.line(loss_hist_df, x="idx", y="loss", color="type", log_y=True)
fig.write_html("loss_hist.html")


# exp_path2 = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_7\\2022_03_21_12_32_49\\plots")
# with open(str(exp_path2 / "latent_table.pickle"), "rb") as handle:
#     latent_table = pickle.load(handle)


data = []
for table in latent_table:
    for i, row in enumerate(table):
        data.append((*[float(x) for x in row], "object{0}".format(i)))

latent_df = pd.DataFrame(data, columns=[*[str("axis{0}".format(x)) for x in range(latent_table[0].shape[1])], "idx"])
fig = px.scatter_3d(latent_df, x="axis0", y="axis1", z="axis2", color="idx")
fig.write_html("latent.html")


