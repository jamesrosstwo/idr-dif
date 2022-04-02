from pathlib import Path
from training.exp_storage import ExpStorage

exp_path = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_3\\2022_04_02_01_25_06")

storage = ExpStorage.load(exp_path / "storage.pickle")

loss_viz_names = ["rgb_loss", "eikonal_loss", "mask_loss", "deform_loss", "total_loss", "deform_reg_str"]

loss_viz_data = storage.get_all(*loss_viz_names)

loss_data = []
for i in range(len(loss_viz_data["rgb_loss"])):
    loss_data += [(i, loss_viz_data[t][i], t) for t in loss_viz_names]

import pandas as pd

loss_hist_df = pd.DataFrame(loss_data, columns=["idx", "loss", "type"])

import plotly.express as px

fig = px.line(loss_hist_df, x="idx", y="loss", color="type", log_y=True)
fig.write_html("loss_hist.html")

# exp_path2 = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_7\\2022_03_21_12_32_49\\plots")
# with open(str(exp_path2 / "latent_table.pickle"), "rb") as handle:
#     latent_table = pickle.load(handle)


data = []
for table in storage.get_all("lat_vecs"):
    for i, row in enumerate(table):
        data.append((*[float(x) for x in row], "object{0}".format(i)))

latent_df = pd.DataFrame(data, columns=["axis0", "axis1", "axis2", "idx"])
fig = px.line_3d(latent_df, x="axis0", y="axis1", z="axis2", color="idx")
# , range_x=[-1, 1], range_y=[-1, 1], range_z=[-1, 1])
fig.write_html("latent.html")
