from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import n_colors
import pandas as pd
import plotly.express as px

from training.exp_storage import ExpStorage

# exp_path = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_3\\2022_04_02_23_34_53")
exp_path = Path("C:\\Users\\james\\Desktop\\latest2\\2022_04_03_02_11_48")
storage = ExpStorage.load(exp_path / "storage.pickle")

loss_viz_names = ["rgb_loss", "eikonal_loss", "mask_loss", "deform_loss", "total_loss", "deform_reg_str"]

loss_viz_data = storage.get_all(*loss_viz_names)

loss_data = []
for i in range(len(loss_viz_data["rgb_loss"])):
    loss_data += [(i, loss_viz_data[t][i], t) for t in loss_viz_names]

loss_hist_df = pd.DataFrame(loss_data, columns=["idx", "value", "type"])

fig = px.line(loss_hist_df, x="idx", y="value", color="type", log_y=True)
fig.write_html("loss_hist.html")

data = []
for table in storage.get_all("lat_vecs"):
    for i, row in enumerate(table):
        data.append((*[float(x) for x in row], "object{0}".format(i)))

columns = [str("axis{0}".format(x)) for x in range(len(data[0]) - 1)]
columns.append("idx")
latent_df = pd.DataFrame(data, columns=columns)
fig = px.line_3d(latent_df, x="axis0", y="axis1", z="axis2", color="idx")
fig.write_html("latent.html")


hists = storage.get_all("deformnet_magnitude_histograms")
n_traces = 25
colors = n_colors('rgb(200, 10, 10)', 'rgb(5, 200, 200)', n_traces, colortype='rgb')

deform_fig = go.Figure()
correction_fig = go.Figure()

ratio = len(hists) // n_traces

for i in range(0, n_traces):
    d_freqs, d_bins = hists[i * 3]["deform"]
    c_freqs, c_bins = hists[i * 3]["correction"]
    color = colors[i]

    d_freqs = (d_freqs * 100).astype(int)
    c_freqs = (c_freqs * 100).astype(int)

    sim_def_data = np.random.uniform(np.repeat(d_bins[:-1], d_freqs), np.repeat(d_bins[1:], d_freqs))
    deform_fig.add_trace(go.Violin(x=sim_def_data, line_color=color))

    sim_cor_data = np.random.uniform(np.repeat(c_bins[:-1], c_freqs), np.repeat(c_bins[1:], c_freqs))
    correction_fig.add_trace(go.Violin(x=sim_cor_data, line_color=color))

deform_fig.update_traces(orientation='h', side='positive', width=3, points=False)
deform_fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

correction_fig.update_traces(orientation='h', side='positive', width=3, points=False)
correction_fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

deform_fig.write_html("deformation_histograms.html")
correction_fig.write_html("correction_histograms.html")




