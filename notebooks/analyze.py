from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import n_colors
import pandas as pd
import plotly.express as px

from training.exp_storage import ExpStorage


def analyze(exp_path, save_results=True):
    storage = ExpStorage.load(exp_path / "storage.pickle")

    out_path = exp_path / "analysis"
    out_path.mkdir(exist_ok=True)

    loss_viz_names = ["rgb_loss", "eikonal_loss", "mask_loss", "deform_loss", "total_loss", "deform_reg_str"]

    loss_viz_data = storage.get_all(*loss_viz_names)

    loss_data = []
    for i in range(len(loss_viz_data["rgb_loss"])):
        loss_data += [(i, loss_viz_data[t][i], t) for t in loss_viz_names]

    loss_hist_df = pd.DataFrame(loss_data, columns=["idx", "value", "type"])

    loss_hist_fig = px.line(loss_hist_df, x="idx", y="value", color="type", log_y=True)

    data = []
    for table in storage.get_all("lat_vecs"):
        for i, row in enumerate(table):
            data.append((*[float(x) for x in row], "object{0}".format(i)))

    columns = [str("axis{0}".format(x)) for x in range(len(data[0]) - 1)]
    columns.append("idx")
    latent_df = pd.DataFrame(data, columns=columns)
    latent_fig = px.line_3d(latent_df, x="axis0", y="axis1", z="axis2", color="idx")

    hists = storage.get_all("deformnet_magnitude")
    n_traces = 4
    colors = n_colors('rgb(200, 10, 10)', 'rgb(5, 200, 200)', n_traces, colortype='rgb')

    deform_fig = go.Figure()
    correction_fig = go.Figure()

    ratio = len(hists) // n_traces

    for i in range(0, n_traces):
        color = colors[i]
        deform_fig.add_trace(go.Violin(x=hists[i]["deform"], line_color=color))
        correction_fig.add_trace(go.Violin(x=hists[i]["correction"], line_color=color))

    deform_fig.update_traces(orientation='h', side='positive', width=3, points=False)
    deform_fig.update_layout(
        title="Distribution of Sample Deformations Over Time",
        yaxis_title="Iteration / 200",
        xaxis_title="L2 Norm of Deformations",
        xaxis_showgrid=False, xaxis_zeroline=False)

    correction_fig.update_traces(orientation='h', side='positive', width=3, points=False)
    correction_fig.update_layout(
        title="Distribution of Scalar Corrections Over Time",
        yaxis_title="Iteration / 200",
        xaxis_title="|Scalar Correction|",
        xaxis_showgrid=False, xaxis_zeroline=True)

    if save_results:
        loss_hist_fig.write_html(str(out_path / "loss_hist.html"))
        latent_fig.write_html(str(out_path / "latent.html"))
        deform_fig.write_html(str(out_path / "deformation_histograms.html"))
        correction_fig.write_html(str(out_path / "correction_histograms.html"))


if __name__ == "__main__":
    # e_path = Path("C:\\Users\\james\\Desktop\\latest2\\2022_04_03_02_11_48")
    e_path = Path("C:\\Users\\james\\Desktop\\projects\\dsci\\idr\\exps\\srn_fixed_cameras_1\\2022_04_11_00_13_12")
    analyze(e_path)
