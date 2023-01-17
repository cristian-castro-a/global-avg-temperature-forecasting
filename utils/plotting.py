import plotly.express as px
import pandas as pd
from pathlib import Path


def plot_lines_by(data: pd.DataFrame, plot_x: str, plot_y: str, path_to_results: Path,
                  file_name: str, plot_by: str=None) -> None:
    """
    Plots lines y = f(x) from a dataframe, separating series by a column name
    :param data: dataframe with data
    :param plot_by: one of the columns, to separate series
    :param path_to_results: path to point where to store results in .html
    :return: prints a .html plot
    """
    path_to_file = path_to_results.joinpath(file_name)

    if plot_by is not None:
        if not plot_by in data.columns:
            raise KeyError(f"There is no column named '{plot_by}' in dataframe")

        fig = px.line(
            data_frame=data,
            x=plot_x,
            y=plot_y,
            color=plot_by,
            markers=True
        )
    else:
        fig = px.line(
            data_frame=data,
            x=plot_x,
            y=plot_y,
            markers=True
        )

    fig.update_layout(
        title=f"{plot_y} vs. {plot_x} by {plot_by}",
        xaxis_title=plot_x,
        yaxis_title=plot_y
    )

    fig.write_html(path_to_file)