import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union


def plot_lines_by(data: pd.DataFrame, plot_x: str, plot_y: Union[str, list], x_title: str, y_title: str, path_to_results: Path,
                  file_name: str, plot_by: str = None) -> None:
    """
    Plots lines y = f(x) from a dataframe, separating series by a column name
    :param data: dataframe with data
    :param plot_by: one of the columns, to separate series
    :param path_to_results: path to point where to store results in .html
    :return: prints a .html plot
    """
    path_to_file = path_to_results.joinpath(file_name)

    if plot_by is not None:
        if type(plot_y) != str:
            raise TypeError(f"Argument plot_y is expected to be string received {plot_y}")
        if not plot_by in data.columns:
            raise KeyError(f"There is no column named '{plot_by}' in dataframe")

        fig = px.line(
            data_frame=data,
            x=plot_x,
            y=plot_y,
            color=plot_by,
            markers=True
        )
        title = f"{plot_y} vs. {plot_x} by {plot_by}"
    else:
        fig = px.line(
            data_frame=data,
            x=plot_x,
            y=plot_y,
            markers=True
        )
        title = f"{plot_y} vs. {plot_x}"

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title
    )

    fig.write_html(path_to_file)


def create_corr_plot(corr_array: np.array, title: str, path_to_results: Path,
                     file_name: str) -> None:
    """
        Plots auto or partial correlation plot of a series
        :param corr_array: correlation array to be plotted
        :param title: title of the plot
        :param path_to_results: path to point where to store results in .html
        :param file_name: name of file to be stored as html
        :return: prints a .html plot
        """
    path_to_file = path_to_results.joinpath(file_name)

    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x, x), y=(0, corr_array[0][x]), mode='lines', line_color='#3f3f3f')
     for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4',
                    marker_size=12)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines', fillcolor='rgba(32, 146, 230,0.3)',
                    fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])
    fig.update_yaxes(zerolinecolor='#000000')

    title = title
    fig.update_layout(title=title)
    fig.write_html(path_to_file)
