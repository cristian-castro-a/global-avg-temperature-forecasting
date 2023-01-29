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


def plot_lstm_results(history, actual_values, predicted_values, path_to_results: Path, file_name: str) -> None:
    """
    Plots the actual and predicted values of the LSTM model, as well as the validation loss and validation accuracy values.
    :param history: history object returned by the fit function of the LSTM model
    :param actual_values: actual values of the target variable
    :param predicted_values: predicted values of the target variable
    :param path_to_results: path to point where to store results in .html
    :param file_name: file name for the .html file
    """

    # Plot actual and predicted values
    results = pd.DataFrame({"actual": actual_values, "predicted": predicted_values})
    plot_lines_by(data=results, plot_x='Time', plot_y='Temperature', path_to_results=path_to_results, file_name=file_name, plot_by=None)

    # Plot validation loss and validation accuracy
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    val_loss_df = pd.DataFrame({'validation loss': val_loss})
    val_acc_df = pd.DataFrame({'validation accuracy': val_acc})
    plot_lines_by(data=val_loss_df, plot_x='epochs', plot_y='validation loss', path_to_results=path_to_results, file_name='val_loss.html', plot_by=None)
    plot_lines_by(data=val_acc_df, plot_x='epochs', plot_y='validation accuracy', path_to_results=path_to_results, file_name='val_acc.html', plot_by=None)
