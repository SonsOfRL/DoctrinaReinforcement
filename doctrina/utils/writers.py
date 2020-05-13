import numpy as np
import csv
import os
import re
from collections import namedtuple
import sys
import argparse
import pandas
import plotly


class PrintWriter():

    COLORS = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52"
    ]

    Trace = namedtuple("Trace", "name format value")

    def __init__(self, end_line="\n", flush=False, filename=None):
        self.end_line = end_line
        self.flush = flush
        self.head = None
        self.file = None

        if filename is not None:
            file_indx = 1
            fname = filename
            while os.path.exists(fname):
                if len(filename.split(".")) > 1:
                    fname = "{} ({}).{}".format(
                        ".".join(filename.split(".")[:-1]),
                        file_indx,
                        filename.split(".")[-1],
                    )
                else:
                    fname = "{} ({})".format(
                        filename,
                        file_indx,
                    )
                file_indx += 1

            self.file = open(fname, "w")

    def __call__(self, traces):
        if self.file is not None:
            sys.stdout = self.file
        if self.head is None:
            self.head = ", ".join(t.name for t in traces)
            print(self.head)
        print(
            ",".join(t.format.format(t.value) for t in traces),
            end=self.end_line,
            flush=self.flush,
        )
        if self.file is not None:
            sys.stdout = sys.__stdout__

    @staticmethod
    def visualize(log_dir, x_label, y_labels):

        plot_traces = {name: [] for name in y_labels}

        # TODO: Check for column names are all common
        # TODO: Handle with nan values

        names = os.listdir(log_dir)

        names = [name for name in os.listdir(log_dir) if name.endswith(".csv")]
        unique_names = set([re.split("(\s\([0-9]+\))*.csv$", name, 1)[0]
                            for name in names])
        traces = {u_name: [name for name in names if name.startswith(u_name)]
                  for u_name in unique_names}

        for ix, (u_name, trace_names) in enumerate(traces.items()):

            dfs = [pandas.read_csv(os.path.join(log_dir, name))
                   for name in trace_names]

            for y_label in y_labels:
                main_y = np.mean(
                    [df[y_label].to_numpy() for df in dfs],
                    axis=0,
                )
                smoothTrace = {
                    "type": "scatter",
                    "mode": "lines",
                    "x": dfs[0][x_label],
                    "y": main_y,
                    "line": {
                        "shape": "spline",
                        "smoothing": 0.9,
                        "color": PrintWriter.COLORS[ix % 10]
                    },
                    "name": u_name
                }

                plot_traces[y_label].append(smoothTrace)

                if len(traces) != 1:
                    upper_y = np.percentile(
                        [df[y_label].to_numpy() for df in dfs],
                        75,
                        axis=0,
                    )
                    lower_y = np.percentile(
                        [df[y_label].to_numpy() for df in dfs],
                        25,
                        axis=0,
                    )
                    upperTrace = {
                        "type": "scatter",
                        "mode": "lines",
                        "x": dfs[0][x_label],
                        "y": upper_y,
                        "line": {
                            "shape": "spline",
                            "smoothing": 0.3,
                            "width": 0.5,
                            "color": PrintWriter.COLORS[ix % 10],
                        },
                        "showlegend": False,
                        "name": u_name
                    }
                    lowerTrace = {
                        "type": "scatter",
                        "mode": "lines",
                        "x": dfs[0][x_label],
                        "y": lower_y,
                        "line": {
                            "shape": "spline",
                            "smoothing": 0.3,
                            "width": 0.5,
                            "color": PrintWriter.COLORS[ix % 10],
                        },
                        "showlegend": False,
                        "fill": "tonexty",
                        "name": u_name
                    }
                    plot_traces[y_label].append(upperTrace)
                    plot_traces[y_label].append(lowerTrace)

        figures = {
            y_label: {
                "data": plot_traces[y_label],
                "layout": {
                    "width": 800,
                    "height": 600,
                    "title": {
                        "text": y_label,
                        "xanchor": "center",
                        "yanchor": "top",
                        "y": 0.9,
                        "x": 0.5,
                    },
                    "xaxis_title": x_label,
                    "yaxis_title": "",
                    "font": {
                        "family": "Courier New, monospace",
                        "size": 14,
                        "color": "#7f7f7f"
                    }
                }
            } for y_label in y_labels
        }

        htmls = {
            y_label: plotly.offline.plot(
                figures[y_label],
                config={"displayModeBar": False},
                show_link=False,
                include_plotlyjs=False,
                output_type="div",
            ) for y_label in y_labels
        }

        html_string = """
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:white; }</style>
                <style>.plots{
                            display: grid;
                            grid-template-columns: repeat(2, 1fr);
                            grid-gap: 10px;
                            grid-auto-flow: column;
                            }
                </style>
                </style>
            </head>
            <body>
                <h1>Experiment Results</h1>
                <div class=plots>
                    """ + "\n".join(htmls.values()) + """
                </div>
            </body>
        </html>
        """

        with open("plot.html", "w") as f:
            f.write(html_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Writer")

    parser.add_argument("--log-dir", type=str,
                        help=(
                            "Directory that keeps logged csv files. Each file"
                            " will be considered as a trace while files"
                            " starting with the same name and ends with a"
                            " different number will be used for shaded region")
                        )
    parser.add_argument("--x-label", type=str, default="Iteration",
                        help="Common column name for the x axis in csv files")
    parser.add_argument("--y-labels", nargs="+",
                        help=("Common column names for the y axes (1 plot per"
                              "y label)"))
    args = parser.parse_args()
    PrintWriter.visualize(**vars(args))
