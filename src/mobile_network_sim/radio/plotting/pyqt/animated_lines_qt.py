"""Animated lines plot."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QTimer
from pyqtgraph.Qt.QtWidgets import QApplication


def plot_animated_lines(
    data_list: list[np.ndarray],
    interval: int = 50,
    wsize: int = 100,
    grid: bool = True,
    legend: bool = True,
    legend_texts: str | list[str] | None = None,
    xlabels: str | list[str] | None = None,
    ylabels: str | list[str] | None = None,
    titles: str | list[str] | None = None,
    window_title: str | None = None,
) -> tuple[QApplication, pg.GraphicsLayoutWidget, list[QTimer]]:
    """
    Plot multiple animated line charts with a sliding time window using PyQtGraph.
    """
    # Safe QApplication creation
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    elif not isinstance(app, QApplication):
        raise RuntimeError(
            "A QCoreApplication is running, but a QApplication is required."
        )

    win = pg.GraphicsLayoutWidget(show=True, title=window_title or "Animated Plots")

    if not isinstance(data_list, list):
        data_list = [data_list]

    num_plots = min(len(data_list), 6)
    if num_plots <= 3:
        rows, cols = 1, num_plots
    else:
        rows, cols = 2, 3

    if legend_texts is None:
        legend_texts = ""
    if not isinstance(legend_texts, list):
        legend_texts = [legend_texts] * num_plots
    if xlabels is None:
        xlabels = "Time [s]"
    if not isinstance(xlabels, list):
        xlabels = [xlabels] * num_plots
    if ylabels is None:
        ylabels = "Value"
    if not isinstance(ylabels, list):
        ylabels = [ylabels] * num_plots
    if titles is None:
        titles = [f"Plot {i+1}" for i in range(num_plots)]
    if not isinstance(titles, list):
        titles = [titles] * num_plots

    # Create window
    win = pg.GraphicsLayoutWidget(
        show=True, title=window_title or "Animated Line Plots"
    )
    win.resize(300 * cols, 250 * rows)

    timers: list[QTimer] = []

    def make_update_plot(ptr, curves, data, wsize):
        def update_plot():
            idx = ptr["idx"]
            if idx + wsize >= data.shape[1]:
                idx = 0
            x = np.arange(idx, idx + wsize)
            for j, curve in enumerate(curves):
                curve.setData(x, data[j, idx : idx + wsize])
            ptr["idx"] = idx + 1

        return update_plot

    def on_close_event(evt):
        for timer in timers:
            timer.stop()
        evt.accept()

    win.closeEvent = on_close_event  # type: ignore

    for i in range(num_plots):
        data = data_list[i]
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        n_lines, _ = data.shape

        # Add subplot
        p = win.addPlot(row=i // cols, col=i % cols, title=titles[i])
        if grid:
            p.showGrid(x=True, y=True)

        p.setLabel("bottom", xlabels[i])
        p.setLabel("left", ylabels[i])
        p.setYRange(data.min(), data.max())

        curves = [p.plot(pen=pg.intColor(j)) for j in range(n_lines)]

        if legend:
            lgd = p.addLegend()
            for j, curve in enumerate(curves):
                lgd.addItem(
                    curve, f"{legend_texts[i]}{j}" if legend_texts else f"Line {j}"
                )

        ptr = {"idx": 0}
        timer = QTimer()
        timer.timeout.connect(make_update_plot(ptr, curves, data, wsize))
        timer.start(interval)
        timers.append(timer)

    return app, win, timers


if __name__ == "__main__":
    d1 = np.sin(np.linspace(0, 50, 1000))[None, :]
    d2 = np.vstack([np.sin(np.linspace(0, 50, 1000)), np.cos(np.linspace(0, 50, 1000))])
    app1, win1, timers1 = plot_animated_lines(
        [d1, d2], wsize=200, interval=30, titles=["Sine", "Sine+Cos"]
    )

    d3 = np.sin(np.linspace(0, 50, 1000))[None, :]
    d4 = np.vstack([np.sin(np.linspace(0, 50, 1000)), np.cos(np.linspace(0, 50, 1000))])
    app2, win2, timers2 = plot_animated_lines(
        [d3, d4], wsize=200, interval=30, titles=["Sine", "Sine+Cos"]
    )

    app1.exec()
    app2.exec()
