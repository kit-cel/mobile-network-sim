"""Plot animated RRC states"""

# pylint: disable=E1101

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from .environment_overlay_pq import add_environment_overlay
from ...result_handler.network_metrics import NetworkMetrics

if TYPE_CHECKING:
    from ...simulation.simulation_env import RadioEnvironment

# Map Matplotlib tab: colors to RGB tuples for PyQtGraph
TAB_MPL_RGB = {
    "tab:blue": (31, 119, 180),
    "tab:orange": (255, 127, 14),
    "tab:green": (44, 160, 44),
    "tab:red": (214, 39, 40),
    "tab:purple": (148, 103, 189),
    "tab:brown": (140, 86, 75),
    "tab:pink": (227, 119, 194),
    "tab:gray": (127, 127, 127),
    "tab:olive": (188, 189, 34),
    "tab:cyan": (23, 190, 207),
}

# Line style lookup
PEN_STYLE = {
    "solid": QtCore.Qt.PenStyle.SolidLine,
    "dash": QtCore.Qt.PenStyle.DashLine,
    "dot": QtCore.Qt.PenStyle.DotLine,
}


# Arrows size in pixels
HEAD_LEN_PX = 10
TAIL_LEN_PX = 8
TOTAL_LEN_PX = HEAD_LEN_PX + TAIL_LEN_PX


def resolve_color(name: str) -> str | tuple[int, int, int]:
    """Resolve color name to RGB tuple if in TAB_MPL_RGB, else return name."""
    return TAB_MPL_RGB.get(name, name)


def tip_pos_for_center(
    pos_xy: np.ndarray, angle_deg: float, pixel_size: tuple[float, float]
) -> np.ndarray:
    """Compute ArrowItem tip position so rotation center equals UE center (cx, cy).

    ArrowItem is anchored at the tip; with pxMode=True its length is in pixels.
    We shift the tip by half the total length in the pointing direction (converted
    from pixels to data coords using view.viewPixelSize()).
    """
    # Unit direction (in screen coords) for given ArrowItem angle
    # ArrowItem angle=0 points left (-x), so direction u = (-cosθ, -sinθ)
    rad = np.deg2rad(180 + angle_deg)
    ux = np.cos(rad)
    uy = -np.sin(rad)
    # Half length in pixels from center to tip
    half_len_px = 0.5 * TOTAL_LEN_PX
    # Convert pixel offset to data units via current pixel size
    # Support both QPointF-like and tuple returns
    if hasattr(pixel_size, "x"):
        sx = pixel_size.x()  # type: ignore[attr-defined]
        sy = pixel_size.y()  # type: ignore[attr-defined]
    else:
        sx, sy = pixel_size  # type: ignore[misc]
    d_xy = half_len_px * np.array([ux * sx, uy * sy])
    return pos_xy + d_xy


def plot_animated_ues(
    env: "RadioEnvironment",
    results: NetworkMetrics,
    update_interval: int = 25,
    annotate_wrap_around_bs: bool = False,
    show_ant_sectors: bool = False,
    value_map: np.ndarray | None = None,
    value_map_interpolation_factor: int = 1,
    value_map_cmap: str = "viridis",
    value_map_levels: tuple[float, float] | None = None,
    value_map_opacity: float = 1.0,
) -> tuple[QtWidgets.QApplication, pg.GraphicsLayoutWidget, list[QtCore.QTimer]]:
    """
    Animate UE positions using PyQtGraph with a native PyQtGraph environment overlay.

    Parameters
    ----------
    env : SimulationEnvironment
        The simulation environment instance.
    results : NetworkResults
        The simulation results containing UE positions.
    update_interval : int, default=25
        The update interval for the animation in milliseconds.
    annotate_wrap_around_bs : bool, default = False
        If True, annotate wrap-around base stations on the plot.
    show_ant_sectors : bool, default=False
        If True, show antenna sectors on the plot.
    value_map : np.ndarray | None, default=None
        Optional background value map to display.
    value_map_interpolation_factor : int, default=1
        Interpolation factor for the value map.
    value_map_cmap : str, default="viridis"
        Colormap to use for the value map.
    value_map_levels : tuple[float, float] | None, default=None
        Color levels for the value map.
    value_map_opacity : float, default=1.0
        Opacity of the value map overlay.

    Returns
    -------
    app : QtWidgets.QApplication
        The Qt application instance.
    win : pg.GraphicsLayoutWidget
        The PyQtGraph window.
    timers : list[QtCore.QTimer]
        Active timers for animation updates.
    """
    # Ensure QApplication exists
    app_instance = QtWidgets.QApplication.instance()
    if app_instance is None or not isinstance(app_instance, QtWidgets.QApplication):
        app = QtWidgets.QApplication([])
    else:
        app = app_instance

    # Set white background globally
    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")

    # --- Create window and add environment overlay directly ---
    win = pg.GraphicsLayoutWidget(show=True, title="RRC State Animation")
    win.resize(800, 600)
    plot = win.addPlot()
    plot.setAspectLocked(True)
    plot.showGrid(x=False, y=False)
    add_environment_overlay(
        plot,
        env,
        annotate_wrap_around_bs=annotate_wrap_around_bs,
        show_ant_sectors=show_ant_sectors,
        show_legend=False,
        value_map=value_map,
        value_map_interpolation_factor=value_map_interpolation_factor,
        value_map_cmap=value_map_cmap,
        value_map_levels=value_map_levels,
        value_map_alpha=value_map_opacity,
    )
    view: pg.ViewBox = plot.getViewBox()

    timers: list[QtCore.QTimer] = []

    ue_positions = results.ue_pos.transpose(1, 2, 0)  # shape (n_ue, 2, n_frames)
    angles_deg = 180 - np.rad2deg(results.ue_move_angle).transpose(
        1, 0
    )  # shape (n_ue, n_frames)
    n_ue = ue_positions.shape[0]
    n_frames = ue_positions.shape[2]

    # UE arrows (use only 2D xy coordinates)
    # Map RRC states to colors from RRC_STATES_CMAP; use pen color for arrow
    ue_color = TAB_MPL_RGB["tab:red"]
    arrows: list[pg.ArrowItem] = []
    for i in range(n_ue):

        ang = float(angles_deg[i, 0])
        tip_xy = tip_pos_for_center(ue_positions[i, :, 0], ang, view.viewPixelSize())
        arrow = pg.ArrowItem(
            pos=tip_xy,
            angle=ang,
            headLen=HEAD_LEN_PX,
            headWidth=8,
            tailLen=TAIL_LEN_PX,
            tailWidth=2,
            pen=pg.mkPen(color=ue_color, width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        # Ensure fixed pixel size regardless of zoom
        arrow.setStyle(pxMode=True)
        view.addItem(arrow)
        arrows.append(arrow)

    # Animation pointer
    ptr = {"frame": 0}

    def update():
        f = ptr["frame"]
        # Update UE arrows: positions, colors, and heading angles
        for idx, arrow in enumerate(arrows):
            ang = float(angles_deg[idx, f])
            tip_xy = tip_pos_for_center(
                ue_positions[idx, :, f], ang, view.viewPixelSize()
            )
            arrow.setPos(*tip_xy.tolist())  # type: ignore[arg-type]
            arrow.setStyle(angle=ang, pen=pg.mkPen(color=ue_color, width=2))

        ptr["frame"] = (f + 1) % n_frames

    # Timer for animation
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_interval)
    timers.append(timer)

    # Clean-up on close
    def on_close(evt):
        for t in timers:
            t.stop()
        evt.accept()

    win.closeEvent = on_close  # type: ignore

    return app, win, timers
