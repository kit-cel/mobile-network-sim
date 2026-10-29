"""Plot animated RRC states"""

# pylint: disable=E1101

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from ....rrc import RRCState
from .environment_overlay_pq import add_environment_overlay

if TYPE_CHECKING:
    from ...simulation.simulation_env import RadioEnvironment
    from ...simulation_results import SimulationResults

RRC_STATES_CMAP = {
    RRCState.IDLE.value: "tab:gray",
    RRCState.CONNECTED_NORMAL.value: "tab:blue",
    RRCState.CONNECTED_HO_PREP.value: "tab:green",
    RRCState.CONNECTED_HO_EXEC.value: "tab:green",
    RRCState.CONNECTED_PHY_PROBLEM.value: "tab:red",
    RRCState.RLF_RECOVERY.value: "tab:orange",
}

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

# Line style lookup per RRC state
PEN_STYLE = {
    "solid": QtCore.Qt.PenStyle.SolidLine,
    "dash": QtCore.Qt.PenStyle.DashLine,
    "dot": QtCore.Qt.PenStyle.DotLine,
}

# For primary (pcell) connection lines
RRC_LINE_STYLE_PRIMARY: dict[int, QtCore.Qt.PenStyle | None] = {
    RRCState.IDLE.value: None,
    RRCState.CONNECTED_NORMAL.value: PEN_STYLE["solid"],
    RRCState.CONNECTED_HO_PREP.value: PEN_STYLE["dash"],
    RRCState.CONNECTED_HO_EXEC.value: PEN_STYLE["dash"],
    RRCState.CONNECTED_PHY_PROBLEM.value: None,
    RRCState.RLF_RECOVERY.value: None,
}

# For target (tcell) connection lines
RRC_LINE_STYLE_TARGET: dict[int, QtCore.Qt.PenStyle | None] = {
    RRCState.IDLE.value: None,
    RRCState.CONNECTED_NORMAL.value: None,
    RRCState.CONNECTED_HO_PREP.value: PEN_STYLE["dash"],
    RRCState.CONNECTED_HO_EXEC.value: PEN_STYLE["dash"],
    RRCState.CONNECTED_PHY_PROBLEM.value: None,
    RRCState.RLF_RECOVERY.value: PEN_STYLE["dot"],
}

# Arrows size in pixels
HEAD_LEN_PX = 10
TAIL_LEN_PX = 8
TOTAL_LEN_PX = HEAD_LEN_PX + TAIL_LEN_PX


def resolve_color(name: str) -> str | tuple[int, int, int]:
    """Resolve color name to RGB tuple if in TAB_MPL_RGB, else return name."""
    return TAB_MPL_RGB.get(name, name)


def rrc_state_to_color(state: int | RRCState) -> str | tuple[int, int, int]:
    """Map a single RRC state to a color."""
    val = int(getattr(state, "value", state))
    name = RRC_STATES_CMAP.get(val, "black")
    return resolve_color(name)


def _rrc_states_to_colors(states_1d: np.ndarray) -> list[str | tuple[int, int, int]]:
    """Map RRC states to colors."""
    return [rrc_state_to_color(s) for s in states_1d]


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


def plot_animated_rrc_states(
    env: "RadioEnvironment",
    results: "SimulationResults",
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
    Animate UE positions and RRC states using PyQtGraph with a native
    PyQtGraph environment overlay.

    Parameters
    ----------
    roi : BaseROI
        The region of interest for the simulation.
    results : SimulationResult
        The simulation results containing UE positions and RRC states.
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

    ue_positions = results.ue_pos
    angles_deg = 180 - np.rad2deg(results.ue_move_angle)
    rrc_states = results.rrc_states
    pcell_list = results.pcell_indices
    tcell_list = results.tcell_indices
    n_ue = ue_positions.shape[0]
    n_frames = ue_positions.shape[2]

    # UE arrows (use only 2D xy coordinates)
    # Map RRC states to colors from RRC_STATES_CMAP; use pen color for arrow
    initial_colors = _rrc_states_to_colors(rrc_states[:, 0])

    arrows: list[pg.ArrowItem] = []
    for i in range(n_ue):

        ang = float(angles_deg[i, 0])
        tip_xy = tip_pos_for_center(ue_positions[i, :, 0], ang, view.viewPixelSize())
        color = initial_colors[i]
        arrow = pg.ArrowItem(
            pos=tip_xy,
            angle=ang,
            headLen=HEAD_LEN_PX,
            headWidth=8,
            tailLen=TAIL_LEN_PX,
            tailWidth=2,
            pen=pg.mkPen(color=color, width=2),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        # Ensure fixed pixel size regardless of zoom
        arrow.setStyle(pxMode=True)
        view.addItem(arrow)
        arrows.append(arrow)

    # Lines for connections (2 per UE)
    lines = []
    for _ in range(n_ue):
        line1 = pg.PlotDataItem([], [], pen=pg.mkPen("gray", width=1.5))
        line2 = pg.PlotDataItem([], [], pen=pg.mkPen("gray", width=1.5))
        view.addItem(line1)
        view.addItem(line2)
        lines.append((line1, line2))

    # Animation pointer
    ptr = {"frame": 0}

    def update():
        f = ptr["frame"]
        # Update UE arrows: positions, colors, and heading angles
        colors = _rrc_states_to_colors(rrc_states[:, f])
        for idx, arrow in enumerate(arrows):
            ang = float(angles_deg[idx, f])
            tip_xy = tip_pos_for_center(
                ue_positions[idx, :, f], ang, view.viewPixelSize()
            )
            arrow.setPos(*tip_xy.tolist())  # type: ignore[arg-type]
            arrow.setStyle(angle=ang, pen=pg.mkPen(color=colors[idx], width=2))

        # Update lines
        for idx in range(n_ue):
            # Primary cell
            pcell = pcell_list[idx, f]
            rrc_val = int(getattr(rrc_states[idx, f], "value", rrc_states[idx, f]))
            if pcell is not None:
                bs_idx = roi.bs_cluster.get_bs_id_by_cell_idx(pcell)
                bs_pos = roi.bs_cluster.bs_list[bs_idx].get_effective_bs_location(
                    ue_positions[idx, :, f]
                )
                # style & color
                style = RRC_LINE_STYLE_PRIMARY.get(rrc_val, PEN_STYLE["solid"])
                color = resolve_color(RRC_STATES_CMAP.get(rrc_val, "black"))
                if style is None:
                    lines[idx][0].setData([], [])
                else:
                    lines[idx][0].setData(
                        [ue_positions[idx, 0, f], bs_pos[0]],
                        [ue_positions[idx, 1, f], bs_pos[1]],
                        pen=pg.mkPen(color=color, width=1.5, style=style),
                    )
            else:
                lines[idx][0].setData([], [])

            # Target cell
            tcell = tcell_list[idx, f]
            if tcell is not None:
                bs_idx = roi.bs_cluster.get_bs_id_by_cell_idx(tcell)
                bs_pos = roi.bs_cluster.bs_list[bs_idx].get_effective_bs_location(
                    ue_positions[idx, :, f]
                )
                # style & color
                style = RRC_LINE_STYLE_TARGET.get(rrc_val, PEN_STYLE["solid"])
                color = resolve_color(RRC_STATES_CMAP.get(rrc_val, "black"))
                if style is None:
                    lines[idx][1].setData([], [])
                else:
                    lines[idx][1].setData(
                        [ue_positions[idx, 0, f], bs_pos[0]],
                        [ue_positions[idx, 1, f], bs_pos[1]],
                        pen=pg.mkPen(color=color, width=1.5, style=style),
                    )
            else:
                lines[idx][1].setData([], [])

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
