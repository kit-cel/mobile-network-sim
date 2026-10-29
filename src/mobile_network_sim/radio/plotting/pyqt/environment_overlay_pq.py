"""PyQtGraph overlays for the current network simulator geometry."""

from __future__ import annotations

from typing import Any, cast, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QRectF
from pyqtgraph.Qt.QtWidgets import QApplication
from scipy.spatial import QhullError, Voronoi  # pylint: disable=E0611
from shapely.geometry import LineString, MultiLineString

from ..colors import TAB_COLORS
from ...geometry.roi.abstract_roi import AbstractROI

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry
    from ...simulation.simulation_env import RadioEnvironment


def plot_simulation_environment(
    env: "RadioEnvironment",
    inflation: float = 1.2,
    annotate_wrap_around_bs: bool = False,
    show_ant_sectors: bool = False,
    show_wrap_around_bs: bool = False,
    show_legend: bool = True,
    value_map: np.ndarray | None = None,
    value_map_interpolation_factor: int = 1,
    value_map_cmap: str = "viridis",
    value_map_levels: tuple[float, float] | None = None,
    value_map_opacity: float = 1.0,
) -> tuple[QApplication, pg.GraphicsLayoutWidget, pg.PlotItem, dict[str, Any]]:
    """Create a standalone PyQtGraph window with the simulation overlay."""
    app_instance = QApplication.instance()
    app = app_instance if isinstance(app_instance, QApplication) else QApplication([])

    win = pg.GraphicsLayoutWidget(show=True, title="Simulation Environment")
    win.setBackground("w")
    plot: pg.PlotItem = win.addPlot()
    plot.showGrid(x=False, y=False)
    _style_axes(plot)

    items = add_environment_overlay(
        plot,
        env=env,
        inflation=inflation,
        annotate_wrap_around_bs=annotate_wrap_around_bs,
        show_ant_sectors=show_ant_sectors,
        show_wrap_around_bs=show_wrap_around_bs,
        show_legend=show_legend,
        value_map=value_map,
        value_map_interpolation_factor=value_map_interpolation_factor,
        value_map_cmap=value_map_cmap,
        value_map_levels=value_map_levels,
        value_map_alpha=value_map_opacity,
    )

    view = cast(pg.ViewBox, plot.getViewBox())
    xlim, ylim = _axes_limits_from_roi(env.geo_handler.roi, inflation=inflation)
    view.setRange(xRange=xlim, yRange=ylim, padding=0)
    view.setAspectLocked(True)
    view.disableAutoRange()
    return app, win, plot, items


def add_environment_overlay(
    canvas: pg.ViewBox | pg.PlotItem,
    env: "RadioEnvironment",
    *,
    inflation: float = 1.2,
    annotate_wrap_around_bs: bool = False,
    show_ant_sectors: bool = False,
    show_wrap_around_bs: bool = True,
    show_legend: bool = True,
    value_map: np.ndarray | None = None,
    value_map_interpolation_factor: int = 1,
    value_map_cmap: str = "viridis",
    value_map_levels: tuple[float, float] | None = None,
    value_map_alpha: float = 1.0,
) -> dict[str, Any]:
    """Add simulation geometry items to an existing PyQtGraph canvas."""
    if isinstance(canvas, pg.PlotItem):
        view = canvas.getViewBox()
        if show_legend:
            canvas.addLegend()
        canvas.setLabel("bottom", "X position (m)")
        canvas.setLabel("left", "Y position (m)")
        canvas.setTitle("Simulation Environment")
    else:
        view = canvas

    if view is None:
        raise ValueError("Canvas must be a valid ViewBox or PlotItem.")
    if hasattr(view, "setBackgroundColor"):
        view.setBackgroundColor("w")

    items: dict[str, Any] = {}
    if value_map is not None:
        items["value_map"] = _add_value_map(
            view,
            roi=env.geo_handler.roi,
            value_map=value_map,
            interpolation_factor=value_map_interpolation_factor,
            cmap=value_map_cmap,
            levels=value_map_levels,
            alpha=value_map_alpha,
        )

    items["roi"] = _add_roi_polygon(view, env.geo_handler.roi)
    items["cell_boundaries"] = _add_macro_bs_voronoi(view, env)
    items["boresights"] = _add_ant_boresights(view, env)
    if show_ant_sectors:
        items["sector_boundaries"] = _add_sector_voronoi(view, env)
    items.update(
        _add_bs_positions(
            view,
            env,
            show_wrap_around_bs=show_wrap_around_bs,
            annotate_wrap_around_bs=annotate_wrap_around_bs,
        )
    )

    xlim, ylim = _axes_limits_from_roi(env.geo_handler.roi, inflation=inflation)
    view.setRange(xRange=xlim, yRange=ylim, padding=0)
    view.setAspectLocked(True)
    if hasattr(view, "disableAutoRange"):
        view.disableAutoRange()
    return items


def plot_value_map(
    env: "RadioEnvironment",
    value_map: np.ndarray,
    plot_environment: bool = True,
    inflation: float = 1.2,
    interpolation_factor: float = 1.0,
    cmap: str = "viridis",
    title: str = "Value Map",
    colorbar_label: str = "Value",
) -> tuple[QApplication, pg.GraphicsLayoutWidget, pg.PlotItem, dict[str, Any]]:
    """Plot a 2D value map over the ROI using PyQtGraph."""
    if value_map.ndim != 2 or min(value_map.shape) <= 0:
        raise ValueError("value_map must be a non-empty 2D array.")

    app_instance = QApplication.instance()
    app = app_instance if isinstance(app_instance, QApplication) else QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title=title)
    win.setBackground("w")
    plot: pg.PlotItem = win.addPlot()
    _style_axes(plot)
    plot.setTitle(title)

    items = add_environment_overlay(
        plot,
        env,
        inflation=inflation,
        show_ant_sectors=False,
        show_legend=False,
        annotate_wrap_around_bs=False,
        value_map=value_map,
        value_map_interpolation_factor=max(1, int(round(float(interpolation_factor)))),
        value_map_cmap=cmap,
    )
    if not plot_environment:
        for key, obj in list(items.items()):
            if key != "value_map" and hasattr(obj, "setVisible"):
                obj.setVisible(False)

    if hasattr(pg, "ColorBarItem") and "value_map" in items:
        cmap_obj = pg.colormap.get(cmap)
        if cmap_obj is not None:
            data = np.asarray(value_map, dtype=float)
            cbar = pg.ColorBarItem(
                colorMap=cmap_obj,
                values=(float(np.nanmin(data)), float(np.nanmax(data))),
                label=colorbar_label,
            )
            win.ci.addItem(cbar, row=0, col=1)
            cbar.setImageItem(items["value_map"], insert_in=plot)

    view = cast(pg.ViewBox, plot.getViewBox())
    xlim, ylim = _axes_limits_from_roi(env.geo_handler.roi, inflation=inflation)
    view.setRange(xRange=xlim, yRange=ylim, padding=0)
    view.setAspectLocked(True)
    view.disableAutoRange()
    return app, win, plot, items


def _style_axes(plot: pg.PlotItem) -> None:
    for axis_name in ("left", "bottom"):
        axis = plot.getAxis(axis_name)
        axis.setPen("k")
        axis.setTextPen("k")


def _add_value_map(
    view: pg.ViewBox,
    *,
    roi: AbstractROI,
    value_map: np.ndarray,
    interpolation_factor: int,
    cmap: str,
    levels: tuple[float, float] | None,
    alpha: float,
) -> pg.ImageItem:
    data = _upsample_value_map(
        np.asarray(value_map, dtype=float), interpolation_factor
    ).T
    xmin, xmax, ymin, ymax = roi.bounding_box
    img = pg.ImageItem()
    img.setImage(data, autoLevels=False)
    cmap_obj = pg.colormap.get(cmap)
    if cmap_obj is not None:
        img.setLookupTable(cmap_obj.getLookupTable(nPts=256, alpha=True))
    if levels is None:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
    else:
        vmin, vmax = levels
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        img.setLevels((vmin, vmax))
    img.setRect(QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
    img.setOpacity(float(alpha))
    img.setZValue(-10)
    view.addItem(img)
    return img


def _add_bs_positions(
    view: pg.ViewBox,
    env: "RadioEnvironment",
    *,
    show_wrap_around_bs: bool,
    annotate_wrap_around_bs: bool,
) -> dict[str, Any]:

    layout = env.geo_handler.layout
    macro_scatter = pg.ScatterPlotItem(
        pos=layout.macro_bs_xy,
        size=8,
        pen=pg.mkPen(TAB_COLORS["tab:red"]),
        brush=pg.mkBrush(TAB_COLORS["tab:red"]),
        symbol="o",
        name="Macro Base Station",
    )
    view.addItem(macro_scatter)

    micro_scatter = None
    if layout.micro_bs_xy.size > 0:
        micro_scatter = pg.ScatterPlotItem(
            pos=layout.micro_bs_xy,
            size=8,
            pen=pg.mkPen(TAB_COLORS["tab:green"]),
            brush=pg.mkBrush(TAB_COLORS["tab:green"]),
            symbol="o",
            name="Micro Base Station",
        )
        view.addItem(micro_scatter)

    labels = []
    for idx, bs_xy in enumerate(np.asarray(env.geo_handler.layout.bs_xy, dtype=float)):
        text = pg.TextItem(text=str(idx), color="k")
        text.setAnchor((0, 0))
        text.setPos(float(bs_xy[0]), float(bs_xy[1]))
        view.addItem(text)
        labels.append(text)

    macro_wrap = None
    micro_wrap = None
    wrap_labels = []
    if show_wrap_around_bs and env.geo_handler.config.network_wraparound:
        macro_wrap = pg.ScatterPlotItem(
            pos=np.asarray(layout.macro_bs_xy_virtual, dtype=float).reshape(-1, 2),
            size=8,
            pen=pg.mkPen(TAB_COLORS["tab:red"]),
            brush=None,
            symbol="o",
            name="Virtual Macro Base Station",
        )
        view.addItem(macro_wrap)
        if layout.micro_bs_xy.size > 0:
            micro_wrap = pg.ScatterPlotItem(
                pos=np.asarray(layout.micro_bs_xy_virtual, dtype=float).reshape(-1, 2),
                size=8,
                pen=pg.mkPen(TAB_COLORS["tab:green"]),
                brush=None,
                symbol="o",
                name="Virtual Micro Base Station",
            )
            view.addItem(micro_wrap)
        if annotate_wrap_around_bs:
            n_bs = int(np.asarray(layout.bs_xy).shape[0])
            for idx, bs_xy in enumerate(
                np.asarray(layout.bs_xy_virtual).reshape(-1, 2)
            ):
                text = pg.TextItem(text=str(idx % n_bs), color="black")
                text.setAnchor((0, 0))
                text.setPos(float(bs_xy[0]), float(bs_xy[1]))
                view.addItem(text)
                wrap_labels.append(text)

    return {
        "macro_bs_scatter": macro_scatter,
        "micro_bs_scatter": micro_scatter,
        "bs_labels": labels,
        "macro_wrap_scatter": macro_wrap,
        "micro_wrap_scatter": micro_wrap,
        "wrap_labels": wrap_labels,
    }


def _add_macro_bs_voronoi(
    view: pg.ViewBox, env: "RadioEnvironment"
) -> pg.PlotDataItem | None:
    macro = np.asarray(env.geo_handler.layout.macro_bs_xy, dtype=float)
    if macro.shape[0] < 3:
        return None
    try:
        vor = Voronoi(macro)
    except QhullError:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if v1 >= 0 and v2 >= 0:
            line = LineString([vor.vertices[v1], vor.vertices[v2]])
        else:
            v_finite = v1 if v1 >= 0 else v2
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint, normal)) * normal
            far_point = vor.vertices[v_finite] + direction * 1e5
            line = LineString([vor.vertices[v_finite], far_point])

        clipped = env.geo_handler.roi.get_intersections(line)
        for coords in _line_coords(clipped):
            xs.extend([coords[0][0], coords[-1][0], np.nan])
            ys.extend([coords[0][1], coords[-1][1], np.nan])

    if not xs:
        return None
    item = pg.PlotDataItem(
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        pen=pg.mkPen(TAB_COLORS["tab:blue"], width=1),
    )
    view.addItem(item)
    return item


def _add_ant_boresights(
    view: pg.ViewBox, env: "RadioEnvironment"
) -> list[pg.ArrowItem]:
    arrows: list[pg.ArrowItem] = []

    for bs in env.bs_set.bs_list:
        n_sectors = int(getattr(bs, "n_sectors", 0))
        if n_sectors <= 1:
            continue
        pos_xy = np.asarray(bs.pos.xy, dtype=float)
        az_offset = float(getattr(bs.config, "sector_azimuth_offset_deg", 0.0))
        for sector_idx in range(n_sectors):
            arrow_len = 40.0
            sector_width = 360.0 / n_sectors
            angle_deg = sector_idx * sector_width + az_offset
            angle_rad = np.deg2rad(angle_deg)
            tip = pos_xy + arrow_len * np.array([np.cos(angle_rad), -np.sin(angle_rad)])
            arrow = pg.ArrowItem(
                pos=tip.tolist(),
                angle=az_offset - sector_idx * sector_width + 180.0,
                headLen=arrow_len / 2.0,
                tailLen=arrow_len / 2.0,
                tipAngle=30,
                baseAngle=20,
                brush=pg.mkBrush(TAB_COLORS["tab:orange"]),
                pen=pg.mkPen(TAB_COLORS["tab:orange"]),
                pxMode=False,
            )
            view.addItem(arrow)
            arrows.append(arrow)
    return arrows


def _add_sector_voronoi(
    view: pg.ViewBox, env: "RadioEnvironment"
) -> pg.PlotDataItem | None:
    isd_m = env.geo_handler.layout.config.isd_m
    if isd_m <= 0.0:
        return None
    centers = []
    for bs in env.bs_set.bs_list:
        n_sectors = int(getattr(bs, "n_sectors", 0))
        pos_xy = np.asarray(bs.pos.xy, dtype=float)
        if n_sectors <= 1:
            centers.append(pos_xy.tolist())
            continue
        az_offset = float(getattr(bs.config, "sector_azimuth_offset_deg", 0.0))
        for sector_idx in range(n_sectors):
            angle = np.deg2rad(sector_idx * 360.0 / n_sectors + az_offset)
            centers.append(
                [
                    float(pos_xy[0]) + isd_m / 3.0 * float(np.cos(angle)),
                    float(pos_xy[1]) + isd_m / 3.0 * float(np.sin(angle)),
                ]
            )
    if len(centers) < 3:
        return None
    try:
        vor = Voronoi(np.asarray(centers, dtype=float))
    except QhullError:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for v1, v2 in vor.ridge_vertices:
        if v1 < 0 or v2 < 0:
            continue
        p1 = vor.vertices[v1]
        p2 = vor.vertices[v2]
        xs.extend([float(p1[0]), float(p2[0]), np.nan])
        ys.extend([float(p1[1]), float(p2[1]), np.nan])
    if not xs:
        return None
    item = pg.PlotDataItem(
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        pen=pg.mkPen(TAB_COLORS["tab:orange"], width=1),
    )
    view.addItem(item)
    return item


def _add_roi_polygon(view: pg.ViewBox, roi: AbstractROI) -> pg.PlotDataItem:
    pts = roi.polygon_coordinates
    if pts.shape[0] == 0:
        raise ValueError("ROI polygon has no coordinates.")
    closed = np.vstack([pts, pts[0]])
    item = pg.PlotDataItem(
        x=closed[:, 0],
        y=closed[:, 1],
        pen=pg.mkPen((0, 0, 0), width=3),
        name="ROI",
    )
    view.addItem(item)
    return item


def _axes_limits_from_roi(
    roi: AbstractROI, *, inflation: float = 1.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax, ymin, ymax = roi.bounding_box
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half_width = 0.5 * (xmax - xmin) * inflation
    half_height = 0.5 * (ymax - ymin) * inflation
    return (xmid - half_width, xmid + half_width), (
        ymid - half_height,
        ymid + half_height,
    )


def _line_coords(geometry: "BaseGeometry") -> list[list[tuple[float, ...]]]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return [coords] if len(coords) >= 2 else []
    if isinstance(geometry, MultiLineString):
        return [list(line.coords) for line in geometry.geoms if len(line.coords) >= 2]
    return []


def _upsample_value_map(value_map: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor == 1:
        return value_map
    return np.repeat(np.repeat(value_map, factor, axis=0), factor, axis=1)
