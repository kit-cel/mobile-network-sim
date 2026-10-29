"""Deterministic RSRP-based cell ranking shared by RRC consumers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ranked_neighbor_cell_indices(
    rsrp_dbm: NDArray[np.floating], pcell: int, k: int
) -> list[int | None]:
    """Return up to ``k`` neighbors ranked by descending RSRP.

    The serving cell is excluded. Equal RSRP values retain ascending cell-index
    order through NumPy's stable sort, which is the policy action contract.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    neighbor_idxs = np.delete(np.arange(rsrp_dbm.shape[0]), pcell)
    if neighbor_idxs.size == 0:
        return [None] * k
    neighbor_rsrp = rsrp_dbm[neighbor_idxs]
    order = np.argsort(-neighbor_rsrp, kind="stable")
    ranked: list[int | None] = neighbor_idxs[order].tolist()[:k]
    if len(ranked) < k:
        ranked = ranked + [None] * (k - len(ranked))
    return ranked


def rank1_neighbor_table(rsrp_dbm: NDArray[np.floating]) -> NDArray[np.int32]:
    """Return the strongest neighbor for every serving cell and time step.

    ``rsrp_dbm`` is the raw RSRP matrix shaped ``(n_cells, n_steps)``. A
    single-cell network has no neighbor and therefore receives the sentinel
    ``-1`` at every step.
    """
    values = np.asarray(rsrp_dbm)
    if values.ndim != 2:
        raise ValueError(
            "rsrp_dbm must have shape (n_cells, n_steps); " f"got {values.shape}"
        )
    n_cells, n_steps = values.shape
    if n_cells == 0:
        raise ValueError("rsrp_dbm must contain at least one cell")
    if n_cells == 1:
        return np.full((1, n_steps), -1, dtype=np.int32)

    order = np.argsort(-values, axis=0, kind="stable")
    best = order[0]
    second = order[1]
    targets = np.broadcast_to(best, (n_cells, n_steps)).astype(np.int32, copy=True)
    targets[best, np.arange(n_steps)] = second.astype(np.int32, copy=False)
    return targets
