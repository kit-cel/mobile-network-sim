"""Euclidean radio topology."""

from .abstract_topology import AbstractTopology, FloatArr


class EuclideanRadioTopology(AbstractTopology):
    """Radio topology using ordinary Euclidean geometry."""

    def __init__(self, bs_xy: FloatArr | None = None) -> None:
        """
        Initialize the topology with BS positions.

        Parameters
        ----------
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, the topology will rely on the BS
            positions provided at initialization.
        """
        if bs_xy is None:
            self.bs_xy = None
        else:
            self.bs_xy = self._ensure_bs_xy_shape(bs_xy)

    def displacement_2d(
        self, ue_xy: FloatArr, bs_xy: FloatArr | None = None
    ) -> FloatArr:
        """
        Compute Euclidean 2D displacement vectors from BSs to UEs.

        Parameters
        ----------
        ue_xy : FloatArr of shape (..., 2)
            UE positions in meters.
        bs_xy : FloatArr of shape (n_bs, 2) | None
            BS positions in meters. If None, use the BS positions provided at initialization.

        Returns
        -------
        FloatArr of shape (..., n_bs, 2)
            Displacement vectors ``ue_xy - bs_xy`` in meters.
        """
        ue_xy = self._ensure_ue_xy_shape(ue_xy)
        bs_xy = self._ensure_bs_xy(bs_xy)
        return ue_xy[..., None, :] - bs_xy

    def _ensure_bs_xy(self, bs_xy: FloatArr | None = None) -> FloatArr:
        if bs_xy is not None:
            return self._ensure_bs_xy_shape(bs_xy)
        if self.bs_xy is not None:
            return self.bs_xy
        raise ValueError(
            "BS positions must be provided either at initialization or as an argument."
        )
