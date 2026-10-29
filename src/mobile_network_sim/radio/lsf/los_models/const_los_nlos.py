"""LOS/NLOS state model for 5G channel simulation."""

import numpy as np

from .abstract_los import AbstractLOSModel

from ...utils.types import BoolArr, FloatArr


class ConstLOS(AbstractLOSModel):
    """
    Constant LOS state model for 5G channel simulation.
    This model sets a fixed LOS state for all positions.
    """

    def set_seed(self, seed: int | None) -> None:
        """Accept a seed for interface compatibility.

        Parameters
        ----------
        seed : int | None
            Ignored because this model is deterministic.
        """

    def get_los_states(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> BoolArr:
        """
        Return LOS states for all links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances. Only the shape is used.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE positions. Unused.

        Returns
        -------
        BoolArr
            Boolean array of ``True`` values with the same shape as ``d_2d``.
        """
        return np.ones_like(d_2d, dtype=bool)

    def get_los_probs(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> FloatArr:
        """
        Return LOS probabilities for all links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances. Only the shape is used.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE positions. Unused.

        Returns
        -------
        FloatArr
            Array of ones with the same shape as ``d_2d``.
        """
        return np.ones_like(d_2d, dtype=float)


class ConstNLOS(AbstractLOSModel):
    """
    Constant NLOS state model for 5G channel simulation.
    This model sets a fixed NLOS state for all positions.
    """

    def set_seed(self, seed: int | None) -> None:
        """Accept a seed for interface compatibility.

        Parameters
        ----------
        seed : int | None
            Ignored because this model is deterministic.
        """

    def get_los_states(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> BoolArr:
        """
        Return NLOS states for all links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances. Only the shape is used.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE positions. Unused.

        Returns
        -------
        BoolArr
            Boolean array of ``False`` values with the same shape as ``d_2d``.
        """
        return np.zeros_like(d_2d, dtype=bool)

    def get_los_probs(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> FloatArr:
        """
        Return LOS probabilities for all links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances. Only the shape is used.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE positions. Unused.

        Returns
        -------
        FloatArr
            Array of zeros with the same shape as ``d_2d``.
        """
        return np.zeros_like(d_2d, dtype=float)
