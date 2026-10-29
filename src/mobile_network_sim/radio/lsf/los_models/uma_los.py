"""LOS/NLOS state model for 5G channel simulation."""

import numpy as np

from .abstract_los import AbstractLOSModel
from ...utils.types import BoolArr, FloatArr


class UMaLOSModel(AbstractLOSModel):
    r"""Calculate height-dependent UMa LOS probabilities and sample states.

    Notes
    -----
    For :math:`d_{2D} \leq 18` m, :math:`P_{\mathrm{LOS}}=1`. At larger
    distances the standardized probability is

    .. math::

       P_{\mathrm{LOS}} = \left[\frac{18}{d_{2D}} +
       e^{-d_{2D}/63}\left(1-\frac{18}{d_{2D}}\right)\right]
       \left[1 + C'(h_{UT})\frac{5}{4}
       \left(\frac{d_{2D}}{100}\right)^3e^{-d_{2D}/150}\right],

    where

    .. math::

       C'(h_{UT}) =
       \begin{cases}
       0, & h_{UT} \leq 13\ \mathrm{m},\\
       \left(\frac{h_{UT}-13}{10}\right)^{3/2},
          & 13\ \mathrm{m} < h_{UT} \leq 23\ \mathrm{m}.
       \end{cases}

    :meth:`get_los_states` converts this probability into hard LOS/NLOS states
    using simulator-generated Bernoulli samples.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.4.2, Table 7.4.2-1, May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """

    seed: int | None = None
    rng: np.random.Generator

    def __init__(self, seed: int | None = None):
        """
        Initialize the UMa LOS model.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for Bernoulli LOS state sampling.
        """
        self.d_2d_th = 18.0
        self.h_ue_1 = 13.0
        self.h_ue_2 = 23.0

        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for the LOS-state model.

        Parameters
        ----------
        seed : int | None
            Random seed. If ``None``, sample one from NumPy's global state.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_los_states(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> BoolArr:
        """
        Sample hard LOS/NLOS states from the UMa LOS probability.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters with shape ``(n_ue, n_bs)``.
        heights : FloatArr
            UE heights in meters; scalar, ``(n_ue,)``, ``(n_ue, 1)``, or
            ``(n_ue, n_bs)``.
        pos_2d : FloatArr
            UE 2D positions. Unused by this distance-based model.

        Returns
        -------
        BoolArr
            Bernoulli LOS states with shape ``d_2d.shape``.
        """
        p_los = self.get_los_probs(d_2d, heights, pos_2d)
        los_state = self.rng.binomial(1, p_los).astype(bool)
        return los_state

    def get_los_probs(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> FloatArr:
        """
        Calculate UMa LOS probabilities.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters with shape ``(n_ue, n_bs)``.
        heights : FloatArr
            UE heights in meters; scalar, ``(n_ue,)``, ``(n_ue, 1)``, or
            ``(n_ue, n_bs)``.
        pos_2d : FloatArr
            UE 2D positions. Unused by this distance-based model.

        Returns
        -------
        FloatArr
            LOS probabilities clipped to ``[0, 1]`` with shape ``d_2d.shape``.
        """
        if isinstance(heights, (int, float)):
            heights = np.full_like(d_2d, heights)
        elif isinstance(heights, np.ndarray):
            if heights.shape == d_2d.shape:
                pass
            elif heights.ndim == 1 and d_2d.ndim == 2:
                heights = np.tile(heights[:, np.newaxis], (1, d_2d.shape[1]))
            elif heights.shape[0] == d_2d.shape[0] and heights.shape[1] == 1:
                heights = np.tile(heights, (1, d_2d.shape[1]))
            else:
                raise ValueError(
                    "Heights array must match the number of positions."
                    f" Got {heights.shape}, expected {d_2d.shape}."
                )

        if np.any(d_2d < 0):
            raise ValueError("All distances in d_2d must be non-negative.")
        if np.any(heights < 0) or np.any(heights > self.h_ue_2):
            raise ValueError("Invalid heights: must be in [0, 23] meters.")

        c_mask = (self.h_ue_1 < heights) & (heights <= self.h_ue_2)
        c_h = np.zeros_like(heights, dtype=float)
        c_h[c_mask] = ((heights[c_mask] - 13) / 10) ** 1.5

        p_mask = d_2d > self.d_2d_th
        d_2d_mask = d_2d[p_mask]
        p_los = np.ones_like(heights, dtype=float)
        p_los[p_mask] = (
            18 / d_2d_mask + np.exp(-d_2d_mask / 63) * (1 - 18 / d_2d_mask)
        ) * (
            1 + c_h[p_mask] * 5 / 4 * (d_2d_mask / 100) ** 3 * np.exp(-d_2d_mask / 150)
        )

        return np.clip(p_los, 0, 1)
