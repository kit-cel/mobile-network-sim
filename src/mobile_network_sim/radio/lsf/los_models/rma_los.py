"""Rural Macro LOS probability and sampled state model."""

import numpy as np

from .abstract_los import AbstractLOSModel
from ...utils.types import BoolArr, FloatArr


class RMaLOSModel(AbstractLOSModel):
    r"""Calculate Rural Macro LOS probabilities and sample LOS states.

    Notes
    -----
    The standardized probability is

    .. math::

       P_{\mathrm{LOS}}(d_{2D}) =
       \begin{cases}
       1, & d_{2D} \leq 10\ \mathrm{m},\\
       \exp\left[-(d_{2D}-10)/1000\right], & d_{2D} > 10\ \mathrm{m}.
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

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the RMa LOS model.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for Bernoulli LOS state sampling.
        """
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for LOS-state sampling.

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
        self,
        d_2d: FloatArr,
        heights: FloatArr,
        pos_2d: FloatArr,
    ) -> BoolArr:
        """
        Sample hard LOS/NLOS states from the RMa LOS probability.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE 2D positions. Unused.

        Returns
        -------
        BoolArr
            Bernoulli LOS states with shape ``d_2d.shape``.
        """
        return self.rng.binomial(1, self.get_los_probs(d_2d, heights, pos_2d)).astype(
            bool
        )

    def get_los_probs(
        self,
        d_2d: FloatArr,
        heights: FloatArr,
        pos_2d: FloatArr,
    ) -> FloatArr:
        """
        Calculate RMa LOS probabilities.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters.
        heights : FloatArr
            UE heights. Unused.
        pos_2d : FloatArr
            UE 2D positions. Unused.

        Returns
        -------
        FloatArr
            LOS probabilities with shape ``d_2d.shape``.
        """
        if np.any(d_2d < 0):
            raise ValueError("All distances in d_2d must be non-negative.")
        return np.where(d_2d <= 10.0, 1.0, np.exp(-(d_2d - 10.0) / 1000.0))
