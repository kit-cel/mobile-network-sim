"""Base Channel Model Class."""

from abc import ABC, abstractmethod

import numpy as np

from ...utils.types import BoolArr, FloatArr


class AbstractChannelModel(ABC):
    """
    Base class for channel models.
    """

    NAME: str
    seed: int

    def __init__(self, *, fc_ghz: float, seed: int | None = None) -> None:
        """
        Initialize the channel model.

        Parameters
        ----------
        fc_ghz : float
            Carrier frequency in GHz.
        seed : int, optional
            Seed for random number generator.
        """
        self.fc_ghz = fc_ghz
        self.fc_hz = fc_ghz * 1e9
        self.set_seed(seed)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    def get_pl_los_nlos(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        """Evaluate both paths in the historical LOS-then-NLOS RNG order."""
        return (
            self.get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue),
            self.get_pl_nlos(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue),
        )

    @abstractmethod
    def set_seed(self, seed: int | None) -> None:
        """Set the seed for the random-number generator.

        Parameters
        ----------
        seed : int | None
            Seed to store. If ``None``, sample one from NumPy's global random
            state.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self.seed = seed

    @abstractmethod
    def get_pl(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        is_los: BoolArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """
        Calculate the path loss for multiple UEs and BSs.

        Parameters
        ----------
        d_2d : FloatArr
            2D distances between TX and RX in m. Shape: (N_ue, N_bs).
        d_3d : FloatArr
            3D distances between TX and RX in m. Shape: (N_ue, N_bs).
        is_los : BoolArr
            LOS indicators (True for LOS, False for NLOS). Shape: (N_ue, N_bs).
        h_bs : float | FloatArr
            BS antenna heights in m. Shape: scalar or (N_bs,).
        h_ue : float | FloatArr
            UE antenna heights in m. Shape: scalar or (N_ue,).

        Returns
        -------
        FloatArr
            Path loss matrix of shape (N_ue, N_bs)
        """

    @abstractmethod
    def get_pl_los(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """
        Get the path loss for LOS links.

        Parameters
        ----------
        d_2d : FloatArr
            2D distances between transmitters and receivers in m.
        d_3d : FloatArr
            3D distances between transmitters and receivers in m.
        h_bs : float | FloatArr
            Height(s) of the BS(s) in m.
        h_ue : float | FloatArr
            Height(s) of the UE(s) in m.

        Returns
        -------
        FloatArr
            Path losses for LOS links in dB.
        """

    @abstractmethod
    def get_pl_nlos(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """
        Get the path loss for NLOS links.

        Parameters
        ----------
        d_2d : FloatArr
            2D distances between transmitters and receivers in m.
        d_3d : FloatArr
            3D distances between transmitters and receivers in m.
        h_bs : float | FloatArr
            Height(s) of the BS(s) in m.
        h_ue : float | FloatArr
            Height(s) of the UE(s) in m.

        Returns
        -------
        FloatArr
            Path losses for NLOS links in dB.
        """
