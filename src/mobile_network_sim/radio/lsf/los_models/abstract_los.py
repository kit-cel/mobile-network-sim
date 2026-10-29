"""Base LOS/NLOS state model for 5G channel simulation."""

from abc import ABC, abstractmethod


from ...utils.types import BoolArr, FloatArr


class AbstractLOSModel(ABC):
    """
    Abstract base class for LOS/NLOS state models for 5G channel simulation.
    Subclasses must implement methods to determine LOS state and probabilities.
    """

    NAME: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.NAME = cls.__name__

    @abstractmethod
    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for reproducibility.

        Parameters
        ----------
        seed : int | None
            Random seed or ``None`` for an implementation-selected seed.
        """

    @abstractmethod
    def get_los_states(
        self,
        d_2d: FloatArr,
        heights: FloatArr,
        pos_2d: FloatArr,
    ) -> BoolArr:
        """
        Calculate the LOS states based on the 2D distance and heights.

        Parameters
        ----------
        d_2d : FloatArr
            2D distance array.
        heights : FloatArr
            Heights of the UEs.
        pos_2d : FloatArr
            2D positions of the UEs.

        Returns
        -------
        BoolArr
            Array of LOS states (True for LOS, False for NLOS).
        """

    @abstractmethod
    def get_los_probs(
        self,
        d_2d: FloatArr,
        heights: FloatArr,
        pos_2d: FloatArr,
    ) -> FloatArr:
        """
        Calculate the LOS probabilities based on the 2D distance and heights.

        Parameters
        ----------
        d_2d : FloatArr
            2D distance array.
        heights : FloatArr
            Heights of the UEs.
        pos_2d : FloatArr
            2D positions of the UEs.

        Returns
        -------
        FloatArr
            Array of LOS probabilities (values between 0 and 1).
        """
