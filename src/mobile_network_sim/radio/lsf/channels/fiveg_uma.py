"""Urban Macro (UMa) Channel Model."""

import warnings
from functools import lru_cache

import numpy as np

from mobile_network_sim.seeding import (
    COMPONENT_CHANNEL_RNG_1,
    COMPONENT_CHANNEL_RNG_2,
    derive_seed,
)

from ...utils.types import BoolArr, FloatArr
from .abstract_channel_model import AbstractChannelModel


@lru_cache(maxsize=64)
def _frequency_term(fc_ghz: float) -> float:
    """Cache only a frequency-dependent scalar, never stochastic state."""
    return float(20.0 * np.log10(fc_ghz))


class UrbanMacro(AbstractChannelModel):
    r"""Calculate Urban Macro LOS and NLOS path loss.

    The LOS and NLOS path-loss expressions follow Table 7.4.1-1 of
    ETSI TR 138 901 V18.0.0. The accepted simulator inputs are carrier
    frequencies from 0.5 to 100 GHz, two-dimensional distances from 10 to
    5000 m, BS heights from 10 to 150 m, and UE heights from 1.5 to 22.5 m.

    Notes
    -----
    Before the breakpoint, LOS path loss is

    .. math::

       PL_{\mathrm{LOS},1} = 28.0 + 22\log_{10}(d_{3D})
           + 20\log_{10}(f_c).

    After the breakpoint, it is

    .. math::

       PL_{\mathrm{LOS},2} = 28.0 + 40\log_{10}(d_{3D})
           + 20\log_{10}(f_c)
           - 9\log_{10}\left((d'_{BP})^2 + (h_{BS}-h_{UT})^2\right).

    The NLOS result is the maximum of the LOS result and

    .. math::

       PL'_{\mathrm{NLOS}} = 13.54 + 39.08\log_{10}(d_{3D})
           + 20\log_{10}(f_c) - 0.6(h_{UT}-1.5).

    Distances are in metres, heights are in metres, and :math:`f_c` is in GHz.
    Effective environment height is sampled using the simulator's fixed
    candidate set ``{12, 15, 18, 21}`` m when the non-unit branch is selected.
    Consequently, this class is not a complete implementation of every
    applicability and height-sampling condition in Table 7.4.1-1.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.4.1, Table 7.4.1-1 and notes 1--3,
       May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """

    FC_GHZ_MIN = 0.5
    FC_GHZ_MAX = 100.0

    D_MIN = 10.0
    D_MAX = 5000.0

    H_BS_MIN = 10.0
    H_BS_MAX = 150.0
    H_UE_MIN = 1.5
    H_UE_MAX = 22.5

    SHADOWING_STD_DEV_LOS = 4.0
    SHADOWING_STD_DEV_NLOS = 6.0
    SHADOWING_STD_DEV_NLOS_OPT = 7.8
    PROPAGATION_SPEED_MPS = 3.0e8

    rng1: np.random.Generator
    rng2: np.random.Generator

    def __init__(
        self,
        fc_ghz: float,
        seed: int | None = None,
        clip: bool = False,
        warn: bool = False,
    ) -> None:
        """
        Initialize the Urban Macro channel model.

        Parameters
        ----------
        fc_ghz : float
            Carrier frequency in GHz.
        seed : int | None, optional
            Random seed used for stochastic effective environment height sampling.
        clip : bool, optional
            If True, clip 2D distances to the model applicability range.
        warn : bool, optional
            If True, warn when distances are outside the applicability range.
        """
        super().__init__(fc_ghz=fc_ghz, seed=seed)

        # Check parameters
        if not self.FC_GHZ_MIN <= fc_ghz <= self.FC_GHZ_MAX:
            raise ValueError(
                f"fc_ghz must be in ({self.FC_GHZ_MIN}, {self.FC_GHZ_MAX})."
            )

        # Speed of light
        self.clip = clip
        self.warn = warn

        # Set the random seed
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the random seed for the channel model.

        Parameters
        ----------
        seed : int | None
            Seed for the model's random-number generators.
        """
        super().set_seed(seed)
        self.rng1 = np.random.default_rng(
            derive_seed(self.seed, COMPONENT_CHANNEL_RNG_1)
        )
        self.rng2 = np.random.default_rng(
            derive_seed(self.seed, COMPONENT_CHANNEL_RNG_2)
        )

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
        Calculate UMa path loss in dB for LOS/NLOS links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters with shape ``(n_ue, n_bs)``.
        d_3d : FloatArr
            3D link distances in meters with shape ``(n_ue, n_bs)``.
        is_los : BoolArr
            LOS mask with shape ``(n_ue, n_bs)``.
        h_bs : float | FloatArr
            BS antenna height(s) in meters; scalar, ``(n_bs,)``, or ``(1, n_bs)``.
        h_ue : float | FloatArr
            UE antenna height(s) in meters; scalar, ``(n_ue,)``, or ``(n_ue, 1)``.

        Returns
        -------
        FloatArr
            Path loss in dB with shape ``(n_ue, n_bs)``.
        """
        d_2d, d_3d, h_ue, h_bs = self._validate_inputs(
            d_2d=d_2d,
            d_3d=d_3d,
            h_ue=h_ue,
            h_bs=h_bs,
        )
        if is_los.shape != d_2d.shape:
            raise ValueError(
                f"is_los shape must match d_2d shape {d_2d.shape}, got {is_los.shape}."
            )

        pl_los = self._get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
        pl_nlos = np.maximum(pl_los, self._get_pl_nlos_formula(d_3d=d_3d, h_ue=h_ue))

        return np.where(is_los, pl_los, pl_nlos)

    def get_pl_los(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """Calculate UMa LOS path loss.

        Parameters
        ----------
        d_2d : FloatArr
            Two-dimensional link distances in meters.
        d_3d : FloatArr
            Three-dimensional link distances in meters.
        h_bs : float | FloatArr
            BS antenna height or heights in meters.
        h_ue : float | FloatArr
            UE antenna height or heights in meters.

        Returns
        -------
        FloatArr
            LOS path loss in dB.
        """
        d_2d, d_3d, h_ue, h_bs = self._validate_inputs(
            d_2d=d_2d,
            d_3d=d_3d,
            h_ue=h_ue,
            h_bs=h_bs,
        )
        return self._get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)

    def get_pl_nlos(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """
        Calculate NLOS path loss in dB.

        The 3GPP UMa NLOS path loss is the maximum of the LOS path loss and
        the NLOS formula.

        Parameters
        ----------
        d_2d : FloatArr
            Two-dimensional link distances in meters.
        d_3d : FloatArr
            Three-dimensional link distances in meters.
        h_bs : float | FloatArr
            BS antenna height or heights in meters.
        h_ue : float | FloatArr
            UE antenna height or heights in meters.

        Returns
        -------
        FloatArr
            NLOS path loss in dB.
        """
        d_2d, d_3d, h_ue, h_bs = self._validate_inputs(
            d_2d=d_2d,
            d_3d=d_3d,
            h_ue=h_ue,
            h_bs=h_bs,
        )
        pl_los = self._get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
        pl_nlos_formula = self._get_pl_nlos_formula(d_3d=d_3d, h_ue=h_ue)
        return np.maximum(pl_los, pl_nlos_formula)

    def get_pl_los_nlos(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        """Share deterministic work but preserve both environment-height draws.

        The NLOS lower bound historically uses a *second* sampled LOS path.
        This matters for elevated UEs; using the first draw changes the model.
        """
        d_2d, d_3d, h_ue, h_bs = self._validate_inputs(
            d_2d=d_2d,
            d_3d=d_3d,
            h_bs=h_bs,
            h_ue=h_ue,
        )
        first_height = self._get_effective_env_height(d_2d=d_2d, h_ue=h_ue)
        second_height = self._get_effective_env_height(d_2d=d_2d, h_ue=h_ue)
        log_distance = np.log10(d_3d)
        frequency = _frequency_term(self.fc_ghz)
        before = 28.0 + 22.0 * log_distance + frequency
        after_base = 28.0 + 40.0 * log_distance + frequency
        height_difference_squared = (h_bs - h_ue) ** 2

        def los(height: FloatArr) -> FloatArr:
            breakpoint = self._get_breakpoint_distance(h_bs=h_bs, h_ue=h_ue, h_e=height)
            return np.where(
                d_2d < breakpoint,
                before,
                after_base - 9.0 * np.log10(breakpoint**2 + height_difference_squared),
            )

        first = los(first_height)
        # Below 13 m both draws are deterministically one, but still performed.
        second = first if np.all(h_ue < 13) else los(second_height)
        nlos = 13.54 + 39.08 * log_distance + frequency - 0.6 * (h_ue - 1.5)
        return first, np.maximum(second, nlos)

    def _validate_inputs(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        if d_2d.ndim != 2:
            raise ValueError(f"d_2d must have shape (n_ue, n_bs), got {d_2d.shape}.")
        if d_3d.shape != d_2d.shape:
            raise ValueError(
                f"d_3d shape must match d_2d shape {d_2d.shape}, got {d_3d.shape}."
            )
        h_ue = self._ensure_correct_format_of_height(h_ue, (d_2d.shape[0], 1))
        h_bs = self._ensure_correct_format_of_height(h_bs, (1, d_2d.shape[1]))

        d_2d = self._ensure_distances_within_range(d_2d)
        self._validate_heights(h_bs=h_bs, h_ue=h_ue)

        return d_2d, d_3d, h_ue, h_bs

    def _ensure_distances_within_range(self, d_2d: FloatArr) -> FloatArr:
        """Ensure that the 2D distances are within the specified range."""
        if np.any((d_2d < self.D_MIN) | (d_2d > self.D_MAX)):
            if self.warn:
                warnings.warn(
                    f"UE distances must be {self.D_MIN} m < d < {self.D_MAX} m. "
                    "Some distances may be outside this range."
                    f"Distances are {'' if self.clip else 'not '}clipped."
                )
            if self.clip:
                return np.clip(d_2d, self.D_MIN, self.D_MAX)
        return d_2d

    def _validate_heights(self, h_bs: FloatArr, h_ue: FloatArr) -> None:
        """Ensure that the UE and BS antenna heights are within the specified range."""
        if np.any((h_bs < self.H_BS_MIN) | (h_bs > self.H_BS_MAX)):
            raise ValueError(
                f"BS antenna heights must be in [{self.H_BS_MIN}, {self.H_BS_MAX}] m."
                f" Actual antenna height(s) is {h_bs} m."
            )
        if np.any((h_ue < self.H_UE_MIN) | (h_ue > self.H_UE_MAX)):
            raise ValueError(
                f"UE antenna heights must be {self.H_UE_MIN} m < h < {self.H_UE_MAX} m."
                f" Actual antenna height(s) is {h_ue} m."
            )

    def _ensure_correct_format_of_height(
        self, height: float | FloatArr, shape: tuple[int, int]
    ) -> FloatArr:
        """Ensure that the input is a numpy array."""
        if isinstance(height, (int, float)):
            return np.full(shape, height)
        if height.ndim == 1 and height.shape[0] in shape:
            return height.reshape(shape)
        if height.ndim == 2 and height.shape == shape:
            return height
        raise ValueError(
            f"Height must be a scalar or an array of shape {shape}. Actual shape is {height.shape}."
        )

    def _get_pl_los(
        self, *, d_2d: FloatArr, d_3d: FloatArr, h_bs: FloatArr, h_ue: FloatArr
    ) -> FloatArr:
        """
        Calculate the path loss for line-of-sight (LOS) for multiple UEs and BSs.
        h_ue: (N_ue,)
        d_2d: (N_ue, N_bs)
        d_3d: (N_ue, N_bs)
        Returns:
        - LOS path loss matrix of shape (N_ue, N_bs)
        """
        # Get effective environment height and breakpoint distance
        h_env_eff = self._get_effective_env_height(d_2d=d_2d, h_ue=h_ue)
        d_break = self._get_breakpoint_distance(h_bs=h_bs, h_ue=h_ue, h_e=h_env_eff)

        # Calculate LOS path loss for each UE-BS pair
        return np.where(
            d_2d < d_break,
            self._pl_los_1(d_3d),
            self._pl_los_2(d_3d=d_3d, d_break=d_break, h_bs=h_bs, h_ue=h_ue),
        )

    def _pl_los_1(self, d_3d: FloatArr) -> FloatArr:
        """Calculate the LOS path loss before the breakpoint distance for multiple UEs and BSs."""
        return 28.0 + 22.0 * np.log10(d_3d) + _frequency_term(self.fc_ghz)

    def _pl_los_2(
        self,
        *,
        d_3d: FloatArr,
        d_break: FloatArr,
        h_bs: FloatArr,
        h_ue: FloatArr,
    ) -> FloatArr:
        """Calculate the LOS path loss after the breakpoint distance for multiple UEs and BSs."""
        return (
            28.0
            + 40.0 * np.log10(d_3d)
            + _frequency_term(self.fc_ghz)
            - 9.0 * np.log10(d_break**2 + (h_bs - h_ue) ** 2)
        )

    def _get_pl_nlos_formula(self, *, d_3d: FloatArr, h_ue: FloatArr) -> FloatArr:
        """Calculate the raw UMa NLOS path-loss formula in dB."""
        return (
            13.54
            + 39.08 * np.log10(d_3d)
            + _frequency_term(self.fc_ghz)
            - 0.6 * (h_ue - 1.5)
        )

    def _get_breakpoint_distance(
        self, *, h_bs: FloatArr, h_ue: FloatArr, h_e: FloatArr
    ) -> FloatArr:
        """
        Calculate the breakpoint distance for multiple UEs and BSs.
        h_e: (N_ue, N_bs)
        h_ue: (N_ue,)
        Returns:
        - breakpoint distance matrix of shape (N_ue, N_bs)
        """
        return 4 * (h_bs - h_e) * (h_ue - h_e) * self.fc_hz / self.PROPAGATION_SPEED_MPS

    def _get_effective_env_height(self, *, d_2d: FloatArr, h_ue: FloatArr) -> FloatArr:
        """
        Calculate the effective environment height for multiple UEs and BSs.
        h_ue: (N_ue,)
        d_2d: (N_ue, N_bs)
        Returns:
        - effective environment height matrix of shape (N_ue, N_bs)
        """
        if np.all(h_ue < 13):
            # Same call types, shapes and order as the general path, including
            # the empty choice. No RNG draw is removed by this specialization.
            self.rng1.binomial(1, np.ones_like(d_2d), size=d_2d.shape)
            self.rng2.choice([12, 15, 18, 21], size=(0,))
            return np.ones_like(d_2d)
        c = self._calculate_c(d_2d=d_2d, h_ue=h_ue)
        h_e = np.ones_like(d_2d)  # Initialize h_e to 1.0 for each UE-BS pair
        # Randomly choose h_e based on the probability from `c`
        binom_result = self.rng1.binomial(1, 1 / (1 + c), size=d_2d.shape)
        non_h_one_indices = np.where(binom_result == 0)
        h_e[non_h_one_indices] = self.rng2.choice(
            [12, 15, 18, 21], size=non_h_one_indices[0].shape
        )
        return h_e

    @staticmethod
    def _calculate_c(*, d_2d: FloatArr, h_ue: FloatArr) -> FloatArr:
        """
        Calculate the probability factor C for multiple UEs and BSs.
        d_2d: (N_ue, N_bs)
        h_ue: (N_ue,)
        Returns:
        - C matrix of shape (N_ue, N_bs)
        """
        g = np.zeros_like(d_2d)
        c = np.zeros_like(d_2d)

        valid_indices = (13 <= h_ue) & (h_ue <= 23) & (d_2d > 18)

        g[valid_indices] = (
            5
            / 4
            * (d_2d[valid_indices] / 100) ** 3
            * np.exp(-d_2d[valid_indices] / 150)
        )

        h_ue_mat = np.tile(h_ue, d_2d.shape[1])
        c[valid_indices] = ((h_ue_mat[valid_indices] - 13) / 10) ** 1.5 * g[
            valid_indices
        ]
        return c
