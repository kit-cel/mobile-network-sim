"""Rural Macro path-loss model from ETSI TR 138 901."""

import warnings

import numpy as np

from .abstract_channel_model import AbstractChannelModel
from ...utils.types import BoolArr, FloatArr


class RuralMacro(AbstractChannelModel):
    r"""Calculate Rural Macro LOS and NLOS path loss.

    Implements the outdoor RMa path-loss expressions and applicability ranges
    from ETSI TR 138 901 V18.0.0, Table 7.4.1-1.

    Notes
    -----
    Before the breakpoint, LOS path loss is

    .. math::

       PL_1 = 20\log_{10}\left(40\pi d_{3D}f_c/3\right)
          + \min(0.03h^{1.72},10)\log_{10}(d_{3D})
          - \min(0.044h^{1.72},14.77)
          + 0.002\log_{10}(h)d_{3D}.

    After the breakpoint, it is

    .. math::

       PL_2 = PL_1(d_{BP}) + 40\log_{10}(d_{3D}/d_{BP}),
       \qquad d_{BP}=\frac{2\pi h_{BS}h_{UT}f_c}{c}.

    The NLOS result is

    .. math::

       PL_{\mathrm{NLOS}} = \max(PL_{\mathrm{LOS}},
       PL'_{\mathrm{NLOS}}),

    where :math:`PL'_{\mathrm{NLOS}}` contains the building-height,
    street-width, BS-height, UE-height, distance, and frequency terms from
    Table 7.4.1-1. Distances are in metres and :math:`f_c` in GHz, except in
    the breakpoint expression where it is in Hz.

    LOS links are accepted from 10 m through 10 km and NLOS links from 10 m
    through 5 km. With ``clip=False``, values outside the applicable range are
    rejected. With ``clip=True``, horizontal distance is clipped to the
    state-specific range and three-dimensional distance is adjusted while
    preserving the vertical separation.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.4.1, Table 7.4.1-1 and notes 5--6,
       May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """

    FC_GHZ_MIN = 0.5
    FC_GHZ_MAX = 30.0

    D_MIN = 10.0
    D_LOS_MAX = 10_000.0
    D_NLOS_MAX = 5_000.0

    H_BS_MIN = 10.0
    H_BS_MAX = 150.0
    H_UE_MIN = 1.0
    H_UE_MAX = 10.0
    H_BUILD_MIN = 5.0
    H_BUILD_MAX = 50.0
    W_STREET_MIN = 5.0
    W_STREET_MAX = 50.0

    SHADOWING_STD_DEV_LOS_BEFORE_BP = 4.0
    SHADOWING_STD_DEV_LOS_AFTER_BP = 6.0
    SHADOWING_STD_DEV_NLOS = 8.0
    PROPAGATION_SPEED_MPS = 3.0e8

    def __init__(
        self,
        *,
        fc_ghz: float,
        street_width_m: float = 20.0,
        avg_building_height: float = 5.0,
        seed: int | None = None,
        clip: bool = False,
        warn: bool = False,
    ) -> None:
        """
        Initialize the RMa channel model.

        Parameters
        ----------
        fc_ghz : float
            Carrier frequency in GHz.
        street_width_m : float, optional
            Average street width ``W`` in meters.
        avg_building_height : float, optional
            Average building height ``h`` in meters.
        seed : int | None, optional
            Accepted for API consistency. The RMa path-loss model is deterministic.
        clip : bool, optional
            Clip distances to the LOS/NLOS applicability range. If ``False``,
            out-of-range distances raise :class:`ValueError`.
        warn : bool, optional
            Warn before clipping or rejecting out-of-range distances.
        """
        if not self.FC_GHZ_MIN < fc_ghz < self.FC_GHZ_MAX:
            raise ValueError(
                f"fc_ghz must be in ({self.FC_GHZ_MIN}, {self.FC_GHZ_MAX}) for RMa."
            )
        super().__init__(fc_ghz=fc_ghz, seed=seed)
        if not self.W_STREET_MIN <= street_width_m <= self.W_STREET_MAX:
            raise ValueError(
                f"street_width_m must be in [{self.W_STREET_MIN}, {self.W_STREET_MAX}] m."
            )
        if not self.H_BUILD_MIN <= avg_building_height <= self.H_BUILD_MAX:
            raise ValueError(
                f"avg_building_height must be in [{self.H_BUILD_MIN}, {self.H_BUILD_MAX}] m."
            )

        self.w_street = float(street_width_m)
        self.h_build = float(avg_building_height)
        self.clip = clip
        self.warn = warn

    def set_seed(self, seed: int | None = None) -> None:
        """Store a seed for API consistency.

        Parameters
        ----------
        seed : int | None, optional
            Seed to store. If ``None``, generate a seed.

        Notes
        -----
        The path-loss calculation itself contains no randomness.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        self.seed = seed

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
        Calculate RMa path loss in dB for LOS/NLOS links.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters with shape ``(n_ue, n_bs)``.
        d_3d : FloatArr
            3D link distances in meters with shape ``(n_ue, n_bs)``.
        is_los : BoolArr
            LOS mask with shape ``(n_ue, n_bs)``.
        h_bs : float | FloatArr
            BS antenna height(s) in meters.
        h_ue : float | FloatArr
            UE antenna height(s) in meters.

        Returns
        -------
        FloatArr
            Path loss in dB with shape ``(n_ue, n_bs)``.
        """
        if is_los.shape != d_2d.shape:
            raise ValueError(
                f"is_los shape must match d_2d shape {d_2d.shape}, got {is_los.shape}."
            )
        max_distance_m = np.where(is_los, self.D_LOS_MAX, self.D_NLOS_MAX)
        d_2d, d_3d, h_ue, h_bs = self._validate_inputs(
            d_2d=d_2d,
            d_3d=d_3d,
            h_ue=h_ue,
            h_bs=h_bs,
            max_distance_m=max_distance_m,
        )

        pl_los = self._get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
        pl_nlos = self._get_pl_nlos(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
        return np.where(is_los, pl_los, pl_nlos)

    def get_pl_los(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
    ) -> FloatArr:
        """Calculate RMa LOS path loss.

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
            max_distance_m=self.D_LOS_MAX,
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
        Calculate RMa NLOS path loss in dB.

        The 3GPP RMa NLOS path loss is the maximum of the RMa LOS path loss
        and the raw RMa NLOS formula.

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
            max_distance_m=self.D_NLOS_MAX,
        )
        return self._get_pl_nlos(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)

    def _validate_inputs(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: float | FloatArr,
        h_ue: float | FloatArr,
        max_distance_m: float | FloatArr,
    ) -> tuple[FloatArr, FloatArr, FloatArr, FloatArr]:
        if d_2d.ndim != 2:
            raise ValueError(f"d_2d must have shape (n_ue, n_bs), got {d_2d.shape}.")
        if d_3d.shape != d_2d.shape:
            raise ValueError(
                f"d_3d shape must match d_2d shape {d_2d.shape}, got {d_3d.shape}."
            )
        if np.any(d_3d < d_2d):
            raise ValueError("d_3d must be greater than or equal to d_2d.")

        h_ue = self._ensure_height_shape(h_ue, (d_2d.shape[0], 1))
        h_bs = self._ensure_height_shape(h_bs, (1, d_2d.shape[1]))

        d_2d, d_3d = self._ensure_distances_within_range(
            d_2d,
            d_3d,
            max_distance_m=max_distance_m,
        )
        self._validate_heights(h_bs=h_bs, h_ue=h_ue)

        return d_2d, d_3d, h_ue, h_bs

    def _ensure_distances_within_range(
        self,
        d_2d: FloatArr,
        d_3d: FloatArr,
        *,
        max_distance_m: float | FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        invalid = (d_2d < self.D_MIN) | (d_2d > max_distance_m)
        if np.any(invalid):
            if self.warn:
                warnings.warn(
                    "RMa 2D distances are outside their LOS/NLOS applicability "
                    f"range. Distances are {'' if self.clip else 'not '}clipped."
                )
            if not self.clip:
                raise ValueError(
                    "RMa 2D distances must be at least 10 m and no greater "
                    "than 10 km for LOS or 5 km for NLOS links."
                )

            vertical_distance_sq = np.maximum(d_3d**2 - d_2d**2, 0.0)
            clipped_d_2d = np.clip(d_2d, self.D_MIN, max_distance_m)
            clipped_d_3d = np.sqrt(clipped_d_2d**2 + vertical_distance_sq)
            return clipped_d_2d, clipped_d_3d
        return d_2d, d_3d

    def _validate_heights(self, h_bs: FloatArr, h_ue: FloatArr) -> None:
        if np.any((h_bs < self.H_BS_MIN) | (h_bs > self.H_BS_MAX)):
            raise ValueError(
                f"BS antenna heights must be in [{self.H_BS_MIN}, {self.H_BS_MAX}] m."
            )
        if np.any((h_ue < self.H_UE_MIN) | (h_ue > self.H_UE_MAX)):
            raise ValueError(
                f"UE antenna heights must be in [{self.H_UE_MIN}, {self.H_UE_MAX}] m."
            )

    @staticmethod
    def _ensure_height_shape(
        height: float | FloatArr,
        shape: tuple[int, int],
    ) -> FloatArr:
        if isinstance(height, (int, float)):
            return np.full(shape, height, dtype=float)
        height = np.asarray(height, dtype=float)
        if height.ndim == 1 and height.shape[0] in shape:
            return height.reshape(shape)
        if height.ndim == 2 and height.shape == shape:
            return height
        raise ValueError(
            f"Height must be a scalar or an array of shape {shape}. Actual shape is {height.shape}."
        )

    def _get_pl_los(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: FloatArr,
        h_ue: FloatArr,
    ) -> FloatArr:
        d_break = self._get_breakpoint_distance(h_bs=h_bs, h_ue=h_ue)
        pl_1 = self._pl_los_1(d_3d=d_3d)
        pl_2 = self._pl_los_2(d_3d=d_3d, d_break=d_break)
        return np.where(d_2d < d_break, pl_1, pl_2)

    def _get_pl_nlos(
        self,
        *,
        d_2d: FloatArr,
        d_3d: FloatArr,
        h_bs: FloatArr,
        h_ue: FloatArr,
    ) -> FloatArr:
        pl_los = self._get_pl_los(d_2d=d_2d, d_3d=d_3d, h_bs=h_bs, h_ue=h_ue)
        pl_nlos_formula = self._pl_nlos_formula(
            d_3d=d_3d,
            h_bs=h_bs,
            h_ue=h_ue,
        )
        return np.maximum(pl_los, pl_nlos_formula)

    def _pl_los_1(self, *, d_3d: FloatArr) -> FloatArr:
        return (
            20.0 * np.log10(40.0 * np.pi * d_3d * self.fc_ghz / 3.0)
            + np.minimum(0.03 * self.h_build**1.72, 10.0) * np.log10(d_3d)
            - np.minimum(0.044 * self.h_build**1.72, 14.77)
            + 0.002 * np.log10(self.h_build) * d_3d
        )

    def _pl_los_2(self, *, d_3d: FloatArr, d_break: FloatArr) -> FloatArr:
        pl_1_at_break = self._pl_los_1(d_3d=d_break)
        return pl_1_at_break + 40.0 * np.log10(d_3d / d_break)

    def _pl_nlos_formula(
        self,
        *,
        d_3d: FloatArr,
        h_bs: FloatArr,
        h_ue: FloatArr,
    ) -> FloatArr:
        return (
            161.04
            - 7.1 * np.log10(self.w_street)
            + 7.5 * np.log10(self.h_build)
            - (24.37 - 3.7 * (self.h_build / h_bs) ** 2) * np.log10(h_bs)
            + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3.0)
            + 20.0 * np.log10(self.fc_ghz)
            - (3.2 * (np.log10(11.75 * h_ue)) ** 2 - 4.97)
        )

    def _get_breakpoint_distance(self, *, h_bs: FloatArr, h_ue: FloatArr) -> FloatArr:
        """Calculate RMa breakpoint distance in meters."""
        return 2.0 * np.pi * h_bs * h_ue * self.fc_hz / self.PROPAGATION_SPEED_MPS
