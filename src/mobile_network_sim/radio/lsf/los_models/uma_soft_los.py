"""LOS/NLOS state model for 5G channel simulation."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from ...utils.consts import LIGHT_SPEED_MPS
from ...utils.types import BoolArr, FloatArr
from ..grf import GaussianRandomField
from .abstract_los import AbstractLOSModel
from .uma_los import UMaLOSModel

if TYPE_CHECKING:
    from ...geometry.mobility.abstract_boundary_handler import AbstractBoundaryHandler
    from ...geometry.roi.abstract_roi import AbstractROI


class UMaLOSModelSoft(GaussianRandomField, AbstractLOSModel):
    r"""Generate spatially correlated soft UMa LOS values.

    Implements the soft LOS-state mapping in equation (7.6-18) of
    ETSI TR 138 901 V18.0.0. The resulting value lies between zero (NLOS) and
    one (LOS).

    Notes
    -----
    For a spatially correlated standard Gaussian value :math:`G`, wavelength
    :math:`\lambda`, and distance-dependent LOS probability
    :math:`P_{\mathrm{LOS}}(d)`, the soft state is

    .. math::

       \mathrm{LOS}_{\mathrm{soft}} = \frac{1}{2} + \frac{1}{\pi}
       \arctan\left(\sqrt{\frac{20}{\lambda}}[G+F(d)]\right),

    with

    .. math::

       F(d)=\sqrt{2}\,\operatorname{erf}^{-1}
       \left(2P_{\mathrm{LOS}}(d)-1\right).

    Subsequent path-loss interpolation in :class:`LSFModel` is a large-scale
    approximation. It does not implement the complete complex LOS/NLOS channel
    matrix combination in equation (7.6-19).

    The latent Gaussian-map and Cholesky construction is consistent with
    Schwarz et al., equations (5)--(6). Their equations (3)--(4) subsequently
    apply a hard, two-sided LOS/NLOS threshold. That threshold is not used
    here; this class uses the continuous 3GPP mapping above.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clauses 7.6.3.1 and 7.6.3.3,
       Table 7.6.3.1-2 and equations (7.6-18)--(7.6-19), May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    .. [2] S. Schwarz, I. Safiulin, T. Philosof, and M. Rupp, "Gaussian
       Modeling of Spatially Correlated LOS/NLOS Maps for Mobile
       Communications," in *2016 IEEE 84th Vehicular Technology Conference
       (VTC-Fall)*, pp. 1--5, 2016, section II, equations (3)--(6).
       doi:10.1109/VTCFall.2016.7881026.
    """

    los_grf: np.ndarray | None = None

    def __init__(
        self,
        *,
        roi: "AbstractROI",
        boundary_handler: "AbstractBoundaryHandler",
        bs_pos_2d: FloatArr,
        fc_ghz: float,
        grf: GaussianRandomField | None = None,
        grid_spacing: float = 10.0,
        d_decorr: float = 50.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        map_corr_factor: float | None = None,
        progress_bar: bool = False,
        map_dtype: str | np.dtype = "float32",
        seed: int | None = None,
        resource_storage: str = "legacy",
        factor_dtype: str | np.dtype = "float64",
    ):
        """
        Initialize the soft UMa LOS model.

        Parameters
        ----------
        roi : AbstractROI
            Region of interest covered by the LOS random fields.
        boundary_handler : AbstractBoundaryHandler
            Boundary handler used for map sampling and wrapping.
        bs_pos_2d : FloatArr
            Base-station positions in meters with shape ``(n_bs, 2)``.
        fc_ghz : float
            Carrier frequency in GHz.
        grf : GaussianRandomField | None, optional
            Precomputed GRF object whose grid/correlation matrices should be reused.
        grid_spacing : float, optional
            LOS map grid spacing in meters.
        d_decorr : float, optional
            LOS map :math:`1/e` correlation distance in meters. The default is
            the UMa LOS/NLOS-state value from ETSI TR 138 901,
            Table 7.6.3.1-2.
        mu : float, optional
            Mean of the underlying Gaussian random field.
        sigma : float, optional
            Standard deviation of the underlying Gaussian random field.
        map_corr_factor : float | None, optional
            Fixed correlation factor between generated BS maps. If ``None``,
            BS-distance-based correlation is applied after map generation.
        progress_bar : bool, optional
            Whether to show progress bars during matrix creation.
        map_dtype : str | np.dtype, optional
            Floating dtype used to store generated maps.
        seed : int | None, optional
            Random seed for reproducibility.
        """
        if not 0.5 < fc_ghz < 100:
            raise ValueError("Carrier frequency must be in (0.5, 100) GHz.")

        self.bs_pos_2d = bs_pos_2d
        self._uma_los_model = UMaLOSModel(seed)

        if grf is not None:
            if grf.resource_storage == "shared_mmap":
                grf = GaussianRandomField.copy(grf, seed=seed)
            self.__dict__.update(grf.__dict__)  # type: ignore
        else:
            super().__init__(
                roi=roi,
                boundary_handler=boundary_handler,
                grid_spacing=grid_spacing,
                d_decorr=d_decorr,
                mu=mu,
                sigma=sigma,
                field_corr_factor=map_corr_factor,
                roi_extension_factor=1,
                progress_bar=progress_bar,
                map_dtype=map_dtype,
                seed=seed,
                resource_storage=resource_storage,
                factor_dtype=factor_dtype,
            )

        self.lambda_fc = LIGHT_SPEED_MPS / (fc_ghz * 1e9)

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
        super().set_seed(seed)
        self._uma_los_model.set_seed(seed)

    def generate_los_maps(self) -> None:
        """Generate correlated Gaussian fields used by the soft LOS model."""
        fields = self.generate_fields(n_fields=self.bs_pos_2d.shape[0], normalize=True)

        if self.field_corr_factor is None:  # BS-level distance-based correlation
            bs_pos = self.bs_pos_2d
            bs_distance_mat = self._compute_pairwise_distance_matrix(bs_pos)
            bs_cholesky_mat = self._compute_cholesky_matrix(bs_distance_mat)
            fields_ = fields.reshape(fields.shape[0], -1)
            los_grf = (bs_cholesky_mat @ fields_).reshape(fields.shape)
        else:  # Map-level correlation
            los_grf = fields

        self.los_grf = los_grf

    def get_los_states(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr
    ) -> BoolArr:
        """Reject discrete LOS-state requests for this soft model.

        Parameters
        ----------
        d_2d : FloatArr
            Two-dimensional link distances in meters.
        heights : FloatArr
            UE heights in meters.
        pos_2d : FloatArr
            UE positions in meters.

        Returns
        -------
        BoolArr
            This method never returns normally.

        Raises
        ------
        NotImplementedError
            Always; use :meth:`get_los_probs` to obtain soft LOS values.
        """
        raise NotImplementedError("Use get_los_probs method instead.")

    def get_los_probs(
        self, d_2d: FloatArr, heights: FloatArr, pos_2d: FloatArr, *, interpolation=None
    ) -> FloatArr:
        """
        Calculate spatially correlated soft LOS probabilities.

        Parameters
        ----------
        d_2d : FloatArr
            2D link distances in meters with shape ``(n_ue, n_bs)``.
        heights : FloatArr
            UE heights in meters.
        pos_2d : FloatArr
            UE 2D positions in meters with shape ``(n_ue, 2)``.

        Returns
        -------
        FloatArr
            Soft LOS probabilities in ``[0, 1]`` with shape ``d_2d.shape``.
        """
        if self.los_grf is None:
            raise ValueError("LOS maps have not been generated.")
        g_rv = (
            self.get_grf_values(self.los_grf, pos_2d)
            if interpolation is None
            else self.get_grf_values(self.los_grf, pos_2d, interpolation=interpolation)
        ).T

        p_los = self._uma_los_model.get_los_probs(d_2d, heights, pos_2d)
        f_d = np.sqrt(2) * erfinv(2 * p_los - 1)

        los_soft_probs = 1 / 2 + 1 / np.pi * np.arctan(
            np.sqrt(20 / self.lambda_fc) * (g_rv + f_d)
        )

        return los_soft_probs
