"""Large scale fading (LSF) module for the network simulator."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from mobile_network_sim.seeding import (
    COMPONENT_LSF_CHANNEL,
    COMPONENT_LSF_LOS,
    COMPONENT_LSF_SHADOWING,
    derive_seed,
)

from ..lsf.channels import channel_models
from ..lsf.los_models import los_models
from ..lsf.los_models.const_los_nlos import ConstLOS, ConstNLOS
from ..lsf.los_models.rma_los import RMaLOSModel
from ..lsf.los_models.uma_los import UMaLOSModel
from ..lsf.los_models.uma_soft_los import UMaLOSModelSoft
from ..lsf.shadowing.sfm_cholesky import ShadowFadingCholesky
from ..utils import commuinations as utc
from ..utils.types import BoolArr, FloatArr, IntArr

if TYPE_CHECKING:
    from ..components.bs.bs_collection import BaseStationCollection
    from ..components.ue.ue_collection import UserEquipmentCollection
    from ..config.lsf_config import LargeScaleFadingConfig
    from ..config.rat_config import RATConfig
    from ..geometry.simulation_geometry import SimulationGeometry
    from ..lsf.channels.abstract_channel_model import AbstractChannelModel
    from ..lsf.los_models.abstract_los import AbstractLOSModel


@dataclass(frozen=True, slots=True)
class LSFMetrics:
    """Container for signal metrics."""

    pl_db: FloatArr
    cg_db: FloatArr
    rsrp_dbm: FloatArr
    sinr_db: FloatArr
    los: FloatArr
    sf_db: FloatArr

    def to_dict(self) -> dict[str, FloatArr | np.ma.MaskedArray]:
        """Convert the metrics to a dictionary.

        Returns
        -------
        dict[str, FloatArr | numpy.ma.MaskedArray]
            Mapping from metric names to their arrays.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, FloatArr | np.ma.MaskedArray]) -> LSFMetrics:
        """Create metrics from a dictionary.

        Parameters
        ----------
        data : dict[str, FloatArr | numpy.ma.MaskedArray]
            Mapping containing every metrics dataclass field.

        Returns
        -------
        LSFMetrics
            Reconstructed metrics.

        Raises
        ------
        ValueError
            If a required field is absent.
        """
        update_dict = {}
        for field in fields(cls):
            if field.name not in data:
                raise ValueError(f"Missing field '{field.name}' in data.")
            update_dict[field.name] = data[field.name]
        return cls(**update_dict)

    @classmethod
    def keys(cls) -> list[str]:
        """Return the metric field names.

        Returns
        -------
        list[str]
            Dataclass field names in declaration order.
        """
        return [f.name for f in fields(cls)]


class LSFModel:
    """Large scale fading (LSF) module for the network simulator."""

    SINR_EST_METHOD: str = (
        "mean_beam_gain"
        # "stochastic_beam_gain"
        # "scheduler_beam_gain"
    )

    SCS_KHZ: int = 120

    def __init__(
        self,
        config: LargeScaleFadingConfig,
        rat_config: RATConfig,
        geo: SimulationGeometry,
        bs_set: BaseStationCollection,
        ue_set: UserEquipmentCollection,
        internal_dtype: str | np.dtype = "float64",
        map_dtype: str | np.dtype = "float32",
        grf_dtype: str | np.dtype = "float64",
        *,
        realize_fields: bool = True,
        resource_storage: str = "legacy",
    ) -> None:
        self.config = config
        self.resource_storage = resource_storage
        self.seed = int(config.seed)
        self.rat_config = rat_config
        self.internal_dtype = np.dtype(internal_dtype)
        self.map_dtype = np.dtype(map_dtype)
        self.grf_dtype = np.dtype(grf_dtype)

        self.bs_set = bs_set

        self._prepare_noise_and_tx_power(ue_set)
        self.los_model = self._build_los_model(geo, realize_fields=realize_fields)
        self.sfm = self._build_shadowing_model(geo, realize_fields=realize_fields)
        self.channel_model = self._build_channel_model()

    def _prepare_noise_and_tx_power(self, ue_set: UserEquipmentCollection) -> None:
        """Pre-calculate noise and transmit-power terms."""
        self.n_rb = int(self.rat_config.bw_mhz * 1000 // (self.SCS_KHZ * 12))
        if self.n_rb <= 0:
            raise ValueError(
                f"Invalid number of RBs for bw_mhz={self.rat_config.bw_mhz} and "
                f"SCS_KHZ={self.SCS_KHZ}."
            )
        self.n_sc = 12 * self.n_rb
        sc_offset_db = float(10.0 * np.log10(self.n_sc))

        # Full-band noise powers

        self.p_noise_ue_dbm = self._as_internal(
            utc.noise_power_dbm(self.rat_config.bw_mhz, ue_set.nf_db_ues)
        )
        self.p_noise_bs_dbm = self._as_internal(
            utc.noise_power_dbm(self.rat_config.bw_mhz, self.bs_set.nf_db_cells)
        )

        self.p_noise_ue_w = self._as_internal(utc.dbm_to_w(self.p_noise_ue_dbm))[
            :, None
        ]
        self.p_noise_bs_w = self._as_internal(utc.dbm_to_w(self.p_noise_bs_dbm))[
            None, :
        ]

        # Per-subcarrier noise powers
        self.p_noise_sc_ue_dbm = self.p_noise_ue_dbm - sc_offset_db
        self.p_noise_sc_bs_dbm = self.p_noise_bs_dbm - sc_offset_db

        self.p_noise_sc_ue_w = self._as_internal(utc.dbm_to_w(self.p_noise_sc_ue_dbm))[
            :, None
        ]
        self.p_noise_sc_bs_w = self._as_internal(utc.dbm_to_w(self.p_noise_sc_bs_dbm))[
            None, :
        ]

        self.p_tx_sc_bs_dbm = self.bs_set.p_tx_dbm_cells - sc_offset_db

    def _build_los_model(
        self, geo: SimulationGeometry, *, realize_fields: bool
    ) -> AbstractLOSModel:
        """Build the LOS model based on the environment configuration."""
        seed = derive_seed(self.seed, COMPONENT_LSF_LOS)
        los_model: AbstractLOSModel
        if self.config.los_model == ConstLOS.NAME:
            los_model = ConstLOS()
        elif self.config.los_model == ConstNLOS.NAME:
            los_model = ConstNLOS()
        elif self.config.los_model == UMaLOSModel.NAME:
            los_model = UMaLOSModel(seed=seed)
        elif self.config.los_model == RMaLOSModel.NAME:
            los_model = RMaLOSModel(seed=seed)
        elif self.config.los_model == UMaLOSModelSoft.NAME:
            los_model = UMaLOSModelSoft(
                roi=geo.roi,
                boundary_handler=geo.boundary_handler,
                bs_pos_2d=geo.layout.bs_xy,
                fc_ghz=self.rat_config.fc_ghz,
                grid_spacing=self.config.soft_los_map_grid_spacing_m,
                resource_storage=self.resource_storage,
                d_decorr=self.config.soft_los_d_decorr_m,
                map_dtype=self.map_dtype,
                factor_dtype=self.grf_dtype,
                seed=seed,
            )
            if realize_fields:
                los_model.generate_los_maps()
        else:
            raise ValueError(
                f"Unsupported LOS model: {self.config.los_model}. "
                f"Supported models: {list(los_models.keys())}."
            )
        return los_model

    def _build_shadowing_model(
        self, geo: SimulationGeometry, *, realize_fields: bool
    ) -> ShadowFadingCholesky | None:
        """Build the shadowing model based on the environment configuration."""
        if self.config.shadowing_enabled:
            seed = derive_seed(self.seed, COMPONENT_LSF_SHADOWING)
            sfm = ShadowFadingCholesky(
                roi=geo.roi,
                boundary_handler=geo.boundary_handler,
                bs_pos_2d=geo.layout.bs_xy,
                grid_spacing=self.config.shadowing_map_grid_spacing_m,
                resource_storage=self.resource_storage,
                d_decorr=self.config.shadowing_d_decorr_m,
                sigma=1.0,
                map_dtype=self.map_dtype,
                factor_dtype=self.grf_dtype,
                seed=seed,
            )
            if realize_fields:
                sfm.generate_shadow_fading_maps()
        else:
            sfm = None
        return sfm

    def _build_channel_model(self) -> AbstractChannelModel:
        """Build the channel model based on the environment configuration."""
        cl = channel_models.get(self.config.channel_model, None)
        if cl is None:
            raise ValueError(
                f"Unsupported channel model: {self.config.channel_model}. "
                f"Supported models: {list(channel_models.keys())}."
            )
        return cl(
            fc_ghz=self.rat_config.fc_ghz,
            seed=derive_seed(self.seed, COMPONENT_LSF_CHANNEL),
        )

    def reset(self, seed: int, *, regenerate_fields: bool = True) -> None:
        """Reset every stochastic LSF component for a new episode.

        The seed denotes one radio-map realization. LOS, shadowing, and
        channel-internal randomness are derived onto independent sub-streams.
        Spatial fields remain constant during the episode. Explicitly setting
        ``regenerate_fields=False`` retains initialized LOS/shadowing maps while
        reseeding the channel and component RNGs for the new episode.
        """
        self.seed = int(seed)

        los_seed = derive_seed(self.seed, COMPONENT_LSF_LOS)
        self.los_model.set_seed(los_seed)
        if regenerate_fields and isinstance(self.los_model, UMaLOSModelSoft):
            self.los_model.generate_los_maps()

        if self.sfm is not None:
            self.sfm.set_seed(derive_seed(self.seed, COMPONENT_LSF_SHADOWING))
            if regenerate_fields:
                self.sfm.generate_shadow_fading_maps()

        self.channel_model.set_seed(derive_seed(self.seed, COMPONENT_LSF_CHANNEL))

    def _random_generators(self) -> list[np.random.Generator]:
        """Return every mutable RNG used while calculating LSF metrics."""
        generators: list[np.random.Generator] = []
        components = [self.los_model, self.sfm, self.channel_model]
        nested_los = getattr(self.los_model, "_uma_los_model", None)
        if nested_los is not None:
            components.append(nested_los)
        for component in components:
            if component is None:
                continue
            for name in ("rng", "np_rng", "rng1", "rng2"):
                generator = getattr(component, name, None)
                if isinstance(generator, np.random.Generator) and all(
                    generator is not existing for existing in generators
                ):
                    generators.append(generator)
        return generators

    @contextmanager
    def preserve_random_state(self) -> Iterator[None]:
        """Prevent diagnostic coverage-map creation from advancing live RNGs."""
        states = [
            (generator, deepcopy(generator.bit_generator.state))
            for generator in self._random_generators()
        ]
        try:
            yield
        finally:
            for generator, state in states:
                generator.bit_generator.state = state

    def calculate_lsf_metrics(
        self,
        pos_3d: FloatArr,
        d_2d: FloatArr,
        d_3d: FloatArr,
        phi_az_deg: FloatArr,
        theta_el_deg: FloatArr,
        candidate_d_2d: FloatArr | None = None,
        candidate_d_3d: FloatArr | None = None,
        candidate_phi_az_deg: FloatArr | None = None,
        candidate_theta_el_deg: FloatArr | None = None,
        beam_gains: tuple[FloatArr, FloatArr] | None = None,
    ) -> LSFMetrics:
        """
        Calculate path loss, coupling gain, RSRP-like received power, and SINR.

        Parameters
        ----------
        pos_3d : FloatArr
            UE positions with shape ``(..., 3)``.
        d_2d : FloatArr
            2D distances to the nearest BS image with shape ``(..., n_bs)``.
        d_3d : FloatArr
            3D distances to the nearest BS image with shape ``(..., n_bs)``.
        phi_az_deg : FloatArr
            Azimuth angles to the nearest BS image in degrees with shape
            ``(..., n_bs)``.
        theta_el_deg : FloatArr
            Zenith angles to the nearest BS image in degrees with shape
            ``(..., n_bs)``.
        candidate_d_2d : FloatArr | None, optional
            2D distances to all candidate BS images with shape
            ``(..., n_images, n_bs)``. Used only by
            ``wraparound_coupling_method='max_coupling_over_images'``.
        candidate_d_3d : FloatArr | None, optional
            3D distances to all candidate BS images.
        candidate_phi_az_deg : FloatArr | None, optional
            Azimuth angles to all candidate BS images in degrees.
        candidate_theta_el_deg : FloatArr | None, optional
            Zenith angles to all candidate BS images in degrees.

        Returns
        -------
        LSFMetrics
            Path loss, coupling gain, RSRP-like received power, SINR, LOS state,
            and shadow fading. All arrays have shape ``(..., n_cells)``.

        Notes
        -----
        With ``wraparound_coupling_method='nearest_image'``, all sectors of a
        base station use its geometrically closest periodic image. This is
        analogous to the geographic-distance method of Kini et al. With
        ``'max_coupling_over_images'``, each UE-cell link instead retains the
        image with the greatest coupling gain, following their radio-distance
        principle. The implementation generalizes the paper's fixed
        19-site/57-sector layout to the configured topology and candidate
        images.

        References
        ----------
        .. [1] A. V. Kini, M. Hosseinian, M. Lee, and J. Stern-Berkowitz,
           "Reevaluating Cell Wraparound Techniques for 3D Channel Model
           Based System-Level Simulations," in *2015 IEEE Wireless
           Communications and Networking Conference (WCNC)*, pp. 171--176,
           2015, sections I and III. doi:10.1109/WCNC.2015.7127464.
        """
        if self.config.wraparound_coupling_method == "max_coupling_over_images":
            return self._calculate_lsf_metrics_max_coupling_over_images(
                pos_3d=pos_3d,
                fallback_d_2d=d_2d,
                fallback_d_3d=d_3d,
                fallback_phi_az_deg=phi_az_deg,
                fallback_theta_el_deg=theta_el_deg,
                candidate_d_2d=candidate_d_2d,
                candidate_d_3d=candidate_d_3d,
                candidate_phi_az_deg=candidate_phi_az_deg,
                candidate_theta_el_deg=candidate_theta_el_deg,
                beam_gains=beam_gains,
            )

        interpolation = None
        if (
            isinstance(self.los_model, UMaLOSModelSoft)
            and self.sfm is not None
            and self.los_model.boundary_handler is self.sfm.boundary_handler
            and self.los_model.grid.interpolation_key == self.sfm.grid.interpolation_key
        ):
            interpolation = self.los_model.grid.prepare_interpolation(
                pos_3d[..., :2], self.los_model.boundary_handler
            )
        pl_db, los = self._calculate_path_loss(
            pos_3d, d_2d, d_3d, interpolation=interpolation
        )

        sf_db = self._calculate_shadow_fading_db(
            pos_2d=pos_3d[..., :2],
            los=los,
            target_shape=pl_db.shape,
            interpolation=interpolation,
        )

        g_bs_db_max, g_bs_db_mean = (
            self.bs_set.get_max_mean_ant_gains(phi_az_deg, theta_el_deg)
            if beam_gains is None
            else beam_gains
        )
        g_ue_db = np.zeros_like(g_bs_db_max, dtype=self.internal_dtype)

        cg_db = self._get_coupling_gain_db(g_ue_db, g_bs_db_max, pl_db, sf_db)
        rsrp_dbm = self._calculate_rsrp(cg_db)
        sinr_db = self._calculate_sinr(
            phi_az_deg, theta_el_deg, rsrp_dbm, pl_db, sf_db, g_bs_db_mean
        )

        return LSFMetrics(
            pl_db=pl_db,
            cg_db=cg_db,
            rsrp_dbm=rsrp_dbm,
            sinr_db=sinr_db,
            los=los,
            sf_db=sf_db,
        )

    def _calculate_shadow_fading_db(
        self,
        *,
        pos_2d: FloatArr,
        los: FloatArr,
        target_shape: tuple[int, ...],
        interpolation=None,
    ) -> FloatArr:
        """
        Calculate final shadow fading in dB from unit latent shadowing samples.

        Parameters
        ----------
        pos_2d : FloatArr
            UE positions where latent shadowing is sampled, shape ``(..., 2)``.
        los : FloatArr
            LOS state/probability per cell, shape ``target_shape``.
        target_shape : tuple[int, ...]
            Expected output shape, usually ``(..., n_cells)``.

        Returns
        -------
        FloatArr
            Shadow fading in dB with shape ``target_shape``.
        """
        if los.shape != target_shape:
            raise ValueError(
                f"los shape must match target_shape {target_shape}, got {los.shape}."
            )
        return self._calculate_shadow_fading_db_for_los(
            pos_2d=pos_2d, los=los, interpolation=interpolation
        )

    def _calculate_shadow_fading_db_for_los(
        self,
        *,
        pos_2d: FloatArr,
        los: FloatArr,
        interpolation=None,
    ) -> FloatArr:
        """
        Calculate shadow fading in dB for LOS values with optional image axis.

        ``los`` must have shape ``sample_shape + (n_cells,)`` or
        ``sample_shape + (n_images, n_cells)``, where ``sample_shape`` is
        ``pos_2d.shape[:-1]``.
        """
        if self.sfm is None:
            return np.zeros(los.shape, dtype=self.internal_dtype)

        sample_shape = pos_2d.shape[:-1]
        if los.shape[: len(sample_shape)] != sample_shape:
            raise ValueError(
                f"los leading shape must match positions {sample_shape}, got {los.shape}."
            )
        samples = (
            self.sfm.get_sf_values(pos_2d)
            if interpolation is None
            else self.sfm.get_sf_values(pos_2d, interpolation=interpolation)
        )
        z_sf = self.map_bs_values_to_cells(samples)
        if z_sf.shape != (*sample_shape, los.shape[-1]):
            raise ValueError(
                "Latent shadowing shape must match positions and cells: "
                f"expected {(*sample_shape, los.shape[-1])}, got {z_sf.shape}."
            )
        while z_sf.ndim < los.ndim:
            z_sf = z_sf[..., None, :]
        sigma_eff_db = np.sqrt(
            los * self.config.shadowing_std_los_db**2
            + (1.0 - los) * self.config.shadowing_std_nlos_db**2
        )
        return self._as_internal(sigma_eff_db * z_sf)

    def _calculate_path_loss(
        self, pos_3d: FloatArr, d_2d: FloatArr, d_3d: FloatArr, *, interpolation=None
    ) -> tuple[FloatArr, FloatArr]:
        """
        Calculate the path loss for all UEs and cells.

        Parameters
        ----------
        pos_3d : FloatArr
            3D positions with shape (..., 3).
        d_2d : FloatArr
            2D distances with shape (..., n_bs).
        d_3d : FloatArr
            3D distances with shape (..., n_bs).

        Returns
        -------
        tuple[FloatArr, FloatArr]
            Path loss in dB and LOS states with shape (..., n_cells).
        """
        if pos_3d.shape[-1] != 3:
            raise ValueError(f"Position shape must end with 3, got {pos_3d.shape}.")
        if pos_3d.shape[:-1] != d_2d.shape[:-1] or pos_3d.shape[:-1] != d_3d.shape[:-1]:
            raise ValueError(
                f"Shape mismatch: pos_3d={pos_3d.shape}, d_2d={d_2d.shape}, d_3d={d_3d.shape}"
            )
        if d_2d.shape[-1] != self.bs_set.n_bs or d_3d.shape[-1] != self.bs_set.n_bs:
            raise ValueError(
                f"Distance shapes must end with n_bs={self.bs_set.n_bs}, "
                f"got d_2d={d_2d.shape}, d_3d={d_3d.shape}."
            )

        ue_heights = pos_3d[..., 2]
        pos_2d = pos_3d[..., :2]

        los_values: BoolArr | FloatArr
        if isinstance(self.los_model, (ConstLOS, ConstNLOS, UMaLOSModel, RMaLOSModel)):
            los_values = self.los_model.get_los_states(d_2d, ue_heights, pos_2d)
            pl_db = self.channel_model.get_pl(
                d_2d=d_2d,
                d_3d=d_3d,
                is_los=los_values,
                h_bs=self.bs_set.bs_heights,
                h_ue=ue_heights,
            )
        elif isinstance(self.los_model, UMaLOSModelSoft):
            los_values = self.los_model.get_los_probs(
                d_2d, ue_heights, pos_2d, interpolation=interpolation
            )
            pl_los, pl_nlos = self.channel_model.get_pl_los_nlos(
                d_2d=d_2d,
                d_3d=d_3d,
                h_bs=self.bs_set.bs_heights,
                h_ue=ue_heights,
            )
            pl_db = np.asarray(
                utc.lin_to_db(
                    utc.db_to_lin(pl_los) * los_values**2
                    + utc.db_to_lin(pl_nlos) * (1.0 - los_values**2)
                ),
                dtype=self.internal_dtype,
            )
            # Large-scale approximation of soft LOS/NLOS mixing. Equation
            # (7.6-19) combines full complex channel matrices, which this
            # large-scale model does not generate.
        else:
            raise ValueError("Invalid LOS model type.")  # noqa: TRY004 - public contract

        pl = self.map_bs_values_to_cells(self._as_internal(pl_db))
        los = self.map_bs_values_to_cells(self._as_internal(los_values))
        return pl, los

    def _calculate_lsf_metrics_max_coupling_over_images(
        self,
        *,
        pos_3d: FloatArr,
        fallback_d_2d: FloatArr,
        fallback_d_3d: FloatArr,
        fallback_phi_az_deg: FloatArr,
        fallback_theta_el_deg: FloatArr,
        candidate_d_2d: FloatArr | None,
        candidate_d_3d: FloatArr | None,
        candidate_phi_az_deg: FloatArr | None,
        candidate_theta_el_deg: FloatArr | None,
        beam_gains: tuple[FloatArr, FloatArr] | None = None,
    ) -> LSFMetrics:
        """
        Calculate LSF metrics by selecting the strongest wrap-around image.

        Each candidate image is evaluated with its own distance, path loss,
        LOS state/probability, and antenna angle. The final metrics are gathered
        from the candidate image that maximizes coupling gain for each UE-cell
        link.
        """
        if candidate_d_2d is None:
            candidate_d_2d = fallback_d_2d[..., None, :]
        if candidate_d_3d is None:
            candidate_d_3d = fallback_d_3d[..., None, :]
        if candidate_phi_az_deg is None:
            candidate_phi_az_deg = fallback_phi_az_deg[..., None, :]
        if candidate_theta_el_deg is None:
            candidate_theta_el_deg = fallback_theta_el_deg[..., None, :]

        self._validate_candidate_geometry(
            pos_3d=pos_3d,
            candidate_d_2d=candidate_d_2d,
            candidate_d_3d=candidate_d_3d,
            candidate_phi_az_deg=candidate_phi_az_deg,
            candidate_theta_el_deg=candidate_theta_el_deg,
        )

        pl_img_bs, los_img_bs = self._calculate_candidate_path_loss(
            pos_3d=pos_3d,
            candidate_d_2d=candidate_d_2d,
            candidate_d_3d=candidate_d_3d,
        )
        pl_img_db = self.map_bs_values_to_cells(self._as_internal(pl_img_bs))
        los_img = self.map_bs_values_to_cells(self._as_internal(los_img_bs))
        sf_img_db = self._calculate_shadow_fading_db_for_los(
            pos_2d=pos_3d[..., :2],
            los=los_img,
        )

        g_bs_img_db_max, g_bs_img_db_mean = (
            self.bs_set.get_max_mean_ant_gains(
                candidate_phi_az_deg, candidate_theta_el_deg
            )
            if beam_gains is None
            else beam_gains
        )
        g_ue_img_db = np.zeros_like(g_bs_img_db_max, dtype=self.internal_dtype)
        cg_img_db = self._get_coupling_gain_db(
            g_ue_img_db,
            g_bs_img_db_max,
            pl_img_db,
            sf_img_db,
        )

        best_image_idx = np.argmax(cg_img_db, axis=-2)
        cg_db = self._gather_candidate_image(cg_img_db, best_image_idx)
        pl_db = self._gather_candidate_image(pl_img_db, best_image_idx)
        los = self._gather_candidate_image(los_img, best_image_idx)
        sf_db = self._gather_candidate_image(sf_img_db, best_image_idx)

        rsrp_dbm = self._calculate_rsrp(cg_db)
        sinr_db = self._calculate_sinr_from_selected_images(
            p_rx_dbm=rsrp_dbm,
            pl_img_db=pl_img_db,
            sf_img_db=sf_img_db,
            phi_az_img_deg=candidate_phi_az_deg,
            theta_el_img_deg=candidate_theta_el_deg,
            best_image_idx=best_image_idx,
            g_bs_img_db_mean=g_bs_img_db_mean,
        )

        return LSFMetrics(
            pl_db=pl_db,
            cg_db=cg_db,
            rsrp_dbm=rsrp_dbm,
            sinr_db=sinr_db,
            los=los,
            sf_db=sf_db,
        )

    def _validate_candidate_geometry(
        self,
        *,
        pos_3d: FloatArr,
        candidate_d_2d: FloatArr,
        candidate_d_3d: FloatArr,
        candidate_phi_az_deg: FloatArr,
        candidate_theta_el_deg: FloatArr,
    ) -> None:
        """Validate candidate-image geometry arrays."""
        if pos_3d.shape[-1] != 3:
            raise ValueError(f"Position shape must end with 3, got {pos_3d.shape}.")
        if candidate_d_2d.ndim < 3:
            raise ValueError(
                "candidate_d_2d must have shape (..., n_images, n_bs), "
                f"got {candidate_d_2d.shape}."
            )
        expected_shape = (
            *pos_3d.shape[:-1],
            candidate_d_2d.shape[-2],
            self.bs_set.n_bs,
        )
        for name, values in {
            "candidate_d_2d": candidate_d_2d,
            "candidate_d_3d": candidate_d_3d,
            "candidate_phi_az_deg": candidate_phi_az_deg,
            "candidate_theta_el_deg": candidate_theta_el_deg,
        }.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {values.shape}."
                )

    def _calculate_candidate_path_loss(
        self,
        *,
        pos_3d: FloatArr,
        candidate_d_2d: FloatArr,
        candidate_d_3d: FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        """
        Calculate BS-level path loss and LOS values for candidate images.

        The channel models operate on 2D matrices of shape ``(n_samples, n_bs)``.
        Candidate images are flattened into the sample axis and reshaped back
        after path-loss calculation.
        """
        sample_shape = pos_3d.shape[:-1]
        candidate_shape = candidate_d_2d.shape
        n_images = candidate_shape[-2]
        n_bs = candidate_shape[-1]
        flat_d_2d = candidate_d_2d.reshape(-1, n_bs)
        flat_d_3d = candidate_d_3d.reshape(-1, n_bs)

        ue_heights = np.broadcast_to(
            pos_3d[..., 2][..., None],
            (*sample_shape, n_images),
        ).reshape(-1)
        pos_2d = np.broadcast_to(
            pos_3d[..., :2][..., None, :],
            (*sample_shape, n_images, 2),
        ).reshape(-1, 2)

        flat_pl, flat_los = self._calculate_path_loss_values_by_bs(
            pos_2d=pos_2d,
            ue_heights=ue_heights,
            d_2d=flat_d_2d,
            d_3d=flat_d_3d,
        )
        return (
            flat_pl.reshape(candidate_shape),
            flat_los.reshape(candidate_shape),
        )

    def _calculate_path_loss_values_by_bs(
        self,
        *,
        pos_2d: FloatArr,
        ue_heights: FloatArr,
        d_2d: FloatArr,
        d_3d: FloatArr,
    ) -> tuple[FloatArr, FloatArr]:
        """Calculate BS-level path loss and LOS values for 2D distance matrices."""
        los_values: BoolArr | FloatArr
        if isinstance(self.los_model, (ConstLOS, ConstNLOS, UMaLOSModel, RMaLOSModel)):
            los_values = self.los_model.get_los_states(d_2d, ue_heights, pos_2d)
            pl_db = self.channel_model.get_pl(
                d_2d=d_2d,
                d_3d=d_3d,
                is_los=los_values,
                h_bs=self.bs_set.bs_heights,
                h_ue=ue_heights,
            )
        elif isinstance(self.los_model, UMaLOSModelSoft):
            los_values = self.los_model.get_los_probs(d_2d, ue_heights, pos_2d)
            pl_los, pl_nlos = self.channel_model.get_pl_los_nlos(
                d_2d=d_2d,
                d_3d=d_3d,
                h_bs=self.bs_set.bs_heights,
                h_ue=ue_heights,
            )
            pl_db = np.asarray(
                utc.lin_to_db(
                    utc.db_to_lin(pl_los) * los_values**2
                    + utc.db_to_lin(pl_nlos) * (1.0 - los_values**2)
                ),
                dtype=self.internal_dtype,
            )
            # Large-scale approximation of soft LOS/NLOS mixing; not the
            # complex channel-matrix combination in equation (7.6-19).
        else:
            raise ValueError("Invalid LOS model type.")  # noqa: TRY004 - public contract

        return self._as_internal(pl_db), self._as_internal(los_values)

    def _calculate_sinr_from_selected_images(
        self,
        *,
        p_rx_dbm: FloatArr,
        pl_img_db: FloatArr,
        sf_img_db: FloatArr,
        phi_az_img_deg: FloatArr,
        theta_el_img_deg: FloatArr,
        best_image_idx: IntArr,
        g_bs_img_db_mean: FloatArr | None = None,
    ) -> FloatArr:
        """Calculate SINR using the same selected image as coupling gain."""
        if self.SINR_EST_METHOD != "mean_beam_gain":
            raise NotImplementedError(
                "Only mean_beam_gain SINR supports max_coupling_over_images."
            )
        g_ue_img_db = np.zeros_like(pl_img_db, dtype=self.internal_dtype)
        if g_bs_img_db_mean is None:
            g_bs_img_db_mean = self.bs_set.get_mean_ant_gains(
                phi_az_img_deg,
                theta_el_img_deg,
            )
        cg_img_db_mean = self._get_coupling_gain_db(
            g_ue_img_db,
            g_bs_img_db_mean,
            pl_img_db,
            sf_img_db,
        )
        cg_db_mean = self._gather_candidate_image(cg_img_db_mean, best_image_idx)
        p_rx_dbm_mean = cg_db_mean + self.p_tx_sc_bs_dbm
        return self._calculate_mean_sinr(p_rx_dbm, p_rx_dbm_mean)

    @staticmethod
    def _gather_candidate_image(values: FloatArr, image_idx: IntArr) -> FloatArr:
        """Gather values from shape ``(..., n_images, n_cells)`` by image index."""
        gathered = np.take_along_axis(values, image_idx[..., None, :], axis=-2)
        return np.squeeze(gathered, axis=-2)

    def _get_coupling_gain_db(
        self,
        g_ue_db: FloatArr,
        g_bs_db: FloatArr,
        pl_db: FloatArr,
        sf_db: FloatArr,
    ) -> FloatArr:
        """
        Calculate the coupling gain for all UEs and cells.

        Parameters
        ----------
        g_ue_db : FloatArr
            UE antenna gain in dB with shape (..., n_cells).
        g_bs_db : FloatArr
            BS antenna gain in dB with shape (..., n_cells).
        pl_db : FloatArr
            Path loss in dB with shape (..., n_cells).
        sf_db : FloatArr
            Shadow fading in dB with shape (..., n_cells).

        Returns
        -------
        FloatArr
            Coupling gain in dB with shape (..., n_cells).
        """
        if (
            g_ue_db.shape != g_bs_db.shape
            or g_ue_db.shape != pl_db.shape
            or g_ue_db.shape != sf_db.shape
        ):
            raise ValueError("Mismatched array shapes for coupling-gain calculation.")
        return self._as_internal(g_ue_db + g_bs_db - pl_db - sf_db)

    def _calculate_rsrp(self, coupling_gain_db: FloatArr) -> FloatArr:
        """
        Calculate the narrowband received-power proxy for all UEs and cells.

        Parameters
        ----------
        coupling_gain_db : FloatArr
            Coupling gain in dB with shape (..., n_cells).

        Returns
        -------
        FloatArr
            RSRP-like received power in dBm with shape (..., n_cells).
        """
        if coupling_gain_db.shape[-1] != self.bs_set.n_cells:
            raise ValueError(
                f"coupling_gain_db must have shape (..., n_cells={self.bs_set.n_cells}), "
                f"got {coupling_gain_db.shape}."
            )

        p_rx_dbm = self.p_tx_sc_bs_dbm + coupling_gain_db
        if self.rat_config.rsrp_clipping_enabled:
            return np.asarray(
                np.clip(
                    p_rx_dbm,
                    self.rat_config.rsrp_dbm_min,
                    self.rat_config.rsrp_dbm_max,
                ),
                dtype=self.internal_dtype,
            )
        return self._as_internal(p_rx_dbm)

    def _get_noise_w_for_shape(
        self,
        target_shape: tuple[int, ...],
        *,
        per_subcarrier: bool,
    ) -> float | FloatArr:
        """
        Return UE noise power in W broadcast-compatible with (..., n_cells).

        For normal UE-wise calculations, returns shape (n_ue, 1). For map creation or
        other mismatched shapes, returns the scalar mean UE noise power.
        """
        if not target_shape:
            raise ValueError("target_shape must not be empty.")

        noise = self.p_noise_sc_ue_w if per_subcarrier else self.p_noise_ue_w

        if len(target_shape) >= 2 and target_shape[0] == noise.shape[0]:
            return noise

        return float(np.mean(noise))

    def _calculate_rsrq(self, rsrp_dbm: FloatArr) -> FloatArr:
        """
        Calculate an approximate RSRQ for all UEs and cells.

        Parameters
        ----------
        rsrp_dbm : FloatArr
            RSRP-like received power in dBm with shape (..., n_cells).

        Returns
        -------
        FloatArr
            Approximate RSRQ in dB with shape (..., n_cells).
        """
        if rsrp_dbm.shape[-1] != self.bs_set.n_cells:
            raise ValueError(
                f"rsrp_dbm shape {rsrp_dbm.shape} does not match n_cells={self.bs_set.n_cells}."
            )

        p_signal_w = self._as_internal(utc.dbm_to_w(rsrp_dbm))
        p_interference_w = np.sum(p_signal_w, axis=-1, keepdims=True) - p_signal_w
        p_noise_w_ue = self._get_noise_w_for_shape(
            p_signal_w.shape,
            per_subcarrier=False,
        )

        rssi_w = self.n_rb * (p_signal_w + p_interference_w) + p_noise_w_ue
        rsrq_lin = self.n_rb * p_signal_w / rssi_w
        return self._as_internal(utc.lin_to_db(rsrq_lin))

    def _calculate_sinr(
        self,
        phi_az: FloatArr,
        theta_el: FloatArr,
        p_rx_dbm: FloatArr,
        pl_db: FloatArr,
        sf_db: FloatArr,
        g_bs_db_mean: FloatArr | None = None,
    ) -> FloatArr:
        """Calculate SINR for all UEs and cells."""
        if (
            phi_az.shape != theta_el.shape
            or phi_az.shape[:-1] != pl_db.shape[:-1]
            or pl_db.shape != sf_db.shape
        ):
            raise ValueError(
                "Mismatched array shapes for SINR calculation: "
                f"phi_az={phi_az.shape}, theta_el={theta_el.shape}, "
                f"pl_db={pl_db.shape}, sf_db={sf_db.shape}."
            )
        if p_rx_dbm.shape != pl_db.shape:
            raise ValueError(
                f"p_rx_dbm shape {p_rx_dbm.shape} must match pl_db shape {pl_db.shape}."
            )

        g_ue_db = np.zeros_like(pl_db, dtype=self.internal_dtype)

        if self.SINR_EST_METHOD == "mean_beam_gain":
            if g_bs_db_mean is None:
                g_bs_db_mean = self.bs_set.get_mean_ant_gains(phi_az, theta_el)
            cg_db_mean = self._get_coupling_gain_db(g_ue_db, g_bs_db_mean, pl_db, sf_db)
            p_rx_dbm_mean = cg_db_mean + self.p_tx_sc_bs_dbm
            return self._calculate_mean_sinr(p_rx_dbm, p_rx_dbm_mean)

        if self.SINR_EST_METHOD == "stochastic_beam_gain":
            g_bs_beam_db = self.bs_set.get_beam_gains(phi_az, theta_el)
            return self._calculate_stochastic_beam_sinr(
                g_ue_db, g_bs_beam_db, pl_db, sf_db
            )

        if self.SINR_EST_METHOD == "scheduler_beam_gain":
            raise NotImplementedError(
                "Scheduler beam gain SINR estimation is not implemented yet."
            )

        raise ValueError(f"Invalid SINR estimation method: {self.SINR_EST_METHOD}")

    def _calculate_mean_sinr(
        self, p_rx_dbm: FloatArr, p_rx_dbm_mean: FloatArr
    ) -> FloatArr:
        """
        Calculate SINR for all UEs and cells.

        Parameters
        ----------
        p_rx_dbm : FloatArr
            Received power in dBm with shape (..., n_cells) using the best beam.
        p_rx_dbm_mean : FloatArr
            Mean received power in dBm with shape (..., n_cells) for interference
            modeling.

        Returns
        -------
        FloatArr
            SINR in dB with shape (..., n_cells).
        """
        if p_rx_dbm.shape != p_rx_dbm_mean.shape:
            raise ValueError("Mismatched array shapes for SINR calculation.")
        if p_rx_dbm.shape[-1] != self.bs_set.n_cells:
            raise ValueError(
                f"p_rx_dbm must have shape (..., {self.bs_set.n_cells}), got {p_rx_dbm.shape}."
            )

        p_signal_w = self._as_internal(utc.dbm_to_w(p_rx_dbm))
        p_rx_mean_w = self._as_internal(utc.dbm_to_w(p_rx_dbm_mean))
        p_interference_w = np.sum(p_rx_mean_w, axis=-1, keepdims=True) - p_rx_mean_w
        p_noise_w_ue = self._get_noise_w_for_shape(
            p_signal_w.shape,
            per_subcarrier=True,
        )

        sinr_lin = p_signal_w / (p_interference_w + p_noise_w_ue)
        return self._as_internal(utc.lin_to_db(sinr_lin))

    def _calculate_stochastic_beam_sinr(
        self,
        g_ue_db: FloatArr,
        g_bs_beam_db: list[FloatArr],
        pl_db: FloatArr,
        sf_db: FloatArr,
    ) -> FloatArr:
        """
        Calculate SINR for all UEs and cells using stochastic beam gains.

        Parameters
        ----------
        g_ue_db : FloatArr
            UE antenna gain in dB with shape (..., n_cells).
        g_bs_beam_db : list[FloatArr]
            BS beam antenna gains in dB with shape (..., n_cells, n_beams).
        pl_db : FloatArr
            Path loss in dB with shape (..., n_cells).
        sf_db : FloatArr
            Shadow fading in dB with shape (..., n_cells).

        Returns
        -------
        FloatArr
            SINR in dB with shape (..., n_cells).
        """
        raise NotImplementedError("Stochastic beam gain SINR not implemented yet.")

    def map_bs_values_to_cells(self, values: FloatArr) -> FloatArr:
        """
        Map values calculated per BS to each cell/sector.

        Parameters
        ----------
        values : FloatArr
            Values with shape (..., n_bs).

        Returns
        -------
        FloatArr
            Mapped values with shape (..., n_cells).
        """
        if values.shape[-1] != self.bs_set.n_bs:
            raise ValueError(
                f"Values must have shape (..., n_bs={self.bs_set.n_bs}), "
                f"got {values.shape}."
            )
        return np.repeat(self._as_internal(values), self.bs_set.n_cells_per_bs, axis=-1)

    def _as_internal(self, values: BoolArr | FloatArr) -> FloatArr:
        """Return values as the configured internal floating dtype."""
        return np.asarray(values, dtype=self.internal_dtype)
