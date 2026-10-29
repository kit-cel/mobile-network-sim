"""Step-based radio simulation environment and result types."""

from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np

from mobile_network_sim.seeding import RadioEpisodeSeeds, radio_episode_seeds

from ..components.bs.bs_collection import BaseStationCollection
from ..components.ue.ue_collection import UserEquipmentCollection
from ..config.env_config import RadioConfig
from ..geometry.simulation_geometry import SimulationGeometry
from ..lsf.lsf import LSFMetrics, LSFModel
from ..lsf.lsf_map import CoverageMaps, MapGenerator
from ..utils.types import FloatArr


@dataclass(frozen=True, slots=True)
class RadioStepResult(LSFMetrics):
    """Store radio geometry and link metrics for one simulation step.

    Attributes
    ----------
    ue_pos_xy : FloatArr
        UE positions in metres, with shape ``(n_ues, 2)``.
    ue_angles_rad : FloatArr
        UE movement headings in radians, with shape ``(n_ues,)``.
    ue_bs_distances_2d : FloatArr
        Horizontal UE-to-BS distances in metres, with shape
        ``(n_ues, n_bs)``.
    ue_bs_distances_3d : FloatArr
        Three-dimensional UE-to-BS distances in metres, with shape
        ``(n_ues, n_bs)``.

    See Also
    --------
    mobile_network_sim.radio.lsf.lsf.LSFMetrics
        Inherited path-loss, coupling-gain, RSRP, SINR, LOS, and shadow-fading
        arrays.
    """

    ue_pos_xy: FloatArr
    ue_angles_rad: FloatArr

    ue_bs_distances_2d: FloatArr
    ue_bs_distances_3d: FloatArr

    def to_dict(self) -> dict[str, FloatArr | np.ma.MaskedArray]:
        """Return the canonical dataset fields used by result loaders.

        Returns
        -------
        dict of str to numpy.ndarray
            Mapping containing positions, distances, RSRP, SINR, and movement
            headings. The returned arrays are not copied.
        """
        return {
            "ue_pos": self.ue_pos_xy,
            "distances_2d": self.ue_bs_distances_2d,
            "distances_3d": self.ue_bs_distances_3d,
            "rsrp": self.rsrp_dbm,
            "sinr": self.sinr_db,
            "ue_move_angle": self.ue_angles_rad,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, FloatArr | np.ma.MaskedArray]
    ) -> "RadioStepResult":
        """Create a step result from dataclass field values.

        Parameters
        ----------
        data : dict of str to numpy.ndarray
            Values keyed by the actual :class:`RadioStepResult` dataclass field
            names, including inherited link-metric fields.

        Returns
        -------
        RadioStepResult
            Reconstructed immutable step result.

        Raises
        ------
        ValueError
            If any required dataclass field is absent.
        """
        update_dict = {}
        for field in fields(cls):
            if field.name not in data:
                raise ValueError(f"Missing field '{field.name}' in data.")
            update_dict[field.name] = data[field.name]
        return cls(**update_dict)


class RadioEnvironment:
    """Manage geometry, nodes, mobility, and radio-link calculations.

    Parameters
    ----------
    config : RadioConfig
        Radio geometry, network, channel, and UE configuration.
    t_res_ms : float
        Duration of one simulation step in milliseconds.
    internal_dtype : str or numpy.dtype, optional
        Floating-point dtype used for link calculations.
    map_dtype : str or numpy.dtype, optional
        Floating-point dtype used for generated coverage maps.
    realize_lsf : bool, optional
        Generate the initial stochastic LOS and shadow-fading fields during
        construction. The default preserves direct-call eager behavior. When
        ``False``, static topology is available immediately, but :meth:`reset`
        must be called before :meth:`step` or :meth:`get_lsf_maps`.

    Attributes
    ----------
    config : RadioConfig
        Active radio configuration.
    t_res_s : float
        Step duration in seconds.
    geo_handler : SimulationGeometry
        Geometry, layout, region, and boundary manager.
    bs_set : BaseStationCollection
        Configured base stations and cells.
    ue_set : UserEquipmentCollection
        Configured UEs and mobility models.
    lsf_model : LSFModel
        Large-scale fading and link-budget model.
    """

    def __init__(
        self,
        config: "RadioConfig",
        t_res_ms: float,
        *,
        internal_dtype: str | np.dtype = "float64",
        map_dtype: str | np.dtype = "float32",
        beam_dtype: str | np.dtype = "float64",
        grf_dtype: str | np.dtype = "float64",
        realize_lsf: bool = True,
        static_resources: tuple | None = None,
    ) -> None:
        self.config = config
        self.t_res_s = t_res_ms / 1000.0
        self.internal_dtype = np.dtype(internal_dtype)
        self.map_dtype = np.dtype(map_dtype)

        self.rng = np.random.default_rng(seed=config.seed)

        resource_key = (
            config.geometry.model_dump_json(),
            config.network.model_dump_json(),
            config.resource_storage,
            np.dtype(beam_dtype).str,
        )
        if static_resources is not None and static_resources[0] == resource_key:
            self.geo_handler, self.bs_set = deepcopy(static_resources[1])
        else:
            self.geo_handler = SimulationGeometry(config.geometry)
            self.bs_set = BaseStationCollection(
                config=config.network,
                geo_handler=self.geo_handler,
                resource_storage=config.resource_storage,
                beam_dtype=beam_dtype,
            )
        # Capture pristine resources before UE initialization can advance an
        # ROI generator. Copies isolate every mutable geometry object per Env.
        self.static_resources = (
            resource_key,
            deepcopy((self.geo_handler, self.bs_set)),
        )
        self.ue_set = UserEquipmentCollection(
            n_ue=config.num_ues, ue_configs=config.ues, geo_handler=self.geo_handler
        )
        self.lsf_model = LSFModel(
            config=config.channel,
            rat_config=config.network.rat,
            geo=self.geo_handler,
            bs_set=self.bs_set,
            ue_set=self.ue_set,
            internal_dtype=self.internal_dtype,
            map_dtype=self.map_dtype,
            grf_dtype=grf_dtype,
            realize_fields=realize_lsf,
            resource_storage=config.resource_storage,
        )

        self.lsf_map_generator = MapGenerator(
            self.lsf_model,
            self.geo_handler,
            map_dtype=self.map_dtype,
        )
        self._is_ready = bool(realize_lsf)

    def _require_ready(self) -> None:
        """Reject physical work before a deferred environment is reset."""
        if not self._is_ready:
            raise RuntimeError(
                "RadioEnvironment.reset() must be called before physical simulation "
                "when realize_lsf=False"
            )

    def step(self) -> RadioStepResult:
        """Move every UE and calculate link conditions for the new positions.

        Returns
        -------
        RadioStepResult
            UE geometry and per-link radio metrics after one time step.
        """
        self._require_ready()
        self.ue_set.move_ues_in_roi(time_increment_s=self.t_res_s)
        return self.finish_radio_step(
            self.radio_geometry(self.ue_set.xy, self.ue_set.ue_heights)
        )

    def radio_geometry(self, positions, heights):
        """Pure geometry with an arbitrary leading UE dimension."""
        d_2d, d_3d, phi_az_rad, theta_el_rad = (
            self.geo_handler.get_distances_and_angles(
                ue_positions_xy=positions,
                ue_heights=heights,
                bs_heights=self.bs_set.bs_heights,
            )
        )
        candidate_kwargs = {}
        if self.config.channel.wraparound_coupling_method == "max_coupling_over_images":
            cand_d_2d, cand_d_3d, cand_phi_az_rad, cand_theta_el_rad = (
                self.geo_handler.get_candidate_distances_and_angles(
                    ue_positions_xy=positions,
                    ue_heights=heights,
                    bs_heights=self.bs_set.bs_heights,
                )
            )
            candidate_kwargs = {
                "candidate_d_2d": cand_d_2d,
                "candidate_d_3d": cand_d_3d,
                "candidate_phi_az_deg": np.rad2deg(cand_phi_az_rad),
                "candidate_theta_el_deg": np.rad2deg(cand_theta_el_rad),
            }

        return dict(
            d_2d=d_2d,
            d_3d=d_3d,
            phi_az_deg=np.rad2deg(phi_az_rad),
            theta_el_deg=np.rad2deg(theta_el_rad),
            **candidate_kwargs,
        )

    def finish_radio_step(self, geometry, beam_gains=None):
        """Keep all stochastic channel calls local to this radio instance."""
        lsf_metrics = self.lsf_model.calculate_lsf_metrics(
            pos_3d=self.ue_set.xyz,
            **geometry,
            beam_gains=beam_gains,
        )
        return RadioStepResult(
            ue_pos_xy=self.ue_set.xy,
            ue_angles_rad=self.ue_set.angles_rad,
            ue_bs_distances_2d=geometry["d_2d"],
            ue_bs_distances_3d=geometry["d_3d"],
            pl_db=lsf_metrics.pl_db,
            cg_db=lsf_metrics.cg_db,
            rsrp_dbm=lsf_metrics.rsrp_dbm,
            sinr_db=lsf_metrics.sinr_db,
            los=lsf_metrics.los,
            sf_db=lsf_metrics.sf_db,
        )

    def reset(
        self,
        seed: int | RadioEpisodeSeeds | None = None,
        *,
        regenerate_fields: bool = True,
        ue_configs=None,
    ) -> None:
        """Reset base stations and UEs.

        Parameters
        ----------
        seed : int, RadioEpisodeSeeds, or None, optional
            Episode root or resolved component streams. If omitted, the node
            collections use fresh unseeded streams and the current LSF
            realization is retained. On the first reset of a deferred
            environment, omission instead realizes the configured seed.

        ``regenerate_fields=False`` is an explicit training ablation: keep the
        initialized spatial maps, but reset mobility and channel RNGs. Optional
        ``ue_configs`` replaces the UE collection, including noise terms, while
        retaining the radio geometry. Callers must stop radio producers first.
        """
        if not regenerate_fields and (seed is None or not self._is_ready):
            raise ValueError(
                "Keeping radio fields requires an initialized radio and explicit episode seed"
            )
        if ue_configs is not None:
            self.config = self.config.model_copy(update={"ues": ue_configs})
            self.ue_set = UserEquipmentCollection(
                n_ue=self.config.num_ues,
                ue_configs=ue_configs,
                geo_handler=self.geo_handler,
            )
            self.lsf_model._prepare_noise_and_tx_power(self.ue_set)
        if seed is None and self._is_ready:
            self.bs_set.reset(seed=None)
            self.ue_set.reset(seed=None)
            return

        self._is_ready = False
        plan = radio_episode_seeds(self.config.seed if seed is None else seed)
        self.rng = np.random.default_rng(plan.root_seed)
        self.bs_set.reset(seed=plan.root_seed)
        if regenerate_fields:
            self.lsf_model.reset(seed=plan.radio_map_seed)
        else:
            self.lsf_model.reset(seed=plan.radio_map_seed, regenerate_fields=False)
        self.ue_set.reset(seed=plan)
        self._is_ready = True

    def reset_ue(self, ue_idxs: int | list[int]) -> None:
        """Reset selected UEs to newly sampled initial states.

        Parameters
        ----------
        ue_idxs : int or list of int
            UE index or indices to reset.
        """
        if isinstance(ue_idxs, int):
            ue_idxs = [ue_idxs]
        for idx in ue_idxs:
            self.ue_set.reset_ue(idx)

    def get_lsf_maps(self) -> CoverageMaps:
        """Create large-scale-fading coverage maps.

        Returns
        -------
        CoverageMaps
            Map grids and link metrics for the current geometry.
        """
        self._require_ready()
        return self.lsf_map_generator.create_maps()

    def get_dummy_step_result(self) -> RadioStepResult:
        """Generate a shape-only synthetic result for initialization.

        Returns
        -------
        RadioStepResult
            Result whose array shapes and dtypes match a physical step.

        Notes
        -----
        This helper uses only static topology. It neither advances mobility nor
        requires or creates a stochastic LSF realization.
        """
        n_ue = self.ue_set.n_ue
        n_bs = self.bs_set.n_bs
        n_cells = self.bs_set.n_cells
        metric = np.zeros((n_ue, n_cells), dtype=self.internal_dtype)
        return RadioStepResult(
            ue_pos_xy=np.zeros((n_ue, 2), dtype=self.internal_dtype),
            ue_angles_rad=np.zeros(n_ue, dtype=self.internal_dtype),
            ue_bs_distances_2d=np.zeros((n_ue, n_bs), dtype=self.internal_dtype),
            ue_bs_distances_3d=np.zeros((n_ue, n_bs), dtype=self.internal_dtype),
            pl_db=metric.copy(),
            cg_db=metric.copy(),
            rsrp_dbm=metric.copy(),
            sinr_db=metric.copy(),
            los=metric.copy(),
            sf_db=metric.copy(),
        )
