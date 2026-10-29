"""Structured, collision-resistant seed derivation.

Every randomized component of a simulation needs its own stream, and
those streams must stay independent no matter how the base seeds were
chosen. Deriving them arithmetically -- ``base + index`` -- does not
achieve that, and the failure is silent.

Why arithmetic derivation fails
-------------------------------
``UECollection.reset_ue`` used to derive UE mobility seeds as
``episode_seed + ue_index + 1``. Each individual seed is fine:
``numpy.random.default_rng`` mixes an integer through ``SeedSequence``,
so consecutive integers still give independent streams.

The problem is *collision between derivations*. Two episode seeds closer
together than the UE count produce overlapping seed ranges, so UEs of
different episodes replay identical trajectories. With sequential episode
seeds and 25 UEs, adjacent episodes duplicated 24 of their 25
trajectories -- while a per-episode uniqueness check still passed,
because within one episode all 25 seeds do differ.

This module derives instead from a *structured path*:
``SeedSequence([base, component, index])``. Different paths produce
independent streams, so ``(100, UE, 1)`` and ``(101, UE, 0)`` are
unrelated even though ``100 + 1 == 101 + 0``. Because the public result is
compressed to 32 bits, equality remains possible in principle; release
generation therefore checks all realized seeds explicitly. Callers no
longer need to space their base seeds, and a future component can be added
without disturbing existing streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Component codes keep independent concerns on independent streams.
#: Values are arbitrary but must never be reassigned: they are part of
#: the reproducibility contract of anything already generated.
COMPONENT_UE_MOBILITY = 0
COMPONENT_ROI = 1
COMPONENT_EPISODE = 2
COMPONENT_RADIO_MAP = 3
COMPONENT_INITIAL_POSITIONS = 4
COMPONENT_MOBILITY_PROCESS = 5
COMPONENT_BOUNDARY = 6

# Sub-stream below the initial-position component root. It is deliberately
# separate from UE mobility: resetting a subset of UEs must reproduce the
# same initial position for a given UE, independent of reset order.
COMPONENT_UE_INITIAL_POSITION = 0

# Sub-streams below a radio-map seed. These codes live at a different level
# of the seed tree than the episode components above, so reusing small integer
# values is intentional and does not alias streams.
COMPONENT_LSF_LOS = 0
COMPONENT_LSF_SHADOWING = 1
COMPONENT_LSF_CHANNEL = 2

# Sub-streams below the channel component seed. UrbanMacro currently owns two
# independent generators for effective-height sampling.
COMPONENT_CHANNEL_RNG_1 = 0
COMPONENT_CHANNEL_RNG_2 = 1

_SEED_UPPER_BOUND = 2**32


def derive_seed(base_seed: int, *path: int) -> int:
    """Derive one reproducible 32-bit seed from a structured path.

    Parameters
    ----------
    base_seed : int
        The run's single root of entropy.
    *path : int
        Structured coordinates, e.g. ``(COMPONENT_UE_MOBILITY, ue_index)``.
        Different paths yield independent, non-colliding streams.

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.

    Examples
    --------
    Neighbouring base seeds no longer share derived seeds, which is what
    the arithmetic derivation got wrong:

    >>> derive_seed(100, COMPONENT_UE_MOBILITY, 1) == derive_seed(
    ...     101, COMPONENT_UE_MOBILITY, 0
    ... )
    False
    """
    sequence = np.random.SeedSequence([int(base_seed), *(int(p) for p in path)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def ue_mobility_seed(base_seed: int, ue_index: int) -> int:
    """Return the mobility seed for one UE of a reset.

    This is the single definition of that derivation. Anything that needs
    to *predict* the seeds a reset will use -- a dataset generator
    recording provenance, a uniqueness check run before simulating --
    must call this rather than reimplement the arithmetic, or it will
    describe a different simulation than the one that runs.
    """
    return derive_seed(base_seed, COMPONENT_UE_MOBILITY, ue_index)


@dataclass(frozen=True, slots=True)
class RadioEpisodeSeeds:
    """Independent random streams used by one radio episode.

    ``root_seed`` identifies the episode. The other fields are the actual
    seeds consumed by the stochastic radio map, initial-position sampler,
    mobility models, and (currently deterministic) reflective boundary.
    Keeping them explicit makes a reset reproducible without coupling the
    number of random draws in one component to any other component.
    """

    root_seed: int
    radio_map_seed: int
    initial_positions_seed: int
    mobility_process_seed: int
    boundary_seed: int

    @classmethod
    def from_root(cls, root_seed: int) -> "RadioEpisodeSeeds":
        """Derive all component streams from one episode root seed."""
        return cls.from_components(
            root_seed=root_seed,
            radio_map_seed=derive_seed(root_seed, COMPONENT_RADIO_MAP),
            initial_positions_seed=derive_seed(
                root_seed, COMPONENT_INITIAL_POSITIONS
            ),
            mobility_process_seed=derive_seed(root_seed, COMPONENT_MOBILITY_PROCESS),
        )

    @classmethod
    def from_components(
        cls,
        *,
        root_seed: int,
        radio_map_seed: int,
        initial_positions_seed: int,
        mobility_process_seed: int,
    ) -> "RadioEpisodeSeeds":
        """Build a plan from externally derived component seeds.

        Dataset releases use this constructor because their component seeds
        are derived directly from ``(release, namespace, split, episode,
        component)`` and must be recorded verbatim in provenance.
        """
        return cls(
            root_seed=int(root_seed),
            radio_map_seed=int(radio_map_seed),
            initial_positions_seed=int(initial_positions_seed),
            mobility_process_seed=int(mobility_process_seed),
            boundary_seed=derive_seed(int(mobility_process_seed), COMPONENT_BOUNDARY),
        )

    def ue_mobility_seed(self, ue_index: int) -> int:
        """Return the mobility RNG seed for one UE in this episode."""
        return ue_mobility_seed(self.mobility_process_seed, ue_index)

    def ue_initial_position_seed(self, ue_index: int) -> int:
        """Return the initial-position RNG seed for one UE in this episode."""
        return derive_seed(
            self.initial_positions_seed,
            COMPONENT_UE_INITIAL_POSITION,
            ue_index,
        )

    @property
    def lsf_los_seed(self) -> int:
        """Return the LOS realization seed below the radio-map root."""
        return derive_seed(self.radio_map_seed, COMPONENT_LSF_LOS)

    @property
    def lsf_shadowing_seed(self) -> int:
        """Return the shadowing realization seed below the radio-map root."""
        return derive_seed(self.radio_map_seed, COMPONENT_LSF_SHADOWING)

    @property
    def lsf_channel_seed(self) -> int:
        """Return the channel-model root seed below the radio-map root."""
        return derive_seed(self.radio_map_seed, COMPONENT_LSF_CHANNEL)

    def channel_rng_seed(self, index: int) -> int:
        """Return one channel-internal RNG seed (currently indices 0 and 1)."""
        if index not in (0, 1):
            raise ValueError("channel RNG index must be 0 or 1")
        component = (
            COMPONENT_CHANNEL_RNG_1 if index == 0 else COMPONENT_CHANNEL_RNG_2
        )
        return derive_seed(self.lsf_channel_seed, component)


def radio_episode_seeds(seed: int | RadioEpisodeSeeds) -> RadioEpisodeSeeds:
    """Normalize an episode root or an already-resolved component plan."""
    if isinstance(seed, RadioEpisodeSeeds):
        return seed
    return RadioEpisodeSeeds.from_root(int(seed))


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Return the reset seed for one episode of a multi-episode run.

    Episodes must differ from each other while the run as a whole stays
    reproducible from ``base_seed`` alone.
    """
    return derive_seed(base_seed, COMPONENT_EPISODE, episode_index)


__all__ = [
    "COMPONENT_BOUNDARY",
    "COMPONENT_CHANNEL_RNG_1",
    "COMPONENT_CHANNEL_RNG_2",
    "COMPONENT_EPISODE",
    "COMPONENT_INITIAL_POSITIONS",
    "COMPONENT_LSF_CHANNEL",
    "COMPONENT_LSF_LOS",
    "COMPONENT_LSF_SHADOWING",
    "COMPONENT_MOBILITY_PROCESS",
    "COMPONENT_RADIO_MAP",
    "COMPONENT_ROI",
    "COMPONENT_UE_MOBILITY",
    "COMPONENT_UE_INITIAL_POSITION",
    "RadioEpisodeSeeds",
    "derive_seed",
    "episode_seed",
    "radio_episode_seeds",
    "ue_mobility_seed",
]
