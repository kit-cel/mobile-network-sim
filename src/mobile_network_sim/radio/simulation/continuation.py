"""Conditional future innovations without resetting the physical radio state."""

from __future__ import annotations

import numpy as np

from mobile_network_sim.seeding import derive_seed


def reseed_innovations(radio, seed: int) -> tuple[str, ...]:
    """Retain realized maps, geometry, waypoints and mobility/channel memory.

    Call only at a synchronous sample boundary with no prefetched samples.
    Spatial shadow/LOS fields are part of the conditioned state, not newly
    sampled maps. Their generator streams are reseeded without regenerating
    fields. The caller owns RRC and observation history, which are untouched.
    """
    radio._require_ready()
    generators = [("radio", radio.rng)]
    generators.extend(
        (f"mobility/{i}", ue.mobility_model.rng)
        for i, ue in enumerate(radio.ue_set.ue_list)
    )
    generators.extend(
        (f"lsf/{i}", rng) for i, rng in enumerate(radio.lsf_model._random_generators())
    )
    for owner, component in (
        ("boundary", radio.geo_handler.boundary_handler),
        ("roi", radio.geo_handler.roi),
    ):
        for attribute in ("rng", "np_rng"):
            rng = getattr(component, attribute, None)
            if isinstance(rng, np.random.Generator):
                generators.append((owner + "/" + attribute, rng))
    seen: set[int] = set()
    names: list[str] = []
    for name, generator in generators:
        if id(generator) in seen:
            continue
        seen.add(id(generator))
        # Keep the generator object, preserving aliases held by other owners.
        sequence = derive_seed(seed, 709, len(names))
        bit_generator = type(generator.bit_generator)(sequence)
        generator.bit_generator.state = bit_generator.state
        names.append(name)
    return tuple(names)
