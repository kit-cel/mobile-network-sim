"""RRC states."""

from enum import IntEnum, auto

RRCStateTransition = tuple["RRCState", str]


class RRCState(IntEnum):
    """Represent the simulated RRC connection and procedure state.

    Attributes
    ----------
    IDLE
        The UE is not connected to a serving cell.
    CONNECTED_NORMAL
        The UE is connected and no handover or failure procedure is active.
    CONNECTED_HO_PREP
        Handover preparation is active while the user plane remains available.
    CONNECTED_HO_EXEC
        Handover execution is active and the user plane is interrupted.
    CONNECTED_PHY_PROBLEM
        A physical-layer problem has been declared.
    RLF_RECOVERY
        Simulated radio-link-failure recovery is active.
    """

    IDLE = auto()

    CONNECTED_NORMAL = auto()
    CONNECTED_HO_PREP = auto()
    CONNECTED_HO_EXEC = auto()
    CONNECTED_PHY_PROBLEM = auto()

    RLF_RECOVERY = auto()

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"
