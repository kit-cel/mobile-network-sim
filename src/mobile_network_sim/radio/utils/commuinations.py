"""Utility functions for communications calculations."""

import numpy as np

from .consts import P_TH_DBM


def noise_power_dbm(b_mhz: float | np.ndarray, nf_db: np.ndarray) -> np.ndarray:
    """Calculate thermal noise power in dBm.

    Parameters
    ----------
    b_mhz : float or np.ndarray
        Bandwidth in MHz.
    nf_db : np.ndarray
        Noise figure in dB.

    Returns
    -------
    np.ndarray
        Thermal noise power in dBm.
    """
    return P_TH_DBM + 10.0 * np.log10(b_mhz * 1e6) + nf_db


### Power Conversion
def dbm_to_w(dbm: np.ndarray) -> np.ndarray:
    """Convert from dBm to Watts.

    Parameters
    ----------
    dbm : np.ndarray
        Input in dBm.

    Returns
    -------
    np.ndarray
        Output in Watts.
    """
    return 10.0 ** ((dbm - 30.0) / 10.0)


def w_to_dbm(w: np.ndarray) -> np.ndarray:
    """Convert from Watts to dBm.

    Parameters
    ----------
    w : np.ndarray
        Input in Watts.

    Returns
    -------
    np.ndarray
        Output in dBm.
    """
    return 10.0 * np.log10(w) + 30.0


def db_to_lin(db: np.ndarray) -> np.ndarray:
    """Convert from decibels to linear scale.

    Parameters
    ----------
    db : np.ndarray
        Input in decibels.

    Returns
    -------
    np.ndarray
        Output in linear scale.
    """
    return 10.0 ** (db / 10.0)


def lin_to_db(lin: np.ndarray) -> np.ndarray:
    """Convert from linear scale to decibels.

    Parameters
    ----------
    lin : np.ndarray
        Input in linear scale.

    Returns
    -------
    np.ndarray
        Output in decibels.
    """
    return 10.0 * np.log10(lin)
