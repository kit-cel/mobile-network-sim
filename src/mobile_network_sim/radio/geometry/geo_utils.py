"""Coordinate system transformations for 3D geometry."""

from functools import lru_cache

import numpy as np

NUMERIC_EPS = 1e-12


@lru_cache(maxsize=256)
def _orientation_terms(downtilt: float, slant: float) -> tuple:
    """Cache only constant rotation terms, never position-dependent angles."""
    beta, gamma = np.radians(downtilt), np.radians(slant)
    return beta, gamma, np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)


def cartesian_to_spherical(
    coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cartesian vectors (..., 3) -> (r, azimuth, elevation)

    - azimuth (rad): atan2(y, x) in [-pi, pi]
    - elevation (rad): atan2(z, sqrt(x^2 + y^2)) in [-pi/2, pi/2]
      (i.e. elevation measured from the xy-plane; 0 = horizon, +90 = zenith)
    """
    if coords.shape[-1] != 3:
        raise ValueError("coords must have last dim = 3")

    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    r = np.sqrt(x * x + y * y + z * z)
    az = np.arctan2(y, x)  # in radians, [-pi, pi]
    hypot_xy = np.hypot(x, y)  # sqrt(x^2 + y^2)
    elev = np.arctan2(z, hypot_xy)  # in radians, [-pi/2, pi/2]

    # define az/elev = 0 where vector is zero (avoids NaNs)
    zero_mask = r == 0
    if np.any(zero_mask):
        az = np.where(zero_mask, 0.0, az)
        elev = np.where(zero_mask, 0.0, elev)
    return r, az, elev


def map_angle_to_180(x: np.ndarray) -> np.ndarray:
    """
    Map angles to the range [-180, 180] degrees using the formula:
    $$\\text{angle} = ((x + 180) \\mod 360) - 180$$

    Parameters
    ----------
    x : np.ndarray
        Input angle(s) in degrees.

    Returns
    -------
    np.ndarray
        Mapped angle(s) in degrees within [-180, 180].
    """
    return np.mod(x + 180, 360) - 180


def to_physics_convention(
    az: np.ndarray, elev: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert (r, azimuth, elevation) to (r, phi, theta) with physics convention:

    - phi (azimuth): [-pi, pi]
    - theta (polar): [0, pi], where
        0   = zenith (positive z-axis),
        pi/2 = horizontal,
        pi   = nadir (negative z-axis)
    """
    phi = az  # already in [-pi, pi]
    theta = (np.pi / 2) - elev  # map elevation -> polar angle
    theta = np.clip(theta, 0.0, np.pi)

    return phi, theta


def _rotation_x_rad(angle_rad: float) -> np.ndarray:
    """Rotation matrix around x-axis."""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_val, -sin_val],
            [0.0, sin_val, cos_val],
        ],
        dtype=np.float64,
    )


def _rotation_y_rad(angle_rad: float) -> np.ndarray:
    """Rotation matrix around y-axis."""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array(
        [
            [cos_val, 0.0, sin_val],
            [0.0, 1.0, 0.0],
            [-sin_val, 0.0, cos_val],
        ],
        dtype=np.float64,
    )


def _rotation_z_rad(angle_rad: float) -> np.ndarray:
    """Rotation matrix around z-axis."""
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    return np.array(
        [
            [cos_val, -sin_val, 0.0],
            [sin_val, cos_val, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def gcs_to_lcs(
    phi_gcs: np.ndarray,
    theta_gcs: np.ndarray,
    *,
    bearing: float = 0.0,
    downtilt: float = 0.0,
    slant: float = 0.0,
    alpha_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Transform angles from GCS to LCS with an arbitrary 3D rotation.

    The inverse orientation rotation is defined by bearing, downtilt, and
    slant. The general calculation follows equations (7.1-7) and (7.1-8) of
    ETSI TR 138 901; the downtilt-only form follows equations (7.1-18) and
    (7.1-19).

    Parameters
    ----------
    phi_gcs : np.ndarray
        Azimuth angle(s) in GCS in degrees (bearing convention: 0° = North).
    theta_gcs : np.ndarray
        Elevation angle(s) in GCS in degrees.
    bearing : float, optional
        Bearing angle (alpha) in degrees. Default is 0.0.
    downtilt : float, optional
        Downtilt angle (beta) in degrees. Default is 0.0.
    slant : float, optional
        Slant angle (gamma) in degrees. Default is 0.0.
    alpha_sign : int, optional
        Sign convention for bearing angle (+1 or -1). Default is +1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed azimuth and elevation angles in LCS in degrees.

    Notes
    -----
    In the cited specification, :math:`\alpha`, :math:`\beta`, and
    :math:`\gamma` denote bearing, downtilt, and slant, respectively. The
    implementation subtracts the signed bearing before evaluating the general
    or simplified angular expressions.

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clauses 7.1.3--7.1.4, May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """
    # Validate theta_gcs is in valid range [0, 180] degrees
    if np.any((theta_gcs < 0) | (theta_gcs > 180)):
        raise ValueError("theta_gcs must be in range [0, 180] degrees. ")

    # Adjust phi_gcs by bearing angle alpha
    phi_alpha_rad = np.radians(map_angle_to_180(phi_gcs - alpha_sign * bearing))

    # For negligible rotations, return early
    beta_rad, gamma_rad, cos_beta, sin_beta, cos_gamma, sin_gamma = _orientation_terms(
        downtilt, slant
    )

    if np.abs(beta_rad) < NUMERIC_EPS and np.abs(gamma_rad) < NUMERIC_EPS:
        return np.rad2deg(phi_alpha_rad), theta_gcs

    theta_rad = np.radians(theta_gcs)

    cos_theta = np.cos(theta_rad)
    cos_phi_alpha = np.cos(phi_alpha_rad)

    sin_theta = np.sin(theta_rad)
    sin_phi_alpha = np.sin(phi_alpha_rad)

    if np.abs(gamma_rad) > NUMERIC_EPS:
        theta_lcs_rad = np.arccos(  # (7.1-7)
            cos_beta * cos_gamma * cos_theta
            + (sin_beta * cos_gamma * cos_phi_alpha - sin_gamma * sin_phi_alpha)
            * sin_theta
        )
        phi_lcs_rad = np.angle(  # (7.1-8)
            cos_beta * sin_theta * cos_phi_alpha
            - sin_beta * cos_theta
            + 1j
            * (
                cos_beta * sin_gamma * cos_theta
                + (sin_beta * sin_gamma * cos_phi_alpha + cos_gamma * sin_phi_alpha)
                * sin_theta
            )
        )
    else:
        theta_lcs_rad = np.arccos(  # (7.1-18)
            cos_phi_alpha * sin_theta * sin_beta + cos_theta * cos_beta
        )
        phi_lcs_rad = np.angle(  # (7.1-19)
            cos_phi_alpha * sin_theta * cos_beta
            - cos_theta * sin_beta
            + 1j * sin_phi_alpha * sin_theta
        )

    return map_angle_to_180(np.rad2deg(phi_lcs_rad)), np.rad2deg(theta_lcs_rad)


def lcs_to_gcs(
    phi_lcs: np.ndarray,
    theta_lcs: np.ndarray,
    *,
    bearing: float = 0.0,
    downtilt: float = 0.0,
    slant: float = 0.0,
    alpha_sign: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform angles from LCS to GCS with an arbitrary 3D rotation.

    This is the inverse of :func:`gcs_to_lcs`. It represents each local
    spherical direction as a Cartesian unit vector, applies the forward
    composite orientation rotation, and converts the resulting global vector
    back to spherical angles.

    Rotation convention:
    - bearing (alpha): rotation around z-axis
    - downtilt (beta): rotation around y-axis
    - slant (gamma): rotation around x-axis

    Parameters
    ----------
    phi_lcs : np.ndarray
        Azimuth angle(s) in LCS in degrees.
    theta_lcs : np.ndarray
        Zenith angle(s) in LCS in degrees, in [0, 180].
    bearing : float, optional
        Bearing angle (alpha) in degrees. Default is 0.0.
    downtilt : float, optional
        Downtilt angle (beta) in degrees. Default is 0.0.
    slant : float, optional
        Slant angle (gamma) in degrees. Default is 0.0.
    alpha_sign : int, optional
        Sign convention for bearing angle (+1 or -1). Default is +1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed azimuth and zenith angles in GCS in degrees.

    Notes
    -----
    The composite and inverse rotation relationships follow equations (7.1-2)
    and (7.1-3) of ETSI TR 138 901. The spherical unit-vector representation
    follows equation (7.1-6).

    References
    ----------
    .. [1] ETSI, *5G; Study on channel model for frequencies from 0.5 to
       100 GHz (3GPP TR 38.901 version 18.0.0 Release 18)*,
       ETSI TR 138 901 V18.0.0, clause 7.1.3, May 2024.
       https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/
       18.00.00_60/tr_138901v180000p.pdf
    """
    phi_lcs = np.asarray(phi_lcs, dtype=np.float64)
    theta_lcs = np.asarray(theta_lcs, dtype=np.float64)

    if np.any((theta_lcs < 0.0) | (theta_lcs > 180.0)):
        raise ValueError("theta_lcs must be in range [0, 180] degrees.")

    # Flatten for vectorized matrix multiplication, restore shape at end
    orig_shape = np.broadcast(phi_lcs, theta_lcs).shape
    phi_lcs_flat = np.broadcast_to(phi_lcs, orig_shape).ravel()
    theta_lcs_flat = np.broadcast_to(theta_lcs, orig_shape).ravel()

    phi_lcs_rad = np.radians(phi_lcs_flat)
    theta_lcs_rad = np.radians(theta_lcs_flat)

    # Local spherical (phi, theta) -> Cartesian unit vector
    # 3GPP / physics convention:
    #   x = sin(theta) cos(phi)
    #   y = sin(theta) sin(phi)
    #   z = cos(theta)
    sin_theta = np.sin(theta_lcs_rad)
    local_vec = np.vstack(
        [
            sin_theta * np.cos(phi_lcs_rad),
            sin_theta * np.sin(phi_lcs_rad),
            np.cos(theta_lcs_rad),
        ]
    )  # shape (3, N)

    alpha_rad = np.radians(alpha_sign * bearing)
    beta_rad = np.radians(downtilt)
    gamma_rad = np.radians(slant)

    # Inverse of:
    #   local = Rx(-gamma) @ Ry(-beta) @ Rz(-alpha) @ global
    # therefore:
    #   global = Rz(alpha) @ Ry(beta) @ Rx(gamma) @ local
    rot = (
        _rotation_z_rad(alpha_rad)
        @ _rotation_y_rad(beta_rad)
        @ _rotation_x_rad(gamma_rad)
    )

    global_vec = rot @ local_vec

    x_g = global_vec[0, :]
    y_g = global_vec[1, :]
    z_g = np.clip(global_vec[2, :], -1.0, 1.0)

    phi_gcs_deg = map_angle_to_180(np.degrees(np.arctan2(y_g, x_g)))
    theta_gcs_deg = np.degrees(np.arccos(z_g))

    return phi_gcs_deg.reshape(orig_shape), theta_gcs_deg.reshape(orig_shape)


def xy_to_global_angles(
    x_grid_m: np.ndarray, y_grid_m: np.ndarray, h_ant_m: float, h_ue_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ground-plane coordinates to global (phi, theta).

    Antenna at (0, 0, h_ant), points at (x, y, h_map) on the ground plane.
    """
    delta_z_m = h_ant_m - h_ue_m
    dist_2d_m = np.hypot(x_grid_m, y_grid_m)
    dist_3d_m = np.sqrt(dist_2d_m**2 + delta_z_m**2)

    phi_global_deg = np.rad2deg(np.arctan2(y_grid_m, x_grid_m))

    cos_theta = delta_z_m / np.maximum(dist_3d_m, NUMERIC_EPS)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_global_deg = np.rad2deg(np.arccos(cos_theta))

    return phi_global_deg, theta_global_deg
