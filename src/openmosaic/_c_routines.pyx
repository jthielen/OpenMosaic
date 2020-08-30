#cython: language_level=3
"""
Cython functions for efficient computation of LLSD parameters (such as AziShear).
"""

from libc.math cimport fmod

cimport cython
import numpy as np

# Constants
cdef float PI = 3.14159265358979

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float azi_shear_for_kernel_uniform_weight(
    int m,
    int n,
    float r,
    float r_delta,
    float azi_delta,
    float[:, :] radar_field
):
    r"""
    Compute value of azimuthal shear (Mahalik et al. 2019) at center of provided kernel using
    uniform weighting.

    Parameters
    ----------
    m, n : int
        Number of gates and rays in kernel (i.e., kernel size in range and azimuth directions)
    r : float
        Range of center of kernel, in meters
    r_delta : float
        Spacing between range gates, in meters
    azi_delta : float
        Spacing between azimuthal rays, in radians
    radar_field : 2D float32 array
        Pre-processed velocity data over kernel

    Notes
    -----
    Equation used diverges from that given in Appendix A of Mahalik et al. 2019 due to
    suspected error (does not match Eq 12a). Instead, uses Eq 12a directly with local weight
    of 1, obtaining

    .. math:: D = (\sum \Delta r_k \Delta \theta_k)^2 \sum 1
                  + \sum \Delta \theta_k^2 (\sum \Delta r_k)^2
                  + (\sum \Delta \theta_k)^2 \sum \Delta r_k^2
                  - 2 \sum \Delta r_k \sum \Delta \theta_k \sum \Delta r_k \Delta \theta_k
                  - \sum \Delta r_k^2 \sum \Delta \theta_k^2 \sum 1
    
    .. math:: u_\theta = \frac{1}{D}\left[
                       \sum \Delta r_k u_k (
                           \sum \Delta r_k \Delta \theta_k \sum 1
                           - \sum \Delta r_k \Delta \theta_k
                       )
                       - \sum \Delta \theta_k u_k (
                           \sum \Delta \theta_k^2 \sum 1
                           - (\sum \Delta \theta_k)^2
                       )
                       + \sum u_k (
                           \sum \Delta \theta_k^2 \sum \Delta r_k
                           - \sum \Delta r_k \Delta \theta_k \sum \Delta \theta_k
                       )
                       \right]
    """

    cdef float sum_r, sum_theta, sum_r_sqr, sum_theta_sqr, sum_r_theta
    cdef float sum_u, sum_r_u, sum_theta_u
    cdef float u, r_dist, theta_dist
    cdef float D
    cdef int i, j

    sum_r = 0
    sum_theta = 0
    sum_r_sqr = 0
    sum_theta_sqr = 0
    sum_r_theta = 0
    sum_u = 0
    sum_r_u = 0
    sum_theta_u = 0

    for i in range(m):
        r_dist = r_delta * (i - (m / 2))
        for j in range(n):
            u = radar_field[i, j]
            theta_dist = azi_delta * (j - (n / 2)) * (r + r_dist)

            # Add to sums
            sum_r += r_dist
            sum_theta += theta_dist
            sum_r_sqr += r_dist * r_dist
            sum_theta_sqr += theta_dist * theta_dist
            sum_r_theta += r_dist * theta_dist
            sum_u += u
            sum_r_u += u * r_dist
            sum_theta_u += u * theta_dist

    # Compute determinant
    D = (
        sum_r_theta * sum_r_theta * m * n
        + sum_theta_sqr * sum_r * sum_r
        + sum_theta * sum_theta * sum_r_sqr
        - 2 * sum_r * sum_theta * sum_r_theta
        - sum_r_sqr * sum_theta_sqr * m * n
    )

    return (
        (
            sum_r_u * (sum_r_theta * m * n - sum_r * sum_theta)
            - sum_theta_u * (sum_theta_sqr * m * n - sum_theta * sum_theta)
            + sum_u * (sum_theta_sqr * sum_r - sum_r_theta * sum_theta)
        ) / D
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef float div_shear_for_kernel_uniform_weight(
    int m,
    int n,
    float r,
    float r_delta,
    float azi_delta,
    float[:, :] radar_field
):
    r"""
    Compute value of radial divergence (Mahalik et al. 2019) at center of provided kernel using
    uniform weighting.

    Parameters
    ----------
    m, n : int
        Number of gates and rays in kernel (i.e., kernel size in range and azimuth directions)
    r : float
        Range of center of kernel, in meters
    r_delta : float
        Spacing between range gates, in meters
    azi_delta : float
        Spacing between azimuthal rays, in radians
    radar_field : 2D float32 array
        Pre-processed velocity data over kernel (ray, gate dim order)

    Notes
    -----
    Equation used diverges from that given in Appendix B of Mahalik et al. 2019 due to
    suspected error (does not match Eq 12b). Instead, uses Eq 12b directly with local weight
    of 1, obtaining

    .. math:: D = (\sum \Delta r_k \Delta \theta_k)^2 \sum 1
                  + \sum \Delta \theta_k^2 (\sum \Delta r_k)^2
                  + (\sum \Delta \theta_k)^2 \sum \Delta r_k^2
                  - 2 \sum \Delta r_k \sum \Delta \theta_k \sum \Delta r_k \Delta \theta_k
                  - \sum \Delta r_k^2 \sum \Delta \theta_k^2 \sum 1
    
    .. math:: u_\theta = \frac{1}{D}\left[
                       - \sum \Delta r_k u_k (
                           \sum \Delta r_k^2 \sum 1
                           - (\sum \Delta r_k)^2
                       )
                       + \sum \Delta \theta_k u_k (
                           \sum \Delta r_k \Delta \theta_k \sum 1
                           - \sum \Delta r_k \sum \Delta \theta_k
                       )
                       - \sum u_k (
                           \sum \Delta r_k \Delta \theta_k \sum \Delta r_k
                           - \sum \Delta r_k^2 \sum \Delta \theta_k
                       )
                       \right]
    """

    cdef float sum_r, sum_theta, sum_r_sqr, sum_theta_sqr, sum_r_theta
    cdef float sum_u, sum_r_u, sum_theta_u
    cdef float u, r_dist, theta_dist
    cdef float D
    cdef int i, j

    sum_r = 0
    sum_theta = 0
    sum_r_sqr = 0
    sum_theta_sqr = 0
    sum_r_theta = 0
    sum_u = 0
    sum_r_u = 0
    sum_theta_u = 0

    for i in range(m):
        r_dist = r_delta * (i - (m / 2))
        for j in range(n):
            u = radar_field[j, i]
            theta_dist = azi_delta * (j - (n / 2)) * (r + r_dist)

            # Add to sums
            sum_r += r_dist
            sum_theta += theta_dist
            sum_r_sqr += r_dist * r_dist
            sum_theta_sqr += theta_dist * theta_dist
            sum_r_theta += r_dist * theta_dist
            sum_u += u
            sum_r_u += u * r_dist
            sum_theta_u += u * theta_dist

    # Compute determinant
    D = (
        sum_r_theta * sum_r_theta * m * n
        + sum_theta_sqr * sum_r * sum_r
        + sum_theta * sum_theta * sum_r_sqr
        - 2 * sum_r * sum_theta * sum_r_theta
        - sum_r_sqr * sum_theta_sqr * m * n
    )

    return (
        (
            - sum_r_u * (sum_r_sqr * m * n - sum_r * sum_r)
            + sum_theta_u * (sum_r_theta * m * n - sum_r * sum_theta)
            - sum_u * (sum_r_theta * sum_r - sum_r_sqr * sum_theta)
        ) / D
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def azi_shear_uniform_weight(
    int ngates,
    int nrays,
    float[:] r,
    float[:] theta,
    float[:, :] vel_field,
    float azi_width,
    float r_depth,
    int azi_max_rays
):
    r"""
    Compute azimuthal shear (Mahalik et al. 2019) across the full radar scan level.

    Assumes full sweep (cyclically indexed in azimuth direction). Discontinuties may occur
    otherwise.

    Parameters
    ----------
    ngates, nrays : int
        Number of gates and rays in sweep (i.e., slice size in range and azimuth directions)
    r : 1D float32 array
        Range of centers of cells, in meters
    theta : 1D float32 array
        Azimuth of centers of cells, in radians
    vel_field : 2D float32 array
        Pre-processed velocity data over sweep/slice
    azi_width : float
        Maximum width of kernel in azimuth direction, in meters
    r_depth : float
        Maximum depth of kernel in range direction, in meters
    azi_max_rays : int
        Maximum count of rays to include in kernel (to not subtend too large an angle near
        radar site)
    """

    # Init return array and view into it
    azi_shear = np.zeros((nrays, ngates), dtype=np.dtype('float32'))
    cdef float[:, :] azi_shear_view = azi_shear
    
    # Init helper variables
    cdef float azi_delta, r_delta, half_m_float, half_n_float
    cdef int m, n, i_min, i_max, j_min, j_max, nray, ngate

    for nray in range(nrays):
        if nray == 0:
            azi_delta = <float>fmod(theta[1] - theta[-1], 2.0 * PI) / 2.0
        elif nray == nrays - 1:
            azi_delta = <float>fmod(theta[0] - theta[-2], 2.0 * PI) / 2.0
        else:
            azi_delta = <float>fmod(theta[nray + 1] - theta[nray - 1], 2.0 * PI) / 2.0
        
        for ngate in range(ngates):
            if (
                ngate == 0
                or ngate == ngates - 1
                or 2.0 * (r[ngate] - r[0]) < r_depth
                or 2.0 * (r[-1] - r[ngate]) < r_depth
            ):
                # Boundary, computation invalid
                continue
            r_delta = (r[ngate + 1] - r[ngate - 1]) / 2.0

            # Compute kernel size
            half_m_float = r_depth / (2.0 * r_delta)
            m = 2 * <int>half_m_float + 1
            if m < 3:
                m = 3
            half_n_float = azi_width / (2.0 * azi_delta * r[ngate])
            n = 2 * <int>half_n_float + 1
            if n < 3:
                n = 3
            elif n > azi_max_rays:
                if azi_max_rays % 2 == 0:
                    n = azi_max_rays
                else:
                    n = azi_max_rays - 1

            # Compute kernel bounds into array (rays must be cyclically indexed)
            i_min = ngate - (m / 2)
            i_max = ngate + (m / 2) + 1
            j_min = nray - (n / 2) - nrays
            j_max = nray + (n / 2) + 1 - nrays

            # Calculate value
            azi_shear_view[nray, ngate] = azi_shear_for_kernel_uniform_weight(
                m,
                n,
                r[ngate],
                r_delta,
                azi_delta,
                vel_field[j_min:j_max, i_min:i_max]
            )

    return azi_shear


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(True)
def div_shear_uniform_weight(
    int ngates,
    int nrays,
    float[:] r,
    float[:] theta,
    float[:, :] vel_field,
    float azi_width,
    float r_depth,
    int azi_max_rays
):
    r"""
    Compute DivShear/radial divergence (Mahalik et al. 2019) across the full radar scan level.

    Assumes full sweep (cyclically indexed in azimuth direction). Discontinuties may occur
    otherwise.

    Parameters
    ----------
    ngates, nrays : int
        Number of gates and rays in sweep (i.e., slice size in range and azimuth directions)
    r : 1D float32 array
        Range of centers of cells, in meters
    theta : 1D float32 array
        Azimuth of centers of cells, in radians
    vel_field : 2D float32 array
        Pre-processed velocity data over sweep/slice
    azi_width : float
        Maximum width of kernel in azimuth direction, in meters
    r_depth : float
        Maximum depth of kernel in range direction, in meters
    azi_max_rays : int
        Maximum count of rays to include in kernel (to not subtend too large an angle near
        radar site)
    """

    # Init return array and view into it
    div_shear = np.zeros((nrays, ngates), dtype=np.dtype('float32'))
    cdef float[:, :] div_shear_view = div_shear
    
    # Init helper variables
    cdef float azi_delta, r_delta, half_m_float, half_n_float
    cdef int m, n, i_min, i_max, j_min, j_max, nray, ngate

    for nray in range(nrays):
        if nray == 0:
            azi_delta = fmod(theta[1] - theta[-1], 2.0 * PI) / 2.0
        elif nray == nrays - 1:
            azi_delta = fmod(theta[0] - theta[-2], 2.0 * PI) / 2.0
        else:
            azi_delta = fmod(theta[nray + 1] - theta[nray - 1], 2.0 * PI) / 2.0
        
        for ngate in range(ngates):
            if (
                ngate == 0
                or ngate == ngates - 1
                or 2.0 * (r[ngate] - r[0]) < r_depth
                or 2.0 * (r[-1] - r[ngate]) < r_depth
            ):
                # Boundary, computation invalid
                continue
            r_delta = (r[ngate + 1] - r[ngate - 1]) / 2.0

            # Compute kernel size
            half_m_float = r_depth / (2.0 * r_delta)
            m = 2 * <int>half_m_float + 1
            if m < 3:
                m = 3
            half_n_float = azi_width / (2.0 * azi_delta * r[ngate])
            n = 2 * <int>half_n_float + 1
            if n < 3:
                n = 3
            elif n > azi_max_rays:
                if azi_max_rays % 2 == 0:
                    n = azi_max_rays
                else:
                    n = azi_max_rays - 1

            # Compute kernel bounds into array (rays must be cyclically indexed)
            i_min = ngate - (m / 2)
            i_max = ngate + (m / 2) + 1
            j_min = nray - (n / 2) - nrays
            j_max = nray + (n / 2) + 1 - nrays

            # Calculate value
            div_shear_view[nray, ngate] = div_shear_for_kernel_uniform_weight(
                m,
                n,
                r[ngate],
                r_delta,
                azi_delta,
                vel_field[j_min:j_max, i_min:i_max]
            )

    return div_shear
