#cython: language_level=3
"""
Cython functions for efficient computation of calculated fields like AziShear and SL3D
"""

cimport cython
import numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int sl3d_column(
    float[:] reflectivity,
    float[:] altitudes,
    float surface_altitude,
    float melting_level,
    bint neighboring_echos,
    float[:] median_reflectivity_in_neighborhood
):
    """Determine SL3D classification based on reflectivity column.
    
    Parameters
    ----------
    reflectivity: 1D float32 array
        Column of reflectivity values (in dBZ)
    altitudes: 1D float32 array
        Evenly-spaced heights ASL of data values, in increasing order (in meters)
    surface_altitude: float
        Altitude (ASL) of surface
    melting_level: float
        Height ASL of melting level
    neighboring_echos: bint
        True if at least six of eight adjacent cells have an echo anywhere in column
    median_reflectivity_in_neighborhood: 1D float32 array
        Median reflectivity within 12km radius of column at each level (in dBZ)
        
    Returns
    -------
    int
        Integer index of SL3D classification. 1: convection, 2: precip stratiform,
        3: non-precip stratiform, 4: anvil, 5: updraft, 0: no result (no echo)
        
    """

    cdef float this_height, this_refl, echo_top, max_refl, peakedness, threshold, refl_grad
    cdef int peakedness_exceedance_count, below_9km_count
    cdef size_t levels, i
    cdef bint high_refl_above_z_melt, any_above_z_melt, exceed_20_at_3, exceed_10_below_3
    cdef bint any_below_5, refl_grad_exceeded

    levels = len(reflectivity)
    echo_top = 0.0
    max_refl = 0.0
    high_refl_above_z_melt = False
    any_above_z_melt = False
    exceed_20_at_3 = False
    exceed_10_below_3 = False
    any_below_5 = False
    refl_grad_exceeded = False
    peakedness_exceedance_count = 0
    below_9km_count = 0

    for i in range(levels):
        this_altitude = altitudes[i]
        this_height = this_altitude - surface_altitude
        this_refl = reflectivity[i]

        # If below 9km, check for peakedness exceedance
        if this_height <= 9.e3:
            below_9km_count += 1
            peakedness = this_refl - median_reflectivity_in_neighborhood[i]
            if peakedness > max(4.0, 10.0 - this_refl**2 / 337.5):
                peakedness_exceedance_count += 1

        # Exceed above melt check
        if this_altitude >= melting_level:
            if this_refl >= 45.:
                high_refl_above_z_melt = True
                any_above_z_melt = True
            elif this_refl > 0.:
                any_above_z_melt = True

        # Setting echo top
        if this_refl >= 25:
            echo_top = this_altitude

        # Setting max refl
        if this_refl > max_refl:
            max_refl = this_refl

        # Check if level nearest 3km
        if (
            i > 0
            and i < levels - 1
            and altitudes[i - 1] - surface_altitude < 3.e3
            and altitudes[i + 1] - surface_altitude > 3.e3
            and this_height < 6.e3 - altitudes[i - 1] - surface_altitude
            and 6.e3 - this_height <= altitudes[i + 1] - surface_altitude
        ):
            if this_refl >= 20:
                exceed_20_at_3 = True

        # Exceed 10 below 3km
        if this_height < 3.e3 and this_refl >= 10:
            exceed_10_below_3 = True

        # Is there an echo at or below 5km
        if this_height <= 5.e3 and this_refl > 0.:
            any_below_5 = True

        # Max refl gradient
        if i > 0 and i < levels - 1:
            refl_grad = (
                (reflectivity[i + 1] - reflectivity[i - 1])
                / (altitudes[i + 1] - altitudes[i - 1])
            )
            if refl_grad > 8.e-3:
                refl_grad_exceeded = True

    # Determine classification
    if max_refl >= 40. and refl_grad_exceeded and neighboring_echos:
        return 5  # updraft
    elif (
        echo_top >= 1.e4
        or high_refl_above_z_melt
        or peakedness_exceedance_count >= (<float>below_9km_count) / 2.
    ):
        return 1  # convection
    elif exceed_20_at_3 or exceed_10_below_3:
        return 2  # precip stratiform
    elif any_below_5 and not exceed_20_at_3:
        return 3  # non-precip stratiform
    elif not any_below_5 and any_above_z_melt:
        return 4  # anvil
    else:
        return 0  # no echo/null result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint check_neighboorhood(float[:, :, :] field):
    """Check field K, 3, 3 for positive values surrounding central column."""
    cdef size_t i, j, k, K, count
    cdef bint any_positive

    K = field.shape[0]
    count = 0
    for j in range(3):
        for i in range(3):
            if j != 1 and i != 1:
                any_positive = False
                for k in range(K):
                    any_positive = any_positive or field[k, j, i] > 0.
                if any_positive:
                    count += 1
    
    return count >= 6


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def sl3d_grid(
    float[:, :, :] reflectivity,
    float[:] altitudes,
    float[:, :] surface_altitudes,
    float[:, :] melting_level,
    float[:, :, :] median_reflectivity_in_neighborhood,
    float mask_below
):
    """SL3D Classification on a full grid.
    
    Z, Y, X dim order.
    
    Docstring TODO
    """
    cdef size_t i, j, k, I, J, K
    cdef bint neighboring_echos, below_threshold

    I = reflectivity.shape[2]
    J = reflectivity.shape[1]
    K = reflectivity.shape[0]

    output = np.zeros((J, I), dtype=np.dtype("i"))
    cdef int [:, :] output_view = output

    for j in range(J):
        for i in range(I):
            if i > 0 and i < I - 1 and j > 0 and j < J - 1:
                neighboring_echos = check_neighboorhood(
                    reflectivity[:, (j - 1):(j + 1), (i - 1):(i + 1)]
                )
            else:
                neighboring_echos = False

            output[j, i] = sl3d_column(
                reflectivity[:, j, i],
                altitudes,
                surface_altitudes[j, i],
                melting_level[j, i],
                neighboring_echos,
                median_reflectivity_in_neighborhood[:, j, i]
            )

            below_threshold = True
            for k in range(K):
                if reflectivity[k, j, i] >= mask_below:
                    below_threshold = False

            if below_threshold:
                output[j, i] = 0

    return output
