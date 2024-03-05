# created by: Pau Ramos (@brugalada) & Anthony Brown
# based on Matlab code by: Lennart Lindegren
# Date: 21/07/2020  (important for the table of coefficients)

import os
import numpy as np
import warnings

__mypath = os.path.dirname(os.path.abspath(__file__))

_file5_currentversion = __mypath + '/coefficients/z5_200720.txt'
_file6_currentversion = __mypath + '/coefficients/z6_200720.txt'


# Definition of functions that load the coefficient tables and initialize the global variables

def _read_table(file, sep=','):
    """
    Extract the coefficients and interpolation limits from the input file provided.

    The first and second rows are assumed to be the indices that govern, respectively, the colour and sinBeta interpolations.

    From the third rows onwards, all belong to the G magnitude interpolation: first column, the phot_g_mean_mag boundaries. The rest of columns, the interpolation coefficients.
    """
    # reads the file (.txt)
    input_file = np.genfromtxt(file, delimiter=sep)

    # auxiliary variables j and k
    j = list(map(int, input_file[0, 1:]))
    k = list(map(int, input_file[1, 1:]))
    # g vector
    g = input_file[2:, 0]
    # coefficients
    q_jk = input_file[2:, 1:]
    # shape
    n, m = q_jk.shape

    return j, k, g, q_jk, n, m


def load_tables(file5=_file5_currentversion, file6=_file6_currentversion, sep=','):
    """
    Initialises the tables containing the coefficients of the interpolations for the Z5 and Z6 functions.

    NOTE: USE THE DEFAULT VALUES unless you are very sure of what you are doing.
    
    Inputs
        file5: path to the file with the Z5 coefficients (.txt or similar)
        file6: path to the file with the Z6 coefficients (.txt or similar)
        sep (optional): separator used to split the lines (default, comma)
    """
    global j_5, k_5, g_5, q_jk5, n_5, m_5, j_6, k_6, g_6, q_jk6, n_6, m_6

    j_5, k_5, g_5, q_jk5, n_5, m_5 = _read_table(file5, sep=sep)
    j_6, k_6, g_6, q_jk6, n_6, m_6 = _read_table(file6, sep=sep)

    return None


# Auxiliary function: calculates the zero-point only for an array of stars with the same number of
# astrometric_params_solved

def _calc_zpt(phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, sinBeta, source_type):
    """ 
    Compute the zero-point parallax for an array of stars.
    
    WARNING! This function is meant to be auxiliary, therefore it assumes that the inputs are well formatted (see
    get_zpt()) and that all the sources have the same value for astrometric_params_solved. That is, either all are 5p
    (source_type: 5) or 6p (source_type: 6). Never 2p.
    """

    # load the right coefficients: 
    if source_type == 5:
        colour = nu_eff_used_in_astrometry
        j, k, g, q_jk, n, m = j_5, k_5, g_5, q_jk5, n_5, m_5
    elif source_type == 6:
        colour = pseudocolour
        j, k, g, q_jk, n, m = j_6, k_6, g_6, q_jk6, n_6, m_6

    # basis functions evaluated at colour and ecl_lat
    c = [np.ones_like(colour),
         np.max((-0.24 * np.ones_like(colour), np.min((0.24 * np.ones_like(colour), colour - 1.48), axis=0)), axis=0),
         np.min((0.24 * np.ones_like(colour), np.max((np.zeros_like(colour), 1.48 - colour), axis=0)), axis=0) ** 3,
         np.min((np.zeros_like(colour), colour - 1.24), axis=0),
         np.max((np.zeros_like(colour), colour - 1.72), axis=0)]
    b = [np.ones_like(sinBeta), sinBeta, sinBeta ** 2 - 1. / 3]

    # coefficients must be interpolated between g(left) and g(left+1)
    # find the bin in g where gMag is
    ig = np.max((np.zeros_like(phot_g_mean_mag),
                 np.min((np.ones_like(phot_g_mean_mag) * (n - 2), np.digitize(phot_g_mean_mag, g, right=False) - 1),
                        axis=0)), axis=0).astype(int)

    # interpolate coefficients to gMag:
    h = np.max((np.zeros_like(phot_g_mean_mag),
                np.min((np.ones_like(phot_g_mean_mag), (phot_g_mean_mag - g[ig]) / (g[ig + 1] - g[ig])), axis=0)),
               axis=0)

    # sum over the product of the coefficients to get the zero-point
    zpt = np.sum([((1 - h) * q_jk[ig, i] + h * q_jk[ig + 1, i]) * c[j[i]] * b[k[i]] for i in range(m)], axis=0)

    return zpt


# Main function: calculates the zero-point for any source in the Gaia catalogue

def get_zpt(phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved,
            _warnings=True):
    """
    Returns the parallax zero point [mas] for a source of given G magnitude, effective wavenumber (nuEff) [1/micron],
    pseudocolour (pc) [1/micron], and ecl_lat [degrees]. It also needs the astrometric_params_solved to discern
    between 5-p and 6-p solutions. Valid for 5- and 6-parameter solutions with 6<G<21 and 1.1<nuEff<1.9,
    1.24<pc<1.72. Outside these ranges, the function can return a very imprecise zero-point.

    The inputs can be either floats or an iterable (ndarray, list or tuple). In case of the later, their shape must
    be the same and equal to (N,), where N is the number of sources.

    Usage: parallax_corrected = parallax_catalogue - zero_point

    Original code: @LL 2020-07-14

    NOTE: if any of the inputs values is NaN, the output will be NaN. Also, if the warnings are turned off and the
    source probided falls outside the valid range specified above, its zero-point will be NaN.

    Input:
        phot_g_mean_mag [mag]
        nu_eff_used_in_astrometry [1/micron]
        pseudocolour [1/micron]
        ecl_lat [degrees]
        astrometric_params_solved (3 -> 2p, 31 -> 5p, 95 -> 6p)

    Output:
        correction in mas (milliarcsecond, not micro).
    """
    inputs_are_floats = False

    # check availability of the tables
    try:
        global j_5, k_5, g_5, q_jk5, n_5, m_5, j_6, k_6, g_6, q_jk6, n_6, m_6
        len(g_5) + len(g_6)
    except:
        raise ValueError("The table of coefficients have not been initialized!!\n Run load_tables().")

    # check input types
    inputs = [phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved]
    inputs_names = ['phot_g_mean_mag', 'nu_eff_used_in_astrometry', 'pseudocolour', 'ecl_lat',
                    'astrometric_params_solved']
    for i, inp in enumerate(inputs):
        # first: check is not an iterable
        if not (isinstance(inp, np.ndarray) or isinstance(inp, list) or isinstance(inp, tuple)):
            # if not an iterable, has to be int or float
            if not (np.can_cast(inp, float) or np.can_cast(inp, int)):
                raise ValueError(
                    """The input '{}' is of an unknown type. 
                       Only types accepted are: float, int, ndarray, list or tuple.""".format(inputs_names[i]))

    # check coherence among inputs
    if np.isscalar(phot_g_mean_mag):
        inputs_are_floats = True
        try:
            phot_g_mean_mag = np.array([phot_g_mean_mag])
            nu_eff_used_in_astrometry = np.array([nu_eff_used_in_astrometry])
            pseudocolour = np.array([pseudocolour])
            ecl_lat = np.array([ecl_lat])
            astrometric_params_solved = np.array([astrometric_params_solved])

            for inp in [phot_g_mean_mag, nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved]:
                inp[0]
        except:
            raise ValueError("The variables are not well formated! The types are not coherent among the inputs.")

    else:
        phot_g_mean_mag = np.array(phot_g_mean_mag)
        nu_eff_used_in_astrometry = np.array(nu_eff_used_in_astrometry)
        pseudocolour = np.array(pseudocolour)
        ecl_lat = np.array(ecl_lat)
        astrometric_params_solved = np.array(astrometric_params_solved)

    if not (
            phot_g_mean_mag.shape == nu_eff_used_in_astrometry.shape == pseudocolour.shape == ecl_lat.shape ==
            astrometric_params_solved.shape):
        raise ValueError("Dimension mismatch! At least one of the inputs has a different shape than the rest.")

    # ###### HERE ALL VARIABLES SHOULD BE CORRECT ########

    # check astrometric_params_solved
    if not np.all((astrometric_params_solved == 31) | (astrometric_params_solved == 95)):
        raise ValueError(
            """Some of the sources have an invalid number of the astrometric_params_solved and are not one of the two 
            possible values (31,95). Please provide an acceptable value.""")

    # define 5p and 6p sources
    sources_5p = np.where(astrometric_params_solved == 31)
    sources_6p = np.where(astrometric_params_solved == 95)

    # check magnitude and colour ranges
    if not _warnings:
        # initialise filterning arrays
        gmag_outofrange_ind = None
        nueff_outofrange_ind = None
        pseudocolor_outofrange_ind = None

    if np.any(phot_g_mean_mag >= 21) or np.any(phot_g_mean_mag <= 6):
        if _warnings:
            warnings.warn(
                """The apparent magnitude of one or more of the sources is outside the expected range (6-21 mag). 
                Outside this range, there is no further interpolation, thus the values at 6 or 21 are returned.""",
                UserWarning)
            # raise ValueError('The apparent magnitude of the source is outside the valid range (6-21 mag)')
        else:
            if inputs_are_floats:
                return np.nan
            else:
                # return np.ones_like(phot_g_mean_mag) * np.nan
                gmag_outofrange_ind = np.where((phot_g_mean_mag >= 21) | (phot_g_mean_mag <= 6))

    if (np.any(nu_eff_used_in_astrometry[sources_5p] >= 1.9) or np.any(
            nu_eff_used_in_astrometry[sources_5p] <= 1.1)):
        if _warnings:
            warnings.warn(
                """The nu_eff_used_in_astrometry of some of the 5p source(s) is outside the expected range (1.1-1.9 
                mag). Outside this range, the zero-point calculated can be seriously wrong.""",
                UserWarning)
        else:
            if inputs_are_floats:
                return np.nan
            else:
                nueff_outofrange_ind = np.where((astrometric_params_solved == 31) & (
                        (nu_eff_used_in_astrometry >= 1.9) | (nu_eff_used_in_astrometry <= 1.1)))

    if np.any(pseudocolour[sources_6p] >= 1.72) or np.any(pseudocolour[sources_6p] <= 1.24):
        if _warnings:
            warnings.warn(
                """The pseudocolour of some of the 6p source(s) is outside the expected range (1.24-1.72 mag).
                 The maximum corrections are reached already at 1.24 and 1.72""",
                UserWarning)
        else:
            if inputs_are_floats:
                return np.nan
            else:
                pseudocolor_outofrange_ind = np.where(
                    (astrometric_params_solved == 95) & ((pseudocolour >= 1.72) | (pseudocolour <= 1.24)))

    # initialise answer
    zpt = np.zeros_like(phot_g_mean_mag)

    # compute zero-point for 5p
    zpt[sources_5p] = _calc_zpt(phot_g_mean_mag[sources_5p], nu_eff_used_in_astrometry[sources_5p],
                                pseudocolour[sources_5p], np.sin(np.deg2rad(ecl_lat[sources_5p])), 5)

    # compute zero-point for 5p
    zpt[sources_6p] = _calc_zpt(phot_g_mean_mag[sources_6p], nu_eff_used_in_astrometry[sources_6p],
                                pseudocolour[sources_6p], np.sin(np.deg2rad(ecl_lat[sources_6p])), 6)

    if inputs_are_floats:
        return np.round(zpt * 0.001, 6)[0]  # convert to mas
    else:
        zpt = np.round(zpt * 0.001, 6)  # convert to mas
        # if warnings are turned off, turn to NaN the sources out of range
        if not _warnings:
            if gmag_outofrange_ind is not None:
                zpt[gmag_outofrange_ind] = np.nan * np.ones_like(gmag_outofrange_ind)
            if nueff_outofrange_ind is not None:
                zpt[nueff_outofrange_ind] = np.nan * np.ones_like(nueff_outofrange_ind)
            if pseudocolor_outofrange_ind is not None:
                zpt[pseudocolor_outofrange_ind] = np.nan * np.ones_like(pseudocolor_outofrange_ind)
        return zpt


# A simple pandas wrapper: it should take around 2e-4 seconds/star

def zpt_wrapper(pandas_row):
    """
    Compute the parallax zero-point with get_zpt function for each row of the pandas DataFrame. It assumes that the
    DataFrame has:

    - phot_g_mean_mag: apparent magnitude in the G band
    - nu_eff_used_in_astrometry: effective wavenumber for a 5-parameter solution
    - pseudocolour: effective wavenumber for a 6-parameter solution
    - ecl_lat: ecliptic latitude in degrees
    - astrometric_params_solved (3 -> 2p, 31 -> 5p, 95 -> 6p)
    
    Errors are set to False, therefore stars that are NOT inside the valid range of the interpolators will receive a
    NaN.
    
    Example: df.apply(zpt_wrapper,axis=1)
    """

    return get_zpt(pandas_row.phot_g_mean_mag, pandas_row.nu_eff_used_in_astrometry,
                   pandas_row.pseudocolour,pandas_row.ecl_lat,
                   pandas_row.astrometric_params_solved,
                   _warnings=False)
