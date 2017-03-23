'''
desitarget.photo
================

Implements the photometric transforms between SDSS and DECam using g,r,z
documented in DESI-1788v1
https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1788
'''

def sdss2decam(g_sdss, r_sdss, i_sdss, z_sdss):
    '''
    Converts SDSS magnitudes to DECam magnitudes

    Args:
        [griz]_sdss: SDSS magnitudes (float or arrays of floats)

    Returns:
        g_decam, r_decam, z_decam

    Note: SDSS griz are inputs, but only grz (no i) are output
    '''
    gr = g_sdss - r_sdss
    ri = r_sdss - i_sdss
    iz = i_sdss - z_sdss

    #- DESI-1788v1 equations 4-6
    g_decals = g_sdss + 0.01684 - 0.11169*gr
    r_decals = r_sdss - 0.03587 - 0.14114*ri
    z_decals = z_sdss - 0.00756 - 0.07692*iz

    return g_decals, r_decals, z_decals

def cfht2decam(g_cfht, r_cfht, i_cfht, z_cfht):
    '''
    Converts CFHT magnitudes to DECam magnitudes

    Args:
        [griz]_cfht: CFHT magnitudes (float or arrays of floats)

    Returns:
        g_decam, r_decam, z_decam

    Note: CFHT griz are inputs, but only grz (no i) are output
    '''
    gr = g_cfht - r_cfht
    ri = r_cfht - i_cfht
    iz = i_cfht - z_cfht

    #- DESI-1788v1 equations 1-3
    g_decals = g_cfht - 0.03926 + 0.05736*gr
    r_decals = r_cfht - 0.07371 - 0.13004*ri
    z_decals = z_cfht - 0.08165 - 0.20494*iz

def decam2sdss(g_decam, r_decam, z_decam):
    '''Not yet implemented'''
    raise NotImplementedError

def decam2cfht(g_decam, r_decam, z_decam):
    '''Not yet implemented'''
    raise NotImplementedError
