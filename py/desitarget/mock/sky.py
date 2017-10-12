import desimodel.io
import desimodel.footprint
import numpy as np
import healpy as hp

def _random_theta_phi(nside, pix):
    theta, phi = hp.pix2ang(nside, pix, nest=True)
    dpix = np.sqrt(hp.nside2pixarea(nside))
    theta += np.random.uniform(-dpix/2, dpix/2, size=len(theta))
    phi += np.random.uniform(-dpix/2, dpix/2, size=len(phi)) * np.cos(np.pi/2 - theta)
    return theta % np.pi, phi % (2*np.pi)

def random_sky(nside=2048, tiles=None, maxiter=20):
    '''
    Returns sky locations within healpixels covering tiles

    Options:
        nside (int): healpixel nside; coverage is uniform at this scale
        tiles: DESI tiles to cover, for desimodel.footprint.tiles2pix()
        maxiter (int): maximum number of iterations to ensure coverage

    Generates sky locations that are more uniform than true random,
    such that every healpixel has a point within it.  Note that this
    should *not* be used for mock randoms.
    
    nside=2048 corresponds to about half of a DESI positioner patrol area
    and results in ~18M sky locations over the full footprint.
    '''
    if tiles is None:
        tiles = desimodel.io.load_tiles()
    pix = desimodel.footprint.tiles2pix(nside, tiles)
    theta, phi = _random_theta_phi(nside, pix)

    #- there is probably a more clever way to do this, but iteratively
    #- replace points that scatter outside their original healpixel until
    #- all healpixels are covered
    for i in range(maxiter):
        skypix = hp.ang2pix(nside, theta, phi, nest=True)
        missing = np.in1d(pix, skypix, invert=True)
        ii = np.where(missing)[0]
        if len(ii) == 0:
            break

        tx, px = _random_theta_phi(nside, pix[ii])
        theta[ii] = tx
        phi[ii] = px

    ra = np.degrees(phi)
    dec = 90 - np.degrees(theta)

    return ra, dec
