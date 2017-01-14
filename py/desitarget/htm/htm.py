# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
==============
desitarget.htm
==============

All-Python module for performin HTM look-ups
See here for HTM: http://www.skyserver.org/htm/
"""
from __future__ import (absolute_import, division)
#
from time import time
import numpy as np
from numpy.core.umath_tests import inner1d
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy import units as u

from .. import __version__ as desitarget_version
from .. import gitversion

def char2int(ID):

    """Convert an array of HTM IDs in character format integers
    (see documentation below for onechar2int, the non-array version)

    Parameters
    ----------
    ID :class:`str array` or `str`
       An array of HTM IDs in character format (will also accept an
       individual string to act as an overall wrapper on onechar2int)

    Returns
    -------
    :class:`int`
       The corresponding HTM ID(s) in integer format
    
    """
    #ADM check if an individual string was passed, if so default
    #ADM to non-array version of code
    if isinstance(ID, str):
        return np.array([onechar2int(ID)])
    else:
        return np.array([ onechar2int(i) for i in ID ])


def onechar2int(ID):

    """Convert an HTM ID in character format to a unique corresponding integer

    Parameters
    ----------
    ID :class:`char` 
       An HTM ID in character format,  e.g. 'N333133130'

    Returns
    -------
    :class:`int`
       The corresponding HTM ID in integer format, e.g. 1046492
    
    Caveats
    -------
    This is a trusting routine - it will, without flagging an
    error, return a value for any input string but will only be
    meaningful for HTM index-strings.

    Notes
    -----
    Successive HTMIDs at each tree level have successive integer codes.
    This is useful for sorting the tree, e.g.:

    N333133130 --> 1046492          S00 --> 32
    N333133131 --> 1046493          N33 --> 63
    N333133132 --> 1046494         S000 --> 128
    N333133133 --> 1046495         N333 --> 255
    N333133200 --> 1046496        S0000 --> 512
    N333133201 --> 1046497        N3333 --> 1023

    """

    #ADM The first digit in the binary HTM representation is always 1
    binrep = '1'

    #ADM If the first letter in the ID is 'N', then the second digit in the
    #ADM binary representation is also a "1", otherwise it's a "0"
    binrep += str((ID[0] == 'N')*1)

    #ADM For each ID character (0,1,2 or 3) find the two digits in its binary 
    #ADM representation, concatenate them, and append them to binrep
    binrep += "".join([ '{:02b}'.format(int(num)) for num in ID[1:] ])
    
    #ADM Now we have the binary number that corresponds to the HTM ID, expressed
    #ADM as a string. All that's left to do is to turn it into an integer
    #ADM First turn it into a numpy array via a list...
    binval = np.array([ int(s) for s in binrep ])

    #ADM ...then x by the appropriate power of 2 and sum to get an integer
    binexp = 2**np.arange(2*len(ID))[::-1]

    return np.sum(binval*binexp)

def approx_area(level):
    """Return the APPROXIMATE area of an HTM pixel at a given level of the tree

    Parameters
    ----------
    level : :class:`int`
       Level of the HTM quad-tree (8 initial pixels returned by initri is level 0
       and each level below that splits the pixels into 4 children.

    Returns
    -------
    Approximate area of a single pixel in square degrees at the input level.
    It's approximate because child pixels are not quite equal-area

    """

    spharea = 4.*180.*180./np.pi
    return (spharea/8./4.**level)


def approx_resolution(level):
    """Return the APPROXIMATE resolution of an HTM pixel at a given level of the tree

    Parameters
    ----------
    level : :class:`int`
       Level of the HTM quad-tree (8 initial pixels returned by initri is level 0
       and each level below that splits the pixels into 4 children.

    Returns
    -------
    Approximate resolution as the SIDE length of a single pixel in degrees at the input level. It's
    approximate because child pixels are not quite equal-area and spherical curvature is ignored

    """
    
    area = approx_area(level)
    fac = 4./np.sqrt(3)
    return np.sqrt(fac*area)


def within(testverts,v):
    """Check whether a test point (cartesian vector) lies within a spherical triangle (3 cartesian vectors)
    
    Parameters
    ----------
    testverts : :class:`float array`
       An array of three 3 vectors representing the Cartesian coordinates of the vertices of 
       a spherical triangle can be N-dimensional, e.g.

       array([[[x1,  y1,  z1],       Vertex 1 of first Triangle
               [x2,  y2,  z2],       Vertex 2 of first Triangle
               [x3,  y3,  z3]],      Vertex 3 of first Triangle

              [[Nx1, Ny1,  Mz1],     Vertex 1 of Nth Triangle
               [Nx2, Ny2,  Nz2],     Vertex 2 of Nth Triangle
               [Nx3, Ny3,  Nz3]]])   Vertex 3 of Nth Triangle

    v : :class:`float array'`
        A numpy array representing a point on the sphere (as a Cartesian vector, e.g.

       array([[[x1,  y1,  z1],       Vector 1

               [xN,  yN,  zN]]])     Vector N

        Note that v must have the same dimension as testverts (each point is checked row-by-row against its triangle)

    Returns
    -------
    True if the point represented by v is inside of the spherical triangle represented by testverts. False otherwise.
    Returns an array of Trues and Falses if testverts and v were N-dimensional
    
    """

    #ADM an array of Trues and Falses for the output. Default to True.
    boolwithin = np.ones(len(v),dtype='bool')

    #ADM The algorithm is to check the direction of the projection (dot product) of the test
    #ADM vector onto each vector normal (cross product) to the geodesics (planes) that
    #ADM represent the sides of the triangle.

    vertperm = [0,1,2,0]

    for i in range(3):
        vert1 = vertperm[i]
        vert2 = vertperm[i+1]
        #ADM the inner1d function performs a row-by-row dot product
        test = inner1d(np.cross(testverts[:,vert1],testverts[:,vert2]),v)
        w = np.where(test < 0.0)
        boolwithin[w] = False

    return boolwithin

    
def initri():
    """Initializer that returns a dictionary of the 8 initial (level 1) HTM nodes

    Parameters 
    ----------
    None

    Returns
    -------
    A dictionary that contains the names and (Cartesian) vertices of level 1 of the HTM tree
    """

    #ADM The six possible vertices of the initial eight HTM spherical triangles are
    A = np.array([0.,0.,1.])
    B = np.array([1.,0.,0.])
    C = np.array([0.,1.,0.])
    D = np.array([-1.,0.,0.])
    E = np.array([0.,-1.,0.])
    F = np.array([0.,0.,-1.])

    #ADM Construct a dictionary of the vertices of the eight initial nodes 
    verts = {}

    verts["S0"] = B,F,C                # S0 triangle's vertices
    verts["S1"] = C,F,D                # S1 triangle's vertices
    verts["S2"] = D,F,E                # S2 triangle's vertices
    verts["S3"] = E,F,B                # S3 triangle's vertices
    verts["N0"] = B,A,E                # N0 triangle's vertices
    verts["N1"] = E,A,D                # N1 triangle's vertices
    verts["N2"] = D,A,C                # N2 triangle's vertices
    verts["N3"] = C,A,B                # N3 triangle's vertices                                                                                    

    return verts


def childnode(vert):
    """Return the correctly ordered (anti-clockwise) four child nodes of an HTM node

    Parameters
    ----------
    vert : :class:`float array`
       An array of three 3 vectors representing the Cartesian coordinates of the vertices of an HTM triangular node
       can be N-dimensional, e.g.

       array([[[x1,  y1,  z1],       Vertex 1 of first Triangle
               [x2,  y2,  z2],       Vertex 2 of first Triangle
               [x3,  y3,  z3]],      Vertex 3 of first Triangle

              [[Nx1, Ny1,  Mz1],     Vertex 1 of Nth Triangle
               [Nx2, Ny2,  Nz2],     Vertex 2 of Nth Triangle
               [Nx3, Ny3,  Nz3]]])   Vertex 3 of Nth Triangle
           
    Returns
    -------
    :class:`float array`
       A dictionary containing the vertices of the 4 child nodes, three vertices in (x,y,z) form for each child node.
       The keys of the dictionary are the HTMid extension for that child  node ('0','1','2' or '3')
       Each of the dictionary values has the same format as the input for vert (explained under 'Parameters')
    """

    childverts = {}
    npix = len(vert)

    #ADM Find the midpoint vectors of the parent triangle...
    A = vert[:,1] + vert[:,2]
    B = vert[:,0] + vert[:,2]
    C = vert[:,0] + vert[:,1]
    #ADM ...and normalize these vectors so that they're on the unit sphere
    A /= np.linalg.norm(vert[:,1] + vert[:,2],axis=1,keepdims=True)
    B /= np.linalg.norm(vert[:,0] + vert[:,2],axis=1,keepdims=True)
    C /= np.linalg.norm(vert[:,0] + vert[:,1],axis=1,keepdims=True)
    
    #ADM Now compile test progeny triangles from the midpoints and vertices of the parent triangle
    #ADM and put them in a dictionary where the key is the name of the node
    #ADM The reshape and hstack-ing is just to format the triangles the same as the input format
    childverts[0] = np.hstack(np.array([vert[:,0],C,B])).reshape(npix,3,3)
    childverts[1] = np.hstack(np.array([vert[:,1],A,C])).reshape(npix,3,3)
    childverts[2] = np.hstack(np.array([vert[:,2],B,A])).reshape(npix,3,3)
    childverts[3] = np.hstack(np.array([A,B,C])).reshape(npix,3,3)

    return childverts


def lookup(ra,dec,level=20,charpix=True):
    """Return the HTM pixel for a given RA/Dec

    Parameters
    ----------
    ra : :class:`float` or `array`
        A Right Ascension in degrees (can also be a numpy array with multiple values)
    dec : :class:`float` or `array`
        A Declination in degrees (can also be a numpy array with multiple values)
    level : :class:`int`, optional
        Which level of the HTM tree to pixelize down to
    charpix : :class:`bool`, optional, defaults to True
        If True, return pixels in character format, otherwise return them in integer format

    Returns
    -------
    :class:`char array` or `int array`
        The HTM pixels corresponding to the passed RA/Dec at the requisite level. Will be the same
        length as length of ra and dec

    Timing Notes
    ------------
    
    Performs about 16000 level 20 (resolution ~ 0.4 arcsec) lookups per second on a NERSC login node
    Performs about 25000 level 13 (resolution ~ 0.8 arcmin) lookups per second on a NERSC login node
    Is about a factor of 3 slower for charpix = False

    """

    t0 = time()

    #ADM convert input spherical coordinates to Cartesian
    v = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    v.representation = 'cartesian'

    #ADM convert SkyCoord object to numpy array for speedier manipulation
    #ADM .T is the transpose attribute
    #ADM this conversion also allows either single floats or longer arrays to be passed
    v = np.vstack(np.array([v.x.value,v.y.value,v.z.value])).T

    #ADM we begin to hit 64-bit floating point issues at level 25 but this is small enough for
    #ADM most applications (at level 25 a spherical triangle's longest side is ~1/100 arcsec)
    if level > 25:
        print("WARNING: Module htm.htm.py: pixels too small for 64-bit floats")
        print("LEVEL WILL BE SET TO 25")
        level = 25
    
    testverts = initri()

    #ADM assign the initial node using a bitmask. For example:
    #ADM vx > 0 + vy >= 0 + vz >= 0 = 2^2 + 2^1 + 2^0 = 7 is parent node N3
    #ADM vx > 0 + vy < 0 + vz >=0 = 2^2 + 0*2^1 + 2^0 = 5 is parent node N0

    vecnodenumber = (v[:,0] > 0)*2**2 + (v[:,1] > 0)*2**1 + (v[:,2] > 0)*2**0
    nodenames = np.array(["S2","N1","S1","N2","S3","N0","S0","N3"])

    #ADM these are the name of the parent node and the vertices of the parent nodes
    desig = np.array(nodenames[vecnodenumber])
    vert = np.array([ testverts[i] for i in desig ])

    count = 1

    while count <= level:
        #ADM Knowing the "parent" spherical triangle, we loop through to the
        #ADM desired level by breaking into 4 smaller triangles with vertices
        #ADM defined by the midpoints of the parent, find which triangle the
        #ADM point is in and continue looping until the desired level is reached

        #ADM First, compile test progeny triangles from the midpoints and
        #ADM vertices of the parent triangle
        
        testverts = childnode(vert)
        
        #ADM Now we test which of these triangles our point "v" is in

        #ADM for speed-up don't test 0th triangle. If the point is not in the other
        #ADM three then by definition it has to be in the 0th
        index = np.chararray(len(desig),unicode=True)
        index[:] = '0'
        #ADM default to zeroth triangle. This will update with the actual pixel
        #ADM that the point is in if those nodes are triangle 1, 2 or 3
        vert = testverts[0]
        for i in range(1,4):
            w = np.where(within(testverts[i],v))
            #ADM update
            vert[w] = testverts[i][w]
            index[w] = str(i)

        #ADM We check the triangles in order T0, T1, T2, T3 and break at the
        #ADM first spherical triangle that contains the point of interest

        desig = desig + index
        count +=1

    #ADM if integer format was specified for the pixels, convert to integer format
    if not charpix:
        desig = char2int(desig)

    print('{} HTM lookups at level {} in {:.2f}s'.format(len(ra), level, time()-t0))

    return desig


