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
import fitsio
from glob import glob
from astropy.coordinates import SkyCoord
from astropy import units as u

from .. import __version__ as desitarget_version
from .. import gitversion

def within(testverts,v):
    """Check whether a test point (cartesian vector) lies within a spherical triangle (3 cartesian vectors)
    
    Parameters
    ----------
    testverts : :class:`float`
        An array of 3 numpy arrays, each of which is a Cartesian representation of the vertex of a spherical triangle
    v : :class:`float`
        A numpy array representing a point on the sphere (as a Cartesian vector)

    Returns
    -------
    True if the point represented by v is inside of the spherical triangle represented by testverts. False otherwise
    """

    #ADM The algorithm is to check the direction of the projection (dot product) of the test
    #ADM vector onto each vector normal (cross product) to the geodesics (planes) that
    #ADM represent the sides of the triangle.

    vertperm = [0,1,2,0]

    for i in range(3):
        vert1 = vertperm[i]
        vert2 = vertperm[i+1]
        test = np.dot(np.cross(testverts[vert1],testverts[vert2]),v)
        if test < -1.0e-15:
            return False

    return True

    
def initri():
    """Initializer that returns a dictionary of the 8 initial (level 1) HTM nodes
                                                                                                                                                      
    Parameters 
    ----------
    None

    Returns
    -------
    A dictionary that contains the names and (cartesian) vertices of level 1 of the HTM tree
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

    """Return the correctly ordered (anti-colockwise) four child nodes of an HTM node

    Parameters
    ----------
    vert : :class:`float array`
       An array of three 3 vectors representing the Cartesian coordinates of the vertices of an HTM node
       can be N-dimensional, e.g.

       array([[[x1,  y1,  z1],       Vertex 1 of first Triangle
               [x2,  y2,  z2],       Vertex 2 of first Triangle
               [x3,  y3,  z3]],      Vertex 3 of first Triangle

              [[Nx1, Ny1,  Mz1],     Vertex 1 of Nth Triangle
               [Nx2, Ny2,  Nz2],     Vertex 2 of Nth Triangle
               [Nx3, Ny3,  Nz3]]])   Vertex 3 of Nth Triangle
           
    Returns
    -------
    :class:`float`
       A dictionary containing the vertices of the 4 child nodes, three vertices in (x,y,z) form for each child node.
       The keys of the dictionary are the HTMid extension for that child  node ('0','1','2' or '3')
    """

    childverts = {}

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
    childverts[0] = np.hstack(np.array([vert[:,0],C,B])).reshape(2,3,3)
    childverts[1] = np.hstack(np.array([vert[:,1],A,C])).reshape(2,3,3)
    childverts[2] = np.hstack(np.array([vert[:,2],B,A])).reshape(2,3,3)
    childverts[3] = np.hstack(np.array([A,B,C])).reshape(2,3,3)

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
    :class:`char` or `int``
        The HTM pixels corresponding to the passed RA/Dec at the requisite level. Will be the same
        length as length of ra and dec
    """

    #ADM if inputs are single floats, convert to arrays
    if type(ra) == type(float()):
        ra = np.array([ra])
        dec = np.array([dec])

    #ADM convert input spherical coordinates to Cartesian
    v = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    v.representation = 'cartesian'

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

    vecnodenumber = (v.x > 0)*2**2 + (v.y > 0)*2**1 + (v.z > 0)*2**0
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

        for i in testverts:
            if vec.within(testverts[i],v):
                vert = testverts[i]
                index = str(i)
                break
            if i == 2:
                vert = testverts[3]
                index = "3"
                break

        #ADM We check the triangles in order T0, T1, T2, T3 and break at the
        #ADM first spherical triangle that contains the point of interest

        desig.append(index)
        count +=1

    return "".join(desig), v


