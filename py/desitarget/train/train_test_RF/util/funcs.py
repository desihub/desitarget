#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
import operator
import collections
import numpy as np

# ------------------------------------------------------------------------------
# ***Fonction qui produit un string formaté pour afficher une durée à partir
# d'une quantité donnée en secondes***


def Time2StrFunc(tm):

    tmStr = []
    var_tm = divmod(tm, 60.)
    # [ms]
    if divmod(tm, 1.)[1] > 0.:
        tmStr.append("{:d} ms".format(int(divmod(tm, 1.)[1] * 1000.)))
    # [s]
    if int(var_tm[1]) > 0:
        tmStr.append("{:d} sec".format(int(var_tm[1])))

    var_tm = divmod(var_tm[0], 60.)
    # [min]
    if int(var_tm[1]) > 0:
        tmStr.append("{:d} min".format(int(var_tm[1])))
    # [hour]
    if int(var_tm[0]) > 0:
        tmStr.append("{:d} h".format(int(var_tm[0])))

    tmStr = tmStr[::-1]
    tmStr = reduce(lambda a, b: a + ' - ' + b, tmStr)

    return tmStr


def shift_photo_north(gflux=None, rflux=None, zflux=None):
    """Convert fluxes in the northern (BASS/MzLS) to the southern (DECaLS) system.
    Parameters
    ----------
    gflux, rflux, zflux : :class:`array_like` or `float`
        The flux in nano-maggies of g, r, z bands.
    Returns
    -------
    The equivalent fluxes shifted to the southern system.
    Notes
    -----
    - see also https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=3390;filename=Raichoor_DESI_05Dec2017.pdf;version=1
    """
    # ADM only use the g-band color shift when r and g are non-zero
    gshift = gflux * 10**(-0.4*0.013)
    w = np.where((gflux != 0) & (rflux != 0))
    gshift[w] = (gflux[w] * 10**(-0.4*0.013) * (gflux[w]/rflux[w])**complex(-0.059)).real

    # ADM only use the r-band color shift when r and z are non-zero
    # ADM and only use the z-band color shift when r and z are non-zero
    w = np.where((rflux != 0) & (zflux != 0))
    rshift = rflux * 10**(-0.4*0.007)
    zshift = zflux * 10**(+0.4*0.022)

    rshift[w] = (rflux[w] * 10**(-0.4*0.007) * (rflux[w]/zflux[w])**complex(-0.027)).real
    zshift[w] = (zflux[w] * 10**(+0.4*0.022) * (rflux[w]/zflux[w])**complex(+0.019)).real

    return gshift, rshift, zshift


def _Flux2MagFunc(OBJ_Data_flux, OBJ_Data_mw_transmission):

    OBJ_flux = OBJ_Data_flux / OBJ_Data_mw_transmission
    limitInf = 1.e-04
    OBJ_flux = OBJ_flux.clip(limitInf)
    OBJ_mag = np.where(OBJ_flux > limitInf, 22.5 - 2.5 * np.log10(OBJ_flux), 0.)
    OBJ_mag[np.isnan(OBJ_mag)] = 0.
    OBJ_mag[np.isinf(OBJ_mag)] = 0.

    return OBJ_mag


def Flux2MagFunc(dataArray):

    gflux = dataArray.FLUX_G[:]/dataArray.MW_TRANSMISSION_G[:]
    rflux = dataArray.FLUX_R[:]/dataArray.MW_TRANSMISSION_R[:]
    zflux = dataArray.FLUX_Z[:]/dataArray.MW_TRANSMISSION_Z[:]
    W1flux = dataArray.FLUX_W1[:]/dataArray.MW_TRANSMISSION_W1[:]
    W2flux = dataArray.FLUX_W2[:]/dataArray.MW_TRANSMISSION_W2[:]

    limitInf = 1.e-04
    gflux = gflux.clip(limitInf)
    rflux = rflux.clip(limitInf)
    zflux = zflux.clip(limitInf)
    W1flux = W1flux.clip(limitInf)
    W2flux = W2flux.clip(limitInf)

    g = np.where(gflux > limitInf, 22.5-2.5*np.log10(gflux), 0.)
    r = np.where(rflux > limitInf, 22.5-2.5*np.log10(rflux), 0.)
    z = np.where(zflux > limitInf, 22.5-2.5*np.log10(zflux), 0.)
    W1 = np.where(W1flux > limitInf, 22.5-2.5*np.log10(W1flux), 0.)
    W2 = np.where(W2flux > limitInf, 22.5-2.5*np.log10(W2flux), 0.)

    g[np.isnan(g)] = 0.
    g[np.isinf(g)] = 0.
    r[np.isnan(r)] = 0.
    r[np.isinf(r)] = 0.
    z[np.isnan(z)] = 0.
    z[np.isinf(z)] = 0.
    W1[np.isnan(W1)] = 0.
    W1[np.isinf(W1)] = 0.
    W2[np.isnan(W2)] = 0.
    W2[np.isinf(W2)] = 0.

    return g, r, z, W1, W2


def ColorsFunc(nbEntries, nfeatures, g, r, z, W1, W2):

    colors = np.zeros((nbEntries, nfeatures))

    colors[:, 0] = g-r
    colors[:, 1] = r-z
    colors[:, 2] = g-z
    colors[:, 3] = g-W1
    colors[:, 4] = r-W1
    colors[:, 5] = z-W1
    colors[:, 6] = g-W2
    colors[:, 7] = r-W2
    colors[:, 8] = z-W2
    colors[:, 9] = W1-W2
    colors[:, 10] = r

    return colors


def GetColorsFunc(data, color_names):

    colors = np.zeros((len(data), len(color_names)))

    for i, col_name in enumerate(color_names):

        colors[:, i] = data[col_name]

    return colors


def AreaFunc(dec1, dec2, alpha1, alpha2):
    res = (alpha2 - alpha1) * (np.sin(dec2) - np.sin(dec1)) * (180. / np.pi)**2
    return res


def RA_DEC_AreaFunc(OBJ_RA, OBJ_DEC, binVect_RA, binVect_DEC, N_OBJ_th=2):

    RA_DEC_meshgrid = np.meshgrid(binVect_RA, binVect_DEC)

    OBJ_dNdRAdDEC = np.histogram2d(x=OBJ_DEC, y=OBJ_RA,
                                   bins=[binVect_DEC, binVect_RA])[0]

    med_OBJ_dNdRAdDEC = np.median(OBJ_dNdRAdDEC)

    skyArea_meshgrid = AreaFunc(RA_DEC_meshgrid[1][0:-1, 0:-1] * np.pi / 180.,
                                RA_DEC_meshgrid[1][1:, 0: -1] * np.pi / 180.,
                                RA_DEC_meshgrid[0][0:-1, 0: -1] * np.pi / 180.,
                                RA_DEC_meshgrid[0][0:-1, 1:] * np.pi / 180.)

    area_OK = (OBJ_dNdRAdDEC) >= N_OBJ_th
    res_area = np.sum(skyArea_meshgrid[area_OK])

    return res_area, med_OBJ_dNdRAdDEC


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def speGetFromDict(dataDict, tag):
    mapList = tag.split(':')
    return getFromDict(dataDict, mapList)


def speSetInDict(dataDict, tag, value):
    mapList = tag.split(':')
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def RecHyParamDictExplFunc(hyParamDict):
    # 'RecursiveHyParamDictExplorationFunc' = 'Tree2FlatHyParamDictConversionFunc'

    new_dict = collections.OrderedDict()

    for key, value in hyParamDict.items():

        if (type(value) == dict) or (type(value) == collections.OrderedDict):

            _dict = collections.OrderedDict([
              (':'.join([key, _key]), _value)
              for _key, _value in RecHyParamDictExplFunc(value).items()])

            new_dict.update(_dict)

        else:
            new_dict[key] = value

    return new_dict


def Flat2TreeHyParamConvFunc(coords, hyParamDictTemplate, hyParamSpaceTags, hyParamSpaceItems):
    # 'Flat2TreeHyParamDictConversionFunc'

    for it, indEl in enumerate(coords):

        tag = hyParamSpaceTags[it]
        value = hyParamSpaceItems[it][indEl]

        speSetInDict(hyParamDictTemplate, tag, value)

    return hyParamDictTemplate


def proba_rmag_func(OBJ_rmag, proba_thold=0.5, rmag_thold=[20.], slope=[0.05]):

    n_segments = len(rmag_thold)
    if n_segments > 1:
        test = np.array(rmag_thold)
        test = np.sort(test) == test
        if not(np.all(test)):
            assert(False), "ATTENTION : rmag_thold non ordonné croissant"

    proba_thold_rmag = np.ones(OBJ_rmag.size) * proba_thold
    tmp = np.copy(proba_thold)

    for num_seg in range(n_segments):

        test_rmag_thold = OBJ_rmag > rmag_thold[num_seg]

        if num_seg < (n_segments - 1):

            test = OBJ_rmag > rmag_thold[num_seg + 1]
            test_rmag_thold &= ~test
            tmp -= (rmag_thold[num_seg + 1] - rmag_thold[num_seg]) * slope[num_seg]
            proba_thold_rmag[test] = tmp

        proba_thold_rmag[test_rmag_thold] -= (
            (OBJ_rmag[test_rmag_thold] - rmag_thold[num_seg]) * slope[num_seg])

    return proba_thold_rmag
