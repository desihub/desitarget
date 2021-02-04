#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
import operator
import collections
import numpy as np

from desitarget.cuts import shift_photo_north

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

    #shift North photometry to South photometry:
    is_north = dataArray['IS_NORTH'][:]
    print(f'[INFO] shift photometry for {is_north.sum()} objects')
    gflux[is_north], rflux[is_north], zflux[is_north] = shift_photo_north(gflux[is_north], rflux[is_north], zflux[is_north])

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
    res = (alpha2 - alpha1) * (np.sin(dec2) - np.sin(dec1)) * (180./np.pi)**2
    return res


def RA_DEC_AreaFunc(OBJ_RA, OBJ_DEC, binVect_RA, binVect_DEC, N_OBJ_th=2):
    RA_DEC_meshgrid = np.meshgrid(binVect_RA, binVect_DEC)
    OBJ_dNdRAdDEC = np.histogram2d(x=OBJ_DEC, y=OBJ_RA, bins=[binVect_DEC, binVect_RA])[0]
    med_OBJ_dNdRAdDEC = np.median(OBJ_dNdRAdDEC)
    skyArea_meshgrid = AreaFunc(RA_DEC_meshgrid[1][0:-1, 0:-1] * np.pi/180.,
                                RA_DEC_meshgrid[1][1:, 0:-1] * np.pi/180.,
                                RA_DEC_meshgrid[0][0:-1, 0:-1] * np.pi/180.,
                                RA_DEC_meshgrid[0][0:-1, 1:] * np.pi/180.)
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
