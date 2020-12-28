#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d


class PredCountsFromQLF_Class():
    def __init__(self):
        self.QLF_OK = False
        self.EFF_OK = False
        self.QLF_EFF_OK = False

        # QLF
        self.QLF_nz = 0
        self.QLF_stepz = 0
        # self.QLF_tabz = None
        self.QLF_zlimit = None

        self.QLF_nmag = 0
        self.QLF_stepmag = 0
        self.QLF_tabmag = None
        self.QLF_maglimit = None

        self.QLF_dNdzdmag = None
        self.QLF_Ndzdmag = None

        # EFF
        self.EFF_zlimit = None
        self.EFF_maglimit = None
        self.EFF_dzdmag = None

        # QLF_EFF
        self.QLF_EFF_zlimit = None
        self.QLF_EFF_maglimit = None

        self.interpEFF_dzdmag = None
        self.interpQLF_dNdzdmag = None
        self.interpQLF_EFF_dNdzdmag = None

        self.QLF_EFF_dNdz = None
        self.QLF4Compl_dNdz = None
        self.Compl_dz = None

        self.QLF_EFF_dNdmag = None
        self.QLF4Compl_dNdmag = None
        self.Compl_dmag = None

        self.QLF_EFF_dNdzdmag = None
        self.QLF4Compl_dNdzdmag = None
        self.Compl_dzdmag = None

    def LoadQLF_Data(self, fpn_QLF_Data, mMzred=np.array([0., 6.]), skyArea=10000.):
        # Data loading in "dataStr"
        dataStr = np.loadtxt(fpn_QLF_Data, dtype=str, delimiter='\n')
        self.QLF_nz = len(re.findall(r'\d+(?:\.\d+)?', dataStr[0])) - 1
        self.QLF_nmag = len(dataStr)

        # ZRED
        self.QLF_zlimit = np.linspace(mMzred[0], mMzred[1], self.QLF_nz + 1, endpoint=True)
        self.QLF_stepz = self.QLF_zlimit[1] - self.QLF_zlimit[0]
        # self.QLF_tabz = self.QLF_zlimit[0:-1] + self.QLF_stepz / 2.

        self.QLF_tabmag = np.zeros(self.QLF_nmag)
        self.QLF_dNdzdmag = np.zeros((self.QLF_nmag + 1, self.QLF_nz + 1))

        for nL, line in enumerate(dataStr):
            dNdzdmag = re.findall(r'\d+(?:\.\d+)?', line)
            dNdzdmag = np.asarray(dNdzdmag).astype(np.float)
            self.QLF_tabmag[nL] = dNdzdmag[0]
            self.QLF_dNdzdmag[nL + 1, 1:] = dNdzdmag[1:]

        self.QLF_stepmag = self.QLF_tabmag[1] - self.QLF_tabmag[0]

        # MAG
        self.QLF_maglimit = np.zeros(self.QLF_nmag + 1)
        self.QLF_maglimit[0:-1] = self.QLF_tabmag - self.QLF_stepmag / 2.
        self.QLF_maglimit[-1] = self.QLF_maglimit[-2] + self.QLF_stepmag

        self.QLF_dNdzdmag /= skyArea

        self.QLF_Ndzdmag = np.cumsum(np.cumsum(
            self.QLF_dNdzdmag, axis=0), axis=1)

        self.QLF_OK = True
        self.QLF_EFF_OK = False

    def LoadEffData(self, EFFdata, EFFzlimit, EFFmaglimit):
        self.EFF_dzdmag = np.copy(EFFdata)
        self.EFF_zlimit = np.copy(EFFzlimit)
        self.EFF_maglimit = np.copy(EFFmaglimit)

        self.EFF_OK = True
        self.QLF_EFF_OK = False

    def PrelOpFunc(self):
        if self.QLF_OK & self.EFF_OK & (not self.QLF_EFF_OK):
            # QLF_EFF_zlimit
            self.QLF_EFF_zlimit = np.unique(np.hstack((self.QLF_zlimit, self.EFF_zlimit)))

            maxQLF_EFF_zlimit = min(float(np.max(self.QLF_zlimit)),
                                    float(np.max(self.EFF_zlimit)))
            minQLF_EFF_zlimit = max(float(np.min(self.QLF_zlimit)),
                                    float(np.min(self.EFF_zlimit)))
            test = (self.QLF_EFF_zlimit >= minQLF_EFF_zlimit) & \
                   (self.QLF_EFF_zlimit <= maxQLF_EFF_zlimit)

            self.QLF_EFF_zlimit = self.QLF_EFF_zlimit[test]

            # QLF_EFFmaglimit
            self.QLF_EFF_maglimit = np.unique(
                np.hstack((self.QLF_maglimit,
                           self.EFF_maglimit)))

            maxQLF_EFF_maglimit = min(float(np.max(self.QLF_maglimit)),
                                      float(np.max(self.EFF_maglimit)))
            minQLF_EFF_maglimit = max(float(np.min(self.QLF_maglimit)),
                                      float(np.min(self.EFF_maglimit)))
            test = (self.QLF_EFF_maglimit >= minQLF_EFF_maglimit) & \
                   (self.QLF_EFF_maglimit <= maxQLF_EFF_maglimit)

            self.QLF_EFF_maglimit = self.QLF_EFF_maglimit[test]

            xnew = self.QLF_EFF_zlimit
            ynew = self.QLF_EFF_maglimit

            # EFF
            x = self.EFF_zlimit.flatten()
            y = self.EFF_maglimit.flatten()
            z = self.EFF_dzdmag

# ==============================================================================
#             f2d_EFF = interp2d(x, y, z, kind = 'linear',
#                                 copy = True, bounds_error = True)
#             interpEFF_dzdmag = f2d_EFF(xnew, ynew)
# ==============================================================================

            interpXinds = np.digitize(xnew, x, right=True) - 1
            interpXinds = np.maximum(interpXinds, 0)

            interpYinds = np.digitize(ynew, y, right=True) - 1
            interpYinds = np.maximum(interpYinds, 0)

            interpXYgridInds = np.meshgrid(interpXinds, interpYinds)

            self.interpEFF_dzdmag = z[interpXYgridInds[1],
                                      interpXYgridInds[0]]

            # QLF
            x = self.QLF_zlimit.flatten()
            y = self.QLF_maglimit.flatten()
            z = self.QLF_Ndzdmag

            f2d_QLF = interp2d(x, y, z, kind='linear', copy=True, bounds_error=True)

            interpQLF_Ndzdmag = f2d_QLF(xnew, ynew)

            interpQLF_dNdzdmag = np.copy(interpQLF_Ndzdmag)
            interpQLF_dNdzdmag[:, 1:] -= np.copy(interpQLF_dNdzdmag[:, :-1])
            interpQLF_dNdzdmag[1:, :] -= np.copy(interpQLF_dNdzdmag[:-1, :])

            self.interpQLF_dNdzdmag = interpQLF_dNdzdmag

            interpQLF_EFF_dNdzdmag = np.zeros(self.interpQLF_dNdzdmag.shape)
            interpQLF_EFF_dNdzdmag = self.interpEFF_dzdmag * self.interpQLF_dNdzdmag
            self.interpQLF_EFF_dNdzdmag = interpQLF_EFF_dNdzdmag

            self.QLF_EFF_OK = True

    def ZREDComplEvalFunc(self, zlimit):
        if self.QLF_EFF_OK:
            xnew = self.QLF_EFF_zlimit
            assert(np.min(zlimit) >= np.min(xnew))
            assert(np.max(zlimit) <= np.max(xnew))

            interpQLF_dNdz = np.sum(self.interpQLF_dNdzdmag, axis=0)
            interpQLF_Ndz = np.cumsum(interpQLF_dNdz)

            # QLF_EFF dNdz
            interpQLF_EFF_dNdz = np.sum(self.interpQLF_EFF_dNdzdmag, axis=0)
            interpQLF_EFF_Ndz = np.cumsum(interpQLF_EFF_dNdz)

            f1d_QLF_EFF = interp1d(xnew, interpQLF_EFF_Ndz, kind='linear', copy=True, bounds_error=True)
            f1d_QLF = interp1d(xnew, interpQLF_Ndz, kind='linear', copy=True, bounds_error=True)

            self.QLF_EFF_dNdz = f1d_QLF_EFF(zlimit)
            self.QLF_EFF_dNdz[1:] -= np.copy(self.QLF_EFF_dNdz[:-1])

            self.QLF4Compl_dNdz = f1d_QLF(zlimit)
            self.QLF4Compl_dNdz[1:] -= np.copy(self.QLF4Compl_dNdz[:-1])

            self.Compl_dz = self.QLF_EFF_dNdz[1:] / self.QLF4Compl_dNdz[1:]
            self.Compl_dz[np.isnan(self.Compl_dz)] = 0.

            return self.Compl_dz

    def RComplEvalFunc(self, maglimit):
        if self.QLF_EFF_OK:
            ynew = self.QLF_EFF_maglimit
            assert(np.min(maglimit) >= np.min(ynew))
            assert(np.max(maglimit) <= np.max(ynew))

            interpQLF_dNdmag = np.sum(self.interpQLF_dNdzdmag, axis=1)
            interpQLF_Ndmag = np.cumsum(interpQLF_dNdmag)

            # QLF_EFF dNdmag
            interpQLF_EFF_dNdmag = np.sum(self.interpQLF_EFF_dNdzdmag, axis=1)
            interpQLF_EFF_Ndmag = np.cumsum(interpQLF_EFF_dNdmag)

            f1d_QLF_EFF = interp1d(ynew, interpQLF_EFF_Ndmag, kind='linear', copy=True, bounds_error=True)
            f1d_QLF = interp1d(ynew, interpQLF_Ndmag, kind='linear', copy=True, bounds_error=True)

            self.QLF_EFF_dNdmag = f1d_QLF_EFF(maglimit)
            self.QLF_EFF_dNdmag[1:] -= np.copy(self.QLF_EFF_dNdmag[:-1])

            self.QLF4Compl_dNdmag = f1d_QLF(maglimit)
            self.QLF4Compl_dNdmag[1:] -= np.copy(self.QLF4Compl_dNdmag[:-1])

            self.Compl_dmag = self.QLF_EFF_dNdmag[1:] / self.QLF4Compl_dNdmag[1:]
            self.Compl_dmag[np.isnan(self.Compl_dmag)] = 0.

            return self.Compl_dmag

    def R_ZREDComplEvalFunc(self, zlimit, maglimit):
        if self.QLF_EFF_OK:
            xnew = self.QLF_EFF_zlimit
            assert(np.min(zlimit) >= np.min(xnew))
            assert(np.max(zlimit) <= np.max(xnew))

            ynew = self.QLF_EFF_maglimit
            assert(np.min(maglimit) >= np.min(ynew))
            assert(np.max(maglimit) <= np.max(ynew))

            interpQLF_EFF_Ndzdmag = np.cumsum(np.cumsum(self.interpQLF_EFF_dNdzdmag, axis=0), axis=1)

            f2d_QLF_EFF = interp2d(xnew, ynew, interpQLF_EFF_Ndzdmag, kind='linear', copy=True, bounds_error=True)

            QLF_EFF4Compl_Ndzdmag = f2d_QLF_EFF(zlimit, maglimit)

            QLF_EFF4Compl_dNdzdmag = np.copy(QLF_EFF4Compl_Ndzdmag)
            QLF_EFF4Compl_dNdzdmag[:, 1:] -= np.copy(QLF_EFF4Compl_dNdzdmag[:, :-1])
            QLF_EFF4Compl_dNdzdmag[1:, :] -= np.copy(QLF_EFF4Compl_dNdzdmag[:-1, :])

            self.QLF_EFF4Compl_dNdzdmag = QLF_EFF4Compl_dNdzdmag

            # QLF
            interpQLF_Ndzdmag = np.cumsum(np.cumsum(self.interpQLF_dNdzdmag, axis=0), axis=1)

            f2d_QLF = interp2d(xnew, ynew, interpQLF_Ndzdmag, kind='linear', copy=True, bounds_error=True)

            QLF4Compl_Ndzdmag = f2d_QLF(zlimit, maglimit)

            QLF4Compl_dNdzdmag = np.copy(QLF4Compl_Ndzdmag)
            QLF4Compl_dNdzdmag[:, 1:] -= np.copy(QLF4Compl_dNdzdmag[:, :-1])
            QLF4Compl_dNdzdmag[1:, :] -= np.copy(QLF4Compl_dNdzdmag[:-1, :])

            self.QLF4Compl_dNdzdmag = QLF4Compl_dNdzdmag

            self.Compl_dzdmag = self.QLF_EFF4Compl_dNdzdmag[1:, 1:] / self.QLF4Compl_dNdzdmag[1:, 1:]
            self.Compl_dzdmag[np.isnan(self.Compl_dzdmag)] = 0.

            return self.Compl_dzdmag

    def R_ZRED_EffVarEvalFunc(self, OBJ_QSO_dNdzdmag):
        self.EffVar4Compl_dzdmag = None
        self.Eff4Compl_dzdmag = np.copy(self.Compl_dzdmag)

        if True:
            self.EffVar4Compl_dzdmag = self.Eff4Compl_dzdmag * (1. - self.Eff4Compl_dzdmag)
            self.EffVar4Compl_dzdmag /= OBJ_QSO_dNdzdmag
            self.EffVar4Compl_dzdmag[OBJ_QSO_dNdzdmag == 0.] = 0.
        else:
            self.Count4Complt_Ndzdmag = self.Eff4Compl_dzdmag * OBJ_QSO_dNdzdmag
            self.EffVar4Compl_dzdmag = OBJ_QSO_dNdzdmag - self.Count4Complt_Ndzdmag + 1.
            self.EffVar4Compl_dzdmag *= self.Count4Complt_Ndzdmag + 1.
            self.EffVar4Compl_dzdmag /= (OBJ_QSO_dNdzdmag + 2)**2 * (OBJ_QSO_dNdzdmag + 3)
            self.EffVar4Compl_dzdmag[OBJ_QSO_dNdzdmag == 0.] = 0.

        return self.EffVar4Compl_dzdmag

    def ZRED_EffVarEvalFunc(self):
        self.EffVar4Compl_dz = self.EffVar4Compl_dzdmag * (self.QLF4Compl_dNdzdmag[1:, 1:])**2

        self.EffVar4Compl_dz = np.sum(self.EffVar4Compl_dz, axis=0)
        tmp_var = np.sum(self.QLF4Compl_dNdzdmag[1:, 1:], axis=0)**2
        self.EffVar4Compl_dz /= tmp_var

        self.EffVar4Compl_dz[tmp_var == 0.] = 0.

        return self.EffVar4Compl_dz

    def R_EffVarEvalFunc(self):
        self.EffVar4Compl_dmag = self.EffVar4Compl_dzdmag * (self.QLF4Compl_dNdzdmag[1:, 1:])**2

        self.EffVar4Compl_dmag = np.sum(self.EffVar4Compl_dmag, axis=1)
        tmp_var = np.sum(self.QLF4Compl_dNdzdmag[1:, 1:], axis=1)**2
        self.EffVar4Compl_dmag /= tmp_var

        self.EffVar4Compl_dmag[tmp_var == 0.] = 0.

        return self.EffVar4Compl_dmag
