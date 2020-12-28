#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import BallTree


class RA_DEC_MatchingClass():
    def __init__(self):
        self.RA_CatalogData = None
        self.DEC_CatalogData = None

        self.RA4MatchingData = None
        self.DEC4MatchingData = None

        self.nNeighResInd = None

    def LoadRA_DEC_CatalogData(self, RA, DEC):
        self.RA_CatalogData = RA.flatten()
        self.DEC_CatalogData = DEC.flatten()

    def LoadRA_DEC4MatchingData(self, RA, DEC):
        self.RA4MatchingData = RA.flatten()
        self.DEC4MatchingData = DEC.flatten()

    def __call__(self, radius=1./60., k_nNeigh=1):
        # Métrique choisie de 'minkowski' d'ordre 2 = métrique euclidienne
        # (Dans l'idéal, métrique de 'haversine' car coordonnées sphériques)
        nNeighObj = BallTree(
            np.array([self.RA_CatalogData, self.DEC_CatalogData]).T,
            leaf_size=40, metric='minkowski', p=2)

        nNeighResInd = nNeighObj.query_radius(
            np.array([self.RA4MatchingData, self.DEC4MatchingData]).T,
            r=radius, return_distance=True, count_only=False,
            sort_results=True)[0]

        # Traçabilité
        self.nNeighResInd = np.hstack([np.array([
            el[0: k_nNeigh], np.repeat(i, el[0: k_nNeigh].size)])
            if el[0: k_nNeigh].size > 0
            else np.array([[-1], [i]])
            for i, el in enumerate(nNeighResInd)])
