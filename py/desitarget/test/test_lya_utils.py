# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.lya_utils.
"""
import os
import unittest
import numpy as np
from astropy.table import Table

from desitarget.lya_utils import *

class TestLYA(unittest.TestCase):
    
    def test_qso_weight(self):
        W, r_edges, r_vec, z_edges, z_vec = load_weights()
        self.assertTrue(np.all(np.diag(W) == qso_weight(z_vec,r_vec))) 
        self.assertTrue(W[0][0] == qso_weight(z_vec[0],r_vec[0]))

    def test_lya_priority(self):
        redshift = np.array([2.6, 1.0, 0.5, 1.0])
        rmag = np.array([19.0, 24.0, 23.5, 19.0]) 
        priorities = np.array([3500, 2, 2, 2])
        print(qso_weight(redshift, rmag))
        print(lya_priority(redshift, rmag))
        self.assertTrue(np.all(priorities == lya_priority(redshift, rmag)))
                    
if __name__ == '__main__':
    unittest.main()

def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_lya_utils
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
