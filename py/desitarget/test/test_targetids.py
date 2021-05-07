# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desitarget.targets targetid encode/decode
"""
import unittest
import numpy as np

from desitarget.targets import (encode_negative_targetid,
                                decode_negative_targetid)


class TestTargetID(unittest.TestCase):

    @classmethod
    def setUp(self):
        pass

    def test_encode_negative_targetid(self):
        """
        Test encoding ra,dec -> negative TARGETID
        """
        # Edge cases with scalars; should alsways be negative.
        for ra in (0, 90, 180, 360):
            for dec in (-90, 0, +90):
                for group in (1, 7, 15):
                    t = encode_negative_targetid(ra, dec, group)
                    msg = f'targetid({ra},{dec},{group})'
                    self.assertLess(t, 0, msg)
                    self.assertTrue(np.isscalar(t), msg)

        # Also works with lists and arrays.
        ra = (0, 10, 20)
        dec = (-10, 0, 89.9)
        t = encode_negative_targetid(ra, dec)
        self.assertEqual(len(t), len(ra))
        self.assertTrue(np.all(t < 0))

        t = encode_negative_targetid(np.asarray(ra), np.asarray(dec))
        self.assertEqual(len(t), len(ra))
        self.assertTrue(np.all(t < 0))

        # Test invalid group numbers.
        with self.assertRaises(ValueError):
            encode_negative_targetid(0, 0, 0)

        with self.assertRaises(ValueError):
            encode_negative_targetid(0, 0, 16)

        with self.assertRaises(ValueError):
            encode_negative_targetid(0, 0, -1)

        # 2 milliarcsec differences -> different TARGETID.
        ra, dec = 10.0, 0.0
        delta = 2.0/(3600*1000)

        t1 = encode_negative_targetid(ra, dec)
        t2 = encode_negative_targetid(ra, dec+delta)
        t3 = encode_negative_targetid(ra, dec-delta)
        t4 = encode_negative_targetid(ra+delta, dec)
        t5 = encode_negative_targetid(ra-delta, dec)
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertNotEqual(t1, t4)
        self.assertNotEqual(t1, t5)
        self.assertNotEqual(t2, t3)
        self.assertNotEqual(t2, t4)
        self.assertNotEqual(t2, t5)
        self.assertNotEqual(t3, t4)
        self.assertNotEqual(t3, t5)
        self.assertNotEqual(t4, t5)

    def test_decode_negative_targetid(self):
        """test negative targetid encoding -> decoding round trip"""

        # roundtrip accurate to at least 2 milliarcsec (without cos(dec)).
        n = 1000
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-90, 90, n)
        group = 5

        # include corner cases.
        ra = np.concatenate((ra, [0, 0, 0, 360, 360, 360]))
        dec = np.concatenate((dec, [-90, 0, 90, -90, 0, 90]))

        targetids = encode_negative_targetid(ra, dec, group)
        ra1, dec1, group1 = decode_negative_targetid(targetids)

        delta = 2.0/(3600*1000)
        self.assertTrue(np.all(np.abs(ra-ra1) < delta))
        self.assertTrue(np.all(np.abs(dec-dec1) < delta))
        self.assertTrue(np.all(group1 == group))

        # check group roundtrip, and scalar inputs.
        ra, dec = 20.1, -16.3333
        for group in range(1, 16):
            targetid = encode_negative_targetid(ra, dec, group)
            ra1, dec1, group1 = decode_negative_targetid(targetid)
            self.assertEqual(group1, group)
            self.assertLess(np.abs(ra1-ra), delta)
            self.assertLess(np.abs(dec1-dec), delta)


if __name__ == '__main__':
    unittest.main()


def test_suite():
    """Allows testing of only this module with the command:

        python setup.py test -m desitarget.test.test_geomask
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
