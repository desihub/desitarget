"""
desitarget.sv2.sv2_targetmask
=============================

This looks more like a script than an actual module.
"""
from desiutil.bitmask import BitMask
from desitarget.targetmask import load_mask_bits

_bitdefs = load_mask_bits("sv2")
try:
    desi_mask = BitMask('sv2_desi_mask', _bitdefs)
    mws_mask = BitMask('sv2_mws_mask', _bitdefs)
    bgs_mask = BitMask('sv2_bgs_mask', _bitdefs)
    scnd_mask = BitMask('sv2_scnd_mask', _bitdefs)
    obsmask = BitMask('sv2_obsmask', _bitdefs)
except TypeError:
    desi_mask = object()
    mws_mask = object()
    bgs_mask = object()
    scnd_mask = object()
    obsmask = object()
