"""
desitarget.sv.sv_targetmask
===========================

This looks more like a script than an actual module.
"""
from desiutil.bitmask import BitMask
from desitarget.targetmask import load_mask_bits

_bitdefs = load_mask_bits("sv")
try:
    sv_desi_mask = BitMask('sv_desi_mask', _bitdefs)
    sv_mws_mask = BitMask('sv_mws_mask', _bitdefs)
    sv_bgs_mask = BitMask('sv_bgs_mask', _bitdefs)
    sv_obsmask = BitMask('sv_obsmask', _bitdefs)
except TypeError:
    sv_desi_mask = object()
    sv_mws_mask = object()
    sv_bgs_mask = object()
    sv_obsmask = object()
