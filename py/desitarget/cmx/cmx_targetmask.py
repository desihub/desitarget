"""
desitarget.cmx.cmx_targetmask
=============================

This looks more like a script than an actual module.
"""
from desiutil.bitmask import BitMask
from desitarget.targetmask import load_mask_bits

_bitdefs = load_mask_bits("cmx")
try:
    cmx_mask = BitMask('cmx_mask', _bitdefs)
    cmx_obsmask = BitMask('cmx_obsmask', _bitdefs)
except TypeError:
    cmx_mask = object()
    cmx_obsmask = object()
