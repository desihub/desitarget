"""
desitarget.targetmask
=====================

This looks more like a script than an actual module.
"""
from desiutil.bitmask import BitMask
import yaml
from pkg_resources import resource_filename


def load_mask_bits(prefix=""):
    """Load bit definitions from yaml file.
    """
    us = ""
    if len(prefix) > 0:
        us = '_'
    prename = prefix+us
    fn = "{}/data/{}targetmask.yaml".format(prefix, prename)
    _filepath = resource_filename('desitarget', fn)
    with open(_filepath) as fx:
        bitdefs = yaml.load(fx)
        try:
            bitdefs = _load_mask_priorities(bitdefs, handle="priorities", prename=prename)
        except TypeError:
            pass
        try:
            bitdefs = _load_mask_priorities(bitdefs, handle="numobs", prename=prename)
        except TypeError:
            pass
    return bitdefs


def _load_mask_priorities(bitdefs, handle="priorities", prename=""):
    """Priorities and NUMOBS are defined in the yaml file, but they aren't
    a bitmask and so require some extra processing.
    """
    for maskname, priorities in bitdefs[handle].items():
        for bitname in priorities:
            # -"SAME_AS_XXX" enables one bit to inherit priorities from another
            if isinstance(priorities[bitname], str) and priorities[bitname].startswith('SAME_AS_'):
                other = priorities[bitname][8:]
                priorities[bitname] = priorities[other]

            # -fill in default "more" priority to be same as "unobs"
            # ADM specifically applies to dictionary of priorities
            if handle == 'priorities':
                if isinstance(priorities[bitname], dict):
                    if 'MORE_ZWARN' not in priorities[bitname]:
                        priorities[bitname]['MORE_ZWARN'] = priorities[bitname]['UNOBS']
                    if 'MORE_ZGOOD' not in priorities[bitname]:
                        priorities[bitname]['MORE_ZGOOD'] = priorities[bitname]['UNOBS']

                    # - fill in other states as priority=1
                    for state, blat, foo in bitdefs[prename+'obsmask']:
                        if state not in priorities[bitname]:
                            priorities[bitname][state] = 1
                else:
                    priorities[bitname] = dict()

        # - add to the extra info dictionary for this target mask
        for bitdef in bitdefs[maskname]:
            bitname = bitdef[0]
            bitdef[3][handle] = priorities[bitname]
    return bitdefs


# -convert to BitMask objects
# if bitdefs is None:
#     load_bits()
_bitdefs = load_mask_bits()
try:
    desi_mask = BitMask('desi_mask', _bitdefs)
    mws_mask = BitMask('mws_mask', _bitdefs)
    bgs_mask = BitMask('bgs_mask', _bitdefs)
    obsconditions = BitMask('obsconditions', _bitdefs)
    obsmask = BitMask('obsmask', _bitdefs)
    targetid_mask = BitMask('targetid_mask', _bitdefs)
except TypeError:
    desi_mask = object()
    mws_mask = object()
    bgs_mask = object()
    obsconditions = object()
    obsmask = object()
    targetid_mask = object()

# -------------------------------------------------------------------------
# -Do some error checking that the bitmasks are consistent with each other
# import sys
# error = False
# for mask in desi_target, mws_target, bgs_target:
#     for bitname in targetmask.names:
#         if targetmask[bitname]
#     if bitname not in priorities.keys():
#         print >> sys.stderr, "ERROR: no priority defined for "+bitname
#         error = True
#
# for bitname in priorities.keys():
#     if bitname not in targetmask.names():
#         print >> sys.stderr, "ERROR: priority defined for bogus name "+bitname
#         error = True
#
# if error:
#     raise ValueError("mismatch between priority and targetmask definitions")
