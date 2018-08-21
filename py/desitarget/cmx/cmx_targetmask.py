"""
desitarget.cmx.cmx_targetmask
=============================

This looks more like a script than an actual module.
"""
from desiutil.bitmask import BitMask
import yaml
from pkg_resources import resource_filename


_bitdefs = None


def _load_bits():
    """Load bit definitions from yaml file.
    """
    global _bitdefs
    if _bitdefs is None:
        _filepath = resource_filename('desitarget', "cmx/data/cmx_targetmask.yaml")
        with open(_filepath) as fx:
            _bitdefs = yaml.load(fx)
        try:
            _load_priorities(handle="priorities")
        except TypeError:
            pass
        try:
            _load_priorities(handle="numobs")
        except TypeError:
            pass
    return


def _load_priorities(handle="priorities"):
    """Priorities and NUMOBS are defined in the yaml file, but they aren't
    a bitmask and so require some extra processing.
    """
    global _bitdefs
    for maskname, priorities in _bitdefs[handle].items():
        for bitname in priorities:
            #- "SAME_AS_XXX" enables one bit to inherit priorities from another
            if isinstance(priorities[bitname], str) and \
                priorities[bitname].startswith('SAME_AS_'):
                    other = priorities[bitname][8:]
                    priorities[bitname] = priorities[other]

            #- fill in default "more" priority to be same as "unobs"
            #ADM specifically applies to dictionary of priorities
            if handle=='priorities':
                if isinstance(priorities[bitname], dict):
                    if 'MORE_ZWARN' not in priorities[bitname]:
                        priorities[bitname]['MORE_ZWARN'] = priorities[bitname]['UNOBS']
                    if 'MORE_ZGOOD' not in priorities[bitname]:
                        priorities[bitname]['MORE_ZGOOD'] = priorities[bitname]['UNOBS']

                    #- fill in other states as priority=1
                    for state, blat, foo in _bitdefs['cmx_obsmask']:
                        if state not in priorities[bitname]:
                            priorities[bitname][state] = 1
                else:
                    priorities[bitname] = dict()

        #- add to the extra info dictionary for this target mask
        for bitdef in _bitdefs[maskname]:
            bitname = bitdef[0]
            bitdef[3][handle] = priorities[bitname]
    return

#- convert to BitMask objects
if _bitdefs is None:
    _load_bits()
try:
    cmx_mask = BitMask('cmx_mask', _bitdefs)
    cmx_obsmask = BitMask('cmx_obsmask', _bitdefs)
except TypeError:
    cmx_mask = object()
    cmx_obsmask = object()
