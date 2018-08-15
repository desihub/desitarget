"""
desitarget.comm.commtargetmask
==============================

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
        _filepath = resource_filename('desitarget', "comm/data/commtargetmask.yaml")
        with open(_filepath) as fx:
            _bitdefs = yaml.load(fx)
        try:
            _load_priorities()
        except TypeError:
            pass
    return


def _load_priorities():
    """Priorities are defined in the yaml file, but they aren't a bitmask
    and they require some extra processing.
    """
    global _bitdefs
    for maskname, priorities in _bitdefs['priorities'].items():
        for bitname in priorities:
            #- "SAME_AS_XXX" enables one bit to inherit priorities from another
            if isinstance(priorities[bitname], str) and \
                priorities[bitname].startswith('SAME_AS_'):
                    other = priorities[bitname][8:]
                    priorities[bitname] = priorities[other]

            #- fill in default "more" priority to be same as "unobs"
            if isinstance(priorities[bitname], dict):
                if 'MORE_ZWARN' not in priorities[bitname]:
                    priorities[bitname]['MORE_ZWARN'] = priorities[bitname]['UNOBS']
                if 'MORE_ZGOOD' not in priorities[bitname]:
                    priorities[bitname]['MORE_ZGOOD'] = priorities[bitname]['UNOBS']

                #- fill in other states as priority=1
                for state, blat, foo in _bitdefs['comm_obsmask']:
                    if state not in priorities[bitname]:
                        priorities[bitname][state] = 1
            else:
                priorities[bitname] = dict()

        #- add to the extra info dictionary for this target mask
        for bitdef in _bitdefs[maskname]:
            bitname = bitdef[0]
            bitdef[3]['priorities'] = priorities[bitname]
    return

#- convert to BitMask objects
if _bitdefs is None:
    _load_bits()
try:
    comm_mask = BitMask('comm_mask', _bitdefs)
    comm_obsmask = BitMask('comm_obsmask', _bitdefs)
except TypeError:
    comm_mask = object()
    comm_obsmask = object()
