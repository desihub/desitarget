"""
desitarget.contammask
=====================

Documentation goes here.
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
        _filepath = resource_filename('desitarget', "data/contammask.yaml")
        with open(_filepath) as fx:
            _bitdefs = yaml.load(fx)
    return


# - convert to BitMask objects
if _bitdefs is None:
    _load_bits()
try:
    contam_mask = BitMask('contam_mask', _bitdefs)
except TypeError:
    contam_mask = object()
