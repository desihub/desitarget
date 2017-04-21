"""
desitarget.contammask
=====================

"""
try:
    from desiutil.bitmask import BitMask
except ImportError:
    #
    # This can happen during documentation builds.
    #
    def BitMask(*args): return None

_bitdefs = None
def _load_bits():
    """Load bit definitions from yaml file.
    """
    global _bitdefs
    import yaml
    from pkg_resources import resource_filename
    if _bitdefs is None:
        _filepath = resource_filename('desitarget', "data/contammask.yaml")
        with open(_filepath) as fx:
            _bitdefs = yaml.load(fx)
    return

#- convert to BitMask objects
if _bitdefs is None:
    _load_bits()
contam_mask = BitMask('contam_mask', _bitdefs)
