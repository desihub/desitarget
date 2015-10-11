from desitarget.internal.maskbits import BitMask

import yaml
_bitdefs = yaml.load("""
#- Target type Mask
targetmask:
    - [LRG,       0, "LRG"]
    - [ELG,       1, "ELG"]
    - [BGS,       2, "BGS"]
    - [QSO,       3, "QSO"]
""")
targetmask = BitMask('targetmask', _bitdefs)
