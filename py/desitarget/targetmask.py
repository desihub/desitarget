from desitarget.internal.maskbits import BitMask

#- THESE BIT DEFINITIONS WILL ALMOST CERTAINLY CHANGE

import yaml
_bitdefs = yaml.load("""
#- Target type Mask
targetmask:
    #- Dark Time Survey: Bits 0-31
    - [LRG,         0, "LRG"]
    - [ELG,         1, "ELG"]
    - [QSO,         2, "QSO"]
    
    #- Not yet used, but placeholders for North vs. South selections
    - [LRG_NORTH,   8, "LRG from Bok/Mosaic data"]
    - [ELG_NORTH,   9, "ELG from Bok/Mosaic data"]
    - [QSO_NORTH,   10, "QSO from Bok/Mosaic data"]

    - [LRG_SOUTH,   16, "LRG from DECam data"]
    - [ELG_SOUTH,   17, "ELG from DECam data"]
    - [QSO_SOUTH,   18, "QSO from DECam data"]

    #- Calibration targets used by the Dark Time Survey
    - [FSTD,        24, "F-type standard stars"]
    - [WDSTAR,      25, "White Dwarf stars"]
    - [SKY,         26, "Blank sky locations"]
    
    #- Bright Galaxy Survey: bits 32-47
    - [BGS_FAINT,           32, "BGS faint targets"]
    - [BGS_BRIGHT,          33, "BGS bright targets"]
    - [BGS_KNOWN_COLLIDED,  34, "BGS known SDSS/BOSS fiber collided"]
    - [BGS_KNOWN_SDSS,      35, "BGS known SDSS targets"]
    - [BGS_KNOWN_BOSS,      36, "BGS known BOSS targets"]
    
    #- Milky Way Survey placeholder: bits 48-59
    - [MWS_WD,              48, "Milky Way Survey White Dwarf"]
    - [MWS_PLX,             49, "Milky Way Survey Parallax"]
    
    #- Placeholder flags
    - [BGS_ANY,             60, "Any BGS bit is set"]
    - [MWS_ANY,             61, "Any MWS bit is set"]
    - [ANCILLARY_ANY,       62, "Any ancillary bit is set"]
    
#- Priorities for Unobserved, observed with uncertain redshift, observed with good redshift
bitpriorities:
    LRG : [0, 0, 0]
    ELG : [0, 0, 0]
    QSO : [0, 0, 0]
    BGS_ANY : [0, 0, 0]
    MWS_ANY : [0, 0, 0]
    ANCILLARY_ANY : [0, 0, 0]
""")
targetmask = BitMask('targetmask', _bitdefs)
