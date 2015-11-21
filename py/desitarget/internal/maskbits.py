"""
desiutil.maskbits
=================

Mask bits for the spectro pipeline.

Individual packages will define their own mask bits and use this as a utility
access wrapper.  Typical users will not need to construct BitMask objects on
their own.

Stephen Bailey, Lawrence Berkeley National Lab
Fall 2015

Example::

    #- Creating a BitMask
    from desiutil.bitmask import BitMask

    import yaml
    _bitdefs = yaml.load('''
    ccdmask:
        - [BAD,       0, "Pre-determined bad pixel (any reason)"]
        - [HOT,       1, "Hot pixel"]
        - [DEAD,      2, "Dead pixel"]
        - [SATURATED, 3, "Saturated pixel from object"]
        - [COSMIC,    4, "Cosmic ray"]
    ''')
    ccdmask = BitMask('ccdmask', _bitdefs)

    #- Accessing the mask
    ccdmask.COSMIC | ccdmask.SATURATED  #- 2**4 + 2**3
    ccdmask.mask('COSMIC')     #- 2**4, same as ccdmask.COSMIC
    ccdmask.mask(4)            #- 2**4, same as ccdmask.COSMIC
    ccdmask.COSMIC             #- 2**4, same as ccdmask.mask('COSMIC')
    ccdmask.bitnum('COSMIC')   #- 4
    ccdmask.bitname(4)         #- 'COSMIC'
    ccdmask.names()            #- ['BAD', 'HOT', 'DEAD', 'SATURATED', 'COSMIC']
    ccdmask.names(3)           #- ['BAD', 'HOT']
    ccdmask.comment(0)         #- "Pre-determined bad pixel (any reason)"
    ccdmask.comment('COSMIC')  #- "Cosmic ray"
"""


class _MaskBit(int):
    """
    A single mask bit.  Subclasses int to act like an int, but allows the
    ability to extend with blat.name, blat.comment, blat.mask, blat.bitnum.
    """
    def __new__(cls, name, bitnum, comment, extra=dict()):
        self = super(_MaskBit, cls).__new__(cls, 2**bitnum)
        self.name = name
        self.bitnum = bitnum
        self.mask = 2**bitnum
        self.comment = comment
        for key, value in extra.items():
            assert key not in (
                'bitlength', 'conjugate', 'real', 'imag',
                'numerator', 'denominator'), \
                "key '{}' already in use by int objects".format(key)
            self.__dict__[key] = value
        return self

    def __str__(self):
        return '{:16s} bit {} mask 0x{:X} - {}'.format(
            self.name, self.bitnum, self.mask, self.comment)


#- Class to provide mask bit utility functions
class BitMask(object):
    """BitMask object.
    """
    def __init__(self, name, bitdefs):
        """
        Args:
            name : name of this mask, must be key in bitdefs
            bitdefs : dictionary of different mask bit definitions;
                each value is a list of [bitname, bitnum, comment]

        Typical users are not expected to create BitMask objects directly.
        """
        self._bits = dict()
        self._name = name
        for x in bitdefs[name]:
            bitname, bitnum, comment = x[0:3]
            if len(x) == 4:
                extra = x[3]
            else:
                extra = dict()
            self._bits[bitname] = _MaskBit(bitname, bitnum, comment, extra)
            self._bits[bitnum] = self._bits[bitname]

    def __getitem__(self, bitname):
        return self._bits[bitname]

    def bitnum(self, bitname):
        """Return bit number (int) for bitname (string)"""
        return self._bits[bitname].bitnum

    def bitname(self, bitnum):
        """Return bit name (string) for this bitnum (integer)"""
        return self._bits[bitnum].name

    def comment(self, bitname_or_num):
        """Return comment for this bit name or bit number"""
        return self._bits[bitname_or_num].comment

    def mask(self, name_or_num):
        """Return mask value, e.g.

        bitmask.mask(3)         #- 2**3
        bitmask.mask('BLAT')
        bitmask.mask('BLAT|FOO')
        """
        if isinstance(name_or_num, int):
            return self._bits[name_or_num].mask
        else:
            mask = 0
            for name in name_or_num.split('|'):
                mask |= self._bits[name].mask
            return mask

    def names(self, mask=None):
        """Return list of names of masked bits.
        If mask=None, return names of all known bits.
        """
        names = list()
        if mask is None:
            #- return names in sorted order of bitnum
            bitnums = [x for x in self._bits.keys() if isinstance(x, int)]
            for bitnum in sorted(bitnums):
                names.append(self._bits[bitnum].name)
        else:
            bitnum = 0
            while bitnum**2 <= mask:
                if (2**bitnum & mask):
                    if bitnum in self._bits.keys():
                        names.append(self._bits[bitnum].name)
                    else:
                        names.append('UNKNOWN'+str(bitnum))
                bitnum += 1

        return names

    #- Allow access via mask.BITNAME
    def __getattr__(self, name):
        if name in self._bits:
            return self._bits[name]
        else:
            raise AttributeError('Unknown mask bit name '+name)

    #- What to print
    def __repr__(self):
        result = list()
        result.append(self._name+':')
        #- return names in sorted order of bitnum
        bitnums = [x for x in self._bits.keys() if isinstance(x, int)]
        for bitnum in sorted(bitnums):
            bit = self._bits[bitnum]
            result.append('    - [{:16s} {:2d}, "{}"]'.format(
                bit.name, bit.bitnum, bit.comment))

        return "\n".join(result)
