from desitarget.internal.maskbits import BitMask

import os
import sys
import yaml

#- Load bit definitions from yaml file
_thisdir = os.path.dirname(__file__)
with open(os.path.join(_thisdir, "targetmask.yaml")) as bitdeffile:
    _bitdefs = yaml.load(bitdeffile)

#- convert to BitMask objects
targetmask = BitMask('targetmask', _bitdefs)
obsconditions = BitMask('obsconditions', _bitdefs)
targetstate = BitMask('targetstate', _bitdefs)

#- Convert targetmask.obsconditions strings into masks
for bitname in targetmask.names():
    assert 'obsconditions' in targetmask.extra(bitname).keys()
    condmask = 0
    for cond in targetmask.extra(bitname)['obsconditions'].split('|'):
        if cond in obsconditions.names():
            condmask |= obsconditions.mask(cond)
        else:
            raise ValueError('Unknown obscondition {} for bitname {}'.format(cond, bitname))
    targetmask.extra(bitname)['obsconditions'] = condmask

#- priorities isn't a bitmask, and requires some extra processing
priorities = _bitdefs['priorities']
for bitname in priorities.keys():
    #- fill in "SAME_AS_XXX" for one bit to inherit priorities from another
    if isinstance(priorities[bitname], str) and priorities[bitname].startswith('SAME_AS_'):
        other = priorities[bitname][8:]
        priorities[bitname] = priorities[other]
    
    #- fill in default "more" priority to be same as "unobs"
    if isinstance(priorities[bitname], dict):
        if 'more' not in priorities[bitname]:
            priorities[bitname]['more'] = priorities[bitname]['unobs']

error = False
for bitname in targetmask.names():
    if bitname not in priorities.keys():
        print >> sys.stderr, "ERROR: no priority defined for "+bitname
        error = True
        
for bitname in priorities.keys():
    if bitname not in targetmask.names():
        print >> sys.stderr, "ERROR: priority defined for bogus name "+bitname
        error = True

if error:
    raise ValueError("mismatch between priority and targetmask definitions")