import  os

from    desitarget.geomask  import  hp_in_box
from    desitarget.randoms  import  dr_extension, _pre_or_post_dr8
from    desiutil            import  brick


nside      = 8
bricks     = brick.Bricks(bricksize=0.25)
bricktable = bricks.to_table()
lookupdict = {bt["BRICKNAME"]: hp_in_box(nside, [bt["RA1"], bt["RA2"], bt["DEC1"], bt["DEC2"]]) for bt in bricktable}

pixnum     = 6195
bricknames = [key for key in lookupdict if pixnum in lookupdict[key]]

print(bricknames)

# bricknames  = ['0880p587', '0880p590', '0881p582', '0882p592'] 

# drdir       = '/global/homes/m/mjwilson/desi/survey-validation/svdc-spring2020g-onepercent/legacydir/'
drdir         = '/global/project/projectdirs/cosmo/data/legacysurvey/dr8/'

drdirs        = _pre_or_post_dr8(drdir)

for drdir in drdirs:
  # ADM determine whether the coadd files have extension .gz or .fz based on the DR directory.
  extn, extn_nb = dr_extension(drdir)

  filt          = ['g', 'r', 'z']
  qnames        = ['nexp', 'depth', 'galdepth', 'psfsize', 'image']

  for brickname in bricknames:
    brickname   = brickname.replace('m', 'p')

    rootdir     = os.path.join(drdir,   'coadd', brickname[:3], brickname)
    fileform    = os.path.join(rootdir, 'legacysurvey-{}-{}-{}.fits.{}')
  
    # ADM loop through the filters and store the number of observations
    # ADM etc. at the RA and Dec positions of the passed points.
    for f in ['g', 'r', 'z']:
      # ADM the input file labels, and output column names and output
      # ADM formats for each of the quantities of interest.
      for q in qnames:
        fn     = fileform.format(brickname, q, f, extn)

        cmd    = 'cp {} legacydir/'.format(fn)
        
        # os.system(cmd)

        print(cmd)
      
    break
