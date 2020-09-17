#- SJB This code was used to trim a set of reference run mock targets+truth
#- into a small dataset for testing.

#- make sure an import can't accidentally trigger this
if __name__ == "__main__":
    import os, sys
    import numpy as np
    import fitsio

    truthfile = "/global/cfs/cdirs/desi/datachallenge/reference_runs/20.4/targets/52/5299/dark/truth-dark-64-5299.fits"
    targetfile = "/global/cfs/cdirs/desi/datachallenge/reference_runs/20.4/targets/52/5299/dark/targets-dark-64-5299.fits"

    if not os.path.exists(truthfile):
        print(f"ERROR: Unable to find {truthfile}")
        print("you must run this script at NERSC")
        sys.exit(1)

    truth_data = dict()
    with fitsio.FITS(truthfile) as fx:
        keep_targetids = list()
        for objtype in ['BGS', 'ELG', 'LRG', 'QSO', 'STAR', 'WD']:
            extname = 'TRUTH_'+objtype
            truth = fx[extname].read()[0:3]  #- keep 3 targets per class
            keep_targetids.extend(truth['TARGETID'])
            truth_data[extname] = truth

        truth_hdr = fx['TRUTH'].read_header()
        truth = fx['TRUTH'].read()
        flux = fx['FLUX'].read()
        wave = fx['WAVE'].read()
        hdr = fx['TRUTH'].read_header()

        keep = np.in1d(truth['TARGETID'], keep_targetids)

        #- trim targets and downsample wavelength grid for smaller file
        truth_data['TRUTH'] = truth[keep]
        truth_data['FLUX'] = flux[keep, 0::25]
        truth_data['WAVE'] = wave[0::25]

    #- output filename
    test_truthfile = 't/truth-mocks.fits'
    test_targetsfile = 't/targets-mocks.fits'

    #- Trim targets tile to match truth targets that are kept
    columns = ['TARGETID', 'RA', 'DEC', 'RELEASE',
        'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',
        'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z',
        'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2',
        'PARALLAX', 'PMRA', 'PMDEC', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET',
        ]
    targets, hdr = fitsio.read(targetfile,'TARGETS',header=True,columns=columns)
    keep = np.in1d(targets['TARGETID'], keep_targetids)
    targets = targets[keep]
    fitsio.write(test_targetsfile, targets, extname='TARGETS',
                 header=hdr, clobber=True)

    fitsio.write(test_truthfile, truth_data['TRUTH'], extname='TRUTH',
        header=truth_hdr, clobber=True)

    for extname in ['WAVE', 'FLUX', 'TRUTH_BGS', 'TRUTH_ELG',
                    'TRUTH_LRG', 'TRUTH_QSO', 'TRUTH_STAR', 'TRUTH_WD']:
        fitsio.write(test_truthfile, truth_data[extname],
            extname=extname)

    print(f'Wrote {test_targetsfile} and {test_truthfile}')

