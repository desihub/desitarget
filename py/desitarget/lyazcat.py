"""
desitarget.lyazcat
==============

Post-redrock ML processing for LyA Quasar object identification.
"""

import os
import numpy as np
import sys
from astropy.io import fits
import fitsio
import time

from desitarget.geomask import match, match_to
from desitarget.internal import sharedmem
from desitarget import io
from desispec.io import read_spectra

from quasarnp.io import load_model
from quasarnp.io import load_desi_coadd
from quasarnp.utils import process_preds
from squeze.model import Model
from squeze.common_functions import load_json
from squeze.candidates import Candidates
from squeze.desi_spectrum import DesiSpectrum
from squeze.spectra import Spectra

# ADM set up the DESI default logger.
from desiutil.log import get_logger
log = get_logger()

zcatdatamodel_names = np.array(['TARGETID', 'Z', 'ZWARN', 'DELTACHI2',
                                'Z_COMB', 'Z_COMB_PROB', 'Z_QN', 'Z_QN_CONF',
                                'IS_QSO_QN', 'Z_SQ', 'Z_SQ_CONF', 'Z_ABS',
                                'Z_ABS_CONF'])

zcatdatamodel_formats = np.array(['>i8', '>f8', '>i8', '>f8',
                                  '>f8', '>f8', '>f8', '>f8',
                                  '>i2', '>f8', '>f8', '>f8',
                                  '>f8'])
zcols_copy = np.array(['TARGETID', 'Z', 'ZWARN', 'DELTACHI2'])
n1_cols = np.array(['Z_QN', 'Z_QN_CONF', 'IS_QSO_QN', 'Z_SQ', 'Z_SQ_CONF',
                    'Z_ABS', 'Z_ABS_CONF'])


def tmark(istring):
    """A function to mark the time an operation starts or ends.
    
    Parameters
    ----------
    istring : :class:'str'
        The input string to print to the terminal.
        
    Output
    ------
    :class:'str'
        A string with the date and time in ISO 8061 standard followed
        by the istring.
    """
    t0 = time.time()
    t_start = time.strftime('%Y-%m-%d | %H:%M:%S')
    log.info('\n{}: {}'.format(istring,t_start))

    
def make_new_zcat(zbestname):
    """Make the initial zcat array with redrock data.
    
    Parameters
    ----------
    zbestname : :class:`str`
        Full filename and path for the zbest file to process.
        
    Returns
    -------
    :class:`~numpy.array`
        A zcat in the official format (`zcatdatamodel`) compiled from
        the `tile', 'night', and 'petal_num', in `zcatdir`.
    """
    tmark('    Making redrock zcat')
    try:
        zs = fits.open(zbestname)['ZBEST'].data
        # EBL write out the zcat as a file with the correct data model.
        zcat = np.zeros(len(zs), dtype={'names': zcatdatamodel_names, 'formats': zcatdatamodel_formats})
        for col in zcols_copy:
            zcat[col] = zs[col]
        for col in n1_cols:
            zcat[col][:] = -1

        return zcat
    except (FileNotFoundError, OSError):
        return False

    
def add_qn_data(zcat, coaddname, qnp_model, qnp_lines, qnp_lines_bal):
    """Apply the QuasarNP model to the input zcat and add data to columns.
    
    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    coaddname : class:'str'
        The name of the coadd file corresponding to the zbest file used
        in make_new_zcat()
        
    Returns
    -------
    :class:'~numpy.array'
        The zcat array with QuasarNP data included in the columns:
        * Z_QN
        * Z_QN_CONF
        * IS_QSO_QN
    """
    tmark('    Adding QuasarNP data')
    
    data, w = load_desi_coadd(coaddname)
    data = data[:, :, None]
    p = qnp_model.predict(data)
    c_line, z_line, zbest, c_line_bal, z_line_bal = process_preds(p, qnp_lines,
                                                                  qnp_lines_bal,
                                                                  verbose=False)

    cbest = np.array(c_line[c_line.argmax(axis=0), np.arange(len(zbest))])
    c_thresh = 0.5
    n_thresh = 1
    is_qso = np.sum(c_line > c_thresh, axis=0) >= n_thresh

    zcat['Z_QN'][w] = zbest
    zcat['Z_QN_CONF'][w] = cbest
    zcat['IS_QSO_QN'][w] = is_qso

    return zcat


def add_sq_data(zcat, coaddname, squeze_model):
    """Apply the SQUEzE model to the input zcat and add data to columns.
    
    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    coaddname : class:'str'
        The name of the coadd file corresponding to the zbest file used
        in make_new_zcat()
    squeze_model : :class:'numpy.array'
        The loaded SQUEzE model file
        
    Returns
    -------
    :class:'~numpy.array'
        The zcat array with SQUEzE data included in the columns:
        * Z_SQ
        * Z_SQ_CONF
    """
    tmark('    Adding SQUEzE data')
    mdata = ['TARGETID']
    single_exposure = False
    sq_cols_keep = ['PROB', 'Z_TRY', 'TARGETID']

    tmark('      Reading spectra')
    desi_spectra = read_spectra(coaddname)
    # EBL Initialize squeze Spectra class
    squeze_spectra = Spectra()
    # EBL Get TARGETIDs
    targetid = np.unique(desi_spectra.fibermap['TARGETID'])
    # EBL Loop over TARGETIDs to build the Spectra objects
    for targid in targetid:
        # EBL Select objects
        pos = np.where(desi_spectra.fibermap['TARGETID'] == targid)
        # EBL Prepare column metadata
        metadata = {col.upper(): desi_spectra.fibermap[col][pos[0][0]] for col in mdata}
        # EBL Add the SPECID as the TARGETID
        metadata['SPECID'] = targid
        # EBL Extract the data
        flux = {}
        wave = {}
        ivar = {}
        mask = {}
        for band in desi_spectra.bands:
            flux[band] = desi_spectra.flux[band][pos]
            wave[band] = desi_spectra.wave[band]
            ivar[band] = desi_spectra.ivar[band][pos]
            mask[band] = desi_spectra.mask[band][pos]
            
        # EBL Format each spectrum for the model application
        spectrum = DesiSpectrum(flux, wave, ivar, mask, metadata, single_exposure)
        # EBL Append the spectrum to the Spectra object
        squeze_spectra.append(spectrum)
    
    # EBL Initialize candidate object. This takes a while with no feedback
    # so we want a time output for benchmarking purposes.
    tmark('      Initializing candidates')
    candidates = Candidates(mode='operation', model=squeze_model)
    # EBL Look for candidate objects. This also takes a while.
    tmark('      Looking for candidates')
    candidates.find_candidates(squeze_spectra.spectra_list(), save=False)
    # EBL Compute the probabilities of the line/model matches to the spectra
    tmark('      Computing probabilities')
    candidates.classify_candidates(save=False)
    # EBL Filter the results by removing the duplicate entries for each
    # TARGETID. Merge the remaining with the zcat data.
    tmark('      Merging SQUEzE data with zcat')
    data_frame = candidates.candidates()
    data_frame = data_frame[~data_frame['DUPLICATED']][sq_cols_keep]
    # EBL Strip the pandas data frame structure and put it into a numpy 
    # structured array first.
    sqdata_arr = np.zeros(len(data_frame), dtype=[('TARGETID', 'int64'),
                                                  ('Z_SQ', 'float64'),
                                                  ('Z_SQ_CONF', 'float64')])
    sqdata_arr['TARGETID'] = data_frame['TARGETID'].values
    sqdata_arr['Z_SQ'] = data_frame['Z_TRY'].values
    sqdata_arr['Z_SQ_CONF'] = data_frame['PROB'].values
    # EBL SQUEzE will reorder the objects, so match on TARGETID.
    zcat_args, sqdata_args = match(zcat['TARGETID'],sqdata_arr['TARGETID'])
    zcat['Z_SQ'][zcat_args] = sqdata_arr['Z_SQ'][sqdata_args]
    zcat['Z_SQ_CONF'][zcat_args] = sqdata_arr['Z_SQ_CONF'][sqdata_args]

    return zcat


def add_abs_data(zcat, coaddname):
    """Add the MgII absorption line finder data to the input zcat array.
    
    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    coaddname : class:'str'
        The name of the coadd file corresponding to the zbest file used
        in make_new_zcat()
        
    Returns
    -------
    :class:'~numpy.array'
        The zcat array with SQUEzE data included in the columns:
        * Z_ABS
        * Z_ABS_CONF
    """
    tmark('    MgII Absorption data not yet added.')

    return zcat


def zcomb_selector(zcat, proc_flag=False):
    """Compare results from redrock, QuasarNP, SQUEzE, and MgII data.
    
    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    proc_flag : :class:'bool'
        Turn on extra comparison procedure.
        
    Returns
    -------
    :class:'~numpy.array'
        The zcat array with SQUEzE data included in the columns:
        * Z_COMB
        * Z_COMB_PROB
    """
    zcat['Z_COMB'][:] = zcat['Z']
    zcat['Z_COMB_PROB'][:] = 0.95

    return zcat


def zcat_writer(outputdir, zcat, outputname, qn_flag=False, sq_flag=False, abs_flag=False):
    """Writes the zcat structured array out as a FITS file.
    
    Parameters
    ----------
    outputdir : :class:'str'
        The directory where the zcat file will be written.
    zcat : :class:'~numpy.array'
        stuff
    outputname : :class:'str'
        The filename of the zcat output file.
    qn_flag : :class:'bool'
        Flag if QuasarNP data (or not) was added to the zcat file.
    sq_flag : :class:'bool'
        Flag if SQUEzE data (or not) was added to the zcat file.
    abs_flag : :class:'bool'
        Flag if MgII Absorption data (or not) was added to the zcat file.
        
    Returns
    -------
    :class:'str'
        The filename, with path, of the FITS file written out.
    """
    tmark('    Creating file...')
    full_outputname = os.path.join(outputdir, outputname)
    
    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header['QN_ADDED'] = str(qn_flag)
    prim_hdu.header['SQ_ADDED'] = str(sq_flag)
    prim_hdu.header['AB_ADDED'] = str(abs_flag)
    
    data_hdu = fits.BinTableHDU.from_columns(zcat, name='ZCATALOG')
    data_out = fits.HDUList([prim_hdu, data_hdu])
    
    data_out.writeto(full_outputname)

    return full_outputname


def create_zcat(tile, night, petal_num,
                zcatdir='/global/cfs/cdirs/desi/spectro/redux/daily', 
                outputdir='/global/cfs/cdirs/desi/spectro/redux/daily',
                qn_flag=False, sq_flag=False, abs_flag=False,
                qnp_model='/global/cfs/cdirs/desi/target/catalogs/lya/qn_models/qn_train_coadd_indtrain_0_0_boss10.h5',
                squeze_model='/global/cfs/cdirs/desi/target/catalogs/lya/sq_models/BOSS_train_64plates_model.json',
                qnp_lines=None, qnp_lines_bal=None, vi_flag=False):
    """This will create a single zcat file from a set of user inputs.
    
    Parameters
    ----------
    tile : :class:'str'
        The TILEID of the tile to process.
    night : :class:'str'
        The date associated with the observation of the 'tile' used.
        * Must be in YYYYMMDD format
    petal_num : :class:'int'
        If 'all_petals' isn't used, the single petal to create a zcat for.
    zcatdir : :class:'str'
        The location for the daily redrock output.
    outputdir : :class:'str'
        The filepath to write out the zcat file.
    qn_flag : :class:'bool'
        Flag to add QuasarNP data (or not) to the zcat file.
    sq_flag : :class:'bool'
        Flag to add SQUEzE data (or not) to the zcat file.
    abs_flag : :class:'bool'
        Flag to add MgII Absorption data (or not) to the zcat file.
    qnp_model : :class:'str' or 'h5 array'
        The filename and path for the QuasarNP model file. IF this function
        is run as part of a loop, the calling function will load the model
        file and this will be an array instead.
    squeze_model : :class:'str' or 'numpy.array'
        The filename and path for the SQUEzE model file. IF this function
        is run as part of a loop, the calling function will load the model
        file and this will be an array instead.
    qnp_lines : :class:'list' or None
        The list of lines to use in the QuasarNP model to test against. If
        the script is run in loop mode, the list is passed from the calling
        function, otherwise it's created below.
    qnp_lines : :class:'list' or None
        The list of BAL lines to use for QuasarNP to identify BALs. If
        the script is run in loop mode, the list is passed from the calling
        function, otherwise it's created below.
    vi_flag : :class:'bool'
        Flag to test this script on the VI'd tiles from SV1/Blanc. These
        were created with exposures only totalling ~1000 sec R_DEPTH_EBVAIR,
        so we can use them as a truth set. This will load coadd and zbest
        files from a different directory automatically.
        
    Outputs
    -------
    A FITS catalog that incorporates redrock, QuasarNP, SQUEzE, and MgII
    absorption redshifts and confidence values.
    
    Notes
    -----
    - There will be many here.
    """
    # EBL Load the SQUEzE Model file first. This is a very large file,
    # so if multiple petals are to be processed, we only want to load
    # it into memory once.
    if isinstance(qnp_model, str) and qn_flag:
        tmark('    Loading QuasarNP Model file')
        qnp_lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
        qnp_lines_bal = ['CIV(1548)']
        qnp_model = load_model(qnp_model)
        tmark('      QNP model file loaded')
    if isinstance(squeze_model, str) and sq_flag:
        tmark('    Loading SQUEzE Model file')
        squeze_model = Model.from_json(load_json(squeze_model))
        tmark('      SQUEzE model file loaded')
    
    # EBL Create the filepath for the input tile/night combination
    rootdir = os.path.join(zcatdir, 'tiles', 'cumulative')
    tiledir = os.path.join(rootdir, tile)
    ymdir = os.path.join(tiledir, night)
        
    # EBL Create the filename tag that appends to zbest-*, coadd-*, and zqso-*
    # files.
    filename_tag = f'{petal_num}-{tile}-thru{night}.fits'
    zbestname = f'zbest-{filename_tag}'
    coaddname = f'coadd-{filename_tag}'
    outputname = f'zqso-{filename_tag}'
    
    if vi_flag:
        ymdir = os.path.join(os.getenv('CSCRATCH'), 'graydark')
    
    zcat = make_new_zcat(os.path.join(ymdir, zbestname))
    if isinstance(zcat, bool):
        print('  !!!! Petal Number does not have a corresponding zbest file !!!!')
    else:
        if qn_flag :
            zcat = add_qn_data(zcat, os.path.join(ymdir, coaddname),
                               qnp_model, qnp_lines, qnp_lines_bal)
        if sq_flag:
            zcat = add_sq_data(zcat, os.path.join(ymdir, coaddname), squeze_model)
        if abs_flag:
            zcat = add_abs_data(zcat, os.path.join(ymdir, coaddname))
        
        fzcat = zcomb_selector(zcat)
    
        full_outputname = zcat_writer(outputdir, fzcat, outputname, qn_flag,
                                      sq_flag, abs_flag)
    
        tmark('    --{} written out correctly.'.format(full_outputname))
        print('='*60)

    
if __name__=='__main__':
    # TODO:  FIX TERMINAL FEEDBACK
    # List of handy tiles to test (TILEID, NIGHTID)
    #    -1, 20210406
    #    -84, 20210410
    #    -85, 20210412
    # For the VI'd tiles using the processed r_depth_ebvair of ~1000
    #    TILEIDs: 80605, 80607, 80609, 80620, 80622
    #    NIGHTID: All use 20210302 (when I made those files). Date is
    #             meaningless in this case, just there due to filename
    #             format requirements.
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a zcat file with additional ML data for a single tile/night combination.')
    parser.add_argument('tile', help='TILEID of the tile to process.')
    parser.add_argument('night', help='NIGHTID of the tile to process.')
    parser.add_argument('-i', '--input_dir', default='/global/cfs/cdirs/desi/spectro/redux/daily',
                        help='The input directory that contains redrock output.')
    parser.add_argument('-o', '--output_dir', default='/global/cfs/cdirs/desi/spectro/redux/daily',
                        help='The output directory for the zcat file. Defaults to the zbest directory.')
    parser.add_argument('-a', '--all_petals', action='store_true',
                        help='Run all petals for a given tile/night combination.')
    parser.add_argument('-p', '--petal_num', type=int, metavar='', choices=[0,1,2,3,4,5,6,7,8,9],
                        default=0, help='Run for this petal number only.')
    parser.add_argument('-q', '--add_quasarnp', action='store_true',
                        help='Add QuasarNP data to zcat.')
    parser.add_argument('-s', '--add_squeze', action='store_true',
                        help='Add SQUEzE data to zcat.')
    parser.add_argument('-m', '--add_mgii', action='store_true',
                        help='Add MgII absorption data to zcat.')
    parser.add_argument('-n', '--qnp_modelfn', default='/global/cfs/cdirs/desi/target/catalogs/lya/qn_models/qn_train_coadd_indtrain_0_0_boss10.h5',
                        help='The full path and filename for the SQUEzE model file.')
    parser.add_argument('-e', '--squeze_modelfn', default='/global/cfs/cdirs/desi/target/catalogs/lya/sq_models/BOSS_train_64plates_model.json',
                        help='The full path and filename for the SQUEzE model file.')
    parser.add_argument('-v', '--vi_test_flag', action='store_true',
                        help='Bypasses the daily directory for redrock and uses the reprocessed VI tiles for testing.')
    args = parser.parse_args()
    
    # EBL For temporary testing purposes:
    args.output_dir = os.path.join(os.getenv('CSCRATCH'), 'lya_test', args.tile)
    
    if args.all_petals:
        if args.add_quasarnp:
            tmark('    Loading QuasarNP Model file')
            qnp_lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
            qnp_lines_bal = ['CIV(1548)']
            qnp_model = load_model(args.qnp_modelfn)
            tmark('      QNP model file loaded')
        if args.add_squeze:
            tmark('    Loading SQUEzE Model file')
            sq_model = Model.from_json(load_json(args.squeze_modelfn))
            tmark('      Model file loaded')
        for petal_num in range(10):
            create_zcat(args.tile, args.night, petal_num,
                        zcatdir=args.input_dir, outputdir=args.output_dir,
                        qn_flag=args.add_quasarnp, sq_flag=args.add_squeze,
                        abs_flag=args.add_mgii, qnp_model=qnp_model,
                        squeze_model=sq_model, qnp_lines=qnp_lines,
                        qnp_lines_bal=qnp_lines_bal,
                        vi_flag=args.vi_test_flag)
    else:
        create_zcat(args.tile, args.night, args.petal_num,
                    zcatdir=args.input_dir, outputdir=args.output_dir,
                    qn_flag=args.add_quasarnp, sq_flag=args.add_squeze,
                    abs_flag=args.add_mgii, qnp_model=args.qnp_modelfn,
                    squeze_model=args.squeze_modelfn,
                    vi_flag=args.vi_test_flag)
