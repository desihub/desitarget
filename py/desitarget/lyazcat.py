"""
desitarget.lyazcat
==================

Post-redrock ML processing for LyA Quasar object identification.
"""

import os
import numpy as np
from astropy.io import fits
import time

from desitarget.geomask import match
from desitarget.internal import sharedmem
from desitarget.io import write_with_units
from desispec.io import read_spectra
from desiutil.depend import add_dependencies

from quasarnp.io import load_model
from quasarnp.io import load_desi_coadd
from quasarnp.utils import process_preds
from squeze.model import Model
from squeze.common_functions import load_json
from squeze.candidates import Candidates
from squeze.desi_spectrum import DesiSpectrum
from squeze.spectra import Spectra

from prospect.mycoaddcam import coadd_brz_cameras
from operator import itemgetter
from itertools import groupby
from astropy.modeling import fitting
from astropy.modeling import models
from astropy.table import Table
from scipy.signal import medfilt
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

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

    Notes
    -----
    - A string with the date and time in ISO 8061 standard followed
      by the 'istring'.
    """
    t0 = time.time()
    t_start = time.strftime('%Y-%m-%d | %H:%M:%S')
    log.info('\n{}: {}'.format(istring, t_start))


def make_new_zcat(zbestname):
    """Make the initial zcat array with redrock data.

    Parameters
    ----------
    zbestname : :class:`str`
        Full filename and path for the zbest file to process.

    Returns
    -------
    :class:`~numpy.array` or 'bool'
        A zcat in the official format (`zcatdatamodel`) compiled from
        the `tile', 'night', and 'petal_num', in `zcatdir`. If the zbest
        file for that petal doesn't exist, returns ``False``.
    """
    tmark('    Making redrock zcat')

    try:
        zs = fits.open(zbestname)['ZBEST'].data
        # EBL write out the zcat as a file with the correct data model.
        zcat = np.zeros(len(zs), dtype={'names': zcatdatamodel_names,
                                        'formats': zcatdatamodel_formats})
        for col in zcols_copy:
            zcat[col] = zs[col]
        for col in n1_cols:
            zcat[col][:] = -1
        return zcat

    except FileNotFoundError:
        return False


def get_qn_model_fname(qnmodel_fname=None):
    """Convenience function to grab the $QN_MODEL_FILE environment variable.

    Parameters
    ----------
    qnmodel_fname : :class:`str`, optional, defaults to $QN_MODEL_FILE
        If `qnmodel_fname` is passed, it is returned from this function. If it's
        not passed, the $QN_MODEL_FILE variable is returned.

    Returns
    -------
    :class:`str`
        not passed, the directory stored in the $QN_MODEL_FILE environment
        variable is returned prepended to the default filename.
    """
    if qnmodel_fname is None:
        qnmodel_fname = os.environ.get('QN_MODEL_FILE')
        # EBL check that the $QN_MODEL_FILE environment variable is set.
        if qnmodel_fname is None:
            msg = "Pass qnmodel_fname or set $QN_MODEL_FILE environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return qnmodel_fname


def load_qn_model(model_filename):
    """Convenience function to load the QuasarNP model and line lists.

    Parameters
    ----------
    model_filename : :class:'str'
        The filename and path of the QuasarNP model. Either input by user or defaults
        to get_qn_model_fname().

    Returns
    -------
    :class:'~numpy.array'
        The QuasarNP model file loaded as an array.
    :class:'~numpy.array'
        An array of the emission line names to be used for quasarnp.process_preds().
    :class:'~numpy.array'
        An array of the BAL emission line names to be used by quasarnp.process_preds().
    """
    lines = ['LYA', 'CIV(1548)', 'CIII(1909)', 'MgII(2796)', 'Hbeta', 'Halpha']
    lines_bal = ['CIV(1548)']
    model = load_model(model_filename)

    return model, lines, lines_bal


def add_qn_data(zcat, coaddname, qnp_model, qnp_lines, qnp_lines_bal):
    """Apply the QuasarNP model to the input zcat and add data to columns.

    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    coaddname : :class:'str'
        The name of the coadd file corresponding to the zbest file used
        in make_new_zcat()
    qnp_model : :class:'h5.array'
        The array containing the pre-trained QuasarNP model.
    qnp_lines : :class:'list'
        A list containing the names of the emission lines that
        quasarnp.process_preds() should use.
    qnp_lines_bal : :class:'list'
        A list containing the names of the emission lines to check
        for BAL troughs.

    Returns
    -------
    :class:'~numpy.array'
        The zcat array with QuasarNP data included in the columns:

        * Z_QN        - The best QuasarNP redshift for the object
        * Z_QN_CONF   - The confidence of Z_QN
        * IS_QSO_QN   - A binary flag indicated object is a quasar
    """
    tmark('    Adding QuasarNP data')

    data, w = load_desi_coadd(coaddname)
    data = data[:, :, None]
    p = qnp_model.predict(data)
    c_line, z_line, zbest, *_ = process_preds(p, qnp_lines, qnp_lines_bal,
                                              verbose=False)

    cbest = np.array(c_line[c_line.argmax(axis=0), np.arange(len(zbest))])
    c_thresh = 0.5
    n_thresh = 1
    is_qso = np.sum(c_line > c_thresh, axis=0) >= n_thresh

    zcat['Z_QN'][w] = zbest
    zcat['Z_QN_CONF'][w] = cbest
    zcat['IS_QSO_QN'][w] = is_qso

    return zcat


def get_sq_model_fname(sqmodel_fname=None):
    """Convenience function to grab the $SQ_MODEL_FILE environment variable.

    Parameters
    ----------
    sqmodel_fname : :class:`str`, optional, defaults to $SQ_MODEL_FILE
        If `sqmodel_fname` is passed, it is returned from this function. If it's
        not passed, the $SQ_MODEL_FILE environment variable is returned.

    Returns
    -------
    :class:`str`
        If `sqmodel_fname` is passed, it is returned from this function. If it's
        not passed, the directory stored in the $SQ_MODEL_FILE environment
        variable is returned.
    """
    if sqmodel_fname is None:
        sqmodel_fname = os.environ.get('SQ_MODEL_FILE')
        # EBL check that the $SQ_MODEL_FILE environment variable is set.
        if sqmodel_fname is None:
            msg = "Pass sqmodel_fname or set $SQ_MODEL_FILE environment variable!"
            log.critical(msg)
            raise ValueError(msg)

    return sqmodel_fname


def load_sq_model(model_filename):
    """Convenience function for loading the SQUEzE model file.

    Parameters
    ----------
    model_filename : :class:'str'
        The filename and path of the SQUEzE model file. Either input by user
        or defaults to get_sq_model_fname().

    Returns
    -------
    :class:'~numpy.array'
        A numpy array of the SQUEzE model.

    Notes
    -----
    - The input model file needs to be in the json file format.
    """
    model = Model.from_json(load_json(model_filename))

    return model


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

        * Z_SQ        - The best redshift from SQUEzE for each object.
        * Z_SQ_CONF   - The confidence value of this redshift.
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
    zcat_args, sqdata_args = match(zcat['TARGETID'], sqdata_arr['TARGETID'])
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
        The zcat array with MgII Absorption data included in the columns:

        * Z_ABS        - The highest redshift of MgII absorption
        * Z_ABS_CONF   - The confidence value for this redshift.

    Notes
    -----
    - The original function was written by Lucas Napolitano (LGN) and
      modified for this script by Eleanor Lyke (EBL).
    """
    fitter = fitting.LevMarLSQFitter()
    model = models.Gaussian1D()
    # LGN Define constants
    first_line_wave = 2796.3543
    second_line_wave = 2803.5315
    rf_line_sep = second_line_wave - first_line_wave

    # LGN Define hyperparameters
    rf_err_margain = 0.50
    kernel_smooth = 2
    kernel = Gaussian1DKernel(stddev=kernel_smooth)
    med_filt_size = 19
    snr_threshold = 3.0
    qi_min = 0.01
    sim_fudge = 0.94

    # LGN Intialize output array.
    out_arr = []

    # LGN Read the coadd file and find targetid.
    specobj = read_spectra(coaddname)
    redrockfile = coaddname.replace('coadd', 'redrock').replace('.fits', '.h5')
    # LGN Get all targetids
    tids = specobj.target_ids()
    # LGN Run for every quasar target on the petal.
    num_rows = len(zcat)
    for specnum in range(num_rows):
        # LGN Grab a single targetid.
        targetid = tids[specnum]
        # LGN Open the redrock file and read in model fits for specific
        # targetid.
        targpath = f'/zfit/{targetid}/zfit'
        zalt = Table.read(redrockfile, path=targpath)
        # LGN If best spectype is a star we shouldn't process it.
        if zalt['spectype'][0] == 'STAR':
            out_arr.append([targetid, 0, 0])
            continue

        # LGN Define wavelength range and flux values.
        # LGN Check to see if b,r, and z cameras are already coadded.
        if "brz" in specobj.wave:
            x_spc = specobj.wave["brz"]
            y_flx = specobj.flux["brz"][specnum]
            y_err = np.sqrt(specobj.ivar["brz"][specnum])**(-1.0)
        # LGN If not, coadd them into "brz" using coadd_brz_cameras from
        # prospect docs.
        else:
            wave_arr = [specobj.wave["b"],
                        specobj.wave["r"],
                        specobj.wave["z"]]
            flux_arr = [specobj.flux["b"][specnum],
                        specobj.flux["r"][specnum],
                        specobj.flux["z"][specnum]]
            noise_arr = [np.sqrt(specobj.ivar["b"][specnum])**(-1.0),
                         np.sqrt(specobj.ivar["r"][specnum])**(-1.0),
                         np.sqrt(specobj.ivar["z"][specnum])**(-1.0)]

            x_spc, y_flx, y_err = coadd_brz_cameras(wave_arr, flux_arr,
                                                    noise_arr)

        # LGN Apply a gaussian smoothing kernel using hyperparameters
        # defined above.
        smooth_yflx = convolve(y_flx, kernel)
        # LGN Estimate the continuum using median filter.
        continuum = medfilt(y_flx, med_filt_size)

        # LGN Run the doublet finder.
        residual = continuum - y_flx

        # LGN Generate groups of data with positive residuals.
        # LGN/EBL: The following is from a stackoverlow thread:
        #     https://stackoverflow.com/questions/3149440/python-splitting-list-based-on-missing-numbers-in-a-sequence
        groups = []
        for k, g in groupby(enumerate(np.where(residual > 0)[0]), lambda x: x[0] - x[1]):
            groups.append(list(map(itemgetter(1), g)))

        # LGN Intialize the absorbtion line list.
        absorb_lines = []

        for group in groups:
            # LGN Skip groups of 1 or 2 data vals, these aren't worthwhile
            #    peaks and cause fitting issues.
            if len(group) < 3:
                continue

            # LGN Calculate the S/N value.
            snr = np.sum(residual[group]) * np.sqrt(np.sum(y_err[group]))**(-1.0)
            if snr > snr_threshold:
                # LGN Fit a gaussian model.
                model = models.Gaussian1D(amplitude=np.nanmax(residual[group]),
                                          mean=np.average(x_spc[group]))
                fm = fitter(model=model, x=x_spc[group], y=residual[group])
                # LGN Unpack the model fit data.
                amp, cen, stddev = fm.parameters

                absorb_lines.append([amp, cen, stddev, snr])

        # LGN Extract the highest z feature and associated quality index (QI)
        hz = 0
        hz_qi = 0
        # LGN This is particuarly poorly implemented, using range(len) so
        # I can slice to higher redshift lines only more easily.
        for counter in range(len(absorb_lines)):
            line1 = absorb_lines[counter]
            # LGN Determine redshift from model parameters.
            ztemp = (line1[1] * first_line_wave**(-1.0)) - 1

            # LGN If redshift is in any of the masked regions ignore it.
            if 2.189 < ztemp < 2.191 or 2.36 < ztemp < 2.40:
                continue
            # LGN Determine line seperation and error margain scaled to
            # redshift.
            line_sep = rf_line_sep * (1 + ztemp)
            err_margain = rf_err_margain * (1 + ztemp)
            # LGN for all lines at higher redshifts.
            for line2 in absorb_lines[counter+1:]:
                # LGN calculate error from expected line seperation
                # given the redshift of the first line.
                sep_err = np.abs(line2[1] - line1[1] - line_sep)
                # LGN Keep if within error margains.
                if sep_err < err_margain:
                    # LGN Calculate the QI.
                    # LGN S/N similarity of lines. sim_fudge is defined
                    #    in the hyperparameters above and
                    #    adjusts for the first line being larger,
                    #    kind of a fudge, won't lie.
                    snr_sim = sim_fudge * line1[3] * line2[3]**(-1.0)
                    # LGN Rescale to peak at lines having exact same S/N.
                    if snr_sim > 1:
                        snr_sim = snr_sim**(-1.0)
                    # LGN seperation accuracy
                    #   Is '1' if expected seperation = actual seperation.
                    #   Decreases to 0 outside this.
                    sep_acc = (1 - sep_err) * err_margain**(-1.0)
                    qi = snr_sim * sep_acc
                    if ztemp > hz and qi > qi_min:
                        hz = ztemp
                        hz_qi = qi

        out_arr.append([targetid, hz, hz_qi])

    # EBL Add the redshift and quality index for each targetid to the
    # zcat file passed to the function.
    out_arr = np.array(out_arr)
    zcat_args, abs_args = match(zcat['TARGETID'], out_arr[0])
    zcat['Z_ABS'][zcat_args] = out_arr[1][abs_args]
    zcat['Z_ABS_CONF'][zcat_args] = out_arr[2][abs_args]

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

        * Z_COMB        - The best models-combined redshift for each object.
        * Z_COMB_PROB   - The combined probability value of that redshift.
    """
    zcat['Z_COMB'][:] = zcat['Z']
    zcat['Z_COMB_PROB'][:] = 0.95

    return zcat


def zcat_writer(zcat, outputdir, outputname,
                qn_flag=False, sq_flag=False, abs_flag=False):
    """Writes the zcat structured array out as a FITS file.

    Parameters
    ----------
    zcat : :class:'~numpy.array'
        The structured array that was created by make_new_zcat()
    outputdir : :class:'str'
        The directory where the zcat file will be written.
    outputname : :class:'str'
        The filename of the zqso output file.
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
    tmark('    Creating output file...')

    # ADM create the necessary output directory, if it doesn't exist.
    os.makedirs(outputdir, exist_ok=True)

    # ADM construct the fill filename.
    full_outputname = os.path.join(outputdir, outputname)

    # ADM create the header and add the standard DESI dependencies.
    hdr = {}
    add_dependencies(hdr)

    # ADM add the specific lyazcat dependencies
    hdr['QN_ADDED'] = qn_flag
    hdr['SQ_ADDED'] = sq_flag
    hdr['AB_ADDED'] = abs_flag

    # ADM write out the data to the full file name.
    write_with_units(full_outputname, zcat, extname='QSOZCAT', header=hdr)

    return full_outputname


def create_zcat(tile, night, petal_num, zcatdir, outputdir, qn_flag=False,
                qnp_model=None, qnp_lines=None, qnp_lines_bal=None,
                sq_flag=False, squeze_model=None, abs_flag=False):
    """This will create a single zqso file from a set of user inputs.

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
        The root output directory used to construct the tile/night
        directory to which to write out the zcat file.
    qn_flag : :class:'bool', optional
        Flag to add QuasarNP data (or not) to the zcat file.
    qnp_model : :class:'h5 array', optional
        The QuasarNP model file to be used for line predictions.
    qnp_lines : :class:'list', optional
        The list of lines to use in the QuasarNP model to test against.
    qnp_lines : :class:'list', optional
        The list of BAL lines to use for QuasarNP to identify BALs.
    sq_flag : :class:'bool', optional
        Flag to add SQUEzE data (or not) to the zcat file.
    squeze_model : :class:'numpy.array', optional
        The numpy array for the SQUEzE model file.
    abs_flag : :class:'bool', optional
        Flag to add MgII Absorption data (or not) to the zcat file.

    Notes
    -----
    - Writes a FITS catalog that incorporates redrock, QuasarNP, SQUEzE, and MgII
      absorption redshifts and confidence values. This will write to the same
      directory of the zbest and coadd files unless a different output directory is
      passed.
    """
    # EBL Create the filepath for the input tile/night combination
    tiledir = os.path.join(zcatdir, tile)
    ymdir = os.path.join(tiledir, night)

    # ADM Create the corresponding output directory.
    outputdir = os.path.join(outputdir, tile, night)

    # EBL Create the filename tag that appends to zbest-*, coadd-*,
    # and zqso-* files.
    filename_tag = f'{petal_num}-{tile}-{night}.fits'
    # ADM try a couple of generic options for the file names.
    if not os.path.isfile(os.path.join(ymdir, f'zbest-{filename_tag}')):
        filename_tag = f'{petal_num}-{tile}-thru{night}.fits'

    zbestname = f'zbest-{filename_tag}'
    coaddname = f'coadd-{filename_tag}'
    outputname = f'zqso-{filename_tag}'

    fn = os.path.join(ymdir, zbestname)
    zcat = make_new_zcat(fn)
    if isinstance(zcat, bool):
        log.info('Petal Number has no corresponding zbest file: {}'.format(fn))
        if not os.path.isdir(ymdir):
            msg = "Directory doesn't exist: {}".format(ymdir)
            log.error(msg)
            raise FileNotFoundError(msg)
    else:
        if qn_flag:
            zcat = add_qn_data(zcat, os.path.join(ymdir, coaddname),
                               qnp_model, qnp_lines, qnp_lines_bal)
        if sq_flag:
            zcat = add_sq_data(zcat, os.path.join(ymdir, coaddname), squeze_model)
        if abs_flag:
            zcat = add_abs_data(zcat, os.path.join(ymdir, coaddname))

        fzcat = zcomb_selector(zcat)

        full_outputname = zcat_writer(fzcat, outputdir, outputname, qn_flag,
                                      sq_flag, abs_flag)

        tmark('    --{} written out correctly.'.format(full_outputname))
        log.info('='*79)
