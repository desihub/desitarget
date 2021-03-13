import fitsio
import numpy as np

import joblib

from desitarget.myRF import myRF
from pkg_resources import resource_filename

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# number of variables
nfeatures = 11


def read_file(inputFile):

    sample = fitsio.read(inputFile, columns=['RA', 'DEC', 'TYPE', 'zred', 'g_r',
                                             'r_z', 'g_z', 'g_W1', 'r_W1',
                                             'z_W1', 'g_W2', 'r_W2', 'z_W2',
                                             'W1_W2', 'r'], ext=1)

    # We keep only Correct Candidates.
    reduce_sample = sample[(((sample['TYPE'][:] == 'PSF ') |
                             (sample['TYPE'][:] == 'PSF')) &
                            (sample['r'][:] < 23.0) &
                            (sample['r'][:] > 0.0))]

    print("\n############################################")
    print('Input file = ', inputFile)
    print('Original size: ', len(sample))
    print('Reduce size: ', len(reduce_sample))
    print("############################################\n")

    return reduce_sample


def build_attributes(nbEntries, nfeatures, sample):

    colors = np.zeros((nbEntries, nfeatures))

    colors[:, 0] = sample['g_r'][:]
    colors[:, 1] = sample['r_z'][:]
    colors[:, 2] = sample['g_z'][:]
    colors[:, 3] = sample['g_W1'][:]
    colors[:, 4] = sample['r_W1'][:]
    colors[:, 5] = sample['z_W1'][:]
    colors[:, 6] = sample['g_W2'][:]
    colors[:, 7] = sample['r_W2'][:]
    colors[:, 8] = sample['z_W2'][:]
    colors[:, 9] = sample['W1_W2'][:]
    colors[:, 10] = sample['r'][:]

    return colors


def compute_proba_desitarget(sample):

    attributes = build_attributes(len(sample), nfeatures, sample)

    print("NOT FINAL VERSION 888")
    pathToRF = resource_filename('desitarget', 'data')
    rf_fileName = pathToRF + f'/rf_model_dr9.npz'
    rf_Highz_fileName = pathToRF + f'/rf_model_dr9_HighZ.npz'

    print('Load Old Random Forest : ')
    print('    * ' + rf_fileName)
    print('    * ' + rf_Highz_fileName)
    print('Random Forest over : ', len(attributes), ' objects\n')

    myrf = myRF(attributes, pathToRF, numberOfTrees=500, version=2)
    myrf.loadForest(rf_fileName)
    proba_rf = myrf.predict_proba()

    myrf_Highz = myRF(attributes, pathToRF, numberOfTrees=500, version=2)
    myrf_Highz.loadForest(rf_Highz_fileName)
    proba_Highz_rf = myrf_Highz.predict_proba()

    return proba_rf, proba_Highz_rf


def compute_proba(sample, RF_file, RF_Highz_file):

    attributes = build_attributes(len(sample), nfeatures, sample)

    print('Load Random Forest: ')
    print('    * ' + RF_file)
    print('    * ' + RF_Highz_file)
    print('Random Forest over: ', len(attributes), ' objects\n')

    RF = joblib.load(RF_file)
    proba_rf = RF.predict_proba(attributes)[:, 1]
    feature_imp = RF.feature_importances_

    RF_Highz = joblib.load(RF_Highz_file)
    proba_rf_Highz = RF_Highz.predict_proba(attributes)[:, 1]
    feature_imp_Highz = RF_Highz.feature_importances_

    return proba_rf, feature_imp, proba_rf_Highz, feature_imp_Highz


def plot_importance_feature(feature_imp, feature_imp_Highz, show=True, save=True,
                            savename='Res_Compare/importance_feature.pdf'):

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    labels = ['g_r', 'r_z', 'g_z', 'g_W1', 'r_W1', 'z_W1', 'g_W2', 'r_W2', 'z_W2', 'W1_W2', 'r']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, feature_imp, width, label=f'dr9')
    rects2 = ax.bar(x + width/2, feature_imp_Highz, width, label=f'dr9_Highz')

    ax.set_ylabel('Feature Importance Score')
    ax.set_xlabel('Features')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()


def plot_cut_selection(r, sel, sel_qso, sel_Highz, proba_rf, proba_Highz_rf,
                       show=True, save=True, savename='Res_Compare/cut_selection.pdf'):
    fig = plt.figure(1, figsize=(12.0, 8.0))

    gs = GridSpec(nrows=4, ncols=2, figure=fig)
    ax_hist_21 = fig.add_subplot(gs[0, 0])
    ax_hist_22 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1:, 0])
    ax4 = fig.add_subplot(gs[1:, 1])

    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.92,
                        wspace=0.08, hspace=0.10)

    rmag = np.linspace(17, 23, 1000)

    cut = [0.75 - 0.05*np.tanh(x - 20.5) for x in rmag]
    cut_Highz = [0.5 for x in rmag]

    hist = [r[sel & ~sel_qso], r[sel & sel_qso], r[~sel & sel_qso]]
    ax_hist_21.hist(hist, bins=50, range=(17., 23.), color=['gray', 'blue', 'red'],
                    stacked=True, label=['not QS0', 'QSO', 'QS0 not selected'])
    ax_hist_21.set_ylabel('# objects', labelpad=8)
    ax_hist_21.set_xticklabels([])
    ax_hist_21.set_yticks([0, 1000, 2000])
    ax_hist_21.set_ylim([0, 2500])
    ax_hist_21.xaxis.set_ticks_position('bottom')

    ax3.scatter(r[~sel_qso & ~sel][::10], proba_rf[~sel_qso & ~sel][::10], color='silver', label='not QSOs not selected')
    ax3.scatter(r[~sel_qso & sel][::10], proba_rf[~sel_qso & sel][::10], color='grey', label='not QSOs selected ')
    ax3.scatter(r[sel_qso & ~sel], proba_rf[sel_qso & ~sel], color='red', label='QSOs not selected')
    ax3.scatter(r[sel_qso & sel], proba_rf[sel_qso & sel], color='blue', label='QSOs selected')
    ax3.plot(rmag, cut, '--k', label='cut')
    ax3.legend(loc='lower left')
    ax3.set_xlabel('r mag')
    ax3.xaxis.set_ticks_position('both')
    ax3.set_ylabel(f'RF_DR9s Probability', labelpad=14)

    hist = [r[~sel_qso & ~sel & sel_Highz], r[sel_qso & ~sel & sel_Highz], r[sel_qso & ~sel & ~sel_Highz]]
    ax_hist_22.hist(hist, bins=50, range=(17., 23.), color=['gray', 'blue', 'red'],
                    stacked=True, label=['not QS0', 'QSO', 'QS0 not selected'])
    ax_hist_22.set_ylabel('# objects', rotation=270, labelpad=19)
    ax_hist_22.set_xticklabels([])
    ax_hist_22.set_yticks([0, 100, 200])
    ax_hist_22.set_ylim([0, 250])
    ax_hist_22.xaxis.set_ticks_position('both')
    ax_hist_22.yaxis.set_ticks_position('right')
    ax_hist_22.yaxis.set_label_position('right')

    ax4.scatter(r[~sel_qso & ~sel & ~sel_Highz][::10], proba_Highz_rf[~sel_qso & ~sel & ~sel_Highz][::10], color='silver', label='no QSOs not selected')
    ax4.scatter(r[~sel_qso & ~sel & sel_Highz][::10], proba_Highz_rf[~sel_qso & ~sel & sel_Highz][::10], color='grey', label='no QSOs selected ')
    ax4.scatter(r[sel_qso & ~sel & ~sel_Highz], proba_Highz_rf[sel_qso & ~sel & ~sel_Highz], color='red', label='QSOs not selected after 2 RF')
    ax4.scatter(r[sel_qso & ~sel & sel_Highz], proba_Highz_rf[sel_qso & ~sel & sel_Highz], color='blue', label='QSOs selected')
    ax4.plot(rmag, cut_Highz, '--k', label='cut')
    ax4.legend(loc='lower left')
    ax4.set_xlabel('r mag')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.yaxis.set_ticks_position('right')
    ax4.yaxis.set_label_position('right')
    ax4.set_ylabel(f'RF_DR9s_HighZ Probability', rotation=270, labelpad=22)

    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()


def hist_ratio(n_x, n_y, bins):

    x = n_x.astype(float)
    y = n_y.astype(float)
    ratio = y/x
    bin_centers = (bins[:-1] + bins[1:])/2.
    # error in binomial case
    errors = np.sqrt(y*(x-y)/x**3)
    errors[errors == 0] = y[errors == 0]/(x[errors == 0]+1)**3

    return bin_centers, ratio, errors


def plot_completness(r, zred, sel_qso, sel_tot_1, label1, sel_tot_2=np.zeros(1), label2=None, N_bins_r=10, N_bins_z=10,
                     show=True, save=True, savename='completeness.pdf'):
    fig = plt.figure(1, figsize=(10.0, 4.0))

    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    n_r_ref, bins_r = np.histogram(r[sel_qso], bins=N_bins_r, range=(17., 23.))
    n_r, bins_r = np.histogram(r[sel_qso & sel_tot_1], bins=N_bins_r, range=(17., 23.))
    bin_centers, ratio, errors = hist_ratio(n_r_ref, n_r, bins_r)
    ax1.errorbar(x=bin_centers, y=ratio, yerr=errors, linestyle='none',
                 marker='.', color='green', label=label1)
    if np.size(sel_tot_2) > 1:
        n_r, bins_r = np.histogram(r[sel_qso & sel_tot_2], bins=N_bins_r, range=(17., 23.))
        bin_centers, ratio, errors = hist_ratio(n_r_ref, n_r, bins_r)
        ax1.errorbar(x=bin_centers, y=ratio, yerr=errors, linestyle='none',
                     marker='.', color='red', label=label2)
    ax1.set_xlabel('r mag')
    ax1.set_ylabel('Completeness')
    ax1.legend(loc='lower left')

    n_z_ref, bins_z = np.histogram(zred[sel_qso], bins=N_bins_z, range=(0., 4.))
    n_z, bins_z = np.histogram(zred[sel_qso & sel_tot_1], bins=N_bins_z, range=(0., 4.))
    bin_centers, ratio, errors = hist_ratio(n_z_ref, n_z, bins_z)
    ax2.errorbar(x=bin_centers, y=ratio, yerr=errors, linestyle='none',
                 marker='.', color='green', label=label1)
    if np.size(sel_tot_2) > 1:
        n_z, bins_z = np.histogram(zred[sel_qso & sel_tot_2], bins=N_bins_z, range=(0., 4.))
        bin_centers, ratio, errors = hist_ratio(n_z_ref, n_z, bins_z)
        ax2.errorbar(x=bin_centers, y=ratio, yerr=errors, linestyle='none',
                     marker='.', color='red', label=label2)
    ax2.set_xlabel('redshift')
    ax2.legend(loc='lower left')

    if save:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()


def make_some_tests_and_plots(inputFile, RF_file, RF_Highz_file, rmax, cut=[0.7, 0.05, 20.5, 0.5]):

    # Load data.
    test_sample = read_file(inputFile)

    # RF output.
    proba_rf, feature_imp, proba_Highz_rf, feature_imp_Highz = compute_proba(
        test_sample, RF_file, RF_Highz_file)

    # Magnitude and Geometry.
    zred = test_sample['zred'][:]
    r = test_sample['r'][:]
    ra = test_sample['RA'][:]
    dec = test_sample['DEC'][:]

    r_mag_min, r_mag_max = np.min(r), np.max(r)
    r_mag_min_sel, r_mag_max_sel = 17.0, rmax

    r_sel = (r >= r_mag_min_sel) & (r <= r_mag_max_sel)

    # selection from r_magnitude
    zred = zred[r_sel]
    r = r[r_sel]
    ra = ra[r_sel]
    dec = dec[r_sel]

    proba_rf, proba_Highz_rf = proba_rf[r_sel], proba_Highz_rf[r_sel]

    print("############################################")
    print("R magnitude Elements from Test Sample : ")
    print("R_mag max = ", r_mag_max, " -- R_mag min = ", r_mag_min)
    print("R_mag max selected = ", r_mag_max_sel, " -- R_mag min selected = ", r_mag_min_sel)
    print("############################################\n")

    # on regarde des objets quasiment à l'horizon --> cos(theta) = 1
    ra_min, ra_max = np.min(ra), np.max(ra)
    dec_min, dec_max = np.min(dec), np.max(dec)
    surface = (ra_max - ra_min)*(dec_max - dec_min)

    print("############################################")
    print("Geomeric Elements from Test Sample : ")
    print("RA max = ", ra_max, " -- RA min = ", ra_min)
    print("DEC max = ", dec_max, " -- DEC min = ", dec_min)
    print("Surface = ", surface)
    print("############################################\n")

    sel_qso = zred > 0
    print(f"\n[INFO] There are {sel_qso.sum()} quasars on the test sample ... \n")
    sel_qso_highz = zred > 3
    sel_qso_2 = zred > 2

    print(f"\n[INFO] :  cut = {cut[0]} - {cut[1]}*np.tanh(r - {cut[2]}) & cut_Highz = {cut[3]}\n")
    cut = cut[0] - cut[1]*np.tanh(r - cut[2])
    cut_Highz = cut[3]

    sel = proba_rf > cut
    sel_Highz = (proba_Highz_rf > cut_Highz) & ~sel
    sel_tot = sel + sel_Highz

    density = float(len(r[sel]))/surface
    effi = float(len(r[sel & sel_qso])) / float(len(r[sel_qso]))

    density_Highz = float(len(r[sel_Highz])) / surface
    effi_Highz = float(len(r[sel_Highz & sel_qso])) / float(len(r[sel_qso]))

    density_tot = float(len(r[sel_tot])) / surface
    effi_tot = float(len(r[sel_tot & sel_qso])) / float(len(r[sel_qso]))

    print('\n############################################')
    print(f'density dr9 = ', density, f' deg^-2 completeness dr9', effi)
    print(f'density dr9 Highz = ', density_Highz, f' deg^-2 completeness dr9 Highz', effi_Highz)
    print(f'density dr9 Total = ', density_tot, f' deg^-2 completeness dr9 Total', effi_tot)
    print('############################################\n')

    print('[INFO] TAKE CARE THE TEST REGION IS IN DES --> NEED TO TUNE INDEPENTENDLY EACH FOOTPRINT')
    print('[INFO] TEST REGION A LITTLE BIT CONTAMINATED BY STARS, FORCE HIGHER DENSITY TO GET 260 targets by deg2 IN THE REST OF THE FOOTPRINT\n')

    # Plots.
    plot_importance_feature(feature_imp, feature_imp_Highz, True, False)
    plot_cut_selection(r, sel, sel_qso, sel_Highz, proba_rf, proba_Highz_rf, True, False)
    plot_completness(r, zred, sel_qso, sel_tot, '', N_bins_r=40, N_bins_z=40, show=True, save=False)


def make_some_tests_and_plots_2_training(inputFile, RF_file_1, RF_Highz_file_1, RF_file_2, RF_Highz_file_2, cut1, cut2,
                                         r_mag_max_sel=23.0, surface_vi=True, save=False, label1='1', label2='2'):
    # Load data.
    test_sample = read_file(inputFile)

    # RF output.
    proba_rf_1, feature_imp_1, proba_Highz_rf_1, feature_imp_Highz_1 = compute_proba(
        test_sample, RF_file_1, RF_Highz_file_1)

    # RF output.
    proba_rf_2, feature_imp_2, proba_Highz_rf_2, feature_imp_Highz_2 = compute_proba(
        test_sample, RF_file_2, RF_Highz_file_2)

    # Magnitude and Geometry.
    zred = test_sample['zred'][:]
    r = test_sample['r'][:]
    ra = test_sample['RA'][:]
    dec = test_sample['DEC'][:]

    r_mag_min, r_mag_max = np.min(r), np.max(r)
    r_mag_min_sel = 17.5

    r_sel = (r >= r_mag_min_sel) & (r <= r_mag_max_sel)

    # selection from r_magnitude
    zred = zred[r_sel]
    r = r[r_sel]
    ra = ra[r_sel]
    dec = dec[r_sel]

    proba_rf_1, proba_Highz_rf_1 = proba_rf_1[r_sel], proba_Highz_rf_1[r_sel]
    proba_rf_2, proba_Highz_rf_2 = proba_rf_2[r_sel], proba_Highz_rf_2[r_sel]

    print("############################################")
    print("R magnitude Elements from Test Sample : ")
    print("R_mag max = ", r_mag_max, " -- R_mag min = ", r_mag_min)
    print("R_mag max selected = ", r_mag_max_sel, " -- R_mag min selected = ", r_mag_min_sel)
    print("############################################\n")

    if surface_vi:
        surface = 150 + 1.4  # we add 360 qso --> we say it is has a surface of 1.4 deg^2
    else:
        # on regarde des objets quasiment à l'horizon --> cos(theta) = 1
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)
        surface = (ra_max - ra_min)*(dec_max - dec_min)

        print("############################################")
        print("Geomeric Elements from Test Sample : ")
        print("RA max = ", ra_max, " -- RA min = ", ra_min)
        print("DEC max = ", dec_max, " -- DEC min = ", dec_min)
        print("Surface = ", surface)
        print("############################################\n")

    sel_qso = zred > 0
    print(f"\n[INFO] There are {sel_qso.sum()} quasars on the test sample ... \n")
    sel_qso_highz = zred > 3

    plt.figure()
    plt.hist(zred[sel_qso], bins=30, label="QSOs")
    plt.legend()
    plt.xlabel('zred')
    plt.show()

    print(f"\n[INFO] CUT1 :  cut = {cut1[0]} - {cut1[1]}*np.tanh(r - {cut1[2]}) & cut_Highz = {cut1[3]}\n")
    cut_1 = cut1[0] - cut1[1]*np.tanh(r - cut1[2])
    cut_Highz_1 = cut1[3]

    sel_1 = proba_rf_1 > cut_1
    sel_Highz_1 = (proba_Highz_rf_1 > cut_Highz_1) & ~sel_1
    sel_tot_1 = sel_1 + sel_Highz_1

    density_1 = float(len(r[sel_1]))/surface
    effi_1 = float(len(r[sel_1 & sel_qso])) / float(len(r[sel_qso]))

    density_Highz_1 = float(len(r[sel_Highz_1])) / surface
    effi_Highz_1 = float(len(r[sel_Highz_1 & sel_qso])) / float(len(r[sel_qso]))

    density_tot_1 = float(len(r[sel_tot_1])) / surface
    effi_tot_1 = float(len(r[sel_tot_1 & sel_qso])) / float(len(r[sel_qso]))

    print('\n############################################')
    print(f'density dr9 = ', density_1, f' deg^-2 completeness dr9', effi_1)
    print(f'density dr9 Highz = ', density_Highz_1, f' deg^-2 completeness dr9 Highz', effi_Highz_1)
    print(f'density dr9 Total = ', density_tot_1, f' deg^-2 completeness dr9 Total', effi_tot_1)
    print('############################################\n')

    print(f"\n[INFO] CUT2 :  cut = {cut2[0]} - {cut2[1]}*np.tanh(r - {cut2[2]}) & cut_Highz = {cut2[3]}\n")
    cut_2 = cut2[0] - cut2[1]*np.tanh(r - cut2[2])
    cut_Highz_2 = cut2[3]

    sel_2 = proba_rf_2 > cut_2
    sel_Highz_2 = (proba_Highz_rf_2 > cut_Highz_2) & ~sel_2
    sel_tot_2 = sel_2 + sel_Highz_2

    density_2 = float(len(r[sel_2]))/surface
    effi_2 = float(len(r[sel_2 & sel_qso])) / float(len(r[sel_qso]))

    density_Highz_2 = float(len(r[sel_Highz_2])) / surface
    effi_Highz_2 = float(len(r[sel_Highz_2 & sel_qso])) / float(len(r[sel_qso]))

    density_tot_2 = float(len(r[sel_tot_2])) / surface
    effi_tot_2 = float(len(r[sel_tot_2 & sel_qso])) / float(len(r[sel_qso]))

    print('\n############################################')
    print(f'density dr9 = ', density_2, f' deg^-2 completeness dr9', effi_2)
    print(f'density dr9 Highz = ', density_Highz_2, f' deg^-2 completeness dr9 Highz', effi_Highz_2)
    print(f'density dr9 Total = ', density_tot_2, f' deg^-2 completeness dr9 Total', effi_tot_2)
    print('############################################\n')

    plot_completness(r, zred, sel_qso, sel_tot_1, label1, sel_tot_2=sel_tot_2, label2=label2, N_bins_r=40, N_bins_z=40, show=True, save=save)


def new_training_versus_desitarget(inputFile, RF_file_new, RF_Highz_file_new, cut, r_mag_max_sel=23.0, surface_vi=True):
    # Load data.
    test_sample = read_file(inputFile)

    # RF output.
    proba_rf_new, feature_imp_new, proba_Highz_rf_new, feature_imp_Highz_new = compute_proba(
        test_sample, RF_file_new, RF_Highz_file_new)

    # RF output. trop lent desitarget p***$*
    # proba_rf_ref, proba_Highz_rf_ref = compute_proba_desitarget(test_sample)
    RF_file_ref = '/global/cfs/cdirs/desi/target/analysis/RF/RFmodel/DR9s_LOW/model_DR9s_LOW_z[0.0, 6.0]_MDepth25_MLNodes850_nTrees500.pkl.gz'
    RF_Highz_file_ref = '/global/cfs/cdirs/desi/target/analysis/RF/RFmodel/DR9s_HighZ/model_DR9s_HighZ_z[3.2, 6.0]_MDepth25_MLNodes850_nTrees500.pkl.gz'
    proba_rf_ref, _, proba_Highz_rf_ref, _ = compute_proba(test_sample, RF_file_ref, RF_Highz_file_ref)

    # Magnitude and Geometry.
    zred = test_sample['zred'][:]
    r = test_sample['r'][:]
    ra = test_sample['RA'][:]
    dec = test_sample['DEC'][:]

    r_mag_min, r_mag_max = np.min(r), np.max(r)
    r_mag_min_sel = 17.5

    r_sel = (r >= r_mag_min_sel) & (r <= r_mag_max_sel)

    # selection from r_magnitude
    zred = zred[r_sel]
    r = r[r_sel]
    ra = ra[r_sel]
    dec = dec[r_sel]

    proba_rf_new, proba_Highz_rf_new = proba_rf_new[r_sel], proba_Highz_rf_new[r_sel]
    proba_rf_ref, proba_Highz_rf_ref = proba_rf_ref[r_sel], proba_Highz_rf_ref[r_sel]

    print("############################################")
    print("R magnitude Elements from Test Sample : ")
    print("R_mag max = ", r_mag_max, " -- R_mag min = ", r_mag_min)
    print("R_mag max selected = ", r_mag_max_sel, " -- R_mag min selected = ", r_mag_min_sel)
    print("############################################\n")

    if surface_vi:
        surface = 150 + 1.4  # we add 360 qso --> we say it is has a surface of 1.4 deg^2
        print("############################################")
        print("SURFACE = ", surface)
        print("############################################\n")
    else:
        # on regarde des objets quasiment à l'horizon --> cos(theta) = 1
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)
        surface = (ra_max - ra_min)*(dec_max - dec_min)

        print("############################################")
        print("Geomeric Elements from Test Sample : ")
        print("RA max = ", ra_max, " -- RA min = ", ra_min)
        print("DEC max = ", dec_max, " -- DEC min = ", dec_min)
        print("Surface = ", surface)
        print("############################################\n")

    sel_qso = zred > 0
    print(f"\n[INFO] There are {sel_qso.sum()} quasars on the test sample ... \n")
    sel_qso_2 = zred > 2
    sel_qso_highz = zred > 3

    plt.figure()
    plt.hist(zred[sel_qso], bins=30, label="QSOs")
    plt.legend()
    plt.xlabel('zred')
    plt.show()

    print('\n############ NEW #####################')
    print(f"\n[INFO] FOR NEW :  cut = {cut[0]} - {cut[1]}*np.tanh(r - {cut[2]}) & cut_Highz = {cut[3]}\n")
    cut_new = cut[0] - cut[1]*np.tanh(r - cut[2])
    cut_Highz_new = cut[3]

    sel_new = proba_rf_new > cut_new
    sel_Highz_new = (proba_Highz_rf_new > cut_Highz_new) & ~sel_new
    sel_tot_new = sel_new + sel_Highz_new

    density_new = float(len(r[sel_new]))/surface
    effi_new = float(len(r[sel_new & sel_qso])) / float(len(r[sel_qso]))
    density_test_new = float(len(r[sel_new & sel_qso_highz]))/surface
    effi_test_new = float(len(r[sel_new & sel_qso_highz])) / float(len(r[sel_qso_highz]))

    density_Highz_new = float(len(r[sel_Highz_new])) / surface
    effi_Highz_new = float(len(r[sel_Highz_new & sel_qso])) / float(len(r[sel_qso]))
    density_Highz_test_new = float(len(r[sel_Highz_new & sel_qso_highz])) / surface
    effi_Highz_test_new = float(len(r[sel_Highz_new & sel_qso_highz])) / float(len(r[sel_qso_highz]))

    density_tot_new = float(len(r[sel_tot_new])) / surface
    effi_tot_new = float(len(r[sel_tot_new & sel_qso])) / float(len(r[sel_qso]))

    print(f'density dr9 = ', density_new, f' deg^-2 completeness dr9', effi_new, ' density dr9 QSO>3.0 : ', density_test_new, ' (ie) ', effi_test_new*100, "% de l'echantillong de test")
    print(f'density dr9 Highz = ', density_Highz_new, f' deg^-2 completeness dr9 Highz', effi_Highz_new, ' densisty dr9 Highz QSO>3.0 : ', density_Highz_test_new, ' (ie) ', effi_Highz_test_new*100, "% de l'echantillong de test")
    print(f'density dr9 Total = ', density_tot_new, f' deg^-2 completeness dr9 Total', effi_tot_new)
    print('############################################\n')

    print('\n########## REFERENCE ############################')
    print("\n[INFO] (WARNING : USE THE SAME CUT THAN TEST TO COMPARE WITH THE CUT THAT I WILL FIX FOR DES IN THE NEW TRAINING) reference cut : cut = 0.75 - 0.05*np.tanh(r - 20.5) & cut_Highz = 0.55  \n")

    cut_ref = 0.75 - 0.05*np.tanh(r - 20.5)
    cut_Highz_ref = 0.55

    sel_ref = proba_rf_ref > cut_ref
    sel_Highz_ref = (proba_Highz_rf_ref > cut_Highz_ref) & ~sel_ref
    sel_tot_ref = sel_ref + sel_Highz_ref

    density_ref = float(len(r[sel_ref]))/surface
    effi_ref = float(len(r[sel_ref & sel_qso])) / float(len(r[sel_qso]))
    density_test_ref = float(len(r[sel_ref & sel_qso_highz]))/surface
    effi_test_ref = float(len(r[sel_ref & sel_qso_highz])) / float(len(r[sel_qso_highz]))

    density_Highz_ref = float(len(r[sel_Highz_ref])) / surface
    effi_Highz_ref = float(len(r[sel_Highz_ref & sel_qso])) / float(len(r[sel_qso]))
    density_Highz_test_ref = float(len(r[sel_Highz_ref & sel_qso_highz])) / surface
    effi_Highz_test_ref = float(len(r[sel_Highz_ref & sel_qso_highz])) / float(len(r[sel_qso_highz]))

    density_tot_ref = float(len(r[sel_tot_ref])) / surface
    effi_tot_ref = float(len(r[sel_tot_ref & sel_qso])) / float(len(r[sel_qso]))

    print(f'density dr9 = ', density_ref, f' deg^-2 completeness dr9', effi_ref, ' density dr9 QSO>3.0 : ', density_test_ref, ' (ie) ', effi_test_ref*100, "% de l'echantillong de test")
    print(f'density dr9 Highz = ', density_Highz_ref, f' deg^-2 completeness dr9 Highz', effi_Highz_ref, ' densisty dr9 Highz QSO>3.0 : ', density_Highz_test_ref, ' (ie) ', effi_Highz_test_ref*100, "% de l'echantillong de test")
    print(f'density dr9 Total = ', density_tot_ref, f' deg^-2 completeness dr9 Total', effi_tot_ref)
    print('############################################\n')

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.scatter(r[~sel_qso], proba_rf_new[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(r[sel_qso], proba_rf_new[sel_qso], s=1, color='blue', label='True QSO')
    plt.plot(np.arange(r_mag_min, r_mag_max, 1000), cut[0] - cut[1]*np.tanh(np.arange(r_mag_min, r_mag_max, 1000) - cut[2]), color='black')
    plt.xlabel('r')
    plt.ylabel('proba rf new')
    plt.legend()
    plt.xlim(17, 23)
    plt.ylim(0, 1)

    plt.subplot(122)
    plt.scatter(r[~sel_qso], proba_rf_ref[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(r[sel_qso], proba_rf_ref[sel_qso], s=1, color='blue', label='True QSO')
    # plt.plot(np.arange(r_mag_min, r_mag_max, 1000), cut[0] - cut[1]*np.tanh(np.arange(r_mag_min, r_mag_max, 1000) - cut[2]), color='black')
    plt.xlabel('r')
    plt.ylabel('proba rf ref')
    plt.legend()
    plt.xlim(17, 23)
    plt.ylim(0, 1)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.scatter(proba_rf_ref[~sel_qso], proba_rf_new[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(proba_rf_ref[sel_qso], proba_rf_new[sel_qso], s=1, color='blue', label='True QSO')
    plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlabel('proba rf ref')
    plt.ylabel('proba rf new')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(122)
    plt.scatter(proba_Highz_rf_ref[~sel_qso], proba_Highz_rf_new[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(proba_Highz_rf_ref[sel_qso], proba_Highz_rf_new[sel_qso], s=1, color='blue', label='True QSO')
    plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('proba rf Highz ref')
    plt.ylabel('proba rf Highz new')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.scatter(proba_rf_ref[~sel_qso], proba_Highz_rf_ref[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(proba_rf_ref[sel_qso], proba_Highz_rf_ref[sel_qso], s=1, color='blue', label='True QSO')
    # plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlabel('proba rf ref')
    plt.ylabel('proba rf Highz ref')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(122)
    plt.scatter(proba_rf_new[~sel_qso], proba_Highz_rf_new[~sel_qso], s=1, color='red', label='Stars')
    plt.scatter(proba_rf_new[sel_qso], proba_Highz_rf_new[sel_qso], s=1, color='blue', label='True QSO')
    # plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('proba rf new')
    plt.ylabel('proba rf Highz new')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.scatter(proba_rf_ref[sel_qso_2 & ~sel_qso_highz], proba_Highz_rf_ref[sel_qso_2 & ~sel_qso_highz], s=1, color='gold', label='QSO (z > 2)')
    plt.scatter(proba_rf_ref[sel_qso_highz], proba_Highz_rf_ref[sel_qso_highz], s=1, color='blue', label='QSO (z > 3)')
    # plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlabel('proba rf ref')
    plt.ylabel('proba rf Highz ref')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.subplot(122)
    plt.scatter(proba_rf_new[sel_qso_2 & ~sel_qso_highz], proba_Highz_rf_new[sel_qso_2 & ~sel_qso_highz], s=1, color='gold', label='QSO (z > 2)')
    plt.scatter(proba_rf_new[sel_qso_highz], proba_Highz_rf_new[sel_qso_highz], s=1, color='blue', label='QSO (z > 3)')
    # plt.plot([0, 1], [0, 1], ls='--', color='black')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('proba rf new')
    plt.ylabel('proba rf Highz new')
    plt.show()

    plot_completness(r, zred, sel_qso, sel_tot_new, 'NEW', sel_tot_2=sel_tot_ref, label2='REF', N_bins_r=40, N_bins_z=40, show=True, save=False)
