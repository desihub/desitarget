# ADM code to make some example good and bad veto files.

if __name__ == "__main__":
    import os
    import shutil
    import numpy as np
    from astropy.table import Table
    from desitarget import io
    from desitarget.mtl import _get_mtl_nside

    # ADM tear everything down.
    shutil.rmtree("t/main", ignore_errors=True)

    # ADM output filenames.
    test_goodfn = "t/main/veto/bright1b/good/good-veto-bright1b.ecsv"
    test_badfn = "t/main/veto/bright1b/bad/bad-veto-bright1b.ecsv"

    # ADM test TARGETIDs we know are in the relevant test MTL files.
    tids = [34207057787749937, 34207057796137988, 34207058328814452]

    # ADM mock up some good and bad veto files.
    goodtable = Table()
    goodtable["TARGETID"] = tids
    goodtable["RA"] = [20.91128333333333, 20.646224999999998, 22.07405]
    goodtable["DEC"] = [42.24111666666667, 42.259119444444444, 42.254019444444445]
    goodtable["TIMESTAMP"] = ['2025-10-09T20:21:10+00:00',
                              '2025-10-13T23:33:13+00:00',
                              '2025-10-17T23:20:13+00:00']

    # ADM the bad test data is bad for 3 reasons.
    # ADM it's missing a required column.
    # ADM it contains a duplicate TARGETID.
    # ADM the timestamps are not in chronological order.
    badtable = Table()
    badtable["TARGETID"] = tids[:2] + tids[:1]
    badtable["RA"] = [20.91128333333333, 20.646224999999998, 22.07405]
    badtable["TIMESTAMP"] = ['2025-10-09T20:21:10+00:00',
                              '2025-10-13T23:33:13+00:00',
                              '2025-10-07T23:20:13+00:00']

    # ADM write the good and bad test data.
    os.makedirs("t/main/veto/bright1b/good", exist_ok=True)
    os.makedirs("t/main/veto/bright1b/bad", exist_ok=True)
    goodtable.write(test_goodfn, overwrite=True)
    badtable.write(test_badfn, overwrite=True)

    print(f"Wrote {test_goodfn} and {test_badfn}")

    # ADM some example MTL files we'll need from NERSC.
    mtldir = "/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/mtl"
    egmtl1fn = os.path.join(mtldir, "main/bright1b/mtl-bright1b-hp-661.ecsv")
    egmtl2fn = os.path.join(mtldir, "main/bright1b/mtl-bright1b-hp-704.ecsv")

    # ADM reading the initial ledger states guards against any updates
    # ADM having happened since the unit tests were written.
    egmtl1 = io.read_mtl_ledger(egmtl1fn, initial=True)
    egmtl2 = io.read_mtl_ledger(egmtl2fn, initial=True)

    # ADM mock up little mini-MTLs.
    mtldir = "t"
    nside = _get_mtl_nside()

    ii = np.array([tid in tids for tid in egmtl1["TARGETID"]])
    _, fn1 = io.write_mtl(mtldir, egmtl1[ii], ecsv=True, survey="main",
                        obscon="bright1b", nsidefile=nside, hpxlist=661)

    ii = np.array([tid in tids for tid in egmtl2["TARGETID"]])
    _, fn2 = io.write_mtl(mtldir, egmtl2[ii], ecsv=True, survey="main",
                          obscon="bright1b", nsidefile=nside, hpxlist=704)

    print(f"Also wrote {fn1} and {fn2}")
