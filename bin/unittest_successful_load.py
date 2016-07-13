import psycopg2 
import argparse
import astropy
from astropy.io import fits
from astropy.table import vstack, Table

import sys, os, re, glob, distutils
from distutils.util import strtobool
import numpy as np

from tractor_load import read_lines

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--list_of_cats",action="store",help='list of tractor cats',default='dr3_cats_qso.txt',required=False)
args = parser.parse_args()

# choose 10 cats randomly
fits_files= read_lines(args.list_of_cats)
rand = np.random.RandomState(seed)
ndraws=10
keep= rand.uniform(1, len(fits_files), ndraws).astype(int)-1
fits_files= fits_files[keep]
# for each, choose a random objid and get corresponding row from DB
for fn in [fits_files[0]]:
    t=Table(fits.getdata(fn, 1))
    ndraws=1
    keep= rand.uniform(1, len(t), ndraws).astype(int)-1
    t= t[keep]
    #grab from DB
    query="select * from decam_cand as c JOIN decam_decam as d ON d.cand_id=c.id JOIN decam_aper as a ON a.cand_id=c.id JOIN decam_wise as w ON w.cand_id=c.id WHERE c.brickname like '%s' and c.objid=%d" % (t['brickname'],t['objid'])  
    


print "args.tractor_catalog= ",args.tractor_catalog

fitsbin = args.tractor_catalog
table = fits.open( fitsbin )
# Access the data part of the table.
trac = table[1].data
trac = np.asarray(trac)
ans={}
i=1000
ans['cand']=trac['blob'][i],trac['ra'][i],trac['dec'][i],trac['dchisq'][i][1],trac['dchisq'][i][3]
ans['decam']=trac['decam_nobs'][i][4],trac['decam_flux'][i][1],trac['decam_flux'][i][2],trac['decam_flux'][i][4]
ans['decam_aper']=trac['decam_apflux'][i][1][0],trac['decam_apflux'][i][2][0],trac['decam_apflux'][i][4][0],trac['decam_apflux'][i][4][7]
ans['wise']=trac['wise_flux'][i][0],trac['wise_flux'][i][1],trac['wise_flux'][i][2],trac['wise_flux'][i][3],trac['wise_flux_ivar'][i][3]

#db
con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_user', database='desi')
cur = con.cursor()
def output(cursor,query):
    cursor.execute(query)
    return cursor.fetchall()

sql={}
sql['cand']= output(cur,"SELECT blob,ra,dec,dchisq2,dchisq4,id from dr2.candidate where objid=1000")
sql['decam']= output(cur,"SELECT gnobs,gflux,rflux,zflux from dr2.decam where cand_id=884")
sql['decam_aper']= output(cur,"SELECT gapflux_1,rapflux_1,zapflux_1,zapflux_8 from dr2.decam_aper where cand_id=884")
sql['wise']= output(cur,"SELECT w1flux,w2flux,w3flux,w4flux,w4flux_ivar from dr2.wise where cand_id=884")

for key in ans.keys():
    print '----',key.upper(),'----'
    print 'tractor cat: ',ans[key]
    print 'postgres   : ',sql[key][0]

#con.commit()
print 'done'

