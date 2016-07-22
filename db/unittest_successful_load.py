import psycopg2 
import argparse
import astropy
from astropy.io import fits
from astropy.table import vstack, Table

import sys, os, re, glob, distutils
from distutils.util import strtobool
import numpy as np

from desitarget.db import my_psql
import desitarget.bin.tractor_load as dbload 

def usetype(var,flag):
    if flag == 's': return str(var)
    elif flag == 'i': return int(var)
    elif flag == 'f': return float(var)
    else: raise ValueError

def diff_rows(trac_at,db_dict):
    '''trac_at -- tractor astropy table
    db_dict -- dict with psql text file columns as keys'''
    # DB keys
    db_keys= ['brickname','objid'] + [b+'flux' for b in ['g','r','z']]
    dtypes= ['s']+['i'] + ['f']*3
    # Tractor keys
    trac_keys= ['brickname','objid'] + ['decam_flux']*3
    trac_i= [None]*2 + [1,2,4]
    # Difference
    for db_key,typ,trac_key,i in zip(db_keys,dtypes,trac_keys,trac_i):
        print 'db key,val=',db_key, usetype(db_dict[db_key]), \
              'trac key,i,val=', trac_key,i,trac_at[trac_key][i]

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--list_of_cats",action="store",help='list of tractor cats',default='dr3_cats_qso.txt',required=True)
parser.add_argument("--seed",type=int,action="store",default=1,required=False)
parser.add_argument("--outdir",action="store",default='/project/projectdirs/desi/users/burleigh',required=False)
args = parser.parse_args()

# choose 10 cats randomly
fits_files= dbload.read_lines(args.list_of_cats)
rand = np.random.RandomState(args.seed)
if len(fits_files) > 10:
    ndraws=10
    keep= rand.uniform(1, len(fits_files), ndraws).astype(int)-1
    fits_files= fits_files[keep]
else: fits_files= [fits_files[0]]
# for each, choose a random objid and get corresponding row from DB
for fn in fits_files:
    t=Table(fits.getdata(fn, 1))
    ndraws=1
    keep= rand.uniform(1, len(t), ndraws).astype(int)-1
    for row in keep:
        # grab this row from DB
        brickname= '%s' % t[row]['brickname']
        objid= '%d' % t[row]['objid']
        cmd="select * from decam_table_cand as c JOIN decam_table_flux as f ON f.cand_id=c.id JOIN decam_table_aper as a ON a.cand_id=c.id JOIN decam_table_wise as w ON w.cand_id=c.id WHERE c.brickname like '%s' and c.objid=%s" % (brickname,objid) 
        name='db_row_%d.csv' % row
        print "selecting row %d from db with cmd:\n%s\nand saving output as %s" % \
                (row,cmd,os.path.join(args.outdir,name))
        my_psql.select(cmd,name,outdir=args.outdir)
# Compare Tractor Catalogue data to db
print 'reading in %s' % os.path.join(args.outdir,name)
db=  my_psql.read_psql_csv(os.path.join(args.outdir,name))
print 'comparing to Tractor catalogue'
print 'trac=',trac[row]
print 'db=',db
diff_rows(t[row],db)
    
print 'done'
