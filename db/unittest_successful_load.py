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

def diff_rows(trac_at,db_dict):
    '''trac_at -- tractor astropy table
    db_dict -- dict with psql text file columns as keys'''
    # There are 4 tables in the db
    # Decam table
    for db_key,trac_key,trac_i in zip(*dbload.decam_table_keys()):
        assert(decam[db_key] == tractor[trac_key][:,trac_i].data)
    ## Aperature table
    #for db_key,trac_key,trac_i,ap_i in zip(*dbload.aper_table_keys()):
    #    aper[db_key]= tractor[trac_key][:,trac_i,ap_i].data
    ## Wise table
    #for db_key,trac_key,trac_i in zip(*dbload.wise_default_keys()):
    #    wise[db_key]= tractor[trac_key][:,trac_i].data
    #if 'wise_lc_flux' in tractor.colnames:
    #    for db_key,trac_key,trac_i,epoch_i in zip(*dbload.wise_lc_keys()):
    #        wise[db_key]= tractor[trac_key][:,trac_i,epoch_i].data
    ## Candidate table
    #for db_key,trac_key in zip(*dbload.cand_default_keys()):
    #    cand[db_key]= tractor[trac_key].data 
    #for db_key,trac_key,trac_i in zip(*dbload.cand_array_keys()):
    #    cand[db_key]= tractor[trac_key][:,trac_i].data
    

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
        print "matching to brickname,objid=",t[row]['brickname'].data[0],t[row]['objid'].data[0]
        cmd="select * from decam_table_cand as c JOIN decam_table_flux as f ON f.cand_id=c.id JOIN decam_table_aper as a ON a.cand_id=c.id JOIN decam_table_wise as w ON w.cand_id=c.id WHERE c.brickname like '%s' and c.objid=%d" % (t[row]['brickname'].data[0],t[row]['objid'].data[0])  
        name='auto.txt'
        #my_psql.select(cmd,name,outdir=args.outdir)
# read in psql output
#db= my_psql.read_from_psql_file(os.path.join(args.outdir,name))
# compare tractor cat row to DB row
#print "psql info= ",db
#diff_rows(t,db)
    
print 'done'
