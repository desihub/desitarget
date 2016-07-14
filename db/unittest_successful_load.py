import psycopg2 
import argparse
import astropy
from astropy.io import fits
from astropy.table import vstack, Table

import sys, os, re, glob, distutils
from distutils.util import strtobool
import numpy as np

from thesis_code.targets import read_from_psql_file
import desitarget.bin.tractor_load as dbload 

def diff_rows(trac_at,db_dict):
    '''trac_at -- tractor astropy table
    db_dict -- dict with psql text file columns as keys'''
    # There are 4 tables in the db
    # Decam table
    for db_key,trac_key in zip(*dbload.get_decam_keys()):
        for cnt,band in enumerate(['u','g','r','i','z','Y']): 
            decam[band+db_key]= tractor[trac_key][:,cnt].data
    # Aperature table
    for db_key,trac_key in zip(*dbload.get_aper_keys()):
        for cnt,band in enumerate(['u','g','r','i','z','Y']): 
            for ap in range(8):
                aper[band+db_key+str(ap+1)]= tractor[trac_key][:,cnt,ap].data
    # Wise table
    for db_key,trac_key in zip(*dbload.get_wise_keys()):
        # If light curve, w1,w2 only
        if '_lc_' in trac_key: 
            for cnt,band in enumerate(['w1','w2']):
                for iepoch,epoch in enumerate(['1','2','3','4','5']):
                    wise[band+db_key+epoch]= tractor[trac_key][:,cnt,iepoch].data
        else:
            for cnt,band in enumerate(['w1','w2','w3','w4']):
                wise[band+db_key]=tractor[trac_key][:,cnt].data
    # Candidate table
    for trac_key in dbload.get_cand_keys(trac_at.keys()):
        if trac_key == 'dchisq':
            for i in range(5): 
                cand[trac_key+str(i)]= tractor[trac_key][:,i].data
        else:
            cand[trac_key]= tractor[trac_key].data 

    print 'tractor keys: ',trac_at.colnames
    print 'db keys: ',db_dict.keys()
    print('get_decam_keys')
    for db_key,trac_key in zip(*dbload.get_decam_keys()): print('%s %s' % (db_key,trac_key))
    #dbload.get_cand_keys()
    #dbload.get_decam_keys()
    #dbload.get_aper_keys()
    #dbload.get_wise_keys()
    

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--list_of_cats",action="store",help='list of tractor cats',default='dr3_cats_qso.txt',required=True)
parser.add_argument("--seed",type=int,action="store",default=1,required=False)
args = parser.parse_args()

# choose 10 cats randomly
fits_files= dbload.read_lines(args.list_of_cats)
rand = np.random.RandomState(args.seed)
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
    query="select * from decam_cand as c JOIN decam_decam as d ON d.cand_id=c.id JOIN decam_aper as a ON a.cand_id=c.id JOIN decam_wise as w ON w.cand_id=c.id WHERE c.brickname like '%s' and c.objid=%d" % (t['brickname'].data[0],t['objid'].data[0])  
    print 'query=',query
    db= read_from_psql_file('decam_with_matching_bokmos.txt')
    # compare tractor cat row to DB row
    diff_rows(t,db)
    
print 'done'

#print "args.tractor_catalog= ",args.tractor_catalog
#
#fitsbin = args.tractor_catalog
#table = fits.open( fitsbin )
## Access the data part of the table.
#trac = table[1].data
#trac = np.asarray(trac)
#ans={}
#i=1000
#ans['cand']=trac['blob'][i],trac['ra'][i],trac['dec'][i],trac['dchisq'][i][1],trac['dchisq'][i][3]
#ans['decam']=trac['decam_nobs'][i][4],trac['decam_flux'][i][1],trac['decam_flux'][i][2],trac['decam_flux'][i][4]
#ans['decam_aper']=trac['decam_apflux'][i][1][0],trac['decam_apflux'][i][2][0],trac['decam_apflux'][i][4][0],trac['decam_apflux'][i][4][7]
#ans['wise']=trac['wise_flux'][i][0],trac['wise_flux'][i][1],trac['wise_flux'][i][2],trac['wise_flux'][i][3],trac['wise_flux_ivar'][i][3]
#
##db
#con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_user', database='desi')
#cur = con.cursor()
#def output(cursor,query):
#    cursor.execute(query)
#    return cursor.fetchall()
#
#sql={}
#sql['cand']= output(cur,"SELECT blob,ra,dec,dchisq2,dchisq4,id from dr2.candidate where objid=1000")
#sql['decam']= output(cur,"SELECT gnobs,gflux,rflux,zflux from dr2.decam where cand_id=884")
#sql['decam_aper']= output(cur,"SELECT gapflux_1,rapflux_1,zapflux_1,zapflux_8 from dr2.decam_aper where cand_id=884")
#sql['wise']= output(cur,"SELECT w1flux,w2flux,w3flux,w4flux,w4flux_ivar from dr2.wise where cand_id=884")
#
#for key in ans.keys():
#    print '----',key.upper(),'----'
#    print 'tractor cat: ',ans[key]
#    print 'postgres   : ',sql[key][0]
#
##con.commit()
#print 'done'
#
