import multiprocessing 
import resource
from functools import partial
#from mpi4py import MPI

from astropy.io import fits
from astropy.table import vstack, Table, Column
from argparse import ArgumentParser
import numpy as np
import os
import psycopg2
import sys
import traceback
from subprocess import check_output

def print_flush(words):
    print words
    sys.stdout.flush()

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return list(np.char.strip(lines))

def current_mem_usage():
    '''return mem usage in MB'''
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.**2

def rem_if_exists(name):
    if os.path.exists(name):
        if os.system(' '.join(['rm','%s' % name]) ): raise ValueError

def write_index_cluster_files(schema):
    indexes= indexes_for_tables(args.schema)
    # Write index
    outname= 'index.'+args.schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    for key in indexes.keys(): 
        for coln in indexes[key]: 
            if coln == 'radec': fin.write('CREATE INDEX q3c_%s_idx ON %s (q3c_ang2ipix(ra, dec));\n' % (key,key))
            else: fin.write('CREATE INDEX %s_%s_idx ON %s (%s);\n' % (key,coln,key,coln)) 
    fin.close()   
    print "wrote index file: %s " % outname 
    # Write cluster
    outname= 'cluster.'+args.schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    for key in indexes.keys():
        if 'radec' in indexes[key]:
            fin.write('CLUSTER q3c_%s_idx ON %s;\n' % (key,key))
            fin.write('ANALYZE %s;\n' % key)
    print "wrote cluster file: %s " % outname 


def write_schema(schema,table,keys,sql_dtype,addrows=[]):
    outname= table+'.table.%s' % schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    fin.write('CREATE SEQUENCE %s_id_seq;' % table)
    fin.write('\n\n'+'CREATE TABLE %s (' % table)
    #add indexing names
    for row in addrows: fin.write('\n'+'\t'+row+',')
    #add catalogue's names
    for key in keys:
        stri= '\n'+'\t'+key.lower()+' '+sql_dtype[key]
        if key != keys[-1]: stri+= ','
        fin.write(stri)
    fin.write('\n'+');'+'\n')
    fin.close()

def insert_query(schema,table,ith_row,data,keys,returning=False,newkeys=[],newvals=[]):
    query = 'INSERT INTO %s ( ' % table   
    #column names 
    for nk in newkeys: query+= '%s, ' % nk
    for key in keys:
        query+= key.lower()
        if key != keys[-1]: query+= ', '
    query+= ' ) VALUES ( '
    #catalogue numeric or string entries
    for nv in newvals:
        query+= '%s, ' % str(nv)
    for key in keys:
        # Put strings in quotes
        if np.issubdtype(data[key][ith_row].dtype, str):  
            query+= "'%s'" % data[key][ith_row].strip()
        # Handle NaNs
        elif np.any((np.isnan(data[key][ith_row]),np.isinf(data[key][ith_row])),axis=0): 
            query+= "'NaN'"
        else: query+= "%s" % str(data[key][ith_row])
        if key != keys[-1]: query+= ', '
    if returning: query+= ' ) RETURNING id'
    else: query+= ' )'
    return query

def update_keys(keys,newkeys,oldkey):
    for k in newkeys:
        keys.insert(keys.index(oldkey), k) #insert newkey so will have same index as oldkey
    keys.pop(keys.index(oldkey)) #remove oldkey from ordered keys
        
def replace_key(data,newkey,oldkey):
    '''data is dict of np arrays'''
    data[newkey]= data[oldkey].copy()
    del data[oldkey]

def get_sql_dtype(data):
    sql_dtype={}
    for key in data.keys():
        if key.startswith('RA') or key.startswith('ra') or key.startswith('DEC') or key.startswith('dec'): sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        elif np.issubdtype(data[key].dtype, np.uint8): sql_dtype[key]= 'integer' #binary, store as int for now
        elif np.issubdtype(data[key].dtype, bool): sql_dtype[key]= 'boolean'
        else: 
            print('key, type= ',key,data[key].dtype)
            raise ValueError
    return sql_dtype

def get_table_colnames(fname):
    f=open(fname,'r')
    lines= f.readlines()
    f.close()
    colnams=[]
    for i in range(len(lines)):
        if 'CREATE TABLE' in lines[i]:
            i+=1
            while ');' not in lines[i]:
                li_arr= lines[i].strip().split()
                i+=1
                if 'id' == li_arr[0]: continue
                else: colnams.append(li_arr[0])
    return colnams

def indexes_for_tables(schema):
    '''keywords from args.table choices where "keywords" are dict(keywords)'''
    if schema == 'truth':
        return dict(bricks=['brickid','radec'],\
                    stripe82=['id','radec','z'],\
                    vipers_w4=['id','radec','zflag','zspec'],\
                    deep2_f2=['id','radec','zhelio'],\
                    cfhtls_d2_i=['id','radec','brickid'],\
                    cfhtls_d2_r=['id','radec','brickid'],\
                    cosmos_acs=['id','radec'],\
                    cosmos_zphot=['id','radec']
                    )
    elif schema in 'dr2dr3':
        return dict(decam_cand=['id','brickid','radec'],\
                    decam_decam=['cand_id','gflux','rflux','zflux'],\
                    decam_aper=['cand_id'],\
                    decam_wise=['cand_id','w1flux','w2flux'])
    else:
        raise ValueError

def get_decam_keys():
    '''return zipped tuple for iteration of
    db,decam keys'''
    pre='decam'
    trac_keys=[pre+'_flux',\
               pre+'_flux_ivar',\
               pre+'_fracflux',\
               pre+'_fracmasked',\
               pre+'_fracin',\
               pre+'_rchi2',\
               pre+'_nobs',\
               pre+'_anymask',\
               pre+'_allmask',\
               pre+'_psfsize',\
               pre+'_mw_transmission'] #'decam_depth','decam_galdepth'
    db_keys=['flux',\
             'flux_ivar',\
             'fracflux',\
             'fracmasked',\
             'fracin',\
             '_rchi2',\
             'nobs',\
             '_anymask',\
             '_allmask',\
             '_psfsize',\
             '_ext']
    return [tuple(db_keys),tuple(trac_keys)]

def get_wise_keys():
    '''return zipped tuple for iteration of
    db,decam keys'''
    pre='wise'
    trac_keys=[pre+'_flux',\
               pre+'_flux_ivar',\
               pre+'_fracflux',\
               pre+'_rchi2',\
               pre+'_nobs',\
               pre+'_mw_transmission',\
               pre+'_lc_flux',\
               pre+'_lc_flux_ivar',\
               pre+'_lc_nobs',\
               pre+'_lc_fracflux',\
               pre+'_lc_rchi2',\
               pre+'_lc_mjd']
    db_keys=['flux',\
             'flux_ivar',\
             'fracflux',\
             '_rchi2',\
             'nobs',\
             '_ext',\
             '_lc_flux_',\
             '_lc_flux_ivar_',\
             '_lc_nobs_',\
             '_lc_fracflux_',\
             '_lc_rchi2_',\
             '_lc_mjd_']
    return [tuple(db_keys),tuple(trac_keys)]
 
def get_aper_keys():
    '''return zipped tuple for iteration of
    db,decam keys'''
    pre='decam'
    trac_keys=[pre+'_apflux',\
               pre+'_apflux_ivar',\
               pre+'_apflux_resid']
    db_keys=['apflux_',\
             'apflux_ivar_',\
             'apflux_resid_']
    return [tuple(db_keys),tuple(trac_keys)]


def get_cand_keys(trac_keys):
    '''return difference between all tractor cat keys and decam,wise,aper keys'''
    keys= set(trac_keys).difference(\
                set(get_decam_keys()[1]+\
                    get_wise_keys()[1]+\
                    get_aper_keys()[1]))
    return list(keys)


def tractor_into_db(tractor_cat, schema=None,table=None,\
                                 overw_schema=False,load_db=False):
    '''wrapper around actual function so get traceback for processes'''
    try:
        tractor_into_db2(tractor_cat, schema=schema,table=table,\
                                     overw_schema=overw_schema,load_db=load_db) 
    except: 
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print_flush("Error loading %s" % tractor_cat)
        traceback.print_tb(exc_traceback, file=sys.stdout)

   
def tractor_into_db2(tractor_cat, schema=None,table=None,\
                                 overw_schema=False,load_db=False):
    
    '''load a single Tractor Catalouge or fits truth table into the DB'''
    assert(schema is not None)
    assert(table is not None)
 
    # Read fits table
    tractor = Table(fits.getdata(tractor_cat, 1))
    nrows = len(tractor) 

    # Print cat name so know whether finished
    if load_db: print_flush('preload, nrows=%d, %s' % (nrows,tractor_cat))
    
    # Load data to appropriate schema file
    if table == 'decam':
        # Store as 4 tables in db
        cand,decam,aper,wise= {},{},{},{}            
        # Decam table
        for db_key,trac_key in zip(*get_decam_keys()):
            for cnt,band in enumerate(['u','g','r','i','z','Y']): 
                decam[band+db_key]= tractor[trac_key][:,cnt].data
        # Aperature table
        for db_key,trac_key in zip(*get_aper_keys()):
            for cnt,band in enumerate(['u','g','r','i','z','Y']): 
                for ap in range(8):
                    aper[band+db_key+str(ap+1)]= tractor[trac_key][:,cnt,ap].data
        # Wise table
        for db_key,trac_key in zip(*get_wise_keys()):
            # If light curve, w1,w2 only
            if '_lc_' in trac_key: 
                for cnt,band in enumerate(['w1','w2']):
                    for iepoch,epoch in enumerate(['1','2','3','4','5']):
                        wise[band+db_key+epoch]= tractor[trac_key][:,cnt,iepoch].data
            else:
                for cnt,band in enumerate(['w1','w2','w3','w4']):
                    wise[band+db_key]=tractor[trac_key][:,cnt].data
        # Candidate table
        for trac_key in get_cand_keys(tractor.keys()):
            if trac_key == 'dchisq':
                for i in range(5): 
                    cand[trac_key+str(i)]= tractor[trac_key][:,i].data
            else:
                cand[trac_key]= tractor[trac_key].data 
        # Extracted all info
        del tractor
        # Match each key with its db data type
        sql_dtype= [get_sql_dtype(cand),\
                    get_sql_dtype(decam),\
                    get_sql_dtype(aper),\
                    get_sql_dtype(wise)]
        # Schema
        tables=[table+name for name in ['_cand','_decam','_aper','_wise']]
        keys=[cand.keys(),\
              decam.keys(),\
              aper.keys(),\
              wise.keys()]
        more_cand= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[0])]
        more_decam= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[1]),\
                        "cand_id bigint REFERENCES %s (id)" % (tables[0])]
        more_aper= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[2]),\
                        "cand_id bigint REFERENCES %s (id)" % (tables[0])]
        more_wise= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[3]),\
                        "cand_id bigint REFERENCES %s (id)" % (tables[0])]
        # Write schema format file
        if overw_schema:
            write_schema(schema,tables[0],np.sort(keys[0]),sql_dtype[0],addrows=more_cand) #np.array(k_cand)[np.lexsort(k_cand)]
            write_schema(schema,tables[1],np.sort(keys[1]),sql_dtype[1],addrows=more_decam)
            write_schema(schema,tables[2],np.sort(keys[2]),sql_dtype[2],addrows=more_aper)
            write_schema(schema,tables[3],np.sort(keys[3]),sql_dtype[3],addrows=more_wise)
        # Load data into db
        if load_db: 
            con= psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
            cursor = con.cursor()
            printed_nans=False
            for i in range(nrows):
                # Get insert string and execute
                query_cand= insert_query(schema,tables[0],i,cand,keys[0],returning=True)
                cursor.execute(query_cand) 
                id = cursor.fetchone()[0]
                query_decam= insert_query(schema,tables[1],i,decam,keys[1],newkeys=['cand_id'],newvals=[id])
                query_aper= insert_query(schema,tables[2],i,aper,keys[2],newkeys=['cand_id'],newvals=[id])
                query_wise= insert_query(schema,tables[3],i,wise,keys[3],newkeys=['cand_id'],newvals=[id])
                cursor.execute(query_decam)
                cursor.execute(query_aper) 
                cursor.execute(query_wise)
                # Print if NaNs in catalogue, print only once
                if printed_nans == False:
                    if 'NaN' in query_cand+query_decam+query_aper+query_wise:
                        print_flush('NaN in %s' % tractor_cat)
                        printed_nans = True
            # If successfully loaded all rows, commit and print that finished
            print_flush('commitedload %s' % tractor_cat) 
            con.commit()
            con.close()
        # Or dry run
        else: 
            i=0
            query_cand= insert_query(schema,tables[0],i,cand,keys[0],returning=True)
            id=np.nan #junk b/c just sanity check
            query_decam= insert_query(schema,tables[1],i,decam,keys[1],newkeys=['cand_id'],newvals=[id])
            query_aper= insert_query(schema,tables[2],i,aper,keys[2],newkeys=['cand_id'],newvals=[id])
            query_wise= insert_query(schema,tables[3],i,wise,keys[3],newkeys=['cand_id'],newvals=[id])
            print_flush('insert queries for row=%d:' % i)    
            print_flush('query_cand= \n',query_cand)
            print_flush('query_decam= \n',query_decam) 
            print_flush('query_aper= \n',query_aper)   
            print_flush('query_wise= \n',query_wise)   
    elif table == 'bricks':
        #dtype for each key using as column name
        sql_dtype= get_sql_dtype(keys)
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % table]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0,nrows):
            query= insert_query(schema,table,i,data,keys)
            if load_db: cursor.execute(query) 
        print 'finished loading files into %s' % table
        print 'query looks like this: \n',query    
        if load_db: 
            con.commit()
        print 'done'
    elif table.startswith('vipers'):
        '''http://vipers.inaf.it/data/pdr1/catalogs/README_VIPERS_SPECTRO_PDR1.txt'''
        #rename ra_deep,dec_deep to ra,dec
        replace_key(data,'RA','ALPHA') 
        update_keys(keys,['RA'],'ALPHA')
        replace_key(data,'DEC','DELTA') 
        update_keys(keys,['DEC'],'DELTA')
        replace_key(data,'IAU_ID','ID_IAU')
        update_keys(keys,['IAU_ID'],'ID_IAU')
        #ZFLG contains info in two integers separated by decimal, split this up
        one= np.zeros(data['ZFLG'].shape[0]).astype(np.int32)-1
        two= one.copy()
        for cnt in range(one.shape[0]):
            both=str(data['ZFLG'][cnt]).split(".")
            one[cnt]= int(both[0])
            two[cnt]= int(both[1])
        data['ZFLG_1']= one 
        data['ZFLG_2']= two 
        del data['ZFLG']
        update_keys(keys,['ZFLG_1','ZFLG_2'],'ZFLG')
        #drop keys, add new ones 
        sql_dtype= get_sql_dtype(keys)
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (table)]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0, nrows):
            query= insert_query(schema,table,i,data,keys,returning=False)
            if load_db: 
                cursor.execute(query) 
        if load_db:
            con.commit()
            print 'finished %s load' %table
        print 'query= \n'    
        print query    
    elif table.startswith('cfhtls_d2_'):
        sql_dtype= get_sql_dtype(keys)
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (table)]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0, nrows):
            query= insert_query(schema,table,i,data,keys,returning=False)
            if load_db: 
                cursor.execute(query) 
        if load_db:
            con.commit()
            print 'finished %s load' %table
        print 'query= '    
        print query    
    elif table == 'cosmos_acs':
        '''description of columns: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_acs_colDescriptions.html'''
        sql_dtype= get_sql_dtype(keys)
        print 'WARNING: cosmos-acs.fits if 300+ MB file, this will take a few min!'
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (table)]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0, nrows):
            query= insert_query(schema,table,i,data,keys,returning=False)
            if load_db: 
                cursor.execute(query) 
        if load_db:
            con.commit()
            print 'finished %s load' %table
        print 'query='    
        print query    
    elif table == 'cosmos_zphot':
        '''http://irsa.ipac.caltech.edu/data/COSMOS/datasets.html
        col descriptoin: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html'''
        #rename any 'id' or 'ID' keys to 'catid' since 'id'is reserved for column name of seqeunce in db
        replace_key(data,'catID','ID') 
        update_keys(keys,['catID'],'ID')
        sql_dtype= get_sql_dtype(keys)
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (table)]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0, nrows):
            query1= insert_query(schema,table,i,data,keys,returning=False)
            if load_db: 
                cursor.execute(query1) 
        if load_db:
            con.commit()
            print 'finished %s load' %table
        print 'finished %s load' %table
        print 'query= '    
        print query1   
    elif table == 'stripe82':
        print 'WARNING: stripe82_specz is 300+ MB file, this will take a few min!'
        replace_key(data,'RA','PLUG_RA') 
        update_keys(keys,['RA'],'PLUG_RA')
        replace_key(data,'DEC','PLUG_DEC') 
        update_keys(keys,['DEC'],'PLUG_DEC')
        sql_dtype= get_sql_dtype(keys)
        #schema
        more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (table)]
        if overw_schema:
            write_schema(schema,table,keys,sql_dtype,addrows=more_rows)
        #db
        con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
        cursor = con.cursor()
        if load_db: print 'loading %d rows into %s table' % (nrows,table)
        for i in range(0, nrows):
            query= insert_query(schema,table,i,data,keys,returning=False)
            if load_db: 
                cursor.execute(query) 
        if load_db:
            con.commit()
            print 'finished %s load' %table
        print 'query= '    
        print query 
    else: raise ValueError
    
def main(args,table):
    fits_files= read_lines(args.list_of_cats)
   
    # Index and cluster file for once everything loaded in
    if args.make_index_file:
        write_index_cluster_files(args.schema)

    # serial 
    if args.serial:
        print 'running serial, cores= %d, should be 0' % args.cores
        for fil in [fits_files[0]]:
            tractor_into_db(fil, schema=args.schema,table=table,\
                                 overw_schema=args.overw_schema,load_db=args.load_db)
    # parallel
    else:
        if args.mpi:
            comm = MPI.COMM_WORLD
            if comm.rank == 0: print 'running mpi, cores= %d' % comm.size
            cnt=0
            i=comm.rank+cnt*comm.size
            while i < len(fits_files):
                tractor_into_db(fits_files[i], schema=args.schema,table=table,\
                                               overw_schema=args.overw_schema,load_db=args.load_db)
                print_flush("rank %d in its %dth iteration and loading %dth cat = %s" % (comm.rank, cnt,i,fits_files[i]))
                cnt+=1
                i=comm.rank+cnt*comm.size
        else:
            print 'running multiprocessing, cores= %d' % args.cores
            print 'Global maximum memory usage b4 multiprocessing: %.2f (mb)' % current_mem_usage()
            pool = multiprocessing.Pool(args.cores)
            # The iterable is fits_files
            results=pool.map(partial(tractor_into_db, schema=args.schema,table=table,\
                                                      overw_schema=args.overw_schema,load_db=args.load_db), \
                             fits_files)
            pool.close()
            pool.join()
            del pool
            print 'Global maximum memory usage after multiprocessing: %.2f (mb)' % current_mem_usage()
    if args.load_db:
        if args.mpi == False: 
            print "finished loading these cats"
            for cat in fits_files: print cat
        elif args.mpi and comm.rank == 0:
            print "finished loading these cats"
            for cat in fits_files: print cat
 
if __name__ == '__main__':
    parser = ArgumentParser(description="test")
    parser.add_argument("--mpi",action="store_true",help='set to use mpi, otherwise multiprocessing for parallelism',required=False)
    parser.add_argument("--serial",action="store_true",help='run on 1 tractor cat only',required=False)
    parser.add_argument("--cores",type=int,action="store",default=0,help='',required=False)
    parser.add_argument("--list_of_cats",action="store",help='',required=True)
    parser.add_argument("--schema",choices=['dr2','dr3','truth'],action="store",help='',required=True)
    parser.add_argument("--overw_schema",action="store_true",help='set to write schema to file, overwritting the previous file',required=False)
    parser.add_argument("--load_db",action="store_true",help='set to load and write to db',required=False)
    parser.add_argument("--make_index_file",action="store_true",help='set to write index and cluster files',required=False)
    parser.add_argument("--special_table",choices=['bricks','stripe82','vipers_w4','deep2_f2','deep2_f3','deep2_f4','cfhtls_d2_r','cfhtls_d2_i','cosmos_acs','cosmos_zphot'],action="store",help='',required=False)
    args = parser.parse_args()

    if args.mpi:
        assert(args.overw_schema == False)
        assert(args.make_index_file == False)
    if args.cores > 0:
        assert(args.mpi == False)
        assert(args.serial == False)
    if args.serial:
        assert(args.mpi == False)
        assert(args.cores == 0)

    if args.schema == 'truth':
        assert(args.special_table)
        table= args.special_table
    else:
        table= 'decam'
    
    main(args,table)
