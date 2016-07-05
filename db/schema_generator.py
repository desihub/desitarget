import thesis_code.fits
from astropy.io import fits
from argparse import ArgumentParser
import numpy as np
import os
import psycopg2
import sys
from subprocess import check_output

from thesis_code.fits import tractor_cat

def rem_if_exists(name):
    if os.path.exists(name):
        if os.system(' '.join(['rm','%s' % name]) ): raise ValueError

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
        if np.issubdtype(data[key][i].dtype, str): #put strings in quotes 
            query+= "'%s'" % data[key][i].strip()
        elif np.any((np.isnan(data[key][i]),np.isinf(data[key][i])),axis=0): #NaN
            query+= "'NaN'"
            print "<<<<<<<< WARNING: %s is NaN in row %d >>>>>>>>>>" % (key,i)
        else: query+= "%s" % str(data[key][i])
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

def get_sql_dtype(keys):
    sql_dtype={}
    for key in keys:
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
                    vipers_w4=['id','radec','zflag','zspec','u','g','r','i','z'],\
                    deep2_f2=['id','radec','zhelio','g','r','z'],\
                    cfhtls_d2_i=['id','radec','brickid','u_mag_auto','g_mag_auto','r_mag_auto','i_mag_auto','z_mag_auto'],\
                    cfhtls_d2_r=['id','radec','brickid','u_mag_auto','g_mag_auto','r_mag_auto','i_mag_auto','z_mag_auto'],\
                    cosmos_acs=['id','radec','mag_iso','mag_isocor','mag_petro','mag_auto','mag_best','flux_auto'],\
                    cosmos_zphot=['id','radec','umag','bmag','vmag','gmag','rmag','imag','zmag','icmag','jmag','kmag']
                    )
    elif schema in 'dr2dr3':
        bands= ['u','g','r','i','z','Y']
        optical= dict(decam_cand=['id','brickid','radec'],\
                    decam_decam=[b+'flux' for b in bands]+ [b+'nobs' for b in bands]+ [b+'_anymask' for b in bands],\
                    bok_mzls_cand=['id','brickid','radec'],\
                    bok_mzls_decam=[b+'flux' for b in bands]+ [b+'nobs' for b in bands]+ [b+'_anymask' for b in bands]
                    )
        bands= ['w1','w2','w3','w4']
        ir= dict(decam_wise=[b+'flux' for b in bands]+ [b+'nobs' for b in bands],\
                    bok_mzls_wise=[b+'flux' for b in bands]+ [b+'nobs' for b in bands]
                    )
        optical.update(ir)
        return optical
    else:
        raise ValueError

parser = ArgumentParser(description="test")
parser.add_argument("-fits_file",action="store",help='',required=True)
parser.add_argument("-schema",choices=['dr1','dr2','dr3','truth'],action="store",help='',required=True)
parser.add_argument("-table",choices=['bricks','stripe82','ptf50','ptf100','ptf150','ptf200','bok_mzls','decam','vipers_w4','deep2_f2','deep2_f3','deep2_f4','cfhtls_d2_r','cfhtls_d2_i','cosmos_acs','cosmos_zphot'],action="store",help='',required=True)
parser.add_argument("-overw_schema",action="store",help='set to anything to write schema to file, overwritting the previous file',required=False)
parser.add_argument("-load_db",action="store",help='set to anything to load and write to db',required=False)
parser.add_argument("-index_cluster",action="store",help='set to anything to write index and cluster files',required=False)
args = parser.parse_args()

if args.schema == 'dr1':
    print 'preventing accidental load to dr1 or dr2'
    raise ValueError

#write index and cluster files
if args.index_cluster:
    indexes= indexes_for_tables(args.schema)
    #write index file
    outname= 'index.'+args.schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    #index cmds to file
    print 'indexes= ',indexes 
    for table in indexes.keys(): 
        for coln in indexes[table]: 
            if coln == 'radec': fin.write('CREATE INDEX q3c_%s_idx ON %s (q3c_ang2ipix(ra, dec));\n' % (table,table))
            else: fin.write('CREATE INDEX %s_%s_idx ON %s (%s);\n' % (table,coln,table,coln)) 
    fin.close()    
    #write cluster file
    outname= 'cluster.'+args.schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    for table in indexes.keys():
        if 'radec' in indexes[table]:
            fin.write('CLUSTER q3c_%s_idx ON %s;\n' % (table,table))
            fin.write('ANALYZE %s;\n' % table)


# Read Tractor Cat
#if args.schema in ['dr2','dr3']:  #95% the same, wise lc handles with if statement
#    data= tractor_cat(args.fits_file)
#    nrows = data['ra'].shape[0]  
#else:
a=fits.open(args.fits_file)
#keys is in desired order, data.keys() has all keys but out of order
data,keys= thesis_code.fits.getdata(a,1)
nrows = data[keys[0]].shape[0]            

if args.schema in 'dr2dr3':
    #split up arrays containing ugrizY bands
    for cnt,b in enumerate(['u','g','r','i','z','Y']): 
        #decam
        data[b+'flux']=data['decam_flux'][:,cnt].copy()
        data[b+'flux_ivar']= data['decam_flux_ivar'][:,cnt].copy()
        data[b+'fracflux']= data['decam_fracflux'][:,cnt].copy()
        data[b+'fracmasked']= data['decam_fracmasked'][:,cnt].copy()
        data[b+'fracin']= data['decam_fracin'][:,cnt].copy()
        data[b+'_rchi2']= data['decam_rchi2'][:,cnt].copy()
        data[b+'nobs']= data['decam_nobs'][:,cnt].copy()
        data[b+'_anymask']= data['decam_anymask'][:,cnt].copy()
        data[b+'_allmask']= data['decam_allmask'][:,cnt].copy()
        data[b+'_psfsize']= data['decam_psfsize'][:,cnt].copy()
        data[b+'_ext']= data['decam_mw_transmission'][:,cnt].copy()
        #data[b+'_depth']= data['decam_depth'][:,cnt].copy()
        #data[b+'_galdepth']= data['decam_galdepth'][:,cnt].copy()
        #decam_aper, 8 aperatures
        for ap in range(8):
            data[b+'apflux_'+str(ap+1)]= data['decam_apflux'][:,cnt,ap].copy()
            data[b+'apflux_resid_'+str(ap+1)]= data['decam_apflux_resid'][:,cnt,ap].copy()
            data[b+'apflux_ivar_'+str(ap+1)]= data['decam_apflux_ivar'][:,cnt,ap].copy()
    keys_to_del= ['decam_flux','decam_flux_ivar','decam_fracflux','decam_fracmasked','decam_fracin','decam_rchi2','decam_nobs','decam_anymask','decam_allmask',\
                'decam_psfsize','decam_mw_transmission','decam_apflux','decam_apflux_resid','decam_apflux_ivar'] #'decam_depth','decam_galdepth'
    #split up arrays containing source morphologies
    for i in range(5): 
        data['dchisq'+str(i)]= data['dchisq'][:,i].copy()
    keys_to_del+= ['dchisq']
    #split up arrays with wise bands 
    for cnt,b in enumerate(['w1','w2','w3','w4']):
        data[b+'flux']=data['wise_flux'][:,cnt].copy()
        data[b+'flux_ivar']= data['wise_flux_ivar'][:,cnt].copy()
        data[b+'fracflux']= data['wise_fracflux'][:,cnt].copy()
        data[b+'_ext']= data['wise_mw_transmission'][:,cnt].copy()
        data[b+'nobs']= data['wise_nobs'][:,cnt].copy()
        data[b+'_rchi2']= data['wise_rchi2'][:,cnt].copy()
    keys_to_del+= ['wise_flux','wise_flux_ivar','wise_fracflux','wise_mw_transmission','wise_nobs','wise_rchi2']
    if args.schema == 'dr3': 
        # wise lightcurves, 6/29/2016 max 5 epochs w1,w2
        fields = [s for s in data.keys() if "wise_lc" in s]
        print 'fields = ',fields
        for ifield,field in enumerate(fields):
            for ib,b in enumerate(['w1','w2']):
                for iepoch,epoch in enumerate(['1','2','3','4','5']):
                    data[field+'_'+b+'_'+epoch]= data[field][:,ib,iepoch].copy()
            keys_to_del+= [field]
    # All keys added, delete old names
    print 'keys_to_del=',keys_to_del
    for key in keys_to_del: del data[key]
    #get keys + flattened array keys
    k_dec,k_aper,k_cand,k_wise=[],[],[],[]
    for key in data.keys():
        #if args.schema == 'dr3': k_wise+= [s for s in a.keys() if "wise_lc" in s]
        if np.any(('w1' in key,'w2' in key,'w3' in key,'w4' in key),axis=0): k_wise.append(key)
        elif 'apflux' in key: k_aper.append(key)
        elif 'flux' in key or 'mask' in key or 'depth' in key: k_dec.append(key)
        elif 'rchi' in key or 'nobs' in key or 'psf' in key or 'ext' in key: k_dec.append(key)
        elif key not in k_dec and key not in k_aper: k_cand.append(key)
        else: raise ValueError
    #dtype for each key using as column name
    sql_dtype= get_sql_dtype(data.keys())
    #schema
    tables=[args.table+name for name in ['_cand','_decam','_aper','_wise']]
    keys=[k for k in [k_cand,k_dec,k_aper,k_wise]]
    more_cand= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[0])]
    more_decam= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[1]),\
                    "cand_id bigint REFERENCES %s (id)" % (tables[0])]
    more_aper= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[2]),\
                    "cand_id bigint REFERENCES %s (id)" % (tables[0])]
    more_wise= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (tables[3]),\
                    "cand_id bigint REFERENCES %s (id)" % (tables[0])]
    if args.overw_schema:
        write_schema(args.schema,tables[0],np.sort(keys[0]),sql_dtype,addrows=more_cand) #np.array(k_cand)[np.lexsort(k_cand)]
        write_schema(args.schema,tables[1],np.sort(keys[1]),sql_dtype,addrows=more_decam)
        write_schema(args.schema,tables[2],np.sort(keys[2]),sql_dtype,addrows=more_aper)
        write_schema(args.schema,tables[3],np.sort(keys[3]),sql_dtype,addrows=more_wise)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        #data must be good, so write to db
        query_cand= insert_query(args.schema,tables[0],i,data,keys[0],returning=True)
        if args.load_db: 
            cursor.execute(query_cand) 
            id = cursor.fetchone()[0]
        else: id=2 #junk val so can print what query would look like 
        query_decam= insert_query(args.schema,tables[1],i,data,keys[1],newkeys=['cand_id'],newvals=[id])
        query_aper= insert_query(args.schema,tables[2],i,data,keys[2],newkeys=['cand_id'],newvals=[id])
        query_wise= insert_query(args.schema,tables[3],i,data,keys[3],newkeys=['cand_id'],newvals=[id])
        if args.load_db: 
            cursor.execute(query_decam)
            cursor.execute(query_aper) 
            cursor.execute(query_wise) 
    if args.load_db: 
        con.commit()
    print 'finished %s load' %args.table
    print 'Load/insert queries are:'    
    print 'query_cand= \n',query_cand    
    print 'query_decam= \n',query_decam 
    print 'query_aper= \n',query_aper   
    print 'query_wise= \n',query_wise   
elif args.table == 'bricks':
    #dtype for each key using as column name
    sql_dtype= get_sql_dtype(keys)
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % args.table]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0,nrows):
        query= insert_query(args.schema,args.table,i,data,keys)
        if args.load_db: cursor.execute(query) 
    print 'finished loading files into %s' % args.table
    print 'query looks like this: \n',query    
    if args.load_db: 
        con.commit()
    print 'done'
elif args.table.startswith('vipers'):
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
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query= insert_query(args.schema,args.table,i,data,keys)
        if args.load_db: 
            cursor.execute(query) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'query= \n'    
    print query  
elif args.table.startswith('deep2'):
    '''http://deep.ps.uci.edu/DR4/photo.extended.html'''
    #rename ra_deep,dec_deep to ra,dec
    replace_key(data,'RA','RA_DEEP') 
    update_keys(keys,['RA'],'RA_DEEP')
    replace_key(data,'DEC','DEC_DEEP') 
    update_keys(keys,['DEC'],'DEC_DEEP')
    sql_dtype= get_sql_dtype(keys)
    sql_dtype['OBJNO']= 'bigint'
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query= insert_query(args.schema,args.table,i,data,keys,returning=False)
        if args.load_db: 
            cursor.execute(query) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'query= \n'    
    print query    
elif args.table.startswith('cfhtls_d2_'):
    sql_dtype= get_sql_dtype(keys)
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query= insert_query(args.schema,args.table,i,data,keys,returning=False)
        if args.load_db: 
            cursor.execute(query) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'query= '    
    print query    
elif args.table == 'cosmos_acs':
    '''description of columns: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_acs_colDescriptions.html'''
    sql_dtype= get_sql_dtype(keys)
    print 'WARNING: cosmos-acs.fits if 300+ MB file, this will take a few min!'
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query= insert_query(args.schema,args.table,i,data,keys,returning=False)
        if args.load_db: 
            cursor.execute(query) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'query='    
    print query    
elif args.table == 'cosmos_zphot':
    '''http://irsa.ipac.caltech.edu/data/COSMOS/datasets.html
    col descriptoin: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html'''
    #rename any 'id' or 'ID' keys to 'catid' since 'id'is reserved for column name of seqeunce in db
    replace_key(data,'catID','ID') 
    update_keys(keys,['catID'],'ID')
    sql_dtype= get_sql_dtype(keys)
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query1= insert_query(args.schema,args.table,i,data,keys,returning=False)
        if args.load_db: 
            cursor.execute(query1) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'finished %s load' %args.table
    print 'query= '    
    print query1   
elif args.table == 'stripe82':
    print 'WARNING: stripe82_specz is 300+ MB file, this will take a few min!'
    replace_key(data,'RA','PLUG_RA') 
    update_keys(keys,['RA'],'PLUG_RA')
    replace_key(data,'DEC','PLUG_DEC') 
    update_keys(keys,['DEC'],'PLUG_DEC')
    sql_dtype= get_sql_dtype(keys)
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    if args.load_db: print 'loading %d rows into %s table' % (nrows,args.table)
    for i in range(0, nrows):
        query= insert_query(args.schema,args.table,i,data,keys,returning=False)
        if args.load_db: 
            cursor.execute(query) 
    if args.load_db:
        con.commit()
        print 'finished %s load' %args.table
    print 'query= '    
    print query 
else: raise ValueError

print 'done'
