import thesis_code.fits
from astropy.io import fits
from argparse import ArgumentParser
import numpy as np
import os
import psycopg2
import sys
from subprocess import check_output

def rem_if_exists(name):
    if os.path.exists(name):
        if os.system(' '.join(['rm','%s' % name]) ): raise ValueError

def write_schema(schema,table,keys,sql_dtype,addrows=[],indexing=[]):
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
    for index in indexing: 
        fin.write('\n'+index+';')
        if index == indexing[-1]: fin.write('\n')
    fin.close()

def insert_query(schema,table,ith_row,data,keys,returning=False,newkeys=[],newvals=[]):
    query = 'INSERT INTO %s.%s ( ' % (schema,table)    
    for nk in newkeys: query+= '%s, ' % nk
    for key in keys:
        query+= key.lower()
        if key != keys[-1]: query+= ', '
    query+= ' ) VALUES ( '
    for nv in newvals: query+= '%s, ' % str(nv)
    for key in keys:
        if np.issubdtype(data[key][i].dtype, str): query+= "'%s'" % data[key][i].strip()
        else: query+= "%s" % str(data[key][i])
        if key != keys[-1]: query+= ', '
    if returning: query+= ' ) RETURNING id'
    else: query+= ' )'
    return query

def replace_key_np_record(data,oldkey,newkey): 
    '''renames key in a numpy.record array'''
    i= np.where(np.array(data.dtype.names) == oldkey)[0][0]
    newn= list(data.dtype.names)
    newn[i]= newkey
    data.dtype.names= newn

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

parser = ArgumentParser(description="test")
parser.add_argument("-fits_file",action="store",help='',required=True)
parser.add_argument("-schema",choices=['public','dr1','dr2','truth'],action="store",help='',required=True)
parser.add_argument("-table",choices=['index','bricks','cfhtls_d2_r','cfhtls_d2_i','cosmos_acs','cosmos_zphot'],action="store",help='',required=True)
parser.add_argument("-overw_schema",action="store",help='set to anything to write schema to file, overwritting the previous file',required=False)
parser.add_argument("-load_db",action="store",help='set to anything to load and write to db',required=False)
args = parser.parse_args()

if args.schema == 'dr1' or args.schema == 'dr2':
    print 'preventing accidental load to dr1 or dr2'
    raise ValueError

a=fits.open(args.fits_file)
data,keys= thesis_code.fits.getdata(a,1)
nrows = data.shape[0] 

#write schemas
if args.table == 'index':
    outname= 'index.table.'+args.schema
    rem_if_exists(outname)
    fin=open(outname,'w')
    #index flux values first, then q3c when done
    #list names of all flux tables
    cmd= "find . -maxdepth 1 -type f -name *flux.table.truth"
    flux_tables= check_output(cmd.split())
    flux_tables= flux_tables.strip().replace("./","").split()
    print 'flux_tables= ',flux_tables
    #index non 'id' columns for each flux table
    for fn in flux_tables:
        cnames= get_table_colnames(fn)
        table= fn.split('.')[0]
        for col in cnames: fin.write('CREATE INDEX %s_%s_idx ON %s.%s (%s);\n' % (table,col,args.schema,table,col))
    #q3c indexing
    #list names of all objs tables
    cmd= "find . -maxdepth 1 -type f -name *objs.table.truth"
    objs_tables= check_output(cmd.split())
    objs_tables= objs_tables.strip().replace("./","").split()
    print 'objs_tables= ',objs_tables
    #index non 'id' columns for each flux table
    for fn in objs_tables: 
        table= fn.split('.')[0]
        fin.write('CREATE INDEX q3c_%s_idx ON %s (q3c_ang2ipix(ra,dec));\n' % (table,table))
        fin.write('CLUSTER q3c_%s_idx ON %s;\n' % (table,table))
    #done
    fin.close()    
    #usual psql first
    #for tname in tnames: fin.write('CREATE INDEX name_idx ON %s.%s (col_name);\n' % (schema,tname)
    #q3c indexing
    #fin.close()
#CREATE INDEX cand_q3c_candidate_idx ON candidate (q3c_ang2ipix(ra,dec));
#CLUSTER cand_q3c_candidate_idx on candidate;
#CREATE INDEX cand_brickid_idx ON candidate (brickid);
#CREATE INDEX decam_candid_idx ON decam (cand_id);
#CREATE INDEX decam_aper_candid_idx ON decam_aper (cand_id);
#CREATE INDEX wise_candid_idx ON wise (cand_id);
#CREATE INDEX uflux_idx ON decam (uflux);
#CREATE INDEX gflux_idx ON decam (gflux); 
elif args.table == 'bricks':
    sql_dtype={}
    for key in keys:
        if key.startswith('RA') or key.startswith('ra') or key.startswith('DEC') or key.startswith('dec'): sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        else: raise ValueError
    #schema
    more_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % args.table]
    indexes= ["CREATE INDEX %s_q3c_idx ON %s (q3c_ang2ipix(ra,dec))" % (args.table,args.table),\
                "CLUSTER %s_q3c_idx on %s" % (args.table,args.table),\
                "CREATE INDEX %s_brickid_idx ON %s (brickid)" %(args.table,args.table)]
    if args.overw_schema:
        write_schema(args.schema,args.table,keys,sql_dtype,addrows=more_rows,indexing=indexes)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0,30): #nrows):
        print 'row= ',i
        query= insert_query(args.schema,args.table,i,data,keys)
        if args.load_db: cursor.execute(query) 
    if args.load_db: 
        con.commit()
        print 'finished bricks load'
    print 'query looks like this: \n',query    

elif args.table == 'cfhtls_d2_r' or args.table == 'cfhtls_d2_i':
    obj_keys,fluxes_keys=[],[]
    for key in keys:
        if key == 'I_VERSION': obj_keys.append(key)
        elif key.startswith('U_') or key.startswith('G_') or key.startswith('R_') or key.startswith('I_') or key.startswith('Z_'):
            fluxes_keys.append(key)
        else: obj_keys.append(key)
    sql_dtype={}
    for key in keys:
        if key == 'RA' or key == 'DEC': sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        else: raise ValueError
    #schema
    more_obj_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_objs'),\
                    "brickid integer default null"]#primary key not null default nextval('%s_id_seq'::regclass)," % args.table)
    more_flux_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_flux'),\
                    "cand_id bigint REFERENCES %s (id)" % (args.table+'_objs')]
    if args.overw_schema:
        write_schema(args.schema,args.table+'_objs',obj_keys,sql_dtype,addrows=more_obj_rows)
        write_schema(args.schema,args.table+'_flux',fluxes_keys,sql_dtype,addrows=more_flux_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0, 30): #nrows):
        print 'row= ',i
        query1= insert_query(args.schema,args.table+'_objs',i,data,obj_keys,returning=True)
        if args.load_db: 
            cursor.execute(query1) 
            id = cursor.fetchone()[0]
        else: id=2 #junk val so can print what query would look like 
        query2= insert_query(args.schema,args.table+'_flux',i,data,fluxes_keys,newkeys=['cand_id'],newvals=[id])
        if args.load_db: cursor.execute(query2) 
    if args.load_db: 
        con.commit()
        print 'finished %s load' %args.table
    print '%s query looks like this: \n' % args.table    
    print 'obj query: \n',query1    
    print 'flux query: \n',query2   
elif args.table == 'cosmos_acs':
    '''description of columns: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_acs_colDescriptions.html'''
    flux_keys= keys[1:17] #[0] is running obj number from sextractor catalog says to ignore
    obj_keys= keys[17:]
    sql_dtype={}
    for key in keys:
        if key.lower().startswith('ra') or key.lower().startswith('dec'): sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        else: raise ValueError
    #schema
    more_obj_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_objs'),\
                    "brickid integer default null"]#primary key not null default nextval('%s_id_seq'::regclass)," % args.table)
    more_flux_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_flux'),\
                    "cand_id bigint REFERENCES %s (id)" % (args.table+'_objs')]
    if args.overw_schema:
        write_schema(args.schema,args.table+'_objs',obj_keys,sql_dtype,addrows=more_obj_rows)
        write_schema(args.schema,args.table+'_flux',flux_keys,sql_dtype,addrows=more_flux_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0, 30): #nrows):
        print 'row= ',i
        query1= insert_query(args.schema,args.table+'_objs',i,data,obj_keys,returning=True)
        if args.load_db: 
            cursor.execute(query1) 
            id = cursor.fetchone()[0]
        else: id=2 #junk val so can print what query would look like 
        query2= insert_query(args.schema,args.table+'_flux',i,data,flux_keys,newkeys=['cand_id'],newvals=[id])
        if args.load_db: cursor.execute(query2) 
    if args.load_db: 
        con.commit()
        print 'finished %s load' %args.table
    print '%s query looks like this: \n' % args.table    
    print 'obj query: \n',query1    
    print 'flux query: \n',query2   
elif args.table == 'cosmos_zphot':
    '''see doc: http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html'''
    #rename any 'id' or 'ID' keys to 'catid' since 'id'is reserved for column name of seqeunce in db
    replace_key_np_record(data,'ID','catID') 
    keys= list(data.dtype.names) #update keys
    obj_keys= keys[:22]+keys[42:47]+[keys[50]]
    flux_keys=[]
    for k in keys: 
        if k not in obj_keys: flux_keys+= [k]
    sql_dtype={}
    for key in keys:
        if key.lower().startswith('ra') or key.lower().startswith('dec'): sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        else: raise ValueError
    #schema
    more_obj_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_objs'),\
                    "brickid integer default null"]#primary key not null default nextval('%s_id_seq'::regclass)," % args.table)
    more_flux_rows= ["id bigint primary key not null default nextval('%s_id_seq'::regclass)" % (args.table+'_flux'),\
                    "cand_id bigint REFERENCES %s (id)" % (args.table+'_objs')]
    if args.overw_schema:
        write_schema(args.schema,args.table+'_objs',obj_keys,sql_dtype,addrows=more_obj_rows)
        write_schema(args.schema,args.table+'_flux',flux_keys,sql_dtype,addrows=more_flux_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0, 30): #nrows):
        print 'row= ',i
        query1= insert_query(args.schema,args.table+'_objs',i,data,obj_keys,returning=True)
        if args.load_db: 
            cursor.execute(query1) 
            id = cursor.fetchone()[0]
        else: id=2 #junk val so can print what query would look like 
        query2= insert_query(args.schema,args.table+'_flux',i,data,flux_keys,newkeys=['cand_id'],newvals=[id])
        if args.load_db: cursor.execute(query2) 
    if args.load_db: 
        con.commit()
        print 'finished %s load' %args.table
    print '%s query looks like this: \n' % args.table    
    print 'obj query: \n',query1    
    print 'flux query: \n',query2   
else: raise ValueError
#output schemas
#print 'writing objects table'
#print 'done'

#load data into obj and fluxes tables in db
# Fire up the db
#cursor.execute( "UPDATE bricks set filename = %s, loaded = 'true', tractorvr = %s where brickid = %s", (fimage, tractver, hdrbrickid,)  )

     
#   id = cursor.fetchone()[0]
###
###
#   cursor.execute( query, tuple( [str(elem) for elem in [ id ] + newdata_decam[i]] ) )
#
