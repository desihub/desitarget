import thesis_code.fits
from astropy.io import fits
from argparse import ArgumentParser
import numpy as np
import os
import psycopg2
import sys

def rem_if_exists(name):
    if os.path.exists(name):
        if os.system(' '.join(['rm','%s' % name]) ): raise ValueError

def write_schema(table,keys,sql_dtype,addrows=[],indexing=[]):
    outname= table+'.table'
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

def insert_query(table,ith_row,data,keys,returning=False,newkeys=[],newvals=[]):
    query = 'INSERT INTO %s ( ' % table    
    for nk in newkeys: query+= '%s, ' % nk
    for key in keys:
        query+= key.lower()
        if key != keys[-1]: query+= ', '
    query+= ' ) VALUES ( '
    for nv in newvals: query+= '%s, ' % str(nv)
    for key in keys:
        if np.issubdtype(data[key][i].dtype, str): query+= "'%s'" % data[key][i]
        else: query+= "%s" % str(data[key][i])
        if key != keys[-1]: query+= ', '
    if returning: query+= ' ) RETURNING id'
    else: query+= ' )'
    return query


parser = ArgumentParser(description="test")
parser.add_argument("-fits_file",action="store",help='',required=True)
parser.add_argument("-table",choices=['bricks','cfhtls_d2_r'],action="store",help='',required=True)
parser.add_argument("-overw_schema",action="store",help='set to anything to write schema to file, overwritting the previous file',required=False)
parser.add_argument("-load_db",action="store",help='set to anything to load and write to db',required=False)
args = parser.parse_args()

a=fits.open(args.fits_file)
data,keys= thesis_code.fits.getdata(a,1)
nrows = data.shape[0] 

#write schemas
if args.table == 'bricks':
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
        write_schema(args.table,keys,sql_dtype,addrows=more_rows,indexing=indexes)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0,30): #nrows):
        print 'row= ',i
        query= insert_query(args.table,i,data,keys)
        if args.load_db: cursor.execute(query) 
    if args.load_db: 
        con.commit()
        print 'finished bricks load'
    print 'query looks like this: \n',query    

elif args.table == 'cfhtls_d2_r':
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
        write_schema(args.table+'_objs',obj_keys,sql_dtype,addrows=more_obj_rows)
        write_schema(args.table+'_flux',fluxes_keys,sql_dtype,addrows=more_flux_rows)
    #db
    con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    for i in range(0, 30): #nrows):
        print 'row= ',i
        query1= insert_query(args.table+'_objs',i,data,obj_keys,returning=True)
        if args.load_db: 
            cursor.execute(query1) 
            id = cursor.fetchone()[0]
        else: id=2 #junk val so can print what query would look like 
        query2= insert_query(args.table+'_flux',i,data,fluxes_keys,newkeys=['cand_id'],newvals=[id])
        if args.load_db: cursor.execute(query2) 
    if args.load_db: 
        con.commit()
        print 'finished cfhtls_d2_r load'
    print 'cfhtls_d2_r query looks like this: \n'    
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
