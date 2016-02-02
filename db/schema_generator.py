import thesis_code.fits
from astropy.io import fits
import argparse
import numpy as np

def write_schema(table,keys,sql_dtype,addrows=[],indexing=[]):
    fin=open(table+'.table','w')
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

def insert_query(ith_row,data,keys,args):
    query = 'INSERT INTO %s ( ' % args.table    
    for key in keys:
        query+= key
        if key != keys[-1]: query+= ', '
    query+= ' ) VALUES ( '
    for key in keys:
        query+= '%s' % str(data[key][i])
        if key != keys[-1]: query+= ', '
    query+= ' )' #RETURNING id'
    return query


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-fits_file",action="store",help='',required=True)
parser.add_argument("-table",choices=['bricks','cfhtls_d2_r'],action="store",help='',required=True)
args = parser.parse_args()

a=fits.open(args.fits_file)
data,keys= thesis_code.fits.getdata(a,1)

#write schemas
if args.table == 'bricks':
    sql_dtype={}
    for key in keys:
        if key.startswith('RA') or key.startswith('ra') or key.startswith('DEC') or key.startswith('dec'): sql_dtype[key]= 'double precision' 
        elif np.issubdtype(data[key].dtype, str): sql_dtype[key]= 'text' 
        elif np.issubdtype(data[key].dtype, int): sql_dtype[key]= 'integer'
        elif np.issubdtype(data[key].dtype, float): sql_dtype[key]= 'real'
        else: raise ValueError
    more_rows= ["id bigint default null"]
    indexes= ["CREATE INDEX %s_q3c_idx ON %s (q3c_ang2ipix(ra,dec))" % (args.table,args.table),\
                "CLUSTER %s_q3c_idx on %s" % (args.table,args.table),\
                "CREATE INDEX %s_brickid_idx ON %s (brickid)" %(args.table,args.table)]
    write_schema(args.table,keys,sql_dtype,addrows=more_rows,indexing=indexes)

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

    #indexing names for schemas
    more_obj_rows= ["id bigint default null",\
                    "brickid integer default null"]#primary key not null default nextval('%s_id_seq'::regclass)," % args.table)
    more_flux_rows= ["id bigint default null",\
                    "cand_id bigint default null"] #REFERENCES candidate (id)
    write_schema(args.table+'_objs',obj_keys,sql_dtype,addrows=more_obj_rows)
    write_schema(args.table+'_flux',fluxes_keys,sql_dtype,addrows=more_flux_rows)

else: raise ValueError
#output schemas
#print 'writing objects table'
#print 'done'

#load data into obj and fluxes tables in db
# Fire up the db
#con = psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
#cursor = con.cursor()
#cursor.execute( "UPDATE bricks set filename = %s, loaded = 'true', tractorvr = %s where brickid = %s", (fimage, tractver, hdrbrickid,)  )

nrows = data.shape[0] 
for i in range(0, 30): #nrows):
    query= insert_query(i,data,keys,args)    
    #cursor.execute(query) 
     
#   id = cursor.fetchone()[0]
###
###
#   cursor.execute( query, tuple( [str(elem) for elem in [ id ] + newdata_decam[i]] ) )
#
