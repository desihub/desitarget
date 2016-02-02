import thesis_code.fits
from astropy.io import fits
import argparse
import numpy as np

def write_table(name,keys,sql_dtype,args,indexing=[]):
    fin=open(name,'w')
    fin.write('CREATE SEQUENCE %s_id_seq;' % args.name)
    fin.write('\n\n'+'CREATE TABLE %s (' % args.name)
    #add indexing names
    for index in indexing: fin.write('\n'+'\t'+index+',')
    #add catalogue's names
    for key in keys:
        stri= '\n'+'\t'+key.lower()+' '+sql_dtype[key]
        if key != keys[-1]: stri+= ','
        fin.write(stri)
    fin.write('\n'+');'+'\n')
    fin.close()

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-fits_table",action="store",help='')
parser.add_argument("-name",action="store",help='name to use as prefix in tables')
args = parser.parse_args()

a=fits.open(args.fits_table)
data,keys= thesis_code.fits.getdata(a,1)
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
obj_indexes= ["id bigint default null",\
                "brickid integer default null"]#primary key not null default nextval('%s_id_seq'::regclass)," % args.name)
fluxes_indexes= ["id bigint default null",\
                "cand_id bigint default null"] #REFERENCES candidate (id)

#output schemas
#print 'writing objects table'
write_table('objects_table.txt',obj_keys,sql_dtype,args,indexing=obj_indexes)
#print 'writing fluxes table'
write_table('fluxes_table.txt',fluxes_keys,sql_dtype,args,indexing=fluxes_indexes)
#print 'done'
