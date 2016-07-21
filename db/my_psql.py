import psycopg2
import os
import numpy as np

def select(cmd,outname,outdir='/project/projectdirs/desi/users/burleigh'):
    '''use "cmd" to select data, save output to file "outname"'''
    con= psycopg2.connect(host='scidb2.nersc.gov', user='desi_admin', database='desi')
    cursor = con.cursor()
    cursor.execute('\o %s' % os.path.join(outdir,outname)) 
    cursor.execute(cmd) 
    cursor.execute('\o') 
    con.close()
    print "selected with:\n",cmd
    print "saved result to %s" % os.path.join(outdir,outname)

def read_from_psql_file(fn,use_cols=range(14),str_cols=['type']):
    '''return data dict for DECaLS()
    fn -- file name of psql db txt file
    use_cols -- list of column indices to get, first column is 0
    str_cols -- list of column names that should have type str not float'''
    #get column names
    fin=open(fn,'r')
    cols=fin.readline()
    fin.close()
    cols= np.char.strip( np.array(cols.split('|'))[use_cols] )
    #get data
    arr=np.loadtxt(fn,dtype='str',comments='(',delimiter='|',skiprows=2,usecols=use_cols)
    data={}
    for i,col in enumerate(cols):
        if col in str_cols: data[col]= np.char.strip( arr[:,i].astype(str) )
        else: data[col]= arr[:,i].astype(float)
    return data
