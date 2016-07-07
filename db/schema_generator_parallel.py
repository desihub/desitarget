import multiprocessing 
import resource
import os
import numpy as np
from argparse import ArgumentParser

def current_mem_usage():
	'''return mem usage in MB'''
	return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.**2

def bash(cmd):
	return os.system('%s' % cmd)
	#if ret:
	#	print 'command failed: %s' % cmd
	#	raise ValueError

def work(fits_file):
    name = multiprocessing.current_process().name
    cmd='python schema_generator.py -fits_file %s -schema dr3 -load_db 1' % fits_file
    ret= bash('echo %s executing: %s' % (name,cmd))
    ret= bash(cmd)
    print '%s maximum memory usage: %.2f (mb)' % (name, current_mem_usage())
    return ret
	
if __name__ == '__main__':
    parser = ArgumentParser(description="test")
    parser.add_argument("-cores",type=int,action="store",help='',required=True)
    parser.add_argument("-file_tractor_cats",action="store",help='',required=True)
    args = parser.parse_args()
     
    print "CPU has %d cores" % multiprocessing.cpu_count()
    pool = multiprocessing.Pool(args.cores)
    fits_files= np.loadtxt(args.file_tractor_cats,dtype=str)
    results=pool.map(work, fits_files)
    pool.close()
    pool.join()
    del pool
    print 'Global maximum memory usage: %.2f (mb)' % current_mem_usage()
    err=np.array(results).astype(bool)
    print "These inputs failed:"
    print fits_files[err]

