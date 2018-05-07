# This code is from astrometry.net: 
# https://github.com/dstndstn/astrometry.net 
# as of version 0.74 (git hash a65e168d). 
# This file (util/fits.py) is licensed under the BSD-3 license.
"""
=======================================
desitarget.skyutilities.astrometry.fits
=======================================

Module so desitarget sky fiber code doesn't need explicit astrometry.net dependencies
"""
#ADM Needed for backwards-compatibility with Python 2 print
from __future__ import (print_function)
import numpy as np
import os

def cut_array(val, I, name=None, to=None):
    if type(I) is slice:
        if to is None:
            return val[I]
        else:
            val[I] = to
            return

    if isinstance(val, (np.ndarray, np.core.defchararray.chararray)):
        # You can't slice a two-dimensional, length-zero, numpy array,
        # with an empty array.
        if len(val) == 0:
            return val
        if to is None:
            # Indexing an array with an empty index array works, but
            # ONLY if it is of integer or bool type.
            # Check for __len__ because "I" can be a plain int too.
            if hasattr(I, '__len__') and len(I) == 0:
                return np.array([], val.dtype)
            return val[I]
        else:
            val[I] = to
            return

    inttypes = [int, np.int64, np.int32, np.int]

    if type(val) in [list,tuple] and type(I) in inttypes:
        if to is None:
            return val[I]
        else:
            val[I] = to
            return

    # HACK -- emulate numpy's boolean and int array slicing
    # (when "val" is a normal python list)
    if type(I) is np.ndarray and hasattr(I, 'dtype') and ((I.dtype.type in [bool, np.bool])
                                                             or (I.dtype == bool)):
        try:
            if to is None:
                return [val[i] for i,b in enumerate(I) if b]
            else:
                for i,(b,t) in enumerate(zip(I,to)):
                    if b:
                        val[i] = t
                return
        except:
            print('Failed to slice field', name)
            #setattr(rtn, name, val)
            #continue

    if type(I) is np.ndarray and all(I.astype(int) == I):
        if to is None:
            return [val[i] for i in I]
        else:
            #[val[i] = t for i,t in zip(I,to)]
            for i,t in zip(I,to):
                val[i] = t
                
    if (np.isscalar(I) and hasattr(I, 'dtype') and
        I.dtype in inttypes):
        if to is None:
            return val[int(I)]
        else:
            val[int(I)] = to
            return

    if hasattr(I, '__len__') and len(I) == 0:
        return []

    print('Error slicing array:')
    print('array is')
    print('  type:', type(val))
    print('  ', val)
    print('cut is')
    print('  type:', type(I))
    print('  ', I)
    raise Exception('Error in cut_array')

class tabledata(object):

    class td_iter(object):
        def __init__(self, td):
            self.td = td
            self.i = 0
        def __iter__(self):
            return self
        def next(self):
            if self.i >= len(self.td):
                raise StopIteration
            X = self.td[self.i]
            self.i += 1
            return X
        # py3
        __next__ = next

    def __init__(self, header=None):
        self._length = 0
        self._header = header
        self._columns = []
    def __str__(self):
        return 'tabledata object with %i rows and %i columns' % (len(self), len([k for k in self.__dict__.keys() if not k.startswith('_')]))
    def __repr__(self):
        if len(self) == 1:
            vals = []
            for k in self.columns():
                v = self.get(k)
                if (not np.isscalar(v)) and len(v) == 1:
                    v = v[0]
                vals.append(v)
            return '<tabledata object with %i rows and %i columns: %s>' % (
                len(self), len(self.columns()), ', '.join(['%s=%s' % (k,v) for k,v in zip(self.columns(), vals)]))
        return '<tabledata object with %i rows and %i columns: %s>' % (
            len(self), len(self.columns()), ', '.join(self.columns()))
    
    def about(self):
        keys = [k for k in self.__dict__.keys() if not k.startswith('_')]
        print('tabledata object with %i rows and %i columns:' % (len(self),  len(keys)))
        keys.sort()
        for k in keys:
            print('  ', k, end=' ')
            v = self.get(k)
            print('(%s)' % (str(type(v))), end=' ')
            if np.isscalar(v):
                print(v, end=' ')
            elif hasattr(v, 'shape'):
                print('shape', v.shape, end=' ')
            elif hasattr(v, '__len__'):
                print('length', len(v), end=' ')
            else:
                print(v, end=' ')

            if hasattr(v, 'dtype'):
                print('dtype', v.dtype, end='')
            print()

    def __setattr__(self, name, val):
        object.__setattr__(self, name, val)
        #print('set', name, 'to', val)
        if (self._length == 0) and (not (name.startswith('_'))) and hasattr(val, '__len__') and len(val) != 0 and type(val) != str:
            self._length = len(val)
        if hasattr(self, '_columns') and not name in self._columns:
            self._columns.append(name)
    def set(self, name, val):
        self.__setattr__(name, val)
    def getcolumn(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            # try case-insensitive
            for k,v in self.__dict__.items():
                if k.lower() == name.lower():
                    return v
            raise
        #except:
        #   return self.__dict__[name.lower()]
    def get(self, name):
        return self.getcolumn(name)
    # Returns the list of column names, as they were ordered in the input FITS or text table.
    def get_columns(self, internal=False):
        if internal:
            return self._columns[:]
        return [x for x in self._columns if not x.startswith('_')]
    # Returns the original FITS header.
    def get_header(self):
        return self._header

    def to_dict(self):
        return dict([(k,self.get(k)) for k in self.columns()])

    def to_np_arrays(self):
        for col in self.get_columns():
            self.set(col, np.array(self.get(col)))

    def columns(self):
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    def __len__(self):
        return self._length
    def delete_column(self, c):
        del self.__dict__[c]
        self._columns.remove(c)

    def rename(self, c_old, c_new):
        setattr(self, c_new, getattr(self, c_old))
        self.delete_column(c_old)
        
    def __setitem__(self, I, O):

        #### TEST

        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            cut_array(val, I, name, to=O.get(name))
        return
        ####

        
        if type(I) is slice:
            print('I:', I)
            # HACK... "[:]" -> slice(None, None, None)
            if I.start is None and I.stop is None and I.step is None:
                I = np.arange(len(self))
            else:
                I = np.arange(I.start, I.stop, I.step)
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            # ?
            if np.isscalar(val):
                self.set(name, O.get(name))
                continue
            try:
                val[I] = O.get(name)
            except Exception:
                # HACK -- emulate numpy's boolean and int array slicing...
                ok = False
                if not ok:
                    print('Error in slicing an astrometry.util.pyfits_utils.table_data object:')
                    import pdb; pdb.set_trace()

                    print('While setting member:', name)
                    print(' setting elements:', I)
                    print(' from obj', O)
                    print(' target type:', type(O.get(name)))
                    print(' dest type:', type(val))
                    print('index type:', type(I))
                    if hasattr(I, 'dtype'):
                        print('  index dtype:', I.dtype)
                    print('my length:', self._length)
                    raise Exception('error in fits_table indexing')

    def copy(self):
        rtn = tabledata()
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                #print('copying scalar', name)
                rtn.set(name, val)
                continue
            if type(val) in [np.ndarray, np.core.defchararray.chararray]:
                #print('copying numpy array', name)
                rtn.set(name, val.copy())
                continue
            if type(val) in [list,tuple]:
                #print('copying list', name)
                rtn.set(name, val[:])
                continue
            print('in pyfits_utils: copy(): can\'t copy', name, '=', val[:10], 'type', type(val))
        rtn._header = self._header
        if hasattr(self, '_columns'):
            rtn._columns = [c for c in self._columns]
        return rtn

    def cut(self, I):
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                continue
            C = cut_array(val, I, name)
            self.set(name, C)
            self._length = len(C)

    def __getitem__(self, I):
        rtn = self.__class__()
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                rtn.set(name, val)
                continue
            try:
                C = cut_array(val, I, name)
            except:
                print('Error in cut_array() via __getitem__, name', name)
                raise
            rtn.set(name, C)

            if np.isscalar(I):
                rtn._length = 1
            else:
                rtn._length = len(getattr(rtn, name))
        rtn._header = self._header
        if hasattr(self, '_columns'):
            rtn._columns = [c for c in self._columns]
        return rtn
    def __iter__(self):
        return tabledata.td_iter(self)

    def append(self, X):
        for name,val in self.__dict__.items():
            if name.startswith('_'):
                continue
            if np.isscalar(val):
                continue
            try:
                val2 = X.getcolumn(name)
                if type(val) is list:
                    newX = val + val2
                else:
                    newX = np.append(val, val2, axis=0)
                self.set(name, newX)
                self._length = len(newX)
            except Exception:
                print('exception appending element "%s"' % name)
                raise
                
    def write_to(self, fn, columns=None, header='default', primheader=None,
                 use_fitsio=True, append=False, append_to_hdu=None,
                 fits_object=None,
                 **kwargs):

        fitsio = None
        if use_fitsio:
            try:
                import fitsio
            except:
                pass

        if columns is None:
            columns = self.get_columns()

        if fitsio:
            arrays = [self.get(c) for c in columns]
            if fits_object is not None:
                fits = fits_object
            else:
                fits = fitsio.FITS(fn, 'rw', clobber=(not append))

            arrays = [np.array(a) if isinstance(a,list) else a
                      for a in arrays]
            # py3
            if b' ' != ' ':
                aa = []
                for a in arrays:
                    if 'U' in str(a.dtype):
                        aa.append(a.astype(np.bytes_))
                    else:
                        aa.append(a)
                arrays = aa
            
            if header == 'default':
                header = None
            try:
                if append and append_to_hdu is not None:
                    fits[append_to_hdu].append(arrays, names=columns, header=header, **kwargs)
                else:
                    if primheader is not None:
                        fits.write(None, header=primheader)
                    fits.write(arrays, names=columns, header=header, **kwargs)

                # If we were passed in a fits object, don't close it.
                if fits_object is None:
                    fits.close()
            except:
                print('Failed to write FITS table')
                print('Columns:')
                for c,a in zip(columns, arrays):
                    print('  ', c, type(a), end='')
                    try:
                        print(a.dtype, a.shape, end='')
                    except:
                        pass
                    print()
                raise
            return

        fc = self.to_fits_columns(columns)
        T = pyfits.BinTableHDU.from_columns(fc)
        if header == 'default':
            header = self._header
        if header is not None:
            add_nonstructural_headers(header, T.header)
        if primheader is not None:
            P = pyfits.PrimaryHDU()
            add_nonstructural_headers(primheader, P.header)
            pyfits.HDUList([P, T]).writeto(fn, clobber=True)
        else:
            pyfits_writeto(T, fn)

    writeto = write_to

    def normalize(self, columns=None):
        if columns is None:
            columns = self.get_columns()
        for c in columns:
            X = self.get(c)
            X = normalize_column(X)
            self.set(c, X)

    def to_fits_columns(self, columns=None):
        cols = []

        fmap = {np.float64:'D',
                np.float32:'E',
                np.int32:'J',
                np.int64:'K',
                np.uint8:'B', #
                np.int16:'I',
                #np.bool:'X',
                #np.bool_:'X',
                np.bool:'L',
                np.bool_:'L',
                np.string_:'A',
                }

        if columns is None:
            columns = self.get_columns()
                
        for name in columns:
            if not name in self.__dict__:
                continue
            val = self.get(name)

            if type(val) in [list, tuple]:
                val = np.array(val)

            try:
                val = normalize_column(val)
            except:
                pass

            try:
                fitstype = fmap.get(val.dtype.type, 'D')
            except:
                print('Table column "%s" has no "dtype"; skipping' % name)
                continue

            if fitstype == 'X':
                # pack bits...
                pass
            if len(val.shape) > 1:
                fitstype = '%i%s' % (val.shape[1], fitstype)
            elif fitstype == 'A' and val.itemsize > 1:
                # strings
                fitstype = '%i%s' % (val.itemsize, fitstype)
            else:
                fitstype = '1'+fitstype
            #print('fits type', fitstype)
            try:
                col = pyfits.Column(name=name, array=val, format=fitstype)
            except:
                print('Error converting column', name, 'to a pyfits column:')
                print('fitstype:', fitstype)
                try:
                    print('numpy dtype:')
                    print(val.dtype)
                except:
                    pass
                print('value:', val)
                raise
            cols.append(col)
        return cols

    def add_columns_from(self, X, dup=None):
        assert(len(self) == len(X))
        mycols = self.get_columns()
        for c in X.get_columns():
            if c in mycols:
                if dup is None:
                    print('Not copying existing column', c)
                    continue
                else:
                    self.set(dup + c, X.get(c))
            else:
                self.set(c, X.get(c))

def fits_table(dataorfn=None, rows=None, hdunum=1, hdu=None, ext=None,
               header='default',
               columns=None,
               column_map=None,
               lower=True,
               mmap=True,
               normalize=True,
               use_fitsio=True,
               tabledata_class=tabledata):
    '''
    If 'columns' (a list of strings) is passed, only those columns
    will be read; otherwise all columns will be read.
    '''

    if dataorfn is None:
        return tabledata_class(header=header)

    fitsio = None
    if use_fitsio:
        try:
            import fitsio
        except:
            pass

    pf = None
    hdr = None
    # aliases
    if hdu is not None:
        hdunum = hdu
    if ext is not None:
        hdunum = ext
    if isinstance(dataorfn, str):

        if fitsio:
            F = fitsio.FITS(dataorfn)
            data = F[hdunum]
            hdr = data.read_header()
        else:
            global pyfits
            pf = pyfits.open(dataorfn, memmap=mmap)
            data = pf[hdunum].data
            if header == 'default':
                hdr = pf[hdunum].header
            del pf
            pf = None
    else:
        data = dataorfn

    if data is None:
        return None
    T = tabledata_class(header=hdr)

    T._columns = []

    if fitsio:
        isrecarray = False
        try:
            import pyfits.core
            # in a try/catch in case pyfits isn't available
            isrecarray = (type(data) == pyfits.core.FITS_rec)
        except:
            try:
                from astropy.io import fits as pyfits
                isrecarray = (type(data) == pyfits.core.FITS_rec)
            except:
                #import traceback
                #traceback.print_exc()
                pass
        if not isrecarray:
            try:
                import pyfits.fitsrec
                isrecarray = (type(data) == pyfits.fitsrec.FITS_rec)
            except:
                try:
                    from astropy.io import fits as pyfits
                    isrecarray = (type(data) == pyfits.fitsrec.FITS_rec)
                except:
                    #import traceback
                    #traceback.print_exc()
                    pass
        #if not isrecarray:
        #    if type(data) == np.recarray:
        #        isrecarray = True


    if fitsio and not isrecarray:
        # fitsio sorts the rows and de-duplicates them, so compute
        # permutation vector 'I' to undo that.
        I = None
        if rows is not None:
            rows,I = np.unique(rows, return_inverse=True)

        if type(data) == np.ndarray:
            dd = data
            if columns is None:
                columns = data.dtype.fields.keys()
        else:
            if data.get_exttype() == 'IMAGE_HDU':
                # This can happen on empty tables (eg, empty SDSS photoObjs)
                return None
            try:
                dd = data.read(rows=rows, columns=columns, lower=True)
            except:
                import sys
                print('Error reading from FITS object', type(data), data, 'dataorfn', dataorfn, file=sys.stderr)
                raise
            if dd is None:
                return None

        if columns is None:
            try:
                columns = data.get_colnames()
            except:
                columns = data.colnames

            if lower:
                columns = [c.lower() for c in columns]

        for c in columns:
            X = dd[c.lower()]
            if I is not None:
                # apply permutation
                X = X[I]
            if column_map is not None:
                c = column_map.get(c, c)
            if lower:
                c = c.lower()
            T.set(c, X)
        
    else:
        if columns is None:
            columns = data.dtype.names
        for c in columns:
            col = data.field(c)
            if rows is not None:
                col = col[rows]
            if normalize:
                col = normalize_column(col)
            if column_map is not None:
                c = column_map.get(c, c)
            if lower:
                c = c.lower()
            T.set(c, col)

    # py3: convert FITS strings from Python bytes to strings.
    if b' ' != ' ':
        # py3
        for c in columns:
            X = T.get(c)
            t = str(X.dtype)
            if 'S' in t:
                X = X.astype(np.str)
                T.set(c, X)
                #print('Converted', c, 'from', t, 'to', X.dtype)

    return T

table_fields = fits_table
