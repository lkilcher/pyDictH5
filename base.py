"""
A base data module for compound data types that supports I/O
to/from hdf5.

This data type is based on dict. The keys of the data must be strings.

The data type is smart, such that you can 'subset' the data by
indexing it with integers, slices, np.ndarrays, or lists. Or, you
can get fields by indexing those.

Objects that are added to the dictionary will automatically be
stored when using one of the i/o routines, e.g.:

    >>> d = pydata.data()
    >>> d['time'] = np.arange(10)
    >>> d.to_hdf5('test.h5')
    >>> d_copy = load('test.h5')
    >>> d_copy['time']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

You can also access data as either attributes, or items:

    >>> d_copy['time'] == d.time
    True

However, only dict entries are stored:
    >>> d.time2 = np.arange(1, 11)
    >>> d.to_hdf5('test2.h5')

    >>> d_copy = load('test2.h5')
    KeyError: 'time2'

    >>> d_copy.time2
    '<data>' object has no attribute 'time2'

    >>> d_copy['time2']
"""
import numpy as np
import pandas as pd
import h5py
import cPickle
from . import _version as ver

indx_subset_valid = (slice, np.ndarray, list, int)


def load_hdf5(buf, group=None, dat_class=None):
    """
    Load a data object from an hdf5 file.
    """
    fl = None
    if isinstance(buf, basestring):
        fl = buf = h5py.File(buf, 'r')
    if group is not None:
        buf = buf[group]
    if dat_class is None:
        try:
            out = cPickle.loads(buf.attrs['__pyclass__'])()
        except:
            out = data()
    else:
        out = dat_class()
    if hasattr(buf, 'iteritems'):
        for nm, dat in buf.iteritems():
            if dat.__class__ is h5py.Group:
                out[nm] = load_hdf5(dat)
            else:
                out[nm] = np.array(dat)
    else:
        out = np.array(buf)
    if fl:
        fl.close()
    return out


class data(dict):

    def append(self, other):
        for nm, dat in self.iteritems():
            if isinstance(dat, np.ndarray):
                self[nm] = np.concatenate((self[nm],
                                           other[nm]),
                                          axis=0)
            else:
                dat.append(other[nm])

    def __getattr__(self, nm):
        try:
            return self[nm]
        except KeyError:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__, nm))

    def __getitem__(self, indx):
        if isinstance(indx, indx_subset_valid + (tuple, )):
            return self.subset(indx)
        else:
            return dict.__getitem__(self, indx)

    def to_hdf5(self, buf, chunks=True, compression='gzip'):
        if isinstance(buf, basestring):
            isfile = True
            buf = h5py.File(buf, 'w')
            buf.attrs['__package_name__'] = ver.__package__
            buf.attrs['__version__'] = ver.__version__
        else:
            isfile = False
        buf.attrs['__pyclass__'] = cPickle.dumps(self.__class__)
        for nm, dat in self.iteritems():
            if isinstance(dat, data):
                dat.to_hdf5(buf.create_group(nm),
                            chunks=chunks, compression=compression)
            else:
                buf.create_dataset(name=nm, data=dat,
                                   chunks=chunks, compression=compression)
        if isfile:
            buf.close()

    def subset(self, inds, **kwargs):
        """
        Take a subset of this data.

        Parameters
        ----------
        inds : {slice, ndarray, list, int}
               The indexing object to use to subset the data.
        **kwargs: name: index
               For compound data types that contain sub-compound data
               types, specify the indices of each sub-data with name:
               index pairs. The 'name' is the name of the sub-data
               field, and the indices should be like `inds`.
        """
        if (inds.__class__ is tuple and len(tuple) == 2
            and isinstance(inds[0], indx_subset_valid)
            and isinstance(inds[1], dict)):
            return self.subset(inds[0], **inds[1])
        out = self.__class__()
        for nm in self:
            if isinstance(self[nm], data):
                if nm in kwargs:
                    out[nm] = self[nm][kwargs[nm]]
            else:
                out[nm] = self[nm][inds]
        return out


class tabular(data):
    """
    A class for holding 'spreadsheet' type data.

    This data-type is assumed to be planar (2-D, or rows and columns)
    only.
    """
    def to_dataframe(self,):
        siteout = None
        for nm, val in self.iteritems():
            if val.ndim == 1:
                if siteout is None:
                    siteout = pd.DataFrame(val, columns=[nm])
                else:
                    siteout.loc[:, nm] = pd.Series(val)
            else:
                siteout[nm] = pd.DataFrame(val)
        return siteout

    def to_excel(self, fname):
        out = {}
        buf = pd.io.excel.ExcelWriter(fname)
        siteout = self.to_dataframe()
        siteout.to_excel(buf, sheet_name='Site')
        for nm in out:
            if np.iscomplex(out[nm]).any():
                out[nm].astype('S').to_excel(buf, sheet_name=nm)
            else:
                out[nm].to_excel(buf, sheet_name=nm)
        buf.close()


class geodat(data):
    """
    A class for holding 'gis' type data.

    This data is assumed to have attributes lat/lon.
    """

    def llrange(self, lon=None, lat=None):
        if lon is not None:
            inds = (lon[0] < self['lon']) & (self['lon'] < lon[1])
        else:
            inds = np.ones(self['lon'].shape, dtype='bool')
        if lat is not None:
            inds &= (lat[0] < self['lat']) & (self['lat'] < lat[1])
        other_inds = {nm: dat.llrange(lon=lon, lat=lat)
                      for nm, dat in self.iteritems()
                      if isinstance(dat, geodat)}
        if len(other_inds) > 0:
            return inds, other_inds
        return inds
