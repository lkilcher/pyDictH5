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
import numpy.testing as nptest
from copy import deepcopy

indx_subset_valid = (slice, np.ndarray, list, int)


def load_hdf5(buf, group=None, dat_class=None):
    """
    Load a data object from an hdf5 file.
    """
    if isinstance(buf, basestring):
        with h5py.File(buf, 'r') as fl:
            return load_hdf5(fl, group=group, dat_class=dat_class)
    if group is not None:
        buf = buf[group]
    if dat_class is None:
        try:
            out = cPickle.loads(buf.attrs['__pyclass__'])()
        except AttributeError:
            print("Warning: Class '{}' not found, defaulting to generic 'pycoda.data'.".format(buf.attrs['__pyclass__']))
            out = data()
    else:
        out = dat_class()
    if hasattr(buf, 'iteritems'):
        for nm, dat in buf.iteritems():
            if dat.__class__ is h5py.Group:
                out[nm] = load_hdf5(dat)
            else:
                if dat.dtype == 'O' and '_type' \
                   in dat.attrs and dat.attrs['_type'] == 'NumPy Object Array':
                    shp = dat.shape
                    out[nm] = np.empty(shp, dtype='O')
                    for idf in xrange(dat.size):
                        ida = np.unravel_index(idf, shp)
                        if dat[ida] == '':
                            out[nm][ida] = None
                        else:
                            out[nm][ida] = cPickle.loads(dat[ida])
                else:
                    out[nm] = np.array(dat)
    else:
        out = np.array(buf)
    return out


def _equiv_dict(d1, d2, print_diff=False):
    """Test whether two dictionary-like are equivalent.

    This includes support for arrays so that you don't get a:

      ValueError: The truth value of an array with more than one
      element is ambiguous. Use a.any() or a.all()

    """
    if type(d1) is not type(d2):
        return False
    if set(d2.keys()) == set(d1.keys()):
        for ky in d1:
            try:
                if isinstance(d1[ky], np.ndarray):
                    assert type(d1[ky]) is type(d2[ky])
                    nptest.assert_equal(d1[ky], d2[ky])
                elif isinstance(d1[ky], dict):
                    assert _equiv_dict(d1[ky], d2[ky],
                                       print_diff=print_diff)
                else:
                    assert d1[ky] == d2[ky]
            except AssertionError:
                if print_diff:
                    print('The values in {} do not match between the data objects.'
                          .format(ky, d1, d2))
                return False
        return True
    if print_diff:
        dif1 = set(d1.keys()) - set(d2.keys())
        dif2 = set(d2.keys()) - set(d1.keys())
        print("The list of items are not the same.\n"
              "Entries in 1 that are not in 2: {}\n"
              "Entries in 2 that are not in 1: {}".format(list(dif1), list(dif2)))
    return False


class data(dict):
    """
    The base PyCoDa class.

    This class supports temporary attribute variables with leading
    underscores (e.g. '_temp'). However, if a dict-entry already
    exists with that value, it *will* point to that entry.

    This class is capable of storing object arrays. This is done by
    pickling each item in the object array.

    """

    def __getitem__(self, indx):
        if '.' not in indx:
            return dict.__getitem__(self, indx)
        else:
            try:
                return dict.__getitem__(self, indx)
            except KeyError:
                tmp = self
                for ky in indx.split('.'):
                    tmp = dict.__getitem__(tmp, ky)
                return tmp

    def __setitem__(self, indx, val):
        if not isinstance(indx, basestring):
            raise IndexError(
                "<class 'PyCoDa.base.data'> objects"
                " only support string indexes.".format(self.__class__))
        if '.' in indx:
            grp, indx = indx.rsplit('.', 1)
            tmp = self[grp]
        else:
            tmp = self
        if indx in dir(tmp):
            raise KeyError("The attribute '{}' exists: Creating a key that "
                           "matches an attribute name is forbidden.".format(indx))
        dict.__setitem__(tmp, indx, val)

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __repr__(self, ):
        outstr = '{}: Data Object with Keys:\n'.format(self.__class__)
        for k in self:
            outstr += '  {}\n'.format(k)
        return outstr

    def iter_subgroups(self, include_hidden=False):
        """Generate the keys for all sub-groups in this data object.

        Parameters
        ----------
        include_hidden : bool (Default: False)
              Whether entries starting with '_' should be included in
              the iteration.
        """
        for ky in self:
            if not include_hidden and ky.startswith('_'):
                continue
            if isinstance(self[ky], data):
                yield ky
                for ky2 in self[ky].iter_subgroups():
                    if not include_hidden and ky2.startswith('_'):
                        continue
                    if isinstance(self[ky][ky2], data):
                        yield '{}.{}'.format(ky, ky2)

    def iter_data(self, include_hidden=False):
        """Generate the keys for all data items in this data object,
        including walking through sub-data objects.

        Parameters
        ----------
        include_hidden : bool (Default: False)
              Whether entries starting with '_' should be included in
              the iteration.
        """
        for ky in self:
            if not include_hidden and ky.startswith('_'):
                continue
            if isinstance(self[ky], data):
                for ky2 in self[ky].iter_data(include_hidden=include_hidden):
                    yield '{}.{}'.format(ky, ky2)
            else:
                yield ky

    ### This needs to mimic os.walk.
    # def walk(self, include_hidden=False):
    #     """Generate the keys for all data items in this data object,
    #     including walking through sub-data objects.

    #     Parameters
    #     ----------
    #     include_hidden : bool (Default: False)
    #           Whether entries starting with '_' should be included in
    #           the iteration.
    #     """
    #     triples = [self._walkthis(), ]
    #     for g in triples[1]:
    #         triples.append(g._walkthis())

    # def _walkthis(self, thispath, walklist, include_hidden=False):
    #     path = ''
    #     data = []
    #     groups = []
    #     for ky in self:
    #         if not include_hidden and ky.startswith('_'):
    #             continue
    #         if isinstance(self[ky], data):
    #             groups.append(ky)
    #         else:
    #             data.append(ky)
    #     return path, groups, data

    def __copy__(self, ):
        return deepcopy(self)

    copy = __copy__

    def __eq__(self, other, print_diff=False):
        """
        Test for equivalence between data objects.
        """
        return _equiv_dict(self, other, print_diff=print_diff)

    def __setattr__(self, nm, val):
        if nm.startswith('_') and (nm not in self):
            # Support for 'temporary variables' that are not added to
            # the dictionary, and therefore not included in I/O
            # operations.
            object.__setattr__(self, nm, val)
        else:
            self.__setitem__(nm, val)

    def __getstate__(self, ):
        return self

    def __getattribute__(self, nm):
        try:
            return dict.__getattribute__(self, nm)
        except AttributeError:
            try:
                return self[nm]
            except KeyError:
                raise AttributeError("'{}' object has no attribute '{}'"
                                     .format(str(self.__class__).split("'")[-2].split('.')[-1],
                                             nm))

    def to_hdf5(self, buf, chunks=True, compression='gzip'):
        """
        Write the data in this object to an hdf5 file.
        """
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
                if dat.dtype == 'O':
                    shp = dat.shape
                    ds = buf.create_dataset(nm, shp, dtype=h5py.special_dtype(vlen=bytes))
                    ds.attrs['_type'] = 'NumPy Object Array'
                    for idf, val in enumerate(dat.flat):
                        ida = np.unravel_index(idf, shp)
                        ds[ida] = cPickle.dumps(val)
                else:
                    buf.create_dataset(name=nm, data=dat,
                                       chunks=chunks, compression=compression)
        if isfile:
            buf.close()


class SpecData(data):
    """
    A class for storing spectral data.

    The last dimension of all data in this class should be frequency.
    """
    pass


class flat(data):
    """
    This class of data assumes that all data in this class have the
    same shape in the first dimension. This class makes it possible to
    slice, and sub-index the data within it from the object's top
    level.

    Notes
    -----

    For example, if 'dat' is defined as::

        dat = flat()
        dat.time = np.arange(10)
        dat.u = 2 + 0.6 * np.arange(10)
        dat.v = 0.3 * np.ones(10)
        dat.w = 0.1 * np.ones(10)

    The data within that structure can be sub-indexed by::

        subdat = dat[:5]

    Also, if you have a similarly defined data object::

        dat2 = flat()
        dat2.time = np.arange(10, 20)
        dat2.u = 0.8 * np.arange(10)
        dat2.v = 0.6 * np.ones(10)
        dat2.w = 0.2 * np.ones(10)

    One can join these data object by:

        dat.append(dat2)

    """

    def empty_like(self, npt, array_creator=np.empty):
        """
        Create empty arrays with first dimension of length `npt`, with
        other dimensions consistent with this data object.

        `array_creator` may be used to specify the function that creates
        the arrays (e.g. np.zeros, np.ones). The default is np.empty.

        """
        out = self.__class__()
        for nm, dat in self.iteritems():
            if isinstance(dat, np.ndarray):
                shp = list(dat.shape)
                shp[0] = npt
                out[nm] = array_creator(shp, dtype=dat.dtype,)
            elif hasattr(self, 'empty_like'):
                out[nm] = dat.empty_like(npt, array_creator=array_creator)
        return out

    def __getitem__(self, indx):
        if isinstance(indx, indx_subset_valid + (tuple, )):
            return self.subset(indx)
        else:
            return data.__getitem__(self, indx)

    def append(self, other):
        """
        Append another PyCoDa data object to this one.  This method
        assumes all arrays should be appended (concatenated) along
        axis 0.

        The appended object must have matching keys and values with
        the same data types.

        Overload this method to implement alternate appending schemes.
        """
        for nm, dat in self.iteritems():
            if isinstance(dat, np.ndarray):
                self[nm] = np.concatenate((self[nm],
                                           other[nm]),
                                          axis=0)
            else:
                dat.append(other[nm])

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
        if (inds.__class__ is tuple and len(inds) == 2 and
                isinstance(inds[0], indx_subset_valid) and
                isinstance(inds[1], dict)):
            return self.subset(inds[0], **inds[1])
        out = self.__class__()
        for nm in self:
            if isinstance(self[nm], data):
                if nm in kwargs:
                    out[nm] = self[nm][kwargs[nm]]
                elif isinstance(self[nm], flat):
                    out[nm] = self[nm][inds]
                #else:
                    #print nm, self[nm].__class__
            else:
                out[nm] = self[nm][inds]
        return out


class TimeBased(data):
    """
    This class of data assumes that all data in an instance has the
    same last dimension. This makes it possible to slice and sub-index
    the data within it from the object's top level.

    Notes
    -----

    For example, if 'dat' is defined as::

        dat = TimeBased()
        dat.time = np.arange(10)
        dat.u = np.vstack(2 + 0.6 * np.arange(10),
                          0.3 * np.ones(10),
                          0.1 * np.ones(10))

    The data within that structure can be sub-indexed by::

        subdat = dat[:5]

    Also, if you have a similarly defined data object::

        dat2 = flat()
        dat2.time = np.arange(10, 20)
        dat2.u = np.vstack(0.8 * np.arange(10),
                           0.6 * np.ones(10),
                           0.2 * np.ones(10))

    One can join these data object by:

        dat.append(dat2)

    """

    def __getitem__(self, indx):
        if isinstance(indx, indx_subset_valid + (tuple, )):
            return self.subset(indx)
        else:
            return dict.__getitem__(self, indx)


class tabular(flat):
    """
    A class for holding tabular (e.g. 'spreadsheet') type data.

    This data-type is assumed to be planar (2-D, rows and columns)
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


class geodat(flat):
    """
    A class for holding 'gis' type data.

    This data is assumed to have lat/lon attributes.
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


class marray(np.ndarray):

    def __new__(cls, input_array, meta={}):
        # Input array is an already formed ndarray instance
        # We first cast it to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.meta = meta
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        # tmp = getattr(obj, 'meta', None)
        # self.meta = deepcopy(tmp)
