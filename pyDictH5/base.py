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
import numpy.testing as nptest
from copy import deepcopy
try:
    # PY 3
    from . import io
except ImportError:
    # PY 2
    import io
import six


debug_level = 0
arrayEQ_tols = dict(rtol=1e-3, atol=1e-6)


_RaiseKeyError = object()


class indexer(object):

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, indx):
        return self.parent._subset(indx)


def _equiv_dict(d1, d2):
    if set(d2.keys()) == set(d1.keys()):
        retval = True
        for ky in d1:
            try:
                if isinstance(d1[ky], np.ndarray):
                    assert type(d1[ky]) is type(d2[ky])  # nopep8
                    assert d1[ky].shape == d2[ky].shape
                    if (not np.issubdtype(d1[ky].dtype, np.inexact)) or \
                       arrayEQ_tols == dict(rtol=0, atol=0):
                        nptest.assert_equal(d1[ky], d2[ky])
                    else:
                        assert np.allclose(d1[ky], d2[ky],
                                           equal_nan=True, **arrayEQ_tols)
                elif isinstance(d1[ky], dict):
                    assert _equiv_dict(d1[ky], d2[ky])
                else:
                    assert d1[ky] == d2[ky]
            except AssertionError:
                retval = False
                if debug_level > 0:
                    if isinstance(d1[ky], np.ndarray):
                        if d1[ky].shape != d2[ky].shape:
                            print('The shapes of the arrays do not match. '
                                  '({}, vs. {}).'.format(d1[ky].shape,
                                                         d2[ky].shape))
                        else:
                            frac = np.float((~np.isclose(
                                d1[ky], d2[ky], equal_nan=True,
                                **arrayEQ_tols)).sum()) / d1[ky].size
                            print('{:0.2f}% of the values in {} do not match.'
                                  .format(frac * 100, ky))
                        try:
                            assert np.allclose(d1[ky], d2[ky],
                                               rtol=1e-3, equal_nan=True)
                            print(' ... but they are close.')
                        except:
                            pass
                    else:
                        print('The values in {} do not match.'
                              .format(ky))
                else:
                    return False
        return retval
    if debug_level > 0:
        dif1 = set(d1.keys()) - set(d2.keys())
        dif2 = set(d2.keys()) - set(d1.keys())
        print("The list of items are not the same.\n"
              "Entries in 1 that are not in 2: {}\n"
              "Entries in 2 that are not in 1: {}".format(list(dif1),
                                                          list(dif2)))
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

    @property
    def subset(self, ):
        """Subset is an `indexer` for creating subsets of the data
        object using Python slice syntax.

        For example, you can do::

            dat2 = dat.subset[10:500]

        Which returns a new data object that contains the 10th through
        500th entry for each variable in the data object. This also
        operates recursively through sub-objects.
        """
        return indexer(self)

    def _subset(self, indx, raise_on_empty_array=False, copy=[]):
        out = self.__class__()
        for nm in self:
            if nm in copy:
                out[nm] = deepcopy(self[nm])
            elif isinstance(self[nm], data):
                out[nm] = self[nm]._subset(
                    indx,
                    raise_on_empty_array=raise_on_empty_array,
                    copy=copy)
            elif isinstance(self[nm], np.ndarray):
                out[nm] = self[nm][indx]
                if raise_on_empty_array and \
                   (np.array(out[nm].shape) == 0).any():
                    raise IndexError("The indexing object yields "
                                     "empty arrays for field '{}'.".format(nm))
            else:
                out[nm] = self[nm]
        return out

    def __getitem__(self, indx):
        if not isinstance(indx, six.string_types) or '.' not in indx:
            return dict.__getitem__(self, indx)
        else:
            try:
                return dict.__getitem__(self, indx)
            except KeyError:
                tmp = self
                for ky in indx.split('.'):
                    tmp = dict.__getitem__(tmp, ky)
                return tmp

    def append(self, other, array_axis=0):
        """
        Append another PyCoDa data object to this one.  This method
        assumes all arrays should be appended (concatenated) along
        axis 0.

        The appended object must have matching keys and values with
        the same data types.

        Overload this method to implement alternate appending schemes.
        """
        for nm, dat in self.items():
            if isinstance(dat, np.ndarray):
                self[nm] = np.concatenate((self[nm],
                                           other[nm]),
                                          axis=array_axis)
            elif not hasattr(dat, 'append') or isinstance(self, (PropData, list)):
                assert dat == other[nm], ("Properties in {} do not match.".format(nm))
            else:
                if 'array_axis' in dat.append.__code__.co_varnames:
                    dat.append(other[nm], array_axis=array_axis)
                else:
                    dat.append(other[nm])

    def pop(self, indx, d=_RaiseKeyError):
        if not isinstance(indx, six.string_types):
            raise IndexError(
                "<class 'PyCoDa.base.data'> objects"
                " only support string indexing.")
        if indx not in self:
            if d is _RaiseKeyError:
                dict.pop(self, indx)
            else:
                return d
        if '.' in indx:
            grp, indx = indx.rsplit('.', 1)
            tmp = self[grp]
        else:
            tmp = self
        return dict.pop(tmp, indx)

    def __setitem__(self, indx, val):
        if not isinstance(indx, six.string_types):
            raise IndexError(
                "<class 'PyCoDa.base.data'> objects"
                " only support string indexing.")
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
        """Generate the keys for all sub-groups in this data object,
        including walking through sub-groups.

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

    def __copy__(self, ):
        """Create a copy of the data object.
        """
        return deepcopy(self)

    copy = __copy__

    def __eq__(self, other):
        """
        Test for equivalence between data objects.
        """
        return self.__class__ is other.__class__ and _equiv_dict(self, other)

    def __setattr__(self, nm, val):
        if nm.startswith('_') and (nm not in self):
            # Support for 'temporary variables' that are not added to
            # the dictionary, and therefore not included in I/O
            # operations.
            object.__setattr__(self, nm, val)
        else:
            self.__setitem__(nm, val)

    # def __getstate__(self, ):
    #     return self

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
        io.hdf5_write(buf, self, chunks=chunks, compression=compression)


class PropData(data):

    def _subset(self, indx):
        return self

    def append(self, other, array_axis=0):
        """
        """
        assert self == other, ("These data items cannot be joined because "
                               "their properties do not match.")
        return self


class geodat(data):
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
                      for nm, dat in self.items()
                      if isinstance(dat, geodat)}
        if len(other_inds) > 0:
            return inds, other_inds
        return inds
