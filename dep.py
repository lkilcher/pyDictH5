from .base import data
import numpy as np

indx_subset_valid = (slice, np.ndarray, list, int)


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
