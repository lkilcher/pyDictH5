import h5py
from . import _version as ver
import cPickle as pkl
import base as bm
import numpy as np


def hdf5_write(buf, dat, chunks=True, compression='gzip'):
    if isinstance(buf, basestring):
        isfile = True
        buf = h5py.File(buf, 'w')
        buf.attrs['__package_name__'] = ver.__package__
        buf.attrs['__version__'] = ver.__version__
    else:
        isfile = False
    buf.attrs['__pyclass__'] = pkl.dumps(dat.__class__)
    for nm, dat in dat.iteritems():
        if isinstance(dat, bm.data):
            dat.to_hdf5(buf.create_group(nm),
                        chunks=chunks, compression=compression)
        elif isinstance(dat, dict):
            tmp = bm.data(dat)
            tmp.to_hdf5(buf.create_group(nm),
                        chunks=chunks, compression=compression)
            buf[nm].attrs['__pyclass__'] = pkl.dumps(dict)
        else:
            if isinstance(dat, np.ndarray):
                if dat.dtype == 'O':
                    shp = dat.shape
                    ds = buf.create_dataset(nm, shp, dtype=h5py.special_dtype(vlen=bytes))
                    ds.attrs['_type'] = 'NumPy Object Array'
                    for idf, val in enumerate(dat.flat):
                        ida = np.unravel_index(idf, shp)
                        ds[ida] = pkl.dumps(val)
                else:
                    ds = buf.create_dataset(name=nm, data=dat,
                                            chunks=chunks, compression=compression)
            else:
                try:
                    ds = buf.create_dataset(nm, (), data=dat)
                    ds.attrs['_type'] = 'non-array scalar'
                except:
                    ds = buf.create_dataset(nm, (), data=pkl.dumps(dat),
                                            dtype=h5py.special_dtype(vlen=bytes))
                    ds.attrs['_type'] = 'pickled object'
            ds.attrs['__pyclass__'] = pkl.dumps(type(dat))
    if isfile:
        buf.close()


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
            out = pkl.loads(buf.attrs['__pyclass__'])()
        except AttributeError:
            print("Warning: Class '{}' not found, defaulting to "
                  "generic 'pycoda.data'.".format(buf.attrs['__pyclass__']))
            out = bm.data()
    else:
        out = dat_class()
    if hasattr(buf, 'iteritems'):
        for nm, dat in buf.iteritems():
            type_str = dat.attrs.get('_type', None)
            if dat.__class__ is h5py.Group:
                out[nm] = load_hdf5(dat)
            else:
                cls = dat.attrs.get('__pyclass__', np.ndarray)
                if cls is not np.ndarray:
                    cls = pkl.loads(cls)
                if type_str == 'pickled object':
                    out[nm] = pkl.loads(dat[()])
                elif type_str == 'non-array scalar':
                    out[nm] = dat[()]
                elif (dat.dtype == 'O' and type_str == 'NumPy Object Array'):
                    shp = dat.shape
                    out[nm] = np.empty(shp, dtype='O')
                    for idf in xrange(dat.size):
                        ida = np.unravel_index(idf, shp)
                        if dat[ida] == '':
                            out[nm][ida] = None
                        else:
                            out[nm][ida] = pkl.loads(dat[ida])
                    if cls is not np.ndarray:
                        out[nm] = out[nm].view(cls)
                else:
                    out[nm] = np.array(dat)
                    if cls is not np.ndarray:
                        out[nm] = out[nm].view(cls)
    else:
        out = np.array(buf)
        cls = buf.attrs.get('__pyclass__', np.ndarray)
        if cls is not np.ndarray:
            out = out.view(pkl.loads(cls))
    return out
