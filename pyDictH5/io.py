import h5py
from . import _version as ver
from . import pkl
try:
    # PY 3
    from . import base as bm
except ImportError:
    # PY 2
    import base as bm
import numpy as np
import six


def hdf5_write(buf, indat, chunks=True, compression='gzip'):
    if isinstance(buf, six.string_types):
        # If it is a filename open the file using `with`.
        with h5py.File(buf, 'w') as h5buf:
            h5buf.attrs['__package_name__'] = ver.__package__
            h5buf.attrs['__version__'] = ver.__version__
            hdf5_write(h5buf, indat, chunks=chunks, compression=compression)
        return
    buf.attrs['__pyclass__'] = pkl.dumps(indat.__class__)
    for nm in indat.keys():
        dat = indat[nm]
        if isinstance(dat, bm.data):
            dat.to_hdf5(buf.create_group(nm),
                        chunks=chunks, compression=compression)
        elif isinstance(dat, dict):
            tmp = bm.data(dat)
            tmp.to_hdf5(buf.create_group(nm),
                        chunks=chunks, compression=compression)
            # This type(dat) so that we can support 
            buf[nm].attrs['__pyclass__'] = pkl.dumps(type(dat))
        else:
            if isinstance(dat, np.ndarray):
                if dat.dtype == 'O':
                    shp = dat.shape
                    ds = buf.create_dataset(
                        nm, shp,
                        dtype=h5py.special_dtype(vlen=bytes))
                    ds.attrs['_type'] = 'NumPy Object Array'
                    for idf, val in enumerate(dat.flat):
                        ida = np.unravel_index(idf, shp)
                        ds[ida] = pkl.dumps(val)
                elif str(dat.dtype).startswith('datetime64'):
                    ds = buf.create_dataset(
                        name=nm, data=dat.astype('S'),
                        chunks=chunks, compression=compression)
                    ds.attrs['_type'] = str(dat.dtype)
                elif dat.dtype.kind == 'U':
                    ds = buf.create_dataset(
                        name=nm, data=dat.astype('S'),
                        chunks=chunks, compression=compression)
                else:
                    try:
                        ds = buf.create_dataset(
                            name=nm, data=dat,
                            chunks=chunks, compression=compression)
                    except TypeError:
                        ds = buf.create_dataset(
                            name=nm, data=dat)
            else:
                try:
                    ds = buf.require_dataset(nm, (), data=dat)
                except (TypeError, ValueError):
                    # Pickle the object.
                    val = pkl.dumps(dat)
                    ds = buf.require_dataset(
                        nm, (), dtype='S{}'.format(len(val))
                    )
                    ds[()] = val
                    ds.attrs['_type'] = 'pickled object'
                else:
                    ds.attrs['_type'] = 'non-array scalar'
            ds.attrs['__pyclass__'] = pkl.dumps(type(dat))


def cls_pklstr_gen(cls_pklstr):
    """A generator function for searching for a class definition
    within packages/subpackages.
    """
    # First try the original string:
    yield cls_pklstr
    ####
    # Now start parsing the string...
    # module, and the rest of the class string.
    mod, cls = cls_pklstr.split(b'\n', 1)
    # Drop the 'c'
    mod = mod[1:]
    while b'.' in mod:
        mod = mod.split(b'.', 1)[-1]
        yield b'c' + mod + b'\n' + cls


def load_hdf5(buf, group=None, dat_class=None):
    """
    Load a data object from an hdf5 file.
    """
    if isinstance(buf, six.string_types):
        with h5py.File(buf, 'r') as fl:
            return load_hdf5(fl, group=group, dat_class=dat_class)
    if isinstance(group, list):
        if '' in group:
            out = load_hdf5(buf, group='')
            group.remove('')
        else:
            out = bm.data()
        for g in group:
            out[g] = load_hdf5(buf, group=g)
        return out
    if group == '':
        pass
    elif group is not None:
        buf = buf[group]
    if dat_class is None:
        outclass = None
        # The try loop focuses on finding the class...
        c_tmp = buf.attrs['__pyclass__']
        for cls_pklstr in cls_pklstr_gen(c_tmp):
            try:
                outclass = pkl.loads(cls_pklstr)
            except ImportError:
                pass
            else:
                # No error, so stop the loop.
                break
        if outclass is None:
            print("Warning: Class '{}' not found, defaulting to "
                  "generic 'pycoda.data'.".format(buf.attrs['__pyclass__']))
            outclass = bm.data
        # Now call it
        out = outclass()
    else:
        out = dat_class()
    if hasattr(buf, 'keys'):
        for nm in buf.keys():
            dat = buf[nm]
            type_str = dat.attrs.get('_type', None)
            try:
                type_str = pkl.decode(type_str)
            except AttributeError:
                pass
            if dat.__class__ is h5py.Group:
                if group != '':
                    out[nm] = load_hdf5(dat)
            else:
                cls = dat.attrs.get('__pyclass__', np.ndarray)
                if cls is not np.ndarray:
                    cls = pkl.loads(cls)
                if type_str == 'pickled object':
                    try:
                        out[nm] = pkl.loads(dat[()])
                    except KeyError:
                        out._set(nm, pkl.loads(dat[()]))
                elif type_str == 'non-array scalar':
                    try:
                        out[nm] = pkl.decode(dat[()])
                    except KeyError:
                        out._set(nm, pkl.decode(dat[()]))
                elif (dat.dtype == 'O' and type_str == 'NumPy Object Array'):
                    shp = dat.shape
                    out[nm] = np.empty(shp, dtype='O')
                    for idf in range(dat.size):
                        ida = np.unravel_index(idf, shp)
                        if dat[ida] == '':
                            out[nm][ida] = None
                        else:
                            tmp = dat[ida]
                            try:
                                out[nm][ida] = pkl.loads(tmp)
                            except:
                                out[nm][ida] = tmp
                    if cls is not np.ndarray:
                        out[nm] = out[nm].view(cls)
                else:
                    out[nm] = np.array(dat)
                    if isinstance(type_str, six.string_types) and \
                       type_str.startswith('datetime64'):
                        out[nm] = out[nm].astype(type_str)
                    if cls is not np.ndarray:
                        out[nm] = out[nm].view(cls)
                    if out[nm].dtype.name.startswith('bytes'):
                        out[nm] = out[nm].astype('<U')
    else:
        out = np.array(buf)
        cls = buf.attrs.get('__pyclass__', np.ndarray)
        if cls is not np.ndarray:
            out = out.view(pkl.loads(cls))
    return out


# These two functions don't use a 'with' statement, so the debugger
# works more cleanly. You do have to close the file explicitly though.
def _debug_write(fname, dat):
    h5buf = h5py.File(fname, 'w')
    h5buf.attrs['__package_name__'] = ver.__package__
    h5buf.attrs['__version__'] = ver.__version__
    hdf5_write(h5buf, dat)
    return h5buf


def _debug_load(fname):
    h5buf = h5py.File(fname, 'r')
    dat = load_hdf5(h5buf)
    return dat
