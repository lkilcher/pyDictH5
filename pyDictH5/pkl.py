import sys
if sys.version_info >= (3, 0):
    import pickle as pkl
    py_ver = 3
elif sys.version_info >= (2, 0):
    import cPickle as pkl
    py_ver = 2
else:
    raise Exception("pyDictH5 only supports python 2 or 3.")


def decode(val):
    if isinstance(val, list):
        for idx in range(len(val)):
            val[idx] = decode(val[idx])
        return val
    try:
        return val.decode('utf-8')
    except (AttributeError, UnicodeDecodeError):
        pass
    return val


def dumps(data):
    return pkl.dumps(data, protocol=0)


def loads(data):
    if py_ver == 2:
        return pkl.loads(data)
    else:
        try:
            out = pkl.loads(data)
        except TypeError:
            out = pkl.loads(bytes(data, encoding='ASCII'))
        except UnicodeDecodeError:
            out = pkl.loads(data, encoding='bytes')
            out = decode(out)
        return out
