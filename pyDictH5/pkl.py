import sys
if sys.version_info >= (3, 0):
    import pickle as pkl
    py_ver = 3
elif sys.version_info >= (2, 0):
    import cPickle as pkl
    py_ver = 2
else:
    raise Exception("pyDictH5 only supports python 2 or 3.")


def dumps(data):
    return pkl.dumps(data, protocol=0)


def loads(data):
    if py_ver == 2:
        return pkl.loads(data)
    else:
        return pkl.loads(data, encoding='bytes')
