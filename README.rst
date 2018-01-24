Introduction
============

.. _NumPy: http://www.numpy.org/
.. _Pandas: http://pandas.pydata.org/
.. _h5py: http://www.h5py.org/
.. _pyDictH5: http://github.com/lkilcher/pyDictH5/
.. _pickle: http://docs.python.org/library/pickle.html

pyDictH5_ is a lightweight framework
and syntax for working with compound data composed primarily of NumPy_
arrays. pyDictH5 utilizes h5py_ to provide efficient file I/O in a
transparent and standardized format.
pyDictH5 uses a standardized syntax for working with arrays of data that
are related in simple to complex ways. 

The primary pyDictH5 data structure, ``pycoda.data``, has the following
key benefits:

#) Under the hood pycoda.data objects are essentially Python dict's,
   with most of the functionality preserved.

#) The keys in pycoda.data objects can be accessed as attributes. This
   makes the pyDictH5 source lightweight and powerful. The lightweight
   source of pycoda.data objects makes sub-classing them simple, so
   that you can implement your own methods for the needs of your data.

#) A standardized syntax and file format that is similar to Matlab's
   'struct'.

pyDictH5 is meant for NumPy users who:

A) Want standardized data (object) file I/O.

B) Frequently utilize N-dimensional data (i.e. find Pandas_ DataFrames
   inadequate),

C) Want a set of unique, simple NumPy arrays for working with data
   (rather than NumPy recarrays, which are compound data types of
   their own),


Data Model
==========

pyDictH5 is designed for use by amateur to advanced developers and data
analysts who want a simple interface for handling N-dimensional data
that reduces the burden of reading/writing data from/to hdf5 files. It
is designed to be lightweight and extensible, as opposed to
multi-purpose and high-level.

pyDictH5 provides a set of simple *base classes* that each provide
unique functionality that is believed to be useful by many
users. These base classes are meant to be used as parent classes for
(inherited from) -- or used to compose -- higher-level data classes.

How is pyDictH5 different from Pandas?
------------------------------------

Pandas_ is great! It provides a suite of flexible and powerful tools
for handling and analyzing 2-dimensional data (i.e. tabular,
spreadsheet-like data). However, if your data is N-dimensional and/or
contains data of mixed size and dimensions, pyDictH5 may be better
suited for your needs. In contrast to Pandas_, pyDictH5 does not provide
high-level interfaces to plotting or other statistical analysis
tools. Instead, it provides a consistent data and I/O structure, and
assumes users will build the higher functionality where it is needed.
A primary advantage of this, compared to Pandas, is that data is not
coerced toward 2 dimensiones (DataFrames).

Why use pyDictH5 vs. Pickle?
--------------------------

ADD TEXT: pyDictH5 (hdf5 I/O via h5py) is faster for large arrays...


Usage
=====

pyDictH5 proposes that constructing data and performing I/O should be
done behind the scenes, so that users can *focus on their data*,
rather than spending time implementing I/O::

  >>> import pycoda as pcd
  >>> import numpy as np
  
Initialize a data object ``my_dat``::

  >>>  my_dat = pcd.data()

Set ``my_dat``'s data::

  >>> my_dat['x'] = np.linspace(0, 1, 3)
  >>> my_dat['y'] = np.linspace(0, 2, 5)

  # Write the data to disk
  >>> my_dat.to_hdf5('my_data.h5')

  # Reload the data
  >>> my_dat_copy = pcd.load('my_data.h5')

  # The data attributes can be accessed using 'attribute references'
  >>> my_dat_copy.x == my_dat.x
  True

Data items can also be added and retrieved by attribute assignment::

  >>> my_dat.time = np.arange(0, 1000, 0.1)
  >>> t0 = my_dat.time[0]

Note, however, that these attributes are actually references to the
dictionary. This means that they will not be an *attribute* of the
data object (FIX THIS?), but are a *member* of it::
  
  >>> hasattr(my_dat, 'time')
  False
  >>> 'time' in my_dat
  True

Temporary Attributes
--------------------

Sometimes it is useful to have temporary attributes associated with a
data object, that are not stored in the output file. To provide this
functionality, attributes with a leading ``'_'`` in the name are not
added to the dictionary, and are attributes of the data object::

  >>> my_dat._tmp = np.arange(5)
  >>> hasattr(my_dat, '_tmp')
  True
  >>> '_tmp' in my_dat
  False

If you want to store data items (non-temporary) with leading ``'_'``,
you must assign them as keys::

  >>> my_dat['_not_tmp'] = np.arange(5)
  >>> my_dat.to_hdf5('my_data.h5')
  >>> my_dat_copy = pcd.load('my_data.h5')
  >>> '_not_tmp' in my_dat_copy
  True
  >>> '_tmp' in my_dat_copy
  False

Note that the ``'_tmp'`` attribute is lost when the data is reloaded::

  >>> hasattr(my_dat_copy, '_tmp')
  False

Sub-data objects
----------------

It is also often useful to be able to store data objects as
sub-objects of other data objects. pyDictH5 can do this too::

  >>> vel_dat['subobj'] = pcd.data()
  >>> vel_dat['subobj']['velocity'] = np.arange(10)
  >>> vel_dat['subobj']['velocity'][3:6] = 4

I/O of these 'compound' data objects are read and written to hdf5
files transparently (as hdf5 *groups*)::

  >>> vel_dat.to_hdf5('vel_data.h5')
  >>> vel_dat_copy = pcd.load('vel_data.h5')
  >>> 'subobj' in vel_dat_copy
  True
  >>> 'velocity' in vel_dat_copy['subobj']
  True

Sub-data objects can also be accessed and created using attribute
reference::

  >>> vel_dat.subobj2 = pcd.data()
  >>> vel_dat.subobj2.velocity2 = vel_dat.subobj.velocity ** 2

It is also possible to access sub-groups and items using dot-group
key-references like this::

  >>> print vel_dat['subobj2.velocity2']
  [ 0  1  4 16 16 16 36 49 64 81]

This is useful for iterating through the group using the ``walk``
function::

  >>> for key in vel_dat.walk():
  ...    print key, vel_dat[key][2]
  subobj.velocity 2
  subobj2.velocity2 4

You can also test whether an item in a sub-group exists using
dot.group key-references::

  >>> 'subobj.velocity' in vel_dat
  True
  >>> 'subobj2.velocity2' in vel_dat
  True
  >>> 'subobj2.junk' in vel_dat
  False

NumPy object arrays
-------------------

pyDictH5 supports NumPy object-array writing (currently this is not
natively supported by h5py_\ ). This is implemented by pickle_\ ing
each object of the array, then writing the pickle-strings into hdf5
*varlen* arrays::
  
  >>> my_dat['obj_arr'] = np.zeros(5, dtype='O')
  >>> my_dat['obj_arr'][1] = np.arange(3)
  >>> my_dat['obj_arr'][3] = {'dog': 'spot', 'cat': 'ruffus', 'one': 1}

Note that this means that you may not want to store large NumPy arrays
*inside* of NumPy object arrays because many of hdf5 performance
advantages (compared to pickle_) will be lost.

Indexing and Appending Data
---------------------------

The ``pcd.flat`` class provides simple functionality for accessing
data, and combining data sets.  For example, assume we define::

  >>> timedat = pcd.flat()
  >>> timedat['time'] = np.arange(10)
  >>> timedat['velocity'] = np.arange(40, 50)
  >>> timedat['accel'] = np.ones(10)

Then we can sub-index the entire data-object by simply doing::
  
  >>> sub_timedat = timedat[1:6]
  >>> print(sub_timedat.time, sub_timedat.velocity)
  (array([1, 2, 3, 4, 5]), array([41, 42, 43, 44, 45]))

You can also combine datasets using ``pcd.flat.append``::

  >>> timedat1 = pcd.flat()
  >>> timedat1['time'] = np.arange(10, 30)
  >>> timedat1['velocity'] = np.arange(40, 80, 2)
  >>> timedat1['accel'] = 2 * np.ones(20)

These two data object can be concatenated by simply doing::

  >>> timedat.append(timedat1)
  >>> print(timedat.time)
  [0, 1, 2, ... 28, 29]

The ``pycoda.data`` object does simple concatenating along the first
(``0``) axis of all arrays. It does no checking to make sure the data
is the same size in this dimension, so if you have data of different
lengths in a single data object, you may get unexpected results.

Sub-classing
------------

A key feature of pyDictH5 is the ability to subclass the ``pycoda.data``
class. For example, if we create a module ``my_data_module.py`` that
contains::

  import pycoda as pcd
  import numpy as np

  class my_data(pcd.data):
      
      def xymesh(self, ):
          return np.meshgrid(self['x'], self['y'])

We can initialize and populate this data type, and utilize the
``xymesh`` method::

  >>> import my_data_module as mdm
  >>> my_dat2 = mdm.my_data()
      
  >>> my_dat2['x'] = np.linspace(0, 1, 3)
  >>> my_dat2['y'] = np.linspace(1, 2, 5)
  >>> xgrid, ygrid = my_dat2.xymesh()
  >>> print(xgrid)
  [[ 0.   0.5  1. ]
   [ 0.   0.5  1. ]
   [ 0.   0.5  1. ]
   [ 0.   0.5  1. ]
   [ 0.   0.5  1. ]]

A major advantage of sub-classing ``pycoda.data`` is that, so long
as the subclass is available consistently between write and read, the
dtype is preserved. This is why it is useful to define sub-classes in
modules (or packages) of their own. Then, so long as those modules or
packages are on the Python path, pyDictH5 will import and utilize those
classes transparently.  For example, if the ``my_data`` class is
defined in a ``my_data_module.py``, the class will be preserved::

  >>> my_dat2.to_hdf5('my_data2.h5')
  >>> my_dat2_copy = pcd.load('my_data2.h5')
  >>> my_dat2_copy.__class__
  my_data_module.my_data

So that we can still do::

  >>> xgrid, ygrid = my_dat2_copy.xymesh()

Furthermore, if we add or modify our sub-classes these changes will be
available when we load the data.  For example, assume we change our
``my_data`` class to be::
  
    class my_data(pcd.data):
    
        # Here we redefine xymesh to be a property and use __xymesh to cache it.
        @property
        def xymesh(self, ):
            if not hasattr(self, '__xymesh'):
                self.__xymesh = np.meshgrid(self['x'], self['y'])
            return self.__xymesh
    
        def distance(self, x, y):
            """
            Calculate the distance between the point `x`,`y`, and all of
            the points in the grid.
            """
            xg, yg = self.xymesh  # xymesh is now a property
            return np.sqrt((xg - x) ** 2 + (yg - y) ** 2)

Now, in a new Python interpreter - so that our module reloads - we can do::

  >>> mydat2 = pcd.load('my_data2.h5')
  >>> dist = mydat2.distance(0, 0.5)
  >>> print(dist)
  [[ 0.5         0.70710678  1.11803399]
   [ 0.          0.5         1.        ]
   [ 0.5         0.70710678  1.11803399]
   [ 1.          1.11803399  1.41421356]
   [ 1.5         1.58113883  1.80277564]]

Is that cool, or what?!

Caveats (gotchas)
-----------------

String keys only
................

In standard Python dictionaries, dictionary keys can be any immutable
object. pyDictH5 -- in order to allow for attribute reference, and
transparent I/O to hdf5, restricts the dictionary keys to be strings::

  >>> my_dat[0] = np.arange(10)
  IndexError: <class 'pyDictH5.base.data'> objects only support string indexes.
  >>> my_dat['0'] = np.arange(10)
  >>> '0' in my_dat
  True
