Introduction
============

The Python Compound Data (PyCoDa) library is a lightweight framework
and syntax for working with compound data composed primarily of NumPy
arrays. PyCoDa utilizes h5py to provide efficient file I/O in a
transparent and standardized format.
PyCoDa uses a standardized syntax for working with arrays of data that
are related in simple to complex ways. 

The primary PyCoDa data structure, ``pycoda.data``, has the following
key benefits:

#) Under the hood pycoda.data objects are essentially Python dict's,
   with most of the functionality preserved.

#) The keys in pycoda.data objects can be accessed as attributes. This
   makes the PyCoDa source lightweight and powerful. The lightweight
   source of pycoda.data objects makes sub-classing them simple, so
   that you can implement your own methods for the needs of your data.

#) A standardized syntax and file format that is similar to Matlab's
   'struct'.

PyCoDa is meant for NumPy users who:

A) Want standardized data (object) file I/O.

B) Frequently utilize N-dimensional data (i.e. find Pandas DataFrames
   inadequate),

C) Want a set of unique, simple NumPy arrays for working with data
   (rather than NumPy recarrays, which are compound data types of
   their own),

Usage
=====

PyCoDa proposes that constructing data and performing I/O should be
done behind the scenes, so that users can *focus on their data*,
rather than spending time implementing I/O::

  >>> import pycoda as pcd

Initialize a data object ``my_dat``::

  >>>  my_dat = pcd.data()

Set ``my_dat``'s data::

  >>> my_dat['time'] = np.arange(10)
  >>> my_dat['x'] = np.linspace(50, 100, 101)
  >>> my_dat['y'] = np.linspace(100, 200, 201)

  # Write the data to disk
  >>> my_dat.to_hdf5('my_data.h5')

  # Reload the data
  >>> my_dat_copy = pcd.load('my_data.h5')

  # The data attributes can be accessed using 'attribute references'
  >>> my_dat_copy.x == my_dat.x
  True

A key feature of PyCoDa is the ability to subclass the pycoda.data
class, for example::

  class my_data(pycoda.data):
      
      def xymesh(self, ):
          return np.meshgrid(self['x'], self['y'])

Now initialize the new data type and populate it::

  >>> my_dat2 = my_data()
      
  >>> my_dat2['x'] = np.linspace(50, 100, 101)
  >>> my_dat2['y'] = np.linspace(100, 200, 201)
  >>> xgrid, ygrid = my_dat2.xymesh()

A major advantage of pycodata.data subclassing is that, as long as the
subclass is available consistently between write and read, the dtype
is preserved. This suggests that sub-classes should be defined in
modules (or packages) of their own. Then, so long as those modules or
packages are on the Python path, PyCoDa will import and utilize those
classes transparently.  For example, if the ``my_data`` class is
defined in a ``my_data_module.py``, the class will be preserved::


  >>> my_dat2.to_hdf5('my_data2.h5')
  >>> my_dat2.__class__
  my_data_module.my_data
