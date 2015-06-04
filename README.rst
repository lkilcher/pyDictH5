Introduction
============

.. _NumPy: http://www.numpy.org/
.. _Pandas: http://pandas.pydata.org/
.. _h5py: http://www.h5py.org/
.. _PyCoDa: http://githumb.com/lkilcher/pyCoDa/

The Python Compound Data (PyCoDa_) library is a lightweight framework
and syntax for working with compound data composed primarily of NumPy_
arrays. PyCoDa utilizes h5py_ to provide efficient file I/O in a
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

B) Frequently utilize N-dimensional data (i.e. find Pandas_ DataFrames
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
  >>> import numpy as np
  
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

Sub-classing
============

A key feature of PyCoDa is the ability to subclass the ``pycoda.data``
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
      
  >>> my_dat2['x'] = np.linspace(50, 100, 101)
  >>> my_dat2['y'] = np.linspace(100, 200, 201)
  >>> xgrid, ygrid = my_dat2.xymesh()

A major advantage of sub-classing ``pycoda.data`` is that, so long
as the subclass is available consistently between write and read, the
dtype is preserved. This is why it is useful to define sub-classes in
modules (or packages) of their own. Then, so long as those modules or
packages are on the Python path, PyCoDa will import and utilize those
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
            xg, yg = self.xymesh
            return np.sqrt((xg - x) ** 2 + (yg - y) ** 2)

Now, in a new Python interpreter - so that our module reloads - we can do::

  >>> mydat2 = pcd.load('my_data2.h5')
  >>> dist = mydat2.distance(50, 150)
  >>> print(dist)
  [[ 50.          50.00249994  50.009999   ...,  70.00714249  70.35801305
     70.71067812]
   [ 49.5         49.50252519  49.51009998 ...,  69.65091528  70.00357134
     70.35801305]
   [ 49.          49.00255095  49.01020302 ...,  69.29646456  69.65091528
     70.00714249]
   ..., 
   [ 49.          49.00255095  49.01020302 ...,  69.29646456  69.65091528
     70.00714249]
   [ 49.5         49.50252519  49.51009998 ...,  69.65091528  70.00357134
     70.35801305]
   [ 50.          50.00249994  50.009999   ...,  70.00714249  70.35801305
     70.71067812]]

Is that cool, or what?!
