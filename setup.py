from distutils.core import setup
import os

# Get the version info We do this to avoid importing __init__, which
# depends on other packages that may not yet be installed.
base_dir = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(base_dir + "/pyDictH5/_version.py") as fp:
    exec(fp.read(), version)

setup(
    name=version['__package__'],
    version=version['__version__'],
    author='Levi Kilcher',
    author_email='levi.kilcher@nrel.gov',
    packages=['pyDictH5'],
    install_requires=['h5py', 'numpy'],
)
