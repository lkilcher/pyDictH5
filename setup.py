from distutils.core import setup
import pyDictH5._version as ver

setup(
    name=ver.__package__,
    version=ver.__version__,
    author='Levi Kilcher',
    author_email='levi.kilcher@nrel.gov',
    packages=['pyDictH5'],
)
