from distutils.core import setup
from Cython.Build import cythonize

setup(name='cutils', ext_modules=cythonize("cutils.pyx"))