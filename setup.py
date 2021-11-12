from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
<<<<<<< HEAD
                  Extension("bispectrum",["bispectrum.pyx"], include_dirs=[numpy.get_include()],)
=======
                  Extension("bispectrum",["bispectrum.pyx"], include_dirs=[numpy.get_include()],
    )
>>>>>>> be3a9d457fcc2ce53c61fdca423959fa91cbd529
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}

setup(ext_modules = ext_modules, cmdclass = {'build_ext': build_ext},)

