from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "laplace_solver_cython",
    sources=["laplace_solver_cython.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize(ext, compiler_directives={'language_level': "3"})
)