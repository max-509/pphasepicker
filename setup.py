from distutils.core import setup, Extension, Command
import numpy as np
import os
import platform
import pathlib
import pybind11

local_sources_path = './src/'
srcs = ['module.cpp', 'butterworth_bandpass.cpp']
abs_sources_path = os.path.abspath(local_sources_path)

if platform.system() == 'Linux':
    compile_args = ['-O3', '-lm', '-march=native', '-fno-math-errno', '-fopenmp', '-Wall' '-g']
elif platform.system() == 'Windows':
	compile_args = ['/MD', '/Ox', '/Ob2', '/arch:AVX2', '/openmp']
else:
    compile_args = []

pphase_picker_module = Extension('P_S_PhasePicker',
                           include_dirs=[str(np.get_include()), str(pybind11.get_include()), abs_sources_path, os.environ['EIGEN3_INCLUDE_DIR']],
                           language='c++',
                           define_macros=['NDEBUG'],
                           extra_compile_args=compile_args,
                           sources=[local_sources_path + f for f in srcs]
                           )


setup(name='P-S-Phase Picker',
      version='1.0',
      description='S- and P-phase picker package for detection begin of event',
      author='Vershinin Maxim',
      author_email='m.vershinin@g.nsu.ru',
      ext_modules=[pphase_picker_module]
)