from distutils.core import setup, Extension
import numpy as np
import os
import platform
import pathlib

local_sources_path = './src/'
srcs = ['module.cpp', 'butterworth_bandpass.cpp']
abs_sources_path = os.path.abspath(local_sources_path)

if platform.system() == 'Linux':
    compile_args = ['-O3', '-lm', '-march=native', '-DNDEBUG', '-fno-math-errno', '-fopenmp', '-Wall']
elif platform.system() == 'Windows':
	compile_args = ['/MD', '/O2', '/Ob2', '/DNDEBUG', '/arch:AVX2', '/openmp']
else:
    compile_args = []

pphase_picker_module = Extension('PphasePicker',
                           include_dirs=[str(np.get_include()), abs_sources_path, os.environ['EIGEN3_INCLUDE_DIR']],
                           language='c++',
                           extra_compile_args=compile_args,
                           sources=[local_sources_path + f for f in srcs]
                           )

setup(name='Pphase Picker',
      version='1.0',
      description='Pphase picker package for detection begin of event',
      author='Vershinin Maxim',
      author_email='m.vershinin@g.nsu.ru',
      ext_modules=[pphase_picker_module]
)