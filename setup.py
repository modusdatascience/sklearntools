from setuptools import setup, find_packages, Extension
import versioneer
import sys
import os


# Determine whether to use Cython
if '--cythonize' in sys.argv:
    from Cython.Build.Dependencies import cythonize
    from Cython.Distutils import build_ext
    cythonize_switch = True
    del sys.argv[sys.argv.index('--cythonize')]
    ext = 'pyx'
    directives = {}
    directives['linetrace'] = False
    directives['binding'] = False
    directives['profile'] = False
    cmdclass = {'build_ext': build_ext}
else:
    from setuptools.command.build_ext import build_ext  # @NoMove @Reimport
    cythonize_switch = False
    ext = 'c'
    cmdclass = {'build_ext': build_ext}

ext_modules = [Extension('sklearntools.interpolation', 
                         [os.path.join('sklearntools', 
                                       'interpolation.%s' % ext)])]

setup(name='sklearntools',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Tools for sklearn models',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      url='https://gitlab.com/jcrudy/sklearntools',
      packages=find_packages(),
      package_data = {'': ['sym/resources/*']},
      ext_modules = cythonize(ext_modules, compiler_directives=directives) if cythonize_switch else ext_modules,
      install_requires = ['scikit-learn>=0.17.0', 'pandas', 'numpy']
     )