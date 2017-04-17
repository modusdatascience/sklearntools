from distutils.core import setup
import versioneer

setup(name='sklearntools',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Tools for sklearn models',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      url='https://gitlab.com/jcrudy/sklearntools',
      packages=['sklearntools'],
      package_data = {'': ['resources/*']},
      install_requires = ['scikit-learn>=0.17.0', 'pandas', 'numpy']
     )