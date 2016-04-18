from distutils.core import setup

setup(name='sklearntools',
      version='0.1',
      description='Tools for sklearn models',
      author='Jason Rudy',
      author_email='jcrudy@gmail.com',
      url='https://gitlab.com/jcrudy/sklearntools',
      packages=['sklearntools', 'sklearntools.calibration', 'sklearntools.feature_selection', 
                'sklearntools.glm', 'sklearntools.model_selection', 'sklearntools.pandables', 
                'sklearntools.quantile', 'sklearntools.scoring', 'sklearntools.sklearntools', 
                'sklearntools.validation'],
     )