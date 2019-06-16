from distutils.core import setup
from skeletor import __version__


setup(name='skeletor',
      version=__version__,
      description='Tools for cremi challenge and neuron segmentation',
      author='Constantin Pape',
      packages=['skeletor',
                'skeletor/io',
                'skeletor/methods'])
