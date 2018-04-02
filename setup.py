from setuptools import setup, find_packages

setup(name='laia', version='0.1', packages=find_packages(),
      scripts=[],
      install_requires=['editdistance',
                        'future;python_version<"3.0"',
                        'mock;python_version<"3.0"',
                        'numpy',
                        'scipy',
                        'torch',
                        'Pillow'],
      extras_require={
          'progress_bar': ['tqdm'],
      },
      author='Joan Puigcerver',
      author_email='joapuipe@gmail.com',
      license='MIT',
      url='https://github.com/jpuigcerver/PyLaia')
