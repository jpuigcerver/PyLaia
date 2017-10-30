from setuptools import setup, find_packages

setup(name='laia', version='0.1', packages=find_packages(),
      scripts=[],
      install_requires=['torch', 'numpy', 'Pillow'],
      author='Joan Puigcerver',
      author_email='joapuipe@gmail.com',
      license='MIT',
      url='https://github.com/jpuigcerver/PyLaia')
