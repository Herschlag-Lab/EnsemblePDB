#!/usr/bin/env python

from setuptools import setup


# This setup is suitable for "python setup.py develop".

setup(name='EnsemblePDB',
      version='0.1',
      description='Make psuedo-ensembles derived from the PDB.',
      author='Herschlag-Lab',
      author_email='rkretsch@stanford.edu',
      url='http://www.herschlaglab.sqaurespace.com',
      packages=['EnsemblePDB'],
      install_requires=['biopython','biopandas','progressbar','mdanalysis','pymol'] #rdkit
      )

#Need to add pymol required