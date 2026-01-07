# EnsemblePDB
A package to create, visualize, and analyze  PDB-derived pseudo-ensembles.

## Install 
```
conda create --name ensemble python=3.10
conda activate ensemble
python setup.py install
conda install pandas=1.4.4
conda install schrodinger::pymol=3.1
pip install rcsbsearchapi==1.6.0
pip install biopandas==0.4.1
pip install tqdm==4.66.1
pip install biopython==1.82
pip install scikit-learn==1.3.2
pip install chardet==5.2.0
```

## Tutorial
To test whether the package has been successfully installed, go to the tutorial folder and run:
```
python3 simple_ensemble.py
```
This will generate a folder with chymotrypsin pseudo-ensemble including PDB metadata and aligned structures.

## Documentation
Documentation can be found here:
[https://htmlpreview.github.io/?https://github.com/Herschlag-Lab/EnsemblePDB/blob/initial_code/doc/build/html/modules.html](https://htmlpreview.github.io/?https://github.com/Herschlag-Lab/EnsemblePDB/blob/initial_code/doc/build/html/modules.html)
