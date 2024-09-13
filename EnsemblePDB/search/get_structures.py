''' EnsemblePDB.search.get_structure

Function to download all pdb files from the PDB database

Authors:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
'''

import pandas as pd
from EnsemblePDB.utils.file_management import get_dir
from EnsemblePDB.utils.biopandas_utils import retrieve_pdbs


def download_pdbs(ensemble_csv, output_folder=None):
    '''
    Given a csv files from a pdb search, downlaod all pdbs.

    Arguments: 
        ensemble_csv (str): csv file resulting from a pdb search should
                        contain the following columns
                                - Entry ID: pdb ids
        output_folder (str): if specified will save pdbs in that folder,
            otherwise will save in unaltered_pdbs in current folder {default: None},
            note will overwrite everything in the folder specified if it exists

    Returns:
        Path, location of the downloaded pdbs

    Output:
        Folder with downloaded, unaltered, pdbs
    '''

    # Retrieve data in the ensemble_csv
    data = pd.read_csv(ensemble_csv)

    # Download all pdbs in a directory
    if output_folder is None:
        directory = get_dir('unaltered_pdbs')
    else:
        directory = get_dir(output_folder, overwrite=True)
    retrieve_pdbs(data, directory)

    return directory
