""" EnsemblePDB.utils.biopandas_utils

Basic functions in manipulating pdb in biopandas format.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

from biopandas.pdb import PandasPdb
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_pdb_struct(reference_pdb):
    '''
    Given a 4-character pdb id fetches the Biopandas.pdb
    Arguments:
        reference_pdb (str): pdb id "XXXX"
    Returns:
        Biopandas.pdb object
    Warnings:
        Returns none if pdb id does not exist, pdb id exists but strcuture has
        not been published, or if connection is poor and it times out. If you
        know structure exists try again or at worst download pdb yourself!
    '''
    try:
        struct = PandasPdb()
        struct.fetch_pdb(reference_pdb)
    except:
        print("Could not retrieve PDB of ", reference_pdb,
              ": this may be an entry that has not published structures yet.")
        return None
    return struct


def get_PandasPDB(file):
    struct = PandasPdb()
    struct.read_pdb(str(file))
    return struct


def get_PandasPDBs(path):
    '''
    Get Pandas PDB object for a set of PDB structures (with aligned coordinates).

    Arguments: 
        path, a directory with aligned PDB files (output from PyMol)

    Returns: 
        a pandas dataframe with 'Entry ID' and 'PandasPDB'
    '''
    filenames = list(Path(path).glob('*.pdb'))
    structures = []
    IDs = []
    for file in tqdm(filenames, total=len(filenames),
                     desc='Reading PDB files'):
        structures.append(get_PandasPDB(file))
        IDs.append(file.stem)
    data = {'Entry ID': IDs, 'PandasPDB': structures}
    PDB_df = pd.DataFrame(data=data)
    return PDB_df


def get_chain_seq(struct, chain, remove_his_tag=True):
    '''
    Gets the sequence of a chain in a structure
    Arguments: 
        struct (Biopandas.pdb)
        chain (str): identity of the chain of interest
        remove_his_tag (bool): remove the 6x his tage if present {default: True}
    Returns: 
        str: sequence of the chain with 1-letter AA code from struct
    '''
    seq = struct.amino3to1()
    sequence = ''.join(seq.loc[seq['chain_id'] == chain, 'residue_name'])

    # if requested remove hisx6 tag
    if remove_his_tag:
        his_tag = "H"*6
        if sequence[:6] == his_tag:
            sequence = sequence[6:]
        if sequence[-6:] == his_tag:
            sequence = sequence[:-6]

    return sequence


def fetch_and_save_pdb(pdbid, directory):
    '''
    Download the given pdb into directory, 
    if already there do not over-write.
    Arguments:
        pdbid (str): 4 character pdbid "XXXX"
        directory (str): directory to save pdbid to
    Output:
        Saves pdb of the pdbid to the directory
    '''
    pdb_file = Path(directory, f"{pdbid.lower()}.pdb")
    if not Path(pdb_file).is_file():
        struct = get_pdb_struct(pdbid)
        if struct is not None:
            struct.to_pdb(path=pdb_file, gz=False)


def retrieve_pdbs(PDB_list, directory):
    '''
    Given a table with pdb ids in 'Entry ID', download
    all the pdb to the directory
    '''
    tqdm.pandas(desc='Retrieve PDBs')
    PDB_list['Entry ID'].progress_apply(
        lambda pdb: fetch_and_save_pdb(pdb, directory))


def get_xyz(PDB_df):
    '''
    Gets xyz coordinates for each atom in a set of PDB structures
    Assumes numberings are consistent.
    '''

    coords_to_concat = []

    for index, row in tqdm(PDB_df.iterrows(), total=len(PDB_df),
                           desc='Getting atom coordinates'):
        Entry_ID = row['Entry ID']
        aa_atom_coord = row['PandasPDB'].df['ATOM']
        hetatm = row['PandasPDB'].df['HETATM']
        nonsolvent_hetatm = hetatm.loc[hetatm['residue_name'] != 'HOH']
        atom_coord = pd.concat([aa_atom_coord, nonsolvent_hetatm])
        atom_coord = atom_coord[['atom_number', 'atom_name', 'residue_name',
                                 "chain_id", 'residue_number', 'insertion',
                                 'x_coord', 'y_coord', 'z_coord', 'occupancy',
                                 'b_factor', 'alt_loc']]
        atom_coord.insert(loc=0, column='Entry ID', value=Entry_ID)
        coords_to_concat.append(atom_coord)

    all_coords = pd.concat(coords_to_concat)
    return all_coords


def clean_multiconformers(atoms):
    '''
    Deletes entries  with multi-conformations.
    '''
    atoms = atoms.reset_index()

    single_atoms = atoms.loc[atoms['occupancy'] == 1.0]

    return single_atoms
