'''EnsemblePDB.refine.rename

Function to rename ambiguous atoms by reference structure.
To ensure correct MDev calculation and visualization, we 
need to rename atoms that are chemically idential and arbitrarily 
labeled in the PDB files (e.g. OD1 and OD2 of ASP atoms.)

Authors:
    Rachael Kretsch (rkretsch@stanford.edu)
    Siyuan Du (dusiyuan@stanford.edu)
    Jacob Parres-Gold
'''
import os
from os import path
import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
from tqdm import tqdm
from shutil import copyfile
from glob import glob

from EnsemblePDB.utils import file_management, biopandas_utils


def rename_ambiguous(directory, reference_pdb=None, output_dir=None):
    '''
    Takes all pdbs in a directory and relabels ambiguous atoms consistently 
    by distance to atoms in the reference structure, saves the renamed PDBs.
    Arguments:
        directory (str): directory with PDBs to rename atoms
        reference_pdb (str): Entry ID of the reference PDB. If None, use the 
            first PDB file in the directory. {default: None}
        output_dir (str): directory to save renamed PDBs to. If None, saves 
            to the parent of directory with suffix _renamed.
    Atoms that will be relabeled:
        Arg NH1, NH2
        Asp OD1, OD2
        Glu OE1, OE2
        Leu CD1, CD2
        Phe CD1, CD2
        Phe CE1, CE2
        Tyr CD1, CD2
        Tyr CE1, CE2
        Val CG1, CG2
    '''
    renaming_guide = {'ARG': [['NH1', 'NH2']],
                      'ASP': [['OD1', 'OD2']],
                      'GLU': [['OE1', 'OE2']],
                      'LEU': [['CD1', 'CD2']],
                      'PHE': [['CD1', 'CD2'], ['CE1', 'CE2']],
                      'TYR': [['CD1', 'CD2'], ['CE1', 'CE2']],
                      'VAL': [['CG1', 'CG2']]}
    # overwrite in current directory if output not specified
    all_files = [x.split('/')[-1] for x in glob(directory+'/*.pdb')]
    if not output_dir:
        output_dir = file_management.get_dir(directory+'_renamed')
    if not reference_pdb:
        reference_pdb = all_files[0][:-4]
        ref_file = path.join(directory, all_files[0])
    else:
        reference_pdb = reference_pdb[:4].lower()+reference_pdb[4:]
        ref_file = path.join(directory, f"{reference_pdb}.pdb")
    all_files.remove(f"{reference_pdb}.pdb")
    copyfile(ref_file, path.join(output_dir, f"{reference_pdb}.pdb"))

    ref_struct = PandasPdb().read_pdb(ref_file).df['ATOM']
    # TODO: tqdm
    for fname in tqdm(all_files, total=len(all_files),
                      desc='Renaming ambiguous atoms'):
        # print(f'Relabelling {file}')
        pdb_file = path.join(directory, fname)
        ppdb = PandasPdb().read_pdb(pdb_file)
        pdb_struct = ppdb.df['ATOM']
        residues = pdb_struct.groupby(
            by=['residue_name', 'residue_number', 'insertion', 'chain_id']).groups.keys()
        for resi in [x for x in residues if x[0] in renaming_guide.keys()]:
            # get atoms rows of this residue
            pdb_rows = pdb_struct[(pdb_struct['chain_id'] == resi[3]) &
                                  (pdb_struct['residue_name'] == resi[0]) &
                                  (pdb_struct['residue_number'] == resi[1]) &
                                  (pdb_struct['insertion'] == resi[2])]
            ref_rows = ref_struct[(ref_struct['chain_id'] == resi[3]) &
                                  (ref_struct['residue_name'] == resi[0]) &
                                  (ref_struct['residue_number'] == resi[1]) &
                                  (ref_struct['insertion'] == resi[2])]
            atom_pairs = renaming_guide[resi[0]]
            for atom_pair in atom_pairs:
                # santiry check
                if not (atom_pair[0] in ref_rows['atom_name'].tolist() and atom_pair[1] in ref_rows['atom_name'].tolist()):
                    continue
                if not (atom_pair[0] in pdb_rows['atom_name'].tolist() and atom_pair[1] in pdb_rows['atom_name'].tolist()):
                    continue
                ref_atom_1 = ref_rows.loc[ref_rows['atom_name']
                                          == atom_pair[0]]
                ref_atom_2 = ref_rows.loc[ref_rows['atom_name']
                                          == atom_pair[1]]
                atom_1 = pdb_rows.loc[pdb_rows['atom_name'] == atom_pair[0]]
                atom_2 = pdb_rows.loc[pdb_rows['atom_name'] == atom_pair[1]]

                # get distances: compare distances of atoms to each ref_atom
                distances = []
                for atom in [atom_1.iloc[0], atom_2.iloc[0]]:
                    for ref in [ref_atom_1.iloc[0], ref_atom_2.iloc[0]]:
                        distance = np.linalg.norm(np.array(
                            ref[['x_coord', 'y_coord', 'z_coord']]) - np.array(atom[['x_coord', 'y_coord', 'z_coord']]))
                        distances.append(distance)
                # compare set of distances
                if distances[0]+distances[3] > distances[1] + distances[2]:
                    # switch atom name. Notice that multiconformer atoms
                    # will be labeled with the same reference atom name.
                    pdb_struct.loc[atom_1.index,
                                   'atom_name'] = ref_atom_2.iloc[0]['atom_name']
                    pdb_struct.loc[atom_2.index,
                                   'atom_name'] = ref_atom_1.iloc[0]['atom_name']
        renamed_pdb = path.join(output_dir, fname)
        ppdb.to_pdb(path=renamed_pdb, gz=False)
    print(f'\nSaved renamed PDBs to {output_dir}')
    return
