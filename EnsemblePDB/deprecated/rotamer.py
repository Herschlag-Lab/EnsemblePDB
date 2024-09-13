'''EnsemblePDB.analyze.rotamer

Calculates all psi, phi and chi angles of PDB files.

Authors: 
    Siyuan Du (dusiyuan@stanford.edu)

'''
import Bio.PDB
import numpy as np
import pandas as pd
import math
from pathlib import Path
from tqdm import tqdm
from glob import glob

from EnsemblePDB.utils.file_management import get_nonexistant_file

# definition of chi angles
# definition of chi angles
CHI1 = {
    'ARG': ['N','CA','CB', 'CG'],
    'ASN': ['N','CA','CB', 'CG'],
    'ASP': ['N','CA','CB', 'CG'],
    'CYS': ['N','CA','CB', 'SG'],
    'GLU': ['N','CA','CB', 'CG'],
    'GLN': ['N','CA','CB', 'CG'],
    'HIS': ['N','CA','CB', 'CG'],
    'ILE': ['N','CA','CB', 'CG1'],
    'LEU': ['N','CA','CB', 'CG'],
    'LYS': ['N','CA','CB', 'CG'],
    'MET': ['N','CA','CB', 'CG'],
    'PHE': ['N','CA','CB', 'CG'],
    'PRO': ['N','CA','CB', 'CG'],
    'SER': ['N','CA','CB', 'OG'],
    'THR': ['N','CA','CB', 'OG1'],
    'TRP':['N','CA','CB', 'CG'],
    'TYR': ['N','CA','CB', 'CG'],
    'VAL': ['N','CA','CB', 'CG1']
}
CHI2 = {
    'ARG': ['CA','CB','CG','CD'],
    'GLU': ['CA','CB','CG','CD'],
    'GLN': ['CA','CB','CG','CD'],
    'LYS': ['CA','CB','CG','CD'],
    'MET': ['CA','CB','CG','SD'],
    'PRO': ['CA','CB', 'CG','CD'], 
    'HIS': ['CA','CB', 'CG','ND1'],
    'LEU': ['CA','CB', 'CG', 'CD1'], #nonrotameric
    'ILE': ['CA','CB', 'CG1','CD1'], #nonrotameric
    'ASN': ['CA','CB','CG', 'OD1'], #nonrotameric
    'ASP': ['CA','CB','CG', 'OD1'], #nonrotameric
    'PHE': ['CA','CB','CG','CD1'], #nonrotameric
    'TRP': ['CA','CB','CG','CD1'], #nonrotameric
    'TYR': ['CA','CB','CG','CD1'] #nonrotameric
    }
CHI3 = {
    'ARG': ['CB','CG','CD','NE'],
    'LYS': ['CB','CG','CD','CE'],
    'MET': ['CB','CG','SD','CE'],
    'GLN': ['CB','CG','CD','OE1'],
    'GLU': ['CB','CG','CD','OE1'],#nonrotameric
}
CHI4 = {
    'ARG': ['CG','CD','NE','CZ'],
    'LYS': ['CG','CD','CE','NZ'],   
}


def get_dihedrals(pdb, all_chains=False,chain_ids=['A'],models=[0],save_to=None):
    '''
    For a given pdb file and a protein chain, calculate all possible psi,phi and chi angles.
        pdb (str): path to PDB file,
        chain_id (str): ID of the chain to calculate,
        models (list): for regular PDBs typically only one model is included and no specification is needed; for NMR or models from MD trajectories, specify the model number, otherwise only the first model is calculated {default: [0]}
    Note: for any dihedral calculation, if any of the four atom is disorderd (multiconfomer), 
            this angle will be skipped (np.nan).
    '''
    if pdb[:-3] == 'pdb':
        parser = Bio.PDB.PDBParser(QUIET=True)
    if pdb[:-3] == 'cif':
        parser = Bio.PDB.MMCIFParser(QUIET=True)
    else:
        print('File type must be .pdb or .cif')
    structure = parser.get_structure("_", pdb)
    all_residue_angles = []
    i = 0
    # for model in tqdm(models, total=len(models),desc='Models'):
    for model in models:
        if model not in [x.get_id() for x in structure.get_models()]:
            print(f'Specified model {model} not found in model ids')
            continue
        if all_chains:
            chain_ids = [x.get_id() for x in structure[model].get_chains()]
        for chain_id in chain_ids:
            if chain_id not in [x.get_id() for x in structure[model].get_chains()]:
                print(f'Specified chain {chain_id} not found in model {model} chains')
                continue
            chain = structure[model][chain_id]
            for residue in chain.get_list():
                resid = residue.get_id()
                # skip if HETATM flag is true
                if resid[0] != " ":
                    continue
                dihedral_angles = {}
                dihedral_angles['model'] = model
                dihedral_angles['chain'] = chain_id
                dihedral_angles['residue_number'] = resid[1]
                dihedral_angles['residue_name'] = residue.resname
                dihedral_angles['insertion'] = resid[2]
                dihedral_angles['phi'] = calcphi(chain, resid)
                dihedral_angles['psi'] = calcpsi(chain, resid)
                # calculate chi angles
                for chi in ['chi1', 'chi2', 'chi3', 'chi4']:
                    dihedral_angles[chi] = np.nan
                for chi_atoms, chi_name in zip([CHI1, CHI2, CHI3, CHI4],
                                            ['chi1', 'chi2', 'chi3', 'chi4']):
                    if residue.resname not in chi_atoms.keys():
                        continue
                    atoms_to_find = chi_atoms[residue.resname]
                    atoms_in_pdb = [a.name for a in residue.get_list()]
                    # check if atom is in PDB & if they are disordered
                    if any(list(map(lambda a: a not in atoms_in_pdb, atoms_to_find))):
                        continue
                    if any(list(map(lambda a: residue[a].is_disordered == 0,
                                    atoms_to_find))):
                        continue
                    atomobj = [residue[a] for a in atoms_to_find]
                    dihedral_angles[chi_name] = calcdihedral(*atomobj)
                all_residue_angles.append(dihedral_angles)
        if i != 0 and i % 50 == 0 and save_to:
            df = pd.DataFrame.from_records(all_residue_angles)
            df.to_csv(save_to+f'rotamers.csv')
            # all_residue_angles = []
        i = i + 1
    df = pd.DataFrame.from_records(all_residue_angles)
    if save_to:
        df.to_csv(save_to+f'rotamers.csv')
    return df

def get_all_rotamers(directory, all_chains=False, chains=['A'], models= [0],output_directory=None,batch=50, restart=0):
    if output_directory is None:
        output_directory = str(Path(directory).parents[0])
    # all_rotamers = {}
    # for chain in chains:
    rotamer = []
    all_pdbs = glob(directory+'/*.pdb')
    i = 0
    for pdb in tqdm(all_pdbs, total=len(all_pdbs),
                    desc=f'Calculating rotamer angles'):
        if i < restart:
            i += 1
            continue
        if i != 0 and i!= restart and i % batch == 0:
            all_df = pd.concat(rotamer)
            all_df.to_csv(f'{output_directory}/rotamers_{i}.csv')
            rotamer = []
        # print(pdb)
        df = get_dihedrals(pdb=pdb, all_chains=all_chains,chain_ids=chains,models=models)
        df['Entry ID'] = pdb.split('/')[-1][:-4]
        rotamer.append(df)
        i += 1
    all_df = pd.concat(rotamer)
        # all_rotamers[chain] = all_df
    # fname = get_nonexistant_file(f'{output_directory}/all_residue_rotamers.csv')
    # all_df.to_csv(fname)
    all_df.to_csv(f'{output_directory}/rotamers_{i}.csv')
    print(f'\nRotamer angle distributions saved to {output_directory}')
    return all_df


def get_residue_by_seqproximity(chain, reference_resid, position):
    '''
    position: +1 or -1 : next residue or previous residue
    '''
    atoms = Bio.PDB.Selection.unfold_entities(chain, 'A')
    ns = Bio.PDB.NeighborSearch(atoms)
    curr_res = chain[reference_resid]
    if position == 1:
        if 'C' in [x.name for x in curr_res]:
            target_atom = curr_res['C']
        else:
            return
    if position == -1:
        if 'N' in [x.name for x in curr_res]:
            target_atom = curr_res['N']
        else:
            return
    cov_atoms = ns.search(target_atom.coord, 2)
    candidates = []
    for a in cov_atoms:
        if a.get_parent().id != reference_resid:
            candidates.append(a)
    neighbor_atom = None
    if len(candidates) == 0:
        return
    elif len(candidates) == 1:
        neighbor_atom = candidates[0]
    else:
        distances = [np.linalg.norm(x.coord-target_atom.coord)
                     for x in candidates]
        neighbor_atom = candidates[np.array(distances).argmin()]
    # check atom identity
    if position == 1:
        if neighbor_atom.name != 'N':
            return
    if position == -1:
        if neighbor_atom.name != 'C':
            return
    return neighbor_atom.get_parent().id


def calcphi(chain, res_id):
    res_id__1 = get_residue_by_seqproximity(
        chain=chain, reference_resid=res_id, position=-1)
    if not res_id__1:
        return np.nan
    try:
        CO_atom = chain[res_id__1]['C']
        N_atom = chain[res_id]['N']
        CA_atom = chain[res_id]['CA']
        C_atom = chain[res_id]['C']
    except KeyError:
        return np.nan
    # check if disordered
    if (CO_atom.is_disordered() != 0 or N_atom.is_disordered() != 0 or
            CA_atom.is_disordered() != 0 or C_atom.is_disordered() != 0):
        # print('DEBUG: disordered atom')
        return np.nan
    C0 = Bio.PDB.vectors.Vector(CO_atom.coord)
    N = Bio.PDB.vectors.Vector(N_atom.coord)
    CA = Bio.PDB.vectors.Vector(CA_atom.coord)
    C = Bio.PDB.vectors.Vector(C_atom.coord)
    phi = Bio.PDB.vectors.calc_dihedral(C0, N, CA, C)
    return phi * 180/math.pi


def calcpsi(chain, res_id):
    res_id_1 = get_residue_by_seqproximity(
        chain=chain, reference_resid=res_id, position=1)
    if not res_id_1:
        return np.nan
    try:
        N_atom = chain[res_id]['N']
        CA_atom = chain[res_id]['CA']
        C_atom = chain[res_id]['C']
        N1_atom = chain[res_id_1]['N']
    except KeyError:
        return np.nan
    if (N_atom.is_disordered() != 0 or CA_atom.is_disordered() != 0 or
            C_atom.is_disordered() != 0 or N1_atom.is_disordered() != 0):
        # print('DEBUG: disordered atom')
        return np.nan
    N = Bio.PDB.vectors.Vector(chain[res_id]['N'].coord)
    CA = Bio.PDB.vectors.Vector(chain[res_id]['CA'].coord)
    C = Bio.PDB.vectors.Vector(chain[res_id]['C'].coord)
    N1 = Bio.PDB.vectors.Vector(chain[res_id_1]['N'].coord)
    psi = Bio.PDB.vectors.calc_dihedral(N, CA, C, N1)
    return psi * 180/math.pi


def calcchi1(chain, res_id, chi1s=CHI1):
    atoms = CHI1[chain[res_id].resname]
    atom_coords = []
    for a in atoms:
        try:
            a_coord = Bio.PDB.vectors.Vector(chain[res_id][a].coord)
        except KeyError:
            return np.nan
        atom_coords.append(a_coord)
    chi1 = Bio.PDB.vectors.calc_dihedral(*atom_coords)
    return chi1 * 180/math.pi


def calcdihedral(a1, a2, a3, a4):
    v1 = Bio.PDB.vectors.Vector(a1.coord)
    v2 = Bio.PDB.vectors.Vector(a2.coord)
    v3 = Bio.PDB.vectors.Vector(a3.coord)
    v4 = Bio.PDB.vectors.Vector(a4.coord)
    dihedral = Bio.PDB.vectors.calc_dihedral(v1, v2, v3, v4)
    return dihedral * 180/math.pi
