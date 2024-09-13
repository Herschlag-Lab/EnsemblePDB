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

# definition of chi angles
# list of rotamers and their ranges follow Shapovalov & Dunbrack 2011
CHI1 = {
    'ARG': [['N','CA','CB', 'CG']],
    'ASN': [['N','CA','CB', 'CG']],
    'ASP': [['N','CA','CB', 'CG']],
    'CYS': [['N','CA','CB', 'SG']],
    'GLU': [['N','CA','CB', 'CG']],
    'GLN': [['N','CA','CB', 'CG']],
    'HIS': [['N','CA','CB', 'CG']],
    'ILE': [['N','CA','CB', 'CG1'], ['N','CA','CB', 'CG2']],
    'LEU': [['N','CA','CB', 'CG']],
    'LYS': [['N','CA','CB', 'CG']],
    'MET': [['N','CA','CB', 'CG']],
    'PHE': [['N','CA','CB', 'CG']],
    'PRO': [['N','CA','CB', 'CG']],
    'SER': [['N','CA','CB', 'OG']],
    'THR': [['N','CA','CB', 'OG1']],
    'TRP':[['N','CA','CB', 'CG']],
    'TYR': [['N','CA','CB', 'CG']],
    'VAL': [['N','CA','CB', 'CG1'], ['N','CA','CB', 'CG2']]
}

CHI2 = {
    'ARG': [['CA','CB','CG','CD']],
    'GLU': [['CA','CB','CG','CD']],
    'GLN': [['CA','CB','CG','CD']],
    'LYS': [['CA','CB','CG','CD']],
    'MET': [['CA','CB','CG','SD']],
    'PRO': [['CA','CB', 'CG','CD']], 
    'HIS': [['CA','CB', 'CG','ND1']],
    'ILE': [['CA','CB', 'CG1','CD1']],
    'LEU': [['CA','CB', 'CG', 'CD1'],['CA','CB', 'CG', 'CD2']], 
    'ASN': [['CA','CB','CG', 'OD1']], #nonrotameric
    'ASP': [['CA','CB','CG', 'OD1'],['CA','CB','CG', 'OD2']],#nonrotameric
    'PHE': [['CA','CB','CG','CD1'],['CA','CB','CG','CD2']], #nonrotameric
    'TRP': [['CA','CB','CG','CD1'],['CA','CB','CG','CD2']], #nonrotameric
    'TYR': [['CA','CB','CG','CD1'],['CA','CB','CG','CD2']],#nonrotameric
    }

CHI3 = {
    'ARG': [['CB','CG','CD','NE']],
    'LYS': [['CB','CG','CD','CE']],
    'MET': [['CB','CG','SD','CE']],
    'GLU': [['CB','CG','CD','OE1'],['CB','CG','CD','OE1']], #nonrotameric
    'GLN': [['CB','CG','CD','OE1']], #nonrotameric
}
CHI4 = {
    'ARG': [['CG','CD','NE','CZ']],
    'LYS': [['CG','CD','CE','NZ']],   
}


def get_all_rotamers(directory, all_chains=False, chains=['A'], models= [0],output_directory=None,batch=50, restart=0):
    '''
    Given a directory of pdb files, calculating all backbone and sidechain torsion angles
    '''
    if output_directory is None:
        output_directory = str(Path(directory).parents[0])
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
    all_df.to_csv(f'{output_directory}/rotamers_{i}.csv')
    print(f'\nRotamer angle distributions saved to {output_directory}')
    return all_df

def get_dihedrals(pdb, all_chains=False,chain_ids=['A'],models=[0],save_to=None):
    '''
    For a given pdb file and a protein chain, calculate all possible psi,phi and chi angles.
        pdb (str): path to PDB or cif file,
        chain_id (str): ID of the chain to calculate,
        models (list): for regular PDBs typically only one model is included and no specification is needed; for NMR or models from MD trajectories, specify the model number, otherwise only the first model is calculated {default: [0]}
    '''
    parser = None
    if pdb[-3:] == 'pdb':
        parser = Bio.PDB.PDBParser(QUIET=True)
    elif pdb[-3:] == 'cif':
        parser = Bio.PDB.MMCIFParser(QUIET=True)
    else:
        print('File type must be .pdb or .cif')
        return
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
                residue_dihedrals = get_residue_dihedrals(chain, resid)
                all_residue_angles.append(residue_dihedrals)
        if i != 0 and i % 50 == 0 and save_to:
            df = pd.concat(all_residue_angles)
            df.to_csv(save_to+f'rotamers.csv')
        i = i + 1
    df = pd.concat(all_residue_angles)
    if save_to:
        df.to_csv(save_to+f'rotamers.csv')
    return df

def get_residue_dihedrals(chain, resid):
    '''
    Calculate dihedrals for each residue. 
    Returns a dataframe with psi, phi and chi angles.
    '''
    if resid[0] != " ": 
        return
    residue = chain[resid]
    atoms_in_pdb = [a.name for a in residue.get_list()]
    dihedral_angles = {'A':{"altloc":"A"}}
    dihedral_angles['A']['phi']= calcphi(chain, resid)
    dihedral_angles['A']['psi']= calcpsi(chain, resid)
    for chi_atoms, chi in zip([CHI1, CHI2, CHI3, CHI4],['chi1', 'chi2', 'chi3', 'chi4']):
        if residue.resname not in chi_atoms.keys(): continue
        list_atoms_to_find = chi_atoms[residue.resname]
        # find atoms in pdb
        for atoms_to_find in list_atoms_to_find:
            if any(list(map(lambda a: a not in atoms_in_pdb, atoms_to_find))):continue
            #determine whether disordered
            atomobj = [residue[x] for x in atoms_to_find]
            altconf_dict = get_altconf_dict(atomobj)
            for key, atoms in altconf_dict.items():
                if len(atoms) != 4: continue
                # print(resid, chi, atoms)
                chi_angle = calcdihedral(*atoms)
                if key not in dihedral_angles.keys():
                    dihedral_angles[key] = {}
                # handle cases where two torsion angle calculations are possible
                # pick the one that is closer to 0
                if chi not in dihedral_angles[key].keys():
                    dihedral_angles[key][chi] = chi_angle
                else:
                    # print(dihedral_angles[key][chi],chi_angle)
                    dihedral_angles[key][chi] = pick_dihedral([dihedral_angles[key][chi],chi_angle])
                dihedral_angles[key]['altloc'] = key
                # print(dihedral_angles)
    # angles
    df = pd.DataFrame.from_dict(dihedral_angles).T
    df.insert(loc=0, column='residue_number', value=residue.id[1])
    df.insert(loc=1, column='residue_name', value=residue.resname)
    df.insert(loc=2, column='insertion', value=residue.id[2])
    return df

def get_altconf_dict(atom_list):
    '''
    Handle disordered atoms
    '''
    altconfs = []
    for x in atom_list:
        if x.is_disordered():
            altconfs += x.disordered_get_id_list()
    altconfs = list(set(altconfs))
    if len(altconfs)==0:
        return {'A':atom_list}
    altconf_dict ={}
    for i,altloc in enumerate(altconfs):
        alt_atom_list = []
        for atom in atom_list:
            # if not disordered, populate all conformers
            if not atom.is_disordered():
                alt_atom_list.append(atom)
            else:
                for a in atom:
                    if a.altloc == altloc:
                        alt_atom_list.append(a)
        altconf_dict[altloc] = alt_atom_list
    return altconf_dict

def pick_dihedral(angles):
    #pick the dihedral angle closest to 0
    return angles[np.argmin([abs(x) for x in angles])]

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


def normalize_nonrotamer(row):
    if row['residue_name'] in ['ASN','TRP','HIS']:
        row['chi2'] = normalize_180(row['chi2'])
    elif row['residue_name'] == 'ASP':
        row['chi2'] = normalize_90(row['chi2'])
    elif row['residue_name'] == 'GLU':
        row['chi3'] = normalize_90(row['chi3'])
    elif row['residue_name'] in ['PHE','TYR']:
        row['chi2'] =  normalize_150(row['chi2'])
    elif row['residue_name'] == 'GLN':
        row['chi3'] = normalize_180(row['chi3'])
    return row

def normalize_90(x):
    if x>0 and x<=90:
        return x
    elif x>90 and x<=270:
        return x-180
    elif x>270:
        return x-360
    else:
        return np.nan
    
def normalize_150(x):
    if x>0 and x<=150:
        return x
    elif x>150 and x<=330:
        return x-180
    elif x>330:
        return x-360
    else:
        return np.nan

def normalize_180(x):
    if x>0 and x<=180:
        return x
    elif x>180:
        return x-360
    else:
        return np.nan