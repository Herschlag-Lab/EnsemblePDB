''' EnsemblePDB.visualize.load_ensemble

Fuction to load pdbs and group if desired

Authors:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
'''

import pymol.cmd as pymolcmd
from os import path
from pathlib import Path
import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier


def load_ensemble(directory, Entry_IDs=None, reference_pdb=None, groups=None, group_by='Crude subensemble', cmd=pymolcmd):
    '''
    Function that loads every pdb in a folder, reference first.
    Arguments: 
           directory (str): Path to the folder in which these
                               pdbs can be found. Note this folder should
                               not contain any pdbs you do not wish to analyse.
            Entry_IDs (list): list of entry ids to load
            reference_pdb (str): pdb id of reference to be loaded first
            groups: dataframe with at least 2 columns, "Entry ID" and input group_by
    '''
    # Checking inputs
    assert (path.isdir(directory)), directory + " is not a folder."

    # Setting up pymol
    cmd.bg_color('white')
    cmd.set('depth_cue', 0)
    cmd.set('ray_trace_fog', 0)

    # print("Using all files, *.pdb, in ", directory)

    # load reference pdb and then load and align other pdbs
    if Entry_IDs is None:
        print("Using all files, *.pdb, in ", directory)
        files = list(Path(directory).glob("*.pdb"))
    else:
        print("Using all files in the given Entry ID list")
        Entry_IDs = [x.upper() for x in Entry_IDs]
        files = [x for x in list(Path(directory).glob(
            "*.pdb")) if str(x).split('/')[-1].split('.')[0].upper() in Entry_IDs]
    assert (files != []), "No pdbs in " + directory

    if reference_pdb is not None:
        reference_pdb = reference_pdb[:4].lower()+reference_pdb[4:]
        ref_file = Path(directory) / f"{reference_pdb}.pdb"
        assert (path.isfile(ref_file)), f"{ref_file} is not a file."
        ref_file = str(ref_file)
        cmd.load(ref_file)

        # if split_chains:
        #     cmd.split_chains(reference_pdb)
        #     cmd.group(reference_pdb + "_", " ".join(
        #         [name for name in cmd.get_names() if name[:4] == reference_pdb]))
        files.remove(ref_file)
    for file in files:
        # print(file)
        cmd.load(str(file))
        # pdb = file.stem
        # if split_chains:
        #     cmd.split_chains(pdb)
        #     cmd.group(
        #         pdb + "_", " ".join([name for name in cmd.get_names() if name[:4] == pdb]))

    if groups is not None:
        pdb_names = [file.stem.upper() for file in files]
        # groups = pd.read_csv(groups)
        groups['Entry ID'] = groups['Entry ID'].apply(lambda x: x.upper())
        # group_pdbs = groups["Entry ID"].str.lower().tolist()
        # check if entries are in annotations
        if Entry_IDs is None:
            extra_entry = groups[~groups["Entry ID"].isin(
                pdb_names)]["Entry ID"].tolist()
            print(
                "Some pdbs in the inputed dataframe are not in the directory: ", extra_entry)
        groups = groups[groups["Entry ID"].isin(pdb_names)]
        extra_pdbs = [
            pdb for pdb in pdb_names if not pdb in groups['Entry ID'].values]
        if len(extra_pdbs) > 0:
            print(
                "WARNING these pdbs were found in the directory but not in the inputted dataframe: ", extra_pdbs)
        for subensemble in groups[group_by].unique():
            sub_group = groups[groups[group_by] == subensemble][
                "Entry ID"].tolist()
            print(subensemble, sub_group)
            # if split_chains:
            #     length = 5
            # else:
            #     length = 4
            cmd.group(subensemble, " ".join(
                [name for name in cmd.get_names() if name.upper() in sub_group]))


pymolcmd.extend("load_all", load_ensemble)
