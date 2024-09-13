'''EnsemblePDB.build.renumber

Functions to renumber PDBs by reference chains or sequences

Authors:
    Rachael Kretsch (rkretsch@stanford.edu)
    Siyuan Du (dusiyuan@stanford.edu)
    Jacob Parres-Gold
'''

from os import path, listdir
from pathlib import Path

import pandas as pd
import numpy as np
from biopandas.pdb import PandasPdb
from tqdm import tqdm
from re import match

from EnsemblePDB.utils.sequence_alignment import specific_pairwise2_align
from EnsemblePDB.utils.file_management import get_dir, get_nonexistant_file
from EnsemblePDB.utils.biopandas_utils import get_chain_seq, retrieve_pdbs
from EnsemblePDB.utils.table_utils import get_reference_chains, reformat_dict, get_table_entry_from_list, get_list_from_table_entry, get_nested_lists, match_names


def download_and_renumber(summary_report_csv, reference_pdb=None,
                          reference_chains=None,
                          sequences=None,
                          downloaded_pdb_folder=None,
                          combine_chains_by=None,
                          multimer_save=False,
                          output_directory=None):
    '''
    Given a csv files from a pdb search, a reference pdb, and its reference
    chains, download all unaltered pdbs in one folder and then relabel
    chains and renumber residues of chains aligned to reference chains
    according to reference chains numbering and save renumbering in a new folder.
    Arguments: 
        summary_report_csv (str): csv file resulting from a pdb search 
                                should contain the following columns
                                    - Entry ID: pdb ids
                                for each chain in reference_chains
                                    - Order of chains
                                    - Chain aligned
        reference_pdb (str): pdb id of the reference structure
        reference_chains (list of str): the reference chains as labeled
                                        in the unaltered reference pdb, other pdbs will be labeled according to these chains.
        sequences(list of str): reference sequences, if input sequences instead of reference_pdb and reference_chains, residues will be number by positions in these sequences. Do not suggest using because proteins have conventional numbering.
        downloaded_pdb_folder (str): If have already downloaded the raw
                                 unaltered pdbs, specify the folder in which 
                                 they can be found. Note this folder should not 
                                 include any *.pdb files you do not wish to 
                                 include {default: None}
        chain_only (bool): If do not want to renumbering aligned chains but
                            only want to rename chains {default: False}
        combine_chains_by (str): Only use when need to combine chains, requires input summary report to contain information from refine.check_chains. options: 'aligned' or 'all' {default: None}
        multimer_save (bool): If True, will save PDBs that contain multimers multiple times, each time with a different aligned chain named "A".
        output_directory (str): directory to save PDBs and updated reports to.

    Returns:
        dataframe
    Output 2 folders, one with downloaded unaltered pdb, unless directory
    already given, and a new directory with pdbs with renumbered residues
    to match the reference pdb and all chains that do not match a reference
    chain labeled L and above.
    '''
    assert ((reference_pdb is not None and reference_chains is not None)
            or sequences is not None and not ((reference_pdb is not None and reference_chains is not None)
                                              and sequences is not None)), "Must specify reference pdb and chains or sequences."
    if sequences:
        reference_chains = []
        for i in range(len(sequences)):
            reference_chains.append("inputted_seq_" + str(i))

    # Retrieve data in the ensemble_csv
    data = pd.read_csv(summary_report_csv)
    # Check validity of csv format
    assert('Entry ID' in data.columns), "Summary csv does not contain column: Entry ID"
    assert(any([True if match('Align ref .*: chains aligned', x) else False for x in data.columns])
           ), "Summary csv does not contain column: Align ref *: chains aligned"
    assert(any([True if match('Align ref .*: order of chains', x) else False for x in data.columns])
           ), "Summary csv does not contain column: Align ref *: order of chains"
    assert(any([True if match('Align ref .*: pairwise align score', x) else False for x in data.columns])
           ), "Summary csv does not contain column: Align ref *: pairwise align score"

    # save data to
    output_directory = str(Path(summary_report_csv).parents[0])
    fstem = str(Path(summary_report_csv).stem)

    # Get references listed in summary report. Note that this may be different from input reference_chains. E.g. If used query_by_sequence before, but now want to use a reference PDB and chain to renumber.
    table_references = get_reference_chains(data)

    # Download all pdbs in a directory
    if downloaded_pdb_folder is None:
        directory = get_dir(output_directory+'/Unaltered_pdbs')
        retrieve_pdbs(data, Path(directory))
    else:
        directory = Path(downloaded_pdb_folder)
    # make a matching folder for renumbered pdb
    renumbered_directory = get_dir(
        output_directory+'/Renumbered_unaligned_pdbs')
    # rename chains
    tqdm.pandas(desc="Renaming chains")
    # Remove pdbs in summary csv but not downloaded
    to_drop = match_names(pdb_dir=directory, summary_df=data)
    # if to_drop in data['Entry ID'].unique().tolist():
    data = data.loc[~data['Entry ID'].isin(to_drop)]
    # data = data.drop(data[data['Entry ID'].apply(lambda y: y.lower() not in [x[:4].lower() for x in listdir(directory)])].index, axis=0)
    # else:
    #     data['Entry ID'] = data['Entry ID'].apply(lambda y: y.lower())
    data['Renamed: old to new chain'] = data.progress_apply(lambda row: _rename_chains(
        row, table_references, directory, renumbered_directory, combine_chains_by), axis=1)
    # get new reference chains
    ref_row = data.loc[(data['Entry ID'] == reference_pdb) | (
        data['Entry ID'] == reference_pdb.upper()) | (data['Entry ID'] == reference_pdb.lower())].iloc[0]
    chain_dict = ref_row['Renamed: old to new chain']
    new_ref_chains = list(set([chain_dict[c] for c in reference_chains]))
    # get list of renamed chains that align to each reference sequence
    data = _get_new_aligned_chains(
        data, reference_pdb, new_ref_chains, table_references, sequences, renumbered_directory)
    # get reference positions to renumber by
    all_residue_positions = _get_reference_positions(
        renumbered_directory, new_ref_chains, reference_pdb, sequences)
    # renumber residues
    tqdm.pandas(desc="Renaming residues")
    data[['Renamed: old to new reidues', 'Renamed: mutations']] = data.progress_apply(
        lambda row: _renumber_residues(row, new_ref_chains, renumbered_directory, all_residue_positions), axis=1)

    if multimer_save:
        tqdm.pandas(desc="Saving multimers separately")
        data.progress_apply(lambda row: _save_multimers(
            row, new_ref_chains, renumbered_directory), axis=1)

    # clear format for csv saving
    data = data.drop('Renamed: old to new reidues', axis=1)
    data['Renamed: old to new chain'] = data['Renamed: old to new chain'].apply(
        reformat_dict)
    data['Renamed: mutations'] = data['Renamed: mutations'].apply(
        get_table_entry_from_list)
    for ref_chain in new_ref_chains:
        data[f'Renamed: Align ref {ref_chain}: chains aligned'] = data[f'Renamed: Align ref {ref_chain}: chains aligned'].apply(get_table_entry_from_list)
        data[f'Renamed: Align ref {ref_chain}: ref seq alignment'] = data[f'Renamed: Align ref {ref_chain}: ref seq alignment'].apply(get_table_entry_from_list)
        data[f'Renamed: Align ref {ref_chain}: chains seq alignment'] = data[f'Renamed: Align ref {ref_chain}: chains seq alignment'].apply(get_table_entry_from_list)
    fname = get_nonexistant_file(
        output_directory+'/'+fstem+'_renumbered.csv')
    data.to_csv(fname)
    print('\n')
    print(f'Original PDBs in {directory}.')
    print(f'Renumbered PDBs saved to {renumbered_directory}')
    print(f'Chains information saved to {fname}.')
    return data

###############################################################################
# Helper functions
###############################################################################


def _swap_chains(row, target):
    if row['chain_id'] == target:
        return 'A'
    elif row['chain_id'] == 'A':
        return target
    else:
        return row['chain_id']


def _get_TER_old_to_new(chain_old_to_new):
    '''
    Get dictionary of old chain: new chain for TER records that need to be changed
    Skip any chains to be combined
    Assume chains to be combined are in order!
    '''
    TER_old_to_new = {}
    if not chain_old_to_new:
        return TER_old_to_new
    last_key = list(chain_old_to_new.keys())[0]
    last_chain = chain_old_to_new[last_key]
    for key, value in chain_old_to_new.items():

        if not last_chain:
            last_chain = value
        if last_chain != value:
            TER_old_to_new[last_key] = last_chain
        last_key = key
        last_chain = value
    end_key = list(chain_old_to_new.keys())[-1]
    end_chain = chain_old_to_new[end_key]
    TER_old_to_new[end_key] = end_chain
    return TER_old_to_new


def _change_TER_record(pdb_struct, chain_old_to_new):
    '''
    Change TER records in a pandas pdb
    Delete TER in combined chains
    Rename other TER lines with new chain id
    '''
    TER_old_to_new = _get_TER_old_to_new(chain_old_to_new)
    if not TER_old_to_new:
        return pdb_struct
    TER = pdb_struct.df['OTHERS'].loc[pdb_struct.df['OTHERS']
                                      ['record_name'] == 'TER']
    new_rows = []
    for index, TER_row in TER.iterrows():
        to_delete = False
        entry = TER_row['entry'].split(' ')
        new_entry = []
        for e in entry:
            # get chain id
            to_append = e
            if len(e) == 1 and e.isalpha() == True:
                if e not in TER_old_to_new.keys():
                    to_delete = True
                else:
                    to_append = TER_old_to_new[e]
            new_entry.append(to_append)
        if not to_delete:
            TER_row['entry'] = ' '.join(new_entry)
            new_rows.append(TER_row.to_frame().T)
    if len(new_rows) > 0:
        new_TER = pd.concat(new_rows)
        pdb_struct.df['OTHERS'] = pd.concat(
            [pdb_struct.df['OTHERS'].loc[pdb_struct.df['OTHERS']['record_name'] != 'TER'], new_TER])
    else:
        print('WARNING: All TER records deleted')
    return pdb_struct


def _rename_chains(row, reference_chains, directory, new_directory, combine_chains_by=None):
    '''
    Rename protein chains (aligned to ref) by alphabetical order (starting from 'A')
    Rename ligand chains by alphabetical order (starting from 'L')
    Combine chains of a PDB structure given a summary report dataframe with "Check ref {}: suggest combine chains to groups"
    Saves new PDB to a new directory
    '''
    pdb_file = path.join(directory, f"{row['Entry ID'].lower()}.pdb")
    pdb_struct = PandasPdb().read_pdb(pdb_file)
    chain_old_to_new = {}
    i = 65  # protein start from A
    l = 76  # ligand start from L
    no_combinations_suggested = False
    for ref_chain in reference_chains:
        order_of_chains = get_list_from_table_entry(row[f'Align ref {ref_chain}: order of chains'])
        # first consider chains aligned to reference
        if combine_chains_by:
            to_combine_ = row[f'Check ref {ref_chain}: suggest combine chains to groups ({combine_chains_by})']
            if not pd.isnull(to_combine_) or to_combine_ in ['nan', 'Null', 'None', 'NaN']:
                to_combine = get_nested_lists(to_combine_)
                for group in to_combine:
                    new_chain = chr(i)
                    i += 1
                    for chain in group:
                        chain_old_to_new[chain] = new_chain
            else:
                no_combinations_suggested = True

        if (not combine_chains_by) or no_combinations_suggested:
            aligned_to_rename = [x for x in get_list_from_table_entry(row[f'Align ref {ref_chain}: chains aligned']) if x != 'nan']
            aligned_order = [order_of_chains.index(
                c) for c in aligned_to_rename]
            sorted_aligned = [x for _, x in sorted(zip(
                aligned_order, aligned_to_rename), key=lambda pair: pair[0])]  # sort by order of chains
            for aligned_chain in sorted_aligned:
                new_protein_chain = chr(i)
                i += 1
                chain_old_to_new[aligned_chain] = new_protein_chain

        non_aligned = [x for x in get_list_from_table_entry(
            row['Align: Non-aligned chains']) if x not in chain_old_to_new.keys() and x != 'nan' and x in order_of_chains]
        other_nonaligned = [x for x in get_list_from_table_entry(
            row['Align: Non-aligned chains']) if x not in chain_old_to_new.keys() and (x not in order_of_chains or x == 'nan')]
        # these weirdly labeled nonaligned chains will be added at the end.
        non_aligned_order = [order_of_chains.index(
            c) for c in non_aligned] + other_nonaligned
        sorted_non_aligned = [x for _, x in sorted(
            zip(non_aligned_order, non_aligned), key=lambda pair: pair[0])]
        for non_aligned_chain in sorted_non_aligned:
            new_ligand_chain = chr(l)
            l += 1
            chain_old_to_new[non_aligned_chain] = new_ligand_chain

    pdb_struct.df['ATOM']['chain_id'] = pdb_struct.df['ATOM']['chain_id'].map(
        chain_old_to_new).fillna("NAN")
    pdb_struct.df['HETATM']['chain_id'] = pdb_struct.df['HETATM']['chain_id'].map(
        chain_old_to_new).fillna("NAN")
    pdb_struct.df['ANISOU']['chain_id'] = pdb_struct.df['ANISOU']['chain_id'].map(
        chain_old_to_new).fillna("NAN")
    # TER records, delete combined chain TER lines and change others
    # if not no_combinations_suggested:
    pdb_struct = _change_TER_record(pdb_struct, chain_old_to_new)

    renamed_pdb = path.join(new_directory, row['Entry ID']+'.pdb')
    pdb_struct.to_pdb(path=renamed_pdb, gz=False)

    return chain_old_to_new


def _get_new_aligned_chains(data, reference_pdb, new_ref_chains, table_references, sequences, renumbered_directory):
    '''
    For renamed chains, get chains that aligns to the reference chain and add new columns to the dataframe
    '''
    reference_seqs = []
    # get reference sequences
    if sequences:
        reference_seqs = sequences
        new_ref_chains = table_references
    else:
        ref_file = path.join(renumbered_directory, f"{reference_pdb}.pdb")
        ref_struct = PandasPdb().read_pdb(ref_file)
        # get the new reference_chains
        for ref_chain in new_ref_chains:
            reference_seqs.append(get_chain_seq(
                ref_struct, chain=ref_chain, remove_his_tag=False))

    for i, ref_seq in enumerate(reference_seqs):
        all_chains_aligned = []
        all_ref_alignments = []
        all_seq_alignments = []
        for index, row in tqdm(data.iterrows(), total=len(data), desc=f'Get new aligned chains for reference {i}'):
            chains_aligned = []
            ref_alignment = []
            seq_alignment = []
            pdb_file = path.join(renumbered_directory, f"{row['Entry ID']}.pdb")
            pdb_struct = PandasPdb().read_pdb(pdb_file)
            for chain in pdb_struct.df['ATOM']['chain_id'].unique():
                seq = get_chain_seq(
                    pdb_struct, chain=chain, remove_his_tag=False)
                alignment = specific_pairwise2_align(ref_seq, seq,
                                                     {'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})
                if len(alignment) > 1:
                    alignment = alignment[0]
                else:
                    continue
                max_previous_score = max([float(x) for x in get_list_from_table_entry(row[f'Align ref {table_references[i]}: pairwise align score']) if x != ''])
                # heuristics, if similar to previous alignment score then should be aligned. arbitrary allowance given 10% of sequence length
                if alignment.score >= (max_previous_score-0.1*len(seq)):
                    chains_aligned.append(chain)
                    ref_alignment.append(alignment.seqA)
                    seq_alignment.append(alignment.seqB)
            all_chains_aligned.append(chains_aligned)
            all_ref_alignments.append(ref_alignment)
            all_seq_alignments.append(seq_alignment)
        data[f'Renamed: Align ref {new_ref_chains[i]}: chains aligned'] = all_chains_aligned
        data[f'Renamed: Align ref {new_ref_chains[i]}: ref seq alignment'] = all_ref_alignments
        data[f'Renamed: Align ref {new_ref_chains[i]}: chains seq alignment'] = all_seq_alignments

    return data


def _get_reference_positions(renumbered_directory, reference_chains, reference_pdb=None, sequences=None):
    '''
    Given reference pdb and chains or reference sequences, get a dictionary of reference residue positions to use for renumbering.
    Note that the input directory must contain reference PDB with renamed chains (contains original reference chain.)
    '''
    all_ref_positions = []
    if sequences:
        print('Warning: using input sequences and exact amino acid orders. With some proteins that have gapped numbering or insertions, using exact sequences may lead to different numbering from the common annotations. Consider using a correctly numbered reference PDB.')
        for ref_seq in sequences:
            ref_positions = {}
            for pos, residue in enumerate(ref_seq):

                ref_positions[pos] = (pos+1, '', residue)
        all_ref_positions.append(ref_positions)

    else:
        ref_file = path.join(renumbered_directory, f"{reference_pdb}.pdb")
        ref_struct_atoms = PandasPdb().read_pdb(ref_file).df['ATOM']
        for ref_chain in reference_chains:
            ref_positions = {}
            ref_chain_atoms = ref_struct_atoms.loc[ref_struct_atoms['chain_id'] == ref_chain]
            pos = 0
            for index, r in ref_chain_atoms.iterrows():
                atomID = (r['residue_number'],
                          r['insertion'], r['residue_name'])
                if atomID not in ref_positions.values():
                    ref_positions[pos] = atomID
                    pos += 1
        all_ref_positions.append(ref_positions)
    return all_ref_positions


def _renumber_residues(row, reference_chains, renumbered_directory, all_ref_positions):
    '''
    Given a list of reference positions, get a dictionary that specifies the orginal residue ID and the residue ID to set to. (number, insertion,residue name).
    Then, renumber all residues based on this map.
    Saves pdb and overwrites files already in renumbered directory
    '''
    pdb_file = path.join(renumbered_directory, f"{row['Entry ID']}.pdb")
    pdb_struct = PandasPdb().read_pdb(pdb_file)

    mutations = []
    positions_map = {}
    for i, ref_chain in enumerate(reference_chains):
        ref_positions = all_ref_positions[i]
        ref_aligns = row[f'Renamed: Align ref {ref_chain}: ref seq alignment']
        seq_aligns = row[f'Renamed: Align ref {ref_chain}: chains seq alignment']
        for index, chain in enumerate(row[f'Renamed: Align ref {ref_chain}: chains aligned']):
            ref_align = ref_aligns[index]
            seq_align = seq_aligns[index]
            # get the sequence of residues from the structure
            pdb_chain = pdb_struct.df['ATOM'].loc[pdb_struct.df['ATOM']
                                                  ['chain_id'] == chain]
            struct_residues = []
            for index, row in pdb_chain.iterrows():
                residue = (row['residue_number'],
                           row['insertion'], row['residue_name'])
                if residue not in struct_residues:
                    struct_residues.append(residue)
            ref_position = 0
            struct_position = 0
            insert = 65  # start insertion code at A
            seqIDtoset = (np.nan, '', '')
            for ref, seq in zip(ref_align, seq_align):
                # set current insertion code (if previous ID contains insertion code, use the next character in alphabet)
                if seqIDtoset[1] != '':
                    insert = ord(seqIDtoset[1])+1
                else:
                    insert = 65  # start from A
                # check current position. (if sequence left with no reference, add insertion code. if reference left with no sequence, end. )
                if struct_position >= len(struct_residues):
                    break
                if ref_position >= len(ref_positions):
                    seqID = struct_residues[struct_position]
                    seqIDtoset = (
                        ref_positions[ref_position-1][0], chr(insert), seqID[2])
                else:
                    # first, get atomID of the ref residue from biopandas (chain, num, insert) using ref_positions
                    refID = ref_positions[ref_position]
                    # then, get atomID of the seq residue from pdb_struct]
                    seqID = struct_residues[struct_position]
                    seqIDtoset = (np.nan, '', '')
                    if (ref == "-" or ref == "?") and (seq == "-" or seq == "?"):
                        continue
                    elif (ref == "-" or ref == "?") and ~ (seq == "-" or seq == "?"):
                        if ref_position == 0:
                            seqIDtoset = (0, chr(insert), seqID[2])
                            struct_position += 1
                        else:
                            # if reference seq is a gap,
                            seqIDtoset = (
                                ref_positions[ref_position-1][0], chr(insert), seqID[2])
                            # do not count ref index since this is not in reference
                            struct_position += 1
                    elif ~(ref == "-" or ref == "?") and (seq == "-" or seq == "?"):
                        # if seq is gap but reference is not, leave seqID blank.
                        # i += 1
                        ref_position += 1
                    else:
                        seqIDtoset = (refID[0], refID[1], seqID[2])
                        ref_position += 1
                        struct_position += 1
                        if refID[2] != seqID[2]:
                            mutations.append(
                                refID[2]+str(refID[0])+refID[1].lower()+seqID[2])
                positions_map[(chain,)+seqID] = (chain,)+seqIDtoset

    pdb_struct.df['ATOM']['ID'] = pdb_struct.df['ATOM'].apply(lambda x: (
        x['chain_id'], x['residue_number'], x['insertion'], x['residue_name']), axis=1)
    pdb_struct.df['ATOM']['ID'] = pdb_struct.df['ATOM']['ID'].map(
        positions_map).fillna(pdb_struct.df['ATOM']['ID'])
    pdb_struct.df['ATOM'][['chain_id', 'residue_number', 'insertion']
                          ] = pdb_struct.df['ATOM']['ID'].apply(lambda x: pd.Series([x[0], x[1], x[2]]))
    pdb_struct.df['ATOM'] = pdb_struct.df['ATOM'].drop("ID", axis=1)
    pdb_struct.to_pdb(path=pdb_file, gz=False)
    return pd.Series([positions_map, mutations])


def _save_multimers(row, reference_chains, renumbered_directory):
    '''
    If multiple chains aligned to ref sequence/chain, save this PDB multiple times, each with a different aligned chain named to 'A'.
    '''
    pdb_file = path.join(renumbered_directory, row['Entry ID']+'.pdb')
    for ref_chain in reference_chains:
        protein_chains = row[f'Renamed: Align ref {ref_chain}: chains aligned']
        for chain in protein_chains[1:]:
            pdb_copy = PandasPdb().read_pdb(pdb_file)
            pdb_copy.df['ATOM']['chain_id'] = pdb_copy.df['ATOM'].apply(
                lambda x: _swap_chains(x, target=chain), axis=1)
            if len(pdb_copy.df['ANISOU']) != 0:
                pdb_copy.df['ANISOU']['chain_id'] = pdb_copy.df['ANISOU'].apply(
                    lambda x: _swap_chains(x, target=chain), axis=1)
            if len(pdb_copy.df['HETATM']) != 0:
                pdb_copy.df['HETATM']['chain_id'] = pdb_copy.df['HETATM'].apply(
                    lambda x: _swap_chains(x, target=chain), axis=1)
            copy_fname = path.join(renumbered_directory,
                                   row['Entry ID']+'_'+chain+'.pdb')
            pdb_copy.to_pdb(path=copy_fname, gz=False)
