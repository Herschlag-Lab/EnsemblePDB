'''EnsemblePDB.refine.check_chains

Functions to check whether the aligned chains needs to be combined, if there 
are multiple chains in a PDB entry that aligned to each reference sequences or 
chains. For example, some proteases have the sequence split to multiple chains 
in author annotated PDBs, but for alignment we need to make all chain lengths 
consistent.

Authors:
    Rachael Kretsch(rkretsch@stanford.edu), 
    Siyuan Du (dusiyuan@stanford.edu),
    Jacob Parres-Gold
'''

import re
from pathlib import Path
import pandas as pd
from EnsemblePDB.utils import file_management, table_utils, sequence_alignment


def check_multimer_chains(ensemble_csv, allowance=10, output_directory=None):
    '''
    Given a summary report (filtered), suggests chains to group and combine 
    for each reference sequence. Assumes that only the chains in order 
    (as indicated by the PDB authors) can be combined to groups.
    Arguments:
        ensemble_csv = str: path of the summary report csv
        output_directory = str: path to save output dataframes as csvs 
            {default: None}
    Outputs:
        dictionary of dataframes with chain alignment information and 
        suggestions where key is the reference chain name in the dataframes, 
        column "align chain groups" refers to the best group of chains picked 
        from all aligned chains (highest alignment score to reference chain 
        when combined in sequence), column "suggest chain groups" checks any 
        additional non-aligned chains that can be added to aligned chain groups 
        to get better alignment scores.
    '''
    data = pd.read_csv(ensemble_csv)
    reference_chains = table_utils.get_reference_chains(data)

    # get lists of aligned chains, and expected lengths
    all_seq_align = {}
    for chain in reference_chains:
        pdbs = data["Entry ID"].tolist()
        aligned_chains = [table_utils.get_list_from_table_entry(item) for item in data[f"Align ref {chain}: chains aligned"].fillna("").tolist()]
        order_of_chains = [table_utils.get_list_from_table_entry(item) for item in data[f"Align ref {chain}: order of chains"].fillna("").tolist()]
        aligned_seqs = [table_utils.get_list_from_table_entry(item) for item in data[f"Align ref {chain}: aligned sequence"].fillna("").tolist()]
        expected_length = int(data.loc[0, f"Search ref {chain}: query_length"])
        ref_seq = table_utils.get_list_from_table_entry(data.loc[0, f"Align ref {chain}: aligned ref sequence"])[0].replace(" ", "").replace("-", "")
        chains_lengths = data.apply(lambda l: make_length_dict(
            l, reference_chains[0]), axis=1).tolist()
        df = pd.DataFrame({'pdb': pdbs, 'aligned_chains': aligned_chains,
                           'order_of_chains': order_of_chains,
                           'aligned_seqs': aligned_seqs, 'expected_length': expected_length,
                           'ref_seq': ref_seq, 'chains_lengths': chains_lengths})
        df['align_matched_positions'] = df.apply(make_positions_dict, axis=1)
        # sanity check for defect data:
        mask1 = df.apply(lambda x: True if len(x['order_of_chains']) == len(
            x['aligned_seqs']) else False, axis=1)
        mask2 = pd.isna(df['order_of_chains'])
        mask3 = pd.isna(df['aligned_seqs'])
        defect = df.loc[(~mask1) | mask2 | mask3]
        if len(defect) > 0:
            print(f"Warning: PDBs{defect['pdb'].tolist()} have invalid values in order of chains and aligned sequences. Filtered, suggest manual check.")
        df = df.loc[mask1 & (~mask2) & (~mask3)]
        if len(df) == 0:
            continue
        all_seq_align[chain] = df
    if len(all_seq_align) == 0:
        print('No pdbs left to check. Suggest manual check.')
        return

    for ref_chain, chain_df in all_seq_align.items():
        chain_df['aligned_chains'] = chain_df['aligned_chains'].apply(
            lambda x: [] if x == [""] else x)
        no_align = chain_df.loc[chain_df['aligned_chains'].apply(
            lambda x: (len(x) == 0))]
        multiple_align = chain_df.loc[chain_df['aligned_chains'].apply(
            lambda x: len(x) > 1)]
        if len(multiple_align) > 0:
            print(f'You have {len(multiple_align)} PDBs that contain multiple chains that aligned with reference {ref_chain}. Continue to suggest.')
        if len(no_align) > 0:
            print(f"Warning: PDBs {no_align['pdb'].tolist()} have no chain that aligns to ref {ref_chain}")

    for ref_chain, df in all_seq_align.items():
        df['align_chain_groups'] = df.apply(
            lambda x: get_align_chain_groups(x, allowance), axis=1)
        df['suggest_chain_groups'] = df.apply(
            lambda x: get_suggest_chain_groups(x, allowance), axis=1)

    for ref_chain, df in all_seq_align.items():
        multimers = df.loc[df['suggest_chain_groups'].apply(
            lambda x: len(x) > 1)]
        if len(multimers['pdb'].tolist()) > 0:
            print(f"Notice that PDBs {multimers['pdb'].tolist()} have multiple chain groups that align to ref {ref_chain} (multimers)")
        else:
            print("All structures are monomers")
        to_check = df.loc[df['suggest_chain_groups'].apply(
            check_group_members)]
        if len(to_check['pdb'].tolist()) > 0:
            print(f"Manually check PDBs {to_check['pdb'].tolist()}, combined group have different number of chains.")

    # reformat summary report and save
    for chain, df in all_seq_align.items():
        data[f'Check ref {chain}: suggest combine chains to groups (aligned)'] = df['align_chain_groups'].apply(table_utils.reformat_nested_lists)
        data[f'Check ref {chain}: suggest combine chains to groups (all)'] = df['suggest_chain_groups'].apply(table_utils.reformat_nested_lists)
    if output_directory:
        filename = file_management.get_nonexistant_file(
            output_directory+'/'+str(Path(ensemble_csv).stem)+'_checked.csv')
    else:
        filename = file_management.get_nonexistant_file(str(
            Path(ensemble_csv).parents[0])+'/'+str(Path(ensemble_csv).stem)+'_checked.csv')
    data.to_csv(filename)
    print(f'Checked chains to combine and saved information to {filename}')

    return data


###############################################################################
# helper functions


def make_positions_dict(row):
    '''
    Apply on rows of ensemble csv to create a dictionary where key is chain 
    name and value is the positions where the chain aligns to the reference 
    chain/sequence.
    '''
    positions = {}
    for index, seq in enumerate(row['aligned_seqs']):
        list_of_positions = []
        chain = row['order_of_chains'][index]
        start = end = -1
        for i, char in enumerate(seq):
            if char.isalpha():
                if start == -1:
                    start = i
            else:
                if start != -1:
                    end = i - 1
                    list_of_positions.append((start, end))
                    start = end = -1
        if start != -1:
            list_of_positions.append((start, len(seq)-1))
        positions[chain] = list_of_positions
    return positions


def get_align_chain_groups(row, allowance):
    '''
    Assume only chains in order can be combined, determine whether there are 
    multiple repetitive chains(multimers) and group them into lists.
    '''
    groups = []

    group = []
    to_check = False
    # #DEBUG
    # print(row['pdb'])
    for index, chain in enumerate(row['aligned_chains']):
        if chain in group:
            continue
        full_index = row['order_of_chains'].index(chain)
        seq = row['aligned_seqs'][full_index]
        seq = re.compile('[^a-zA-Z]').sub('', seq)
        curr_score = sequence_alignment.specific_pairwise2_align(row['ref_seq'], seq, {
                                                                 'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})[0].score
        best_score = curr_score
        curr_length = row['chains_lengths'][chain]
        group = [chain]

        i = index
        while (curr_length < row['expected_length']-allowance) and (i < len(row['aligned_chains'])-1):
            # try combine with next aligned chain
            i += 1
            full_index += 1
            next_chain = row['aligned_chains'][i]
            # if chains in order are repetitive throw message to manually fix
            if row['chains_lengths'][chain] == row['chains_lengths'][next_chain]:
                to_check = True
            # check whether in order
            if full_index != row['order_of_chains'].index(next_chain):
                break
            # print(f'{chain} length shorter than expected: try combining with {next_chain}')
            seq = seq + row['aligned_seqs'][full_index]
            seq = re.compile('[^a-zA-Z]').sub('', seq)
            curr_score = sequence_alignment.specific_pairwise2_align(row['ref_seq'], seq, {
                                                                     'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})[0].score
            curr_length = curr_length+row['chains_lengths'][next_chain]
            if curr_score < best_score:
                # groups.append(group)
                # print(f'Warning: group {group} has length {curr_length}. Consider unaligned chains.')
                break
            else:
                group.append(next_chain)

        groups.append(group)
    if to_check:
        print(f'Warning: manually check {row["pdb"]}, it may contain chains to combine that are not in order.')
    return groups


def get_suggest_chain_groups(row, allowance):
    '''
    Check if each group in grouped chains match reference sequence length. If 
    not, test combining with a nonaligned chain that is in order.
    Arguments:
        row (Pandas dataframe rows): row of data from ensemble csv
        allowance (int): allowed difference in length for reference sequence 
        and aligned sequence to skip checking
    '''
    # rough check by seq lengths
    new_groups = []
    for group in row['align_chain_groups']:
        new_group = group
        length = 0
        seq = ''
        for chain in group:
            length = length + row['chains_lengths'][chain]
            index = row['order_of_chains'].index(chain)
            seq = seq + row['aligned_seqs'][index]
        curr_score = sequence_alignment.specific_pairwise2_align(row['ref_seq'], seq, {
                                                                 'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})[0].score
        best_score = curr_score
        if length < (row['expected_length']-allowance):
            # first, check the positions missing is at beginning or the end
            start = row['align_matched_positions'][group[0]][0][0]
            end = row['align_matched_positions'][group[-1]][-1][-1]
            # prev_index = row['order_of_chains'].index(group[0]) - 1
            if start > allowance:
                # get the chain before start chain
                prev_index = row['order_of_chains'].index(group[0]) - 1
                if prev_index >= 0:
                    # continue
                    # print(f"PDB:{row['pdb']} chain (group) {group} failed to suggest non-aligned chains to combine. Manually check.")
                    # else:
                    prev_chain = row['order_of_chains'][prev_index]
                    # preppend the chain and check score
                    seq = re.compile('[^a-zA-Z]').sub('',
                                                      row['aligned_seqs'][prev_index] + seq)
                    curr_score = sequence_alignment.specific_pairwise2_align(row['ref_seq'], seq, {
                                                                             'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})[0].score
                    if curr_score > best_score:
                        best_score = curr_score
                        new_group = [prev_chain] + group
                    else:
                        new_group = group
            # this part hasn't been tested yet..
            if end < row['expected_length']-allowance:
                # get the chain next to last chain
                next_index = row['order_of_chains'].index(group[-1]) + 1
                if next_index < len(row['order_of_chains']):
                    # print(f"PDB:{row['pdb']} chain (group) {group} failed to suggest non-aligned chains to combine. Manually check.")
                    # else:
                    next_chain = row['order_of_chains'][next_index]
                    # preppend the chain and check score
                    seq = re.compile('[^a-zA-Z]').sub('',
                                                      seq+row['aligned_seqs'][next_index])
                    curr_score = sequence_alignment.specific_pairwise2_align(row['ref_seq'], seq, {
                                                                             'alignment_type': "global", 'gap_open_score': -0.5, 'gap_extend_score': -0.1})[0].score
                    if curr_score > best_score:
                        best_score = curr_score
                        new_group = group + [next_chain]
        elif length > (row['expected_length']+allowance):
            print(f"PDB:{row['pdb']} chain (group) {group} exceeds the full ref seq length. Manually check for split.")
        new_groups.append(new_group)
    return new_groups


def make_length_dict(data, ref_chain):
    x_list = table_utils.get_list_from_table_entry(data[f"Align ref {ref_chain}: order of chains"])
    y_list = get_chain_length_from_align(data[f"Align ref {ref_chain}: aligned sequence"])
    # correct for delete_all_one_value
    if len(y_list) == 1 and len(x_list) != 1:
        y_list = y_list*len(x_list)
    return dict(zip(x_list, y_list))


def get_chain_length_from_align(s):
    s = table_utils.get_list_from_table_entry(s)
    list_lens = []
    for i in s:
        list_lens.append(len([c for c in i if c != "-"]))
    return list_lens


def check_group_members(row):
    lengths = [len(x) for x in row]
    status = False
    for i, leng in enumerate(lengths):
        if i < len(lengths) - 1:
            if leng != lengths[i+1]:
                status = True
    return status
