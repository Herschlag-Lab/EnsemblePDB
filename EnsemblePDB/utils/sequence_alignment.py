""" EnsemblePDB.utils.sequence_alignment

Generalized way to easily call pairwise2 with any options desired.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""
from pathlib import Path
import pandas as pd
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from EnsemblePDB.utils.file_management import get_nonexistant_file
from EnsemblePDB.utils.biopandas_utils import get_pdb_struct
from EnsemblePDB.utils.table_utils import get_overlap, get_union, get_table_entry_from_list


def specific_pairwise2_align(seqA, seqB, pw_align_opts):
    '''use pairwise2 with any options
    For information on pairwise2 please see https://biopython.org/docs/1.75/api/Bio.pairwise2.html.
    Arguments:
        seqA (str): first sequence to align (ref)
        seqB (str): second sequence to align (seq)
        pw_align_opts (dict): dictionary of alignment options. Keys are:
            "alignment_type": "local" or "global". First, you can give a score for a
            a match and mismatch ("match_score" and "mismatch_score") or
            specify a dictionary that has scores for every possible pair
            ("score_match_dict"). Or you can specify a function for 
            how to score pairing on ref and seq ("score_match_func_ref" 
            and "score_match_func_seq"). Second, you can decide how
            gaps are scored. Either a set open and extend score (
            "gap_open_score" and "gap_extend_score") or you can vary
            how gaps are scored on the reference sequence and input sequence 
            ("gap_open_score_ref","gap_extend_score_ref","gap_open_score_seq",
            "gap_extend_score_seq"). Finally you can define your own
            arbitary functions ("score_gap_func_ref","score_gap_func_seq")
    Returns:
        Alignment objects
    '''
    # Check all inputrs
    assert (pw_align_opts['alignment_type'] in ["local", "global"]
            ), "alignment_type must be global or local"
    if 'match_score' in pw_align_opts or 'mismatch_score' in pw_align_opts:
        assert ('mismatch_score' in pw_align_opts and 'match_score' in
                pw_align_opts), "Need match score and mismatch score."
        assert ('score_match_dict' not in pw_align_opts and
                'score_match_func_ref' not in pw_align_opts and
                'score_match_func_seq' not in pw_align_opts), "Can only either provide a match and mismatch score, a dictionary with match scores, or functions to calculate match scores"
        match_type = 'm'
        match_input = f",{pw_align_opts['match_score']},{pw_align_opts['mismatch_score']}"
    elif 'score_match_dict' in pw_align_opts:
        assert ('score_match_func_ref' not in pw_align_opts and
                'score_match_func_seq' not in pw_align_opts), "Can only either provide a match and mismatch score, a dictionary with match scores, or functions to calculate match scores"
        match_type = 'd'
        match_input = f",{pw_align_opts['score_match_dict']}"
    elif 'score_match_func_ref' in pw_align_opts or 'score_match_func_seq' in pw_align_opts:
        assert('score_match_func_ref' in pw_align_opts and
               'score_match_func_seq' in pw_align_opts), "Need match score function for reference and sequence."
        match_type = 'c'
        match_input = f",{pw_align_opts['score_match_func_ref']},{pw_align_opts['score_match_func_seq']}"
    else:
        match_type = 'x'
        match_input = ""
    if 'gap_open_score' in pw_align_opts or 'gap_extend_score' in pw_align_opts:
        assert ('gap_open_score' in pw_align_opts and 'gap_extend_score' in pw_align_opts), "Need a score for gap opnening and extension."
        assert ('gap_open_score_ref' not in pw_align_opts and 'gap_extend_score_ref' not in pw_align_opts and
                'gap_open_score_seq' not in pw_align_opts and 'gap_extend_score_seq' not in pw_align_opts and
                'score_gap_func_ref' not in pw_align_opts and 'score_gap_func_seq' not in pw_align_opts), "Can only either provide a extend and open gap score for both strands, for each strand individual or function to calculate gap scores."
        gap_type = "s"
        gap_input = f",{pw_align_opts['gap_open_score']},{pw_align_opts['gap_extend_score']}"
    elif ('gap_open_score_ref' in pw_align_opts or 'gap_extend_score_ref' in pw_align_opts or
            'gap_open_score_seq' in pw_align_opts or 'gap_extend_score_seq' in pw_align_opts):
        assert ('gap_open_score_ref' in pw_align_opts and 'gap_extend_score_ref' in pw_align_opts and
                'gap_open_score_seq' in pw_align_opts and 'gap_extend_score_seq' in pw_align_opts), "Need a gap open and gap extend score for reference and sequence."
        assert ('score_gap_func_ref' not in pw_align_opts and 'score_gap_func_seq' not in pw_align_opts), "Can only either provide a extend and open gap score for both strands, for each strand individual or function to calculate gap scores."
        gap_type = "d"
        gap_input = f",{pw_align_opts['gap_open_score_ref']},{pw_align_opts['gap_extend_score_ref']},{pw_align_opts['gap_open_score_seq']},{pw_align_opts['gap_extend_score_seq']}"
    elif 'score_gap_func_ref' in pw_align_opts or 'score_gap_func_seq' in pw_align_opts:
        assert ('score_gap_func_ref' in pw_align_opts and 'score_gap_func_seq' in pw_align_opts), "Need a gap score function for reference and sequence."
        gap_type = "c"
        gap_input = f",{pw_align_opts['score_gap_func_ref']},{pw_align_opts['score_gap_func_seq']}"
    else:
        gap_type = "x"
        gap_input = ""

    # run alignment using correct funciton and inputs
    align = {}
    exec("a= pairwise2.align."+pw_align_opts['alignment_type']+match_type+gap_type +
         "('"+seqA+"','"+seqB+"'"+match_input+gap_input+",one_alignment_only=True)", {'pairwise2': pairwise2}, align)
    return align['a']


def get_all_chain_alignments_for_table_multi(seq_id, input_table, reference_seqs, chain_names, pairwise_align_options, output_dir, min_length):
    '''
    Gets alignments for all matching chains of all pdbs in table, given a list
    of reference_seqs and their corresponding reference chains.
    Chains that do not align to any chain listed in Non_aligned_chains
    '''
    output = input_table.copy()
    never_aligned = None
    all_chains = None

    # For each reference chain get all chain alignments
    for seq, chain in zip(reference_seqs, chain_names):
        search_sum_file = get_nonexistant_file(Path(output_dir, f'sequence_search_summary_{chain}.fasta'))
        output, non_aligned = get_all_chain_alignments_for_table(f"{search_sum_file}", seq_id, output, seq, chain, pairwise_align_options, min_length)

        # Chains not aligned to any of the references make it to never_aligned
        if never_aligned is None:
            never_aligned = pd.Series(non_aligned)
        else:
            never_aligned = never_aligned.combine(
                pd.Series(non_aligned), func=get_overlap)
        if all_chains is None:
            all_chains = output[f"Align ref {chain}: order of chains"]
        else:
            all_chains = all_chains.combine(output[f"Align ref {chain}: order of chains"], func=get_union)

    # # Add column for chains that are not aligned to any refernce
    # output["Align: Non-aligned chains"] = never_aligned
    # output['Chain IDs'] = all_chains
    # fix (10/18/2022): unknown issue pd.Series not able to assign to df column
    output["Align: Non-aligned chains"] = never_aligned.tolist()
    output['Chain IDs'] = all_chains.tolist()
    return output


def get_all_chain_alignments_for_table(search_output_fasta, seq_id, input_table, reference_seq, chain_name, pairwise_align_options, min_length):
    '''
    Given a table with pdbs, adds columns indicating how each chain of each
    pdb aligns to the given reference sequence.
    '''
    chains = f'Align ref {chain_name}: order of chains'
    seq_align_col = f'Align ref {chain_name}: formatted alignment'
    ref_seq_aligned_col = f"Align ref {chain_name}: aligned ref sequence"
    aligned_col = f"Align ref {chain_name}: aligned sequence"
    num_matches_col = f"Align ref {chain_name}: number of AA matches to ref"
    chain_al = f"Align ref {chain_name}: chains aligned"
    ali_score = f"Align ref {chain_name}: pairwise align score"

    # For each row get all relevant chain alignments
    output = input_table.copy()
    output[chains], output[seq_align_col], output[ref_seq_aligned_col], output[aligned_col], output[ali_score], output[
        num_matches_col], output[chain_al], non_aligned = zip(*output['Entry ID'].map(lambda pdb: get_all_chain_alignments(search_output_fasta, seq_id, pdb, reference_seq, pairwise_align_options, min_length)))
    return output, non_aligned


def get_all_chain_alignments(search_output_fasta, seq_id, pdbid, reference_seq, pairwise_align_options, min_length):
    """
    given a pdb id and a reference sequence, returns the alignment
    of every chain of that pdb with the reference sequence
    - chain names
    - formatted alignments output
    - alignments of the reference sequence
    - alignments of the sequences
    - number of AA matches to the refrence chain
    """

    # Get sequence of the structure
    struct = get_pdb_struct(pdbid)
    if struct is None:
        return "", "", "", "", "", "", "", "", ""
    seq = struct.amino3to1()

    # set-up data-collection
    chain, output, ref_aligned, aligned, align_score, num_matches, aligned_chain, non_aligned = [
    ], [], [], [], [], [], [], []

    # For each chain if it is a aligned chain do pairwise
    # sequence alignment with reference
    for chain_id in seq['chain_id'].unique():

        # Get the chains sequence
        seqB = ''.join(
            seq.loc[seq['chain_id'] == chain_id, 'residue_name'])

        # Align according to inputted params
        alignments = specific_pairwise2_align(
            reference_seq, seqB, pairwise_align_options)

        # If alignment successful and to output
        if alignments == []:
            output.append("")
            ref_aligned.append("")
            aligned.append("")
            align_score.append("")
            num_matches.append("")
            non_aligned.append(chain_id)
            chain.append(chain_id)
            continue
        else:
            output.append(format_alignment(
                *alignments[0], full_sequences=True))
            ref_aligned.append(alignments[0].seqA)
            aligned.append(alignments[0].seqB)
            align_score.append(alignments[0].score)
            matches = format_alignment(
                *alignments[0], full_sequences=True).count("|")
            num_matches.append(matches)
            if (matches/min(len(reference_seq), max(min_length, len(seqB)))) >= seq_id:
                aligned_chain.append(chain_id)
                f = open(search_output_fasta, "a")
                f.write(f">{pdbid}_{chain_id}\n{seqB}\n")
                f.close()
            else:
                non_aligned.append(chain_id)
            chain.append(chain_id)

    # Format output and return
    chain = get_table_entry_from_list(chain)
    output = get_table_entry_from_list(output)
    ref_aligned = get_table_entry_from_list(ref_aligned)
    aligned = get_table_entry_from_list(aligned)
    align_score = get_table_entry_from_list(align_score)
    num_matches = get_table_entry_from_list(num_matches)
    aligned_chain = get_table_entry_from_list(aligned_chain)
    non_aligned = get_table_entry_from_list(non_aligned)

    return chain, output, ref_aligned, aligned, align_score, num_matches, aligned_chain, non_aligned
