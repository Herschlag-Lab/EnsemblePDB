""" EnsemblePDB.search.query_pdb

Query PDB to get list of relevent PDB entries.

Authors:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

from pathlib import Path

import pandas as pd

from rcsbsearchapi.search import AttributeQuery, SequenceQuery
from rcsbsearchapi.const import STRUCTURE_ATTRIBUTE_SEARCH_SERVICE

from EnsemblePDB.utils.table_utils import delete_all_one_value
from EnsemblePDB.utils.sequence_alignment import get_all_chain_alignments_for_table_multi
from EnsemblePDB.utils.file_management import get_nonexistant_file
from EnsemblePDB.utils.biopandas_utils import get_pdb_struct, get_chain_seq
from EnsemblePDB.utils.search_utils import search_seq


def query_by_ref_pdb(reference_pdb, reference_chains, label='MyProtein',
                     macromolecule_name=None, seq_id=0.95,
                     xray_only=True, evalue=10,
                     pairwise_align_options={
                         'alignment_type': "local", 'gap_open_score': -0.5, 'gap_extend_score': -0.1},
                     output_dir=".", min_length=20):
    '''
    Will take sequence(s) or reference pdb chain(s) and will search 
    the PDB for structures of similar sequences. 

    Args: 
        reference_pdb (str): pdb id ('XXXX') to be the reference.
        reference_chains (list of str): list of the chains of the reference
            structure to use to seqeunce search.
        label (str): Label to add to end of file names
      (optional)
        macromolecule_name (str): name to search pdb by. May add structures to list 
            not found by sequence search, or identify pdb not
            labeled with the desired name. 
            Reported in column: "From macromolecular search"
        seq_id (float): if specified uses this number as the minimum percent identity 
            of sequences to be added instead of calculating from max_mut
        xray_only (bool): only return x-ray crystallographic structures
            {default: True}
        evalue (float): 
        pairwise_align_options, {default: {'alignment_type':"local",'gap_open_score':-0.5, 'gap_extend_score':-0.1}}
            Options for paiwise sequence alignment scoring.
            Can score alignment anyway with pairwise2:
            https://biopython.org/docs/1.75/api/Bio.pairwise2.html
            Dictionary keys can be:
                alignment_type="local", match_score, mismatch_score, score_match_dict,
                score_match_func_ref, score_match_func_seq,
                gap_open_score, gap_extend_score,
                gap_open_score_ref, gap_extend_score_ref, gap_open_score_seq, gap_extend_score_seq,
                score_gap_func_ref, score_gap_func_seq,
        output_dir (string): if not working in data directory you can specify where
                        to save output {default: "."}

    Returns: 
        Pandas dataframe: summary output of the similar pdb found, 
            including the chains that aligned with to 
            reference and additionally alignment to
            reference for all chains.
        saves the returned data frame to seq_search_output*.csv
        saves the raw output from PDB sequence search to 
        sequence_search_summary*.txt and sequence_search_summary*.fasta
    '''

    ############################## Check inputs ###############################
    assert (type(reference_pdb) == str), "reference_pdb must be a string."
    assert (len(reference_pdb) ==
            4), "Check your reference_pdb, pdbs are 4 characters in length."
    assert (type(reference_chains) ==
            list), "reference_pdb_chains must be a list, eg ['A']"
    assert (all(isinstance(s, str)
                for s in reference_chains)), "Chains must be strings."
    # # print(f"A reference pdb was inputted. Will search PDB for structures"
    #       f" that align to sequence(s) of {reference_pdb} chains "
    #       f"{reference_chains}.")
    if macromolecule_name is not None:
        assert (type(macromolecule_name) ==
                str), "macromolecule_name must be a string."
    if seq_id is not None:
        assert (seq_id >= 0 and seq_id <=
                1), "seq_id must be a percentage (0-1)"
    assert (type(xray_only) == bool), "xray_only must be a bool."
    assert (isinstance(evalue, (int, float))
            ), "evalue should be a number."
    assert(type(pairwise_align_options) ==
           dict), "pairwise_align_options must be a dictionary."
    assert(Path(output_dir).is_dir()), "Output directory must exist"
    ############################## Check inputs ###############################

    # get list of all x-ray structures
    if not xray_only:
        print("Obtaining all structures, including structures from "
              "methods other than X-ray crystallography.")
        xray_pdbs = []
    else:
        print("Looking for X-ray crystallographic structures.")
        query = AttributeQuery("exptl.method", "exact_match", "X-RAY DIFFRACTION",
                    STRUCTURE_ATTRIBUTE_SEARCH_SERVICE # this constant specifies "text" service
                    )
        xray_pdbs = list(query())
        if not xray_pdbs:
            print("No PDB structures found.")
            return
        # assert (xray_pdbs is not None), "PDB search failed, try again."

    # if macrmolecule_name specified get list of pdbs with macromolecular name
    if macromolecule_name is not None:
        print(f"Looking for structures with name {macromolecule_name}")
        search_operator = AttributeQuery(value=macromolecule_name, operator='contains_phrase',
            attribute="rcsb_polymer_entity.rcsb_macromolecular_names_combined.name")
        name_pdbs = list(search_operator('polymer_entity'))
        if not name_pdbs:
            print("No PDB structures found.")
            return
        # assert (name_pdbs is not None), "PDB macromolecule search failed, try again"

        # only keep structures that are x-ray structures
        if xray_only:
            pdb_list = [pdb for pdb in xray_pdbs if pdb in name_pdbs]
        else:
            pdb_list = name_pdbs

        # write dataframe
        macro_output = pd.DataFrame(pdb_list, columns=['Entry ID'])
        macro_output["From macromolecular search"] = True
    else:
        macro_output = None
        pdb_list = xray_pdbs

    # get the sequence of the reference chain(s)
    print(f"Extracting sequence(s) of {reference_pdb}.")
    structure = get_pdb_struct(reference_pdb)
    sequences = []
    for ref_chain in reference_chains:
        sequences.append(get_chain_seq(structure, ref_chain))
    if len(sequences) > 1:
        i = 0
        while i < len(sequences):
            if sequences[i] in sequences[:i]+sequences[i+1:]:
                print(
                    "WARNING: some of your sequences are redudant! You will get redudant information!")
            i += 1

    # search the pdb for the sequence and return those with >=seq_id
    seq_chain_list = None
    for i, seq in enumerate(sequences):
        search_sum_file = get_nonexistant_file(Path(output_dir, f'sequence_search_summary_{reference_chains[i]}.txt'))
        print(f"Saving sequence search raw output in {search_sum_file}.")
        temp_list = search_seq(seq, seq_id, f"{search_sum_file}",
                               reference_chains[i], evalue=evalue)

        if seq_chain_list is None:
            seq_chain_list = temp_list
        else:
            seq_chain_list = seq_chain_list.merge(temp_list, how='outer',
                                                  on='Entry ID')

    # only take x-ray structures
    if xray_only:
        output = seq_chain_list[seq_chain_list['Entry ID'].isin(pdb_list)]
    else:
        output = seq_chain_list

    if macro_output is not None:
        output = output.merge(macro_output, how="outer", on="Entry ID")

    # get all chains and there alignments
    print("Adding all chain alignment information. For large datasets this may"
          " take some time.")
    output = get_all_chain_alignments_for_table_multi(seq_id, output, sequences,
                                                      reference_chains, pairwise_align_options,
                                                      output_dir, min_length)
    output = output.applymap(delete_all_one_value)

    # save and return output
    output_file = get_nonexistant_file(
        Path(output_dir, "seq_search_output_"+label+".csv"))

    print(f"Saved output to {output_file}. To obtain PDB summary info, run "
          f"EnsemblePDB.search.get_data.get_pdb_data('{output_file}')")
    output.to_csv(output_file, index=False)
    return output


def query_by_sequence(sequences,
                      macromolecule_name=None,  label='MyProtein', seq_id=0.95,
                      xray_only=True, evalue=10,
                      pairwise_align_options={'alignment_type': "local",
                                              'gap_open_score': -0.5,
                                              'gap_extend_score': -0.1},
                      output_dir=".", min_length=20):
    '''
    Will take sequence(s) or reference pdb chain(s) and will search 
    the PDB for structures of similar sequences. 
    Args: 
        sequences (list of str): list of sequence(s) to search on.
      (optional)
        macromolecule_name (str): name to search pdb by. May add structures to list 
            not found by sequence search, or identify pdb not
            labeled with the desired name. 
            Reported in column: "From macromolecular search"
        seq_id (float): if specified uses this number as the minimum percent identity 
            of sequences to be added instead of calculating from max_mut
        xray_only (bool): only return x-ray crystallographic structures
            {default: True}
        evalue (float): 
        pairwise_align_options, {default: {'alignment_type':"local",'gap_open_score':-0.5, 'gap_extend_score':-0.1}}
            Options for paiwise sequence alignment scoring.
            Can score alignment anyway with pairwise2:
            https://biopython.org/docs/1.75/api/Bio.pairwise2.html
            Dictionary keys can be:
                alignment_type="local", match_score, mismatch_score, score_match_dict,
                score_match_func_ref, score_match_func_seq,
                gap_open_score, gap_extend_score,
                gap_open_score_ref, gap_extend_score_ref, gap_open_score_seq, gap_extend_score_seq,
                score_gap_func_ref, score_gap_func_seq,
        output_dir, if not working in data directory you can specify where
                        to save output {default: "."}
    Returns: 
        Pandas dataframe: summary output of the similar pdb found, 
            including the chains that aligned with to 
            reference and additionally alignment to
            reference for all chains.
        saves the returned data frame to seq_search_output*.csv
        saves the raw output from PDB sequence search to 
        sequence_search_summary*.txt and sequence_search_summary*.fasta
    '''

    ############################## Check inputs ###############################
    assert (type(sequences) ==
            list), "Sequences must be a list even if with one element."
    assert (all(isinstance(s, str)
                for s in sequences)), "Sequences must be strings."
    # print(f"Sequence(s)inputted. Will search PDB for "
    #       f"{len(sequences)} sequence(s).")
    if macromolecule_name is not None:
        assert (type(macromolecule_name) ==
                str), "macromolecule_name must be a string."
    if seq_id is not None:
        assert (seq_id >= 0 and seq_id <=
                1), "seq_id must be a percentage (0-1)"
    assert (type(xray_only) == bool), "xray_only must be a bool."
    assert (isinstance(evalue, (int, float))
            ), "evalue should be a number."
    assert(type(pairwise_align_options) ==
           dict), "pairwise_align_options must be a dictionary."
    assert(Path(output_dir).is_dir()), "Output directory must exist"
    ############################## Check inputs ###############################

    # get list of all x-ray structures
    if not xray_only:
        # print("WARNING: obtaining all structures, including structures from "
        #   "methods other than X-ray crystallography.")
        xray_pdbs = []
    else:
        # print("Looking for X-ray crystallographic structures.")
        query = AttributeQuery("exptl.method", "exact_match", "X-RAY DIFFRACTION",
                    STRUCTURE_ATTRIBUTE_SEARCH_SERVICE # this constant specifies "text" service
                    )
        xray_pdbs = list(query())
        assert (xray_pdbs is not None), "PDB search failed, try again."

    # if macrmolecule_name specified get list of pdbs with macromolecular name
    if macromolecule_name is not None:
        print(f"Looking for structures with name {macromolecule_name}")
        search_operator = AttributeQuery(value=macromolecule_name, operator='contains_phrase',
            attribute="rcsb_polymer_entity.rcsb_macromolecular_names_combined.name")
        name_pdbs = list(search_operator('polymer_entity'))
        # assert (name_pdbs is not None), "PDB macromolecule search failed, try again"
        if not name_pdbs:
            print("No PDB structures found.")
            return
        # only keep structures that are x-ray structures
        if xray_only:
            pdb_list = [pdb for pdb in xray_pdbs if pdb in name_pdbs]
        else:
            pdb_list = name_pdbs

        # write dataframe
        macro_output = pd.DataFrame(pdb_list, columns=['Entry ID'])
        macro_output["From macromolecular search"] = True
    else:
        macro_output = None
        pdb_list = xray_pdbs

    # get chain names for sequence only queries
    reference_chains = []
    for i in range(len(sequences)):
        reference_chains.append("inputted_seq_" + str(i))
    if len(sequences) > 1:
        i = 0
        while i < len(sequences):
            if sequences[i] in sequences[:i]+sequences[i+1:]:
                print(
                    "WARNING: some of your sequences are redudant! You will get redudant information!")
            i += 1

    # search the pdb for the sequence and return those with >= seqid
    seq_chain_list = None
    for i, seq in enumerate(sequences):
        search_sum_file = get_nonexistant_file(Path(output_dir, f'sequence_search_summary_{reference_chains[i]}.txt'))
        print(f"Saving sequence search raw output in {search_sum_file}.")
        temp_list = search_seq(seq, seq_id, f"{search_sum_file}", reference_chains[i], evalue=evalue)

        if seq_chain_list is None:
            seq_chain_list = temp_list
        else:
            seq_chain_list = seq_chain_list.merge(temp_list, how='outer',
                                                  on='Entry ID')

    # only take x-ray structures
    if xray_only:
        output = seq_chain_list[seq_chain_list['Entry ID'].isin(pdb_list)]
    else:
        output = seq_chain_list

    if macro_output is not None:
        output = output.merge(macro_output, how="outer", on="Entry ID")

    # get all chains and their alignments
    print("Adding all chain alignment information. For large datasets this may"
          " take some time.")
    output = get_all_chain_alignments_for_table_multi(seq_id, output, sequences,
                                                      reference_chains, pairwise_align_options,
                                                      output_dir, min_length)
    output = output.applymap(delete_all_one_value)

    # save and return output
    output_file = get_nonexistant_file(
        Path(output_dir, "seq_search_output_"+label+".csv"))

    print(f"Saved output to {output_file}. To obtain PDB summary info, run "
          f"EnsemblePDB.search_pdb.get_pdb_data('{output_file}')")
    output.to_csv(output_file, index=False)
    return output