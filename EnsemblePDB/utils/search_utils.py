""" EnsemblePDB.utils.search_utils

Utility function for PDB searches.

Authors:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

from rcsbsearchapi.search import SequenceQuery
import requests
import json
import pandas as pd
from EnsemblePDB.utils.table_utils import delete_all_one_value

def search_seq(sequence, seq_id, search_output_file, chain_name, evalue=0.1):
    '''
    Takes in a sequence and a threshold for sequence identity and return a
    table with sequences that reach the threshold with their pdb_id, chaiin
    that matches, and alignment information.
    Arguments: 
        sequence (str): the sequence to search for
        seq_id_min (float): the threshold sequence idenity to include
        chain_name (str): name of reference chain, used only for labeling
    Returns: 
        Pandas Dataframe: with the pdb id, alignment information including
            sequence and the chain(s) that matches
    '''

    # search_operator = pypdb.clients.search.operators.sequence_operators.SequenceOperator(sequence=sequence, evalue_cutoff=evalue, identity_cutoff=seq_id)
    # results = pypdb.search_client.perform_search_with_graph(
    #     query_object=search_operator, return_type=pypdb.search_client.ReturnType.POLYMER_ENTITY, return_raw_json_dict=True)
    results = SequenceQuery(sequence, evalue_cutoff=evalue, identity_cutoff=seq_id, sequence_type='protein')
    cols = ['Entry ID', 'sequence_identity', 'evalue', 'bitscore', 'alignment_length', 'mismatches', 'gaps_opened',
            'query_beg', 'query_end', 'subjnect_beg', 'subject_end', 'query_length', 'subject_length', 'query_aligned_seq', 'subject_aligned_seq']
    data = pd.DataFrame(columns=cols)
    for hit in list(results(results_verbosity='verbose', return_type='polymer_entity')):
        match = hit['services'][0]["nodes"][0]['match_context'][0]
        match["Entry ID"] = hit['identifier'][:4]
        data = data.append(match, ignore_index=True)
    # Write out search results
    f = open(search_output_file, "w")
    f.write(str(data))
    f.close()

    # Make data more readable and return
    pretty_col = {"Entry ID": "Entry ID",
                  "sequence_identity": f'Search ref {chain_name}: sequence_identity',
                  "evalue": f'Search ref {chain_name}: evalue',
                  "bitscore": f'Search ref {chain_name}: bitscore',
                  "alignment_length": f'Search ref {chain_name}: alignment_length',
                  "mismatches": f'Search ref {chain_name}: mismatches',
                  "gaps_opened": f'Search ref {chain_name}: gaps_opened',
                  "query_beg": f'Search ref {chain_name}: query_beg',
                  "query_end": f'Search ref {chain_name}: query_end',
                  "subject_beg": f'Search ref {chain_name}: subject_beg',
                  "subject_end": f'Search ref {chain_name}: subject_end',
                  "query_length": f'Search ref {chain_name}: query_length',
                  "subject_length": f'Search ref {chain_name}: subject_length',
                  "query_aligned_seq": f'Search ref {chain_name}: query_aligned_seq',
                  "subject_aligned_seq": f'Search ref {chain_name}: subject_aligned_seq'}
    data = data.rename(columns=pretty_col)
    data = data.astype(str)
    data = data.groupby('Entry ID').agg(" ~ ".join).reset_index()
    return data

def fetch_data_from_url(url, pdb, alt_url=None):
    ''' fetch data from a url

    Arguments:
        url (str): base url to fetch from
        pdb (str): the extension to url to fetch from
        alt_url (list str): if fail to fetch from original url try to alt urls

    Returns:
        the json information formatted as a dict
    '''
    counter = 0
    while counter < 4:
        try:
            counter += 1
            response = requests.get(url+pdb)
            break
        except requests.exceptions.ConnectionError:
            if counter == 3:
                source = url.split("/")[6].split("_")[-1]
                raise Exception("Too Many Connection Errors for:", source, pdb)
    if response is None or response.status_code != 200:
        if alt_url is not None:
            if len(alt_url) == 1:
                return fetch_data_from_url(alt_url[0], pdb)
            else:
                return fetch_data_from_url(alt_url[0], pdb, alt_url[1:])
        else:
            source = url.split("/")[6].split("_")[-1]
            print('Failed to retrieve data for:', source, pdb)
        return
    return json.loads(response.text)


def fetch_pdb_data(query_type, entry, desired_cols, cols_to_group=[]):
    ''' fetch PDB data regarding the pdb entry

    Arguments: 
        pdb (str): pbd id "XXXX"
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch pubmed data regarding the pdb entry

    Arguments: 
        pdb (str): pbd id "XXXX"
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch PDB data regarding the pdb entity

    Arguments: 
        pdb (str): pbd id "XXXX"
        entity (str): entity number
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch uniprot data regarding the pdb entity

    Arguments: 
        pdb (str): pbd id "XXXX"
        entity (str): entity number
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch PDB data regarding the pdb instance (aka chain)

    Arguments: 
        pdb (str): pbd id "XXXX"
        chain (str): instance (note not author labled chain but PDB labeled)
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch PDB data regarding the pdb assembly

    Arguments: 
        pdb (str): pbd id "XXXX"
        assembly (str): assembly number
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch PDB data regarding a chemical

    Arguments: 
        comp_id (str): unique id of that chemical
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    ''' fetch drugbank data regarding a chemical

    Arguments: 
        comp_id (str): unique id of that chemical
        desired_cols (dict): columns desired as keys
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]

    Returns:
        dict flatten with only information in desired_cols
    '''
    urls = {'pdb_entry': 'https://data.rcsb.org/rest/v1/core/entry/',
            'pubmed_entry': 'https://data.rcsb.org/rest/v1/core/uniprot/',
            'pdb_entity': 'https://data.rcsb.org/rest/v1/core/polymer_entity/',
            'uniprot_entity': 'https://data.rcsb.org/rest/v1/core/uniprot/',
            'pdb_instance': 'https://data.rcsb.org/rest/v1/core/polymer_entity_instance/',
            'assembly': 'https://data.rcsb.org/rest/v1/core/assembly/',
            'chemcomp': 'https://data.rcsb.org/rest/v1/core/chemcomp/',
            'drugbank_chemcomp': 'https://data.rcsb.org/rest/v1/core/drugbank/'}
    alt_urls = {'pdb_entry': None,
                'pubmed_entry': None,
                'pdb_entity': ['https://data.rcsb.org/rest/v1/core/nonpolymer_entity/',
                               'https://data.rcsb.org/rest/v1/core/branched_entity/'],
                'uniprot_entity': None,
                'pdb_instance': ['https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/',
                                 'https://data.rcsb.org/rest/v1/core/branched_entity_instance/'],
                'assembly': None,
                'chemcomp': None,
                'drugbank_chemcomp': None}
    url = 'https://data.rcsb.org/rest/v1/core/entry/'
    return parse_dict_info(fetch_data_from_url(urls[query_type], entry, alt_urls[query_type]), desired_cols,
                           cols_to_group=cols_to_group)


def parse_dict_info(info, desired_cols, prefix="", cols_to_group=[],
                    collapsed_dict=None, check_group=None):
    '''
    Given the raw output of a fetch from pdb recurssively flattens the
    the dictionary and making any multiple instances into a list string sep: (" ~ ")

    Arguments: 
        info (dict): raw output of pypdb.get_info
        desired_cols (dict): columns desired as keys
        prefic (str): only used recursively inside function
        collapsed_dict (dict): only used recursively inside function
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]
        check_group (None or list of bool): Variable used recursively inside function

    Returns: 
        collapsed dictionary with all same information
    '''
    if len(cols_to_group) > 0:
        col_group = {}
        shortest_col = {}
        for group in range(0, len(cols_to_group)):
            col_group[group] = ''
            shortest_col[group] = [col for col in cols_to_group[group] if len(
                col) == min([len(x) for x in cols_to_group[group]])][0]
            for char in range(0, len(shortest_col[group])):
                if sum([(col[char] == shortest_col[group][char]) for col in cols_to_group[group]]) == len(cols_to_group[group]):
                    col_group[group] += (shortest_col[group][char])
                else:
                    break
            if col_group[group][-1] == '.':
                col_group[group] = col_group[group][:-1]
    if collapsed_dict is None:
        collapsed_dict = {}
    if type(info) == dict:
        temp_collapsed_dict = {}
        for key, value in info.items():
            if prefix == "":
                next_prefix = prefix+key
            else:
                # . to match PDB website
                next_prefix = prefix+"."+key
            temp_collapsed_dict = combine_dict_info(temp_collapsed_dict,
                                                    parse_dict_info(value, desired_cols, next_prefix, cols_to_group, collapsed_dict=None), old_info=[])
        if check_group is not None:
            for group in range(0, len(cols_to_group)):
                if check_group[group]:
                    for col in [col for col in cols_to_group[group] if (col not in temp_collapsed_dict)]:
                        temp_collapsed_dict = combine_dict_info(temp_collapsed_dict,
                                                                {col: 'None'}, old_info=[])
        return temp_collapsed_dict
    elif type(info) == list:
        new_collapsed_dict = collapsed_dict.copy()
        old_info = list(collapsed_dict.keys())
        for group in range(0, len(cols_to_group)):
            if col_group[group] == prefix:
                check_group = {}
                check_group[group] = True
        for item in info:
            temp = parse_dict_info(
                item, desired_cols, prefix, cols_to_group, collapsed_dict, check_group)
            new_collapsed_dict = combine_dict_info(
                new_collapsed_dict, temp, old_info)
        return new_collapsed_dict
    else:
        if prefix in desired_cols:
            collapsed_dict[prefix] = str(info)
        return collapsed_dict


def combine_dict_info(new_collapsed_dict, temp, old_info):
    '''
    Given 2 dictionaries and a list of keys not to change
    creates a new dictionary with combined values

    Arguments: 
        new_collapsed_dict (dict): dictionary to add to
        temp (dict): dictionary to add from
        old_info (list of str): list of keys not to change

    Returns: 
        new_collapsed_dict with the information from temp added
    '''
    for key, value in temp.items():
        if key not in old_info:
            if key not in new_collapsed_dict:
                new_collapsed_dict[key] = value
            else:
                new_collapsed_dict[key] = delete_all_one_value(
                    new_collapsed_dict[key]+" ~ "+str(value))
    return new_collapsed_dict
