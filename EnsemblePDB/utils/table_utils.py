""" EnsemblePDB.utils.table_utils

Help interpretting the csv output format, specifically with how lists
of values are represented in the table.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

from re import compile
from os import listdir


def get_list_from_table_entry(s):
    ''' get a tbale entry is list form
    Arguments:
        s (str): the table entry you want the list of
    Returns:
        the list representation of s
    '''
    return str(s).split(" ~ ")


def get_table_entry_from_list(l):
    ''' from a list get the string to represent it in a table
    Arguments:
        l (list): the list you want to put into table
    Returns:
        the string representation of l
    '''
    return " ~ ".join([str(a) for a in l])


def delete_all_one_value(s):
    '''
    Given table entry, return  one value if all values
    are identical, otherwise orginal entry.

    Arguments: 
        s (str): a table entry
    Returns: 
        either the original string or reduced to the one repeated value
    '''
    l = get_list_from_table_entry(s)
    entries = [x.strip() for x in l]
    if len(set(entries)) == 1:
        s = entries[0]
    return s


def get_overlap(x1, x2):
    ''' Given 2 table entries find their intersection

    Arguments: 
        x1 (str): the first table entry
        x2 (str): the second table entry
    Returns: 
        The element that are both in x1 and x2 in their table entry representation
    '''
    l1 = get_list_from_table_entry(x1)
    l2 = get_list_from_table_entry(x2)
    overlap = [chain for chain in l1 if chain in l2]
    return get_table_entry_from_list(overlap)


def get_union(x1, x2):
    ''' Given 2 table entries find their union

    Arguments: 
        x1 (str): the first table entry
        x2 (str): the second table entry
    Returns: 
        The union without repeating their interection in table entry representation
    '''
    l1 = get_list_from_table_entry(x1)
    l2 = get_list_from_table_entry(x2)
    overlap = [chain for chain in l1 if chain in l2]
    l1_only = [chain for chain in l1 if chain not in l2]
    l2_only = [chain for chain in l2 if chain not in l1]
    return " ~ ".join(overlap+l1_only+l2_only)


def get_reference_chains(df):
    ''' Given a ensemble dataframe return which are reference chains

    Arguments: 
        df (Pandas DataFrame): pandas dataframe that has columns "Align ref *: order of chains"
            where * are the reference chains
    Returns: 
        a list of reference chains in df
    '''
    cols = list(df.columns)
    r = compile('.*order of chains')
    ref_chains = []
    for col in filter(r.match, cols):
        ref_chains.append(col.split(":")[0][10:])
    return ref_chains


def get_nested_lists(table_entry):
    '''
    Apply on table entry with + and ~ to get original nested list.

    Argument: 
        table_entry (str): a string with groups separated by + and ~
    Returns:
        nested list
    '''
    l_of_l = []
    for l in str(table_entry).split(' ~ '):
        l_of_l.append(str(l).split(' + '))
    return l_of_l


def reformat_dict(d):
    '''
    Create csv friendly entries of dictionaries
    '''
    entries = []
    for key, val in d.items():
        entries.append(str(key) + ' : ' + str(val))
    return ' ~ '.join(entries)


def get_dict(reformatted_dict):
    '''
    Get dictionary from reformatted csv entry
    '''
    d = {}
    list_of_map = reformatted_dict.split('~')
    for _map in list_of_map:
        original = _map.split(':')[0].strip()
        renamed = _map.split(':')[1].strip()
        d[original] = renamed
    return d


def reformat_nested_lists(l_of_l):
    '''
    Apply on nested lists to get csv friendly format (str separated by + and ~)

    Arguments:
        l_of_l (list of lists of str)
    Returns:
        str
    '''
    l_of_c = []
    for l in l_of_l:
        l_of_c.append(' + '.join(l))
    return ' ~ '.join(l_of_c)


def contains_keyword(x, name, col='rscb-polymer-entity: description', index=0):
    '''
    Apply on rows of dataframes, filter any entry (row) where description of chain does not contain the input content.
    Assumes the first polymer entity is always the major protein (not ligand).
    Case insentitive.
    Arguments:
        x (row of dataframe)
        name (str): the keyword to filter with.
        col (str): column to filter the keyword on
    Returns:
        bool
    '''
    words = x[col].split('~')[0].lower()
    if name.lower() in words:
        return True
    else:
        return False


def wrong_organism(x, orgs):
    '''
    Apply on rows of dataframes to check whether the protein is from a given organism
    Arguments:
        x (row of dataframe)
        name (str): protein name.
        org (list): organism scientific names
    Returns:
        bool
    '''
    organisms = x['pdbx-entity: organism scientific name'].lower().split('~')
    wrong_org = True
    for org in orgs:
        if org.lower() in organisms[0].lower():
            wrong_org = False
    return wrong_org


def check_mutations(x, max_muts):
    '''
    Apply on rows of dataframes to check whether the protein has mutation exceeding (>=) max mutation limit
    Arguments:
        x (row of dataframe)
        name (str): protein name.
        max_muts: maximum mutation number
    Returns:
        bool
    '''
    mutations = str(x['entity-polymer: mutation_count']).split('~')
    if int(mutations[0]) > max_muts:
        return True
    else:
        return False


def match_names(pdb_dir, summary_df):
    '''
    Check if summary dataframe entries are in the PDB directory; assumes the dataframe has "Entry ID" column.
    '''
    Entry_IDs = summary_df['Entry ID'].tolist()
    pdbs = ['.'.join(x.split('.')[:-1]) for x in listdir(pdb_dir)]
    to_drop = []
    for x in Entry_IDs:
        if (x not in pdbs) and (x.lower() not in pdbs) and (x.upper() not in pdbs):
            to_drop.append(x)
    return to_drop
