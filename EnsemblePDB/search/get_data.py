""" pseudo_ensemble.search.get_data

Get information about PDB entries.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), Siyuan Du, Jacob Parres-Gold

Last edited:
    2020-01-14

Last reviewed:
    2021-05-11 SD

Todo:
    print warning when cannot find data column?

    check if same clusters predicted once PDB get its cluster running again...

    TODO Check if stucture comparison worth it

    TODO try to do search just the once??
    AAA = pypdb.search_client.QueryGroup(
    logical_operator=search_client.LogicalOperator.OR, queries=[search_operator])
    prob have to change hit inerpretation a little...
"""
from pathlib import Path
# from pdb import Pdb
import requests
import json
import os
from tqdm import tqdm

import pandas as pd
# from Bio import AlignIO
# import numpy as np
from glob import glob

from EnsemblePDB.utils import file_management, table_utils

###############################################################################
# Global data
###############################################################################

# data can include any of these: https://data.rcsb.org/data-attributes.html

minimum_info = {
    'citation.pdbx_database_id_doi': 'citation: doi',
    'citation.title': 'citation: title',
    'citation.year': 'citation: year',
    'diffrn.ambient_temp': 'expt: diffrn temp',
    'exptl_crystal_grow.method': 'expt: crystal grow method',
    'exptl_crystal_grow.p_h': 'expt: crystal grow pH',
    'refine.ls_rfactor_rwork': 'refine: Rwork',
    'refine.ls_rfactor_rfree': 'refine: Rfree',
    'refine.ls_rfactor_all': 'refine: R all',
    'refine.ls_dres_high': 'refine: resolution high',
    'refine.ls_dres_low': 'refine: resolution low',
    'rcsb_entry_info.resolution_combined': 'rscb-entry: resolution',
    'rcsb_entry_info.molecular_weight': 'rcsb-entry: molecular weight',
    'rcsb_entry_info.polymer_molecular_weight_maximum': 'rcsb-entry: polymer molecular weight max',
    'rcsb_entry_info.polymer_molecular_weight_minimum': 'rcsb-entry: polymer molecular weight min',
    'rcsb_entry_info.polymer_entity_count': 'rcsb-entry: polymer entity count',
    'rcsb_entry_info.polymer_entity_count_protein': 'rcsb-entry: polymer entity protein count',
    'rcsb_entry_info.nonpolymer_entity_count': 'rcsb-entry: nonpolymer entity count',
    'refine.biso_mean': 'refine_bfactor_mean',
    'rcsb_polymer_entity.pdbx_description': 'rscb-polymer-entity: description',
    'rcsb_polymer_entity.pdbx_ec': 'rscb-polymer-entity: EC number',
    'rcsb_polymer_entity.pdbx_mutation': 'rscb-polymer-entity: mutation',
    'rcsb_entry_container_identifiers.entity_ids': 'rscb-entry: entity ids',
    'pdbx_entity_nonpoly.comp_id': 'pdbx-entity-nonpolymer: chem component id',
    'pdbx_entity_nonpoly.entity_id': 'pdbx-entity-nonpolymer: entity id',
    'pdbx_entity_nonpoly.name': 'pdbx-entity-nonpolymer: name',
    'rcsb_nonpolymer_entity.formula_weight': 'rcsb-nonpolymer-entity: formula weight',
    'rcsb_entity_source_organism.scientific_name': 'pdbx-entity: organism scientific name',
    'entity_poly.rcsb_mutation_count': 'entity-polymer: mutation_count'}

other_data = {'cell.angle_alpha': 'unit-cell-angle_alpha (deg)',
              'cell.angle_beta': 'unit-cell-angle_beta (deg)',
              'cell.angle_gamma': 'unit-cell-angle_gamma (deg)',
              'cell.length_a': 'unit-cell-length_a (A)',
              'cell.length_b': 'unit-cell-length_b (A)',
              'cell.length_c': 'unit-cell-length_c (A)',
              'cell.zpdb': 'unit-cell_number_of_chains_z',
              'exptl_crystal_grow.pdbx_details': 'expt-crystal-grow_details',
              'pdbx_vrpt_summary.bfactor_type': 'pdbx-bfactor_type',
              'pdbx_vrpt_summary.dccr': 'pdbx-dcc_R',
              'pdbx_vrpt_summary.dccrfree': 'pdbx-dcc_Rfree',
              'pdbx_vrpt_summary.dccrefinement_program': 'pdbx-dcc_refine_program',
              'pdbx_vrpt_summary.edsr': 'pdbx-eds_Rfactor',
              'pdbx_vrpt_summary.edsresolution': 'pdbx-eds_resolution_highlimit',
              'pdbx_vrpt_summary.edsresolution_low': 'pdbx-eds_resolution_lowlimit',
              'pdbx_vrpt_summary.fo_fc_correlation': 'pdbx-fo-fc_correlation',
              'pdbx_vrpt_summary.pdbresolution_low': 'pdbx-PDB_resolution_lowlimit',
              'pdbx_vrpt_summary.angles_rmsz': 'pdbx-Molprobity-RMSZ_angles',
              'pdbx_vrpt_summary.bonds_rmsz': 'pdbx-Molprobity-RMSZ_bonds',
              'rcsb_id': 'Uniprot_id',
              'rcsb_cluster_membership.cluster_id': 'cluster_id',
              'rcsb_cluster_membership.identity': 'cluster_identity',
              'rcsb_entry_info.deposited_modeled_polymer_monomer_count': 'rcsb-entry_number_monopolymers',
              'rcsb_entry_info.deposited_nonpolymer_entity_instance_count': 'rcsb-entry_number_nonpolymer_entity',
              'rcsb_entry_info.inter_mol_covalent_bond_count': 'rcsb-entry_intermol_covalent_bond_count',
              'rcsb_entry_info.inter_mol_metalic_bond_count': 'rcsb-entry_intermol_metal_bond_count',
              'rcsb_entry_info.solvent_entity_count': 'rcsb-entry_solvent_entity_count',
              'rcsb_uniprot_feature.description': 'feature_descriptions',
              'rcsb_uniprot_feature.feature_positions.beg_seq_id': 'feature_beg_seq_ids',
              'rcsb_uniprot_feature.feature_positions.end_seq_id': 'feature_end_seq_ids',
              'rcsb_uniprot_feature.type': 'feature_types',
              'refine-details': 'refine_details'}

other_data.update(minimum_info)

cols_to_group = [['rcsb_uniprot_feature.feature_positions.beg_seq_id', 'rcsb_uniprot_feature.type',
                  'rcsb_uniprot_feature.feature_positions.end_seq_id', 'rcsb_uniprot_feature.description']]


def get_pdb_data(ensemble_csv, desired_cols=minimum_info, label='MyProtein', info_list=["entry", "entity", "chem"], cols_to_group=[], batches=500, output_dir=None):
    '''
    Given a csv with PDB IDs in the 'Entry ID' column, downloads
    and merges summary reports for those PDB ids from the PDB.

    Arguments: 
        ensemble_csv (str): location of csv file with the PDB IDs
        desired_cols (dict): keys are the name of the PDB information, values are
            what you would like this column to be names
        label (str): Label to add to output file name
        info_list (list of str): list of what type of attributes you want
            note "entry" must be one of them
        cols_to_group (list of lists): Used if there are columns expected to have more than one entry, such as sequence annotations, where each
            site is expected to have a set of data: "type", "description," etc. These data will need to be paired per site, potentially 
            inserting "None" (str) if no data is available for a site to keep everything aligned. Each list contains strings of the column 
            names that will be included in that group. 
            Example: cols_to_group = [['rcsb_uniprot_feature.type','rcsb_uniprot_feature.description'], ...]
    Returns: 
        dataframe of the summary_report
        the saved dataframe (summary_report*.csv)
    '''
    # get pdb ids from csvs
    data = pd.read_csv(ensemble_csv)
    assert ('Entry ID' in data.columns), 'the inputted csv must contain a Entry ID column with pdb ids'

    # Make sure we add the columns they need to get data outside of entry data.
    if "entry" not in info_list:
        print("ERROR: entry needs to be included to get rest of data")
        return
    if "entity" in info_list or "entity: uniprot" in info_list:
        if 'rcsb_entry_container_identifiers.entity_ids' not in desired_cols:
            print("'rcsb_entry_container_identifiers.entity_ids' needed to obtain entity information adding to desired columns")
            desired_cols['rcsb_entry_container_identifiers.entity_ids'] = 'rcsb_entry_container_identifiers.entity_ids'
    if "instance" in info_list:
        if 'rcsb_entry_info.deposited_nonpolymer_entity_instance_count' not in desired_cols:
            print("'rcsb_entry_info.deposited_nonpolymer_entity_instance_count' needed to obtain entity information adding to desired columns")
            desired_cols['rcsb_entry_info.deposited_nonpolymer_entity_instance_count'] = 'rcsb_entry_info.deposited_nonpolymer_entity_instance_count'
        if 'rcsb_entry_info.deposited_polymer_entity_instance_count' not in desired_cols:
            print("'rcsb_entry_info.deposited_polymer_entity_instance_count' needed to obtain entity information adding to desired columns")
            desired_cols['rcsb_entry_info.deposited_polymer_entity_instance_count'] = 'rcsb_entry_info.deposited_polymer_entity_instance_count'
    if "assembly" in info_list:
        if 'rcsb_entry_container_identifiers.assembly_ids' not in desired_cols:
            print("'rcsb_entry_container_identifiers.assembly_ids' needed to obtain entity information adding to desired columns")
            desired_cols['rcsb_entry_container_identifiers.assembly_ids'] = 'rcsb_entry_container_identifiers.assembly_ids'
    if "chem" in info_list or "chem: drugbank" in info_list:
        assert("entity" in info_list), "Need to collect chem names from entity. Please include entity in info_list."
        if 'pdbx_entity_nonpoly.comp_id' not in desired_cols:
            print(
                "'pdbx_entity_nonpoly.comp_id' needed to obtain entity information adding to desired columns")
        if 'rcsb_entry_info.nonpolymer_entity_count' not in desired_cols:
            print("'rcsb_entry_info.nonpolymer_entity_count' needed to determine whether chem data exists, add to desired columns")
            desired_cols['pdbx_entity_nonpoly.comp_id'] = 'pdbx_entity_nonpoly.comp_id'

    info_not_needed = ["entry", "entry: pubmed", "entity",
                       "entity: uniprot", "instance", "assembly", "chem", "chem: drugbank"]
    for info_desired in info_list:
        if info_desired in info_not_needed:
            info_not_needed.remove(info_desired)
        else:
            print("ERROR: your info_list contains an invalid option.")
            return
    if not output_dir:
        output_dir = str(file_management.get_nonexistant_file(
            str(Path(ensemble_csv).parents[0])))
    # get pdb_id list
    pdb_ids = data['Entry ID'].values.tolist()
    all_info = {}

    for i in tqdm(range(len(pdb_ids)), total=len(pdb_ids), desc='Adding data'):
        pdb = pdb_ids[i]
        if "entry" not in info_not_needed:
            info = get_pdb_entry_data(pdb, desired_cols, cols_to_group)
        if "entry: pubmed" not in info_not_needed:
            pubmed_info = get_pubmed_entry_data(
                pdb, desired_cols, cols_to_group)
            info = combine_dict_info(info, pubmed_info, [])

        if "entity" not in info_not_needed or "entity: uniprot" not in info_not_needed:
            old_info = list(info.keys())
            entities = table_utils.get_list_from_table_entry(
                info['rcsb_entry_container_identifiers.entity_ids'])
            for entity in entities:
                if "entity" not in info_not_needed:
                    entity_info = get_pdb_entity_data(
                        pdb, entity, desired_cols, cols_to_group)
                    info = combine_dict_info(info, entity_info, old_info)
                if "entity: uniprot" not in info_not_needed:
                    uniprot_info = get_uniprot_entity_data(
                        pdb, entity, desired_cols, cols_to_group)
                    info = combine_dict_info(info, uniprot_info, old_info)

        if "instance" not in info_not_needed:
            old_info = list(info.keys())
            num_chains = int(info['rcsb_entry_info.deposited_nonpolymer_entity_instance_count'])+int(
                info['rcsb_entry_info.deposited_polymer_entity_instance_count'])
            chains = [chr(i) for i in range(65, 65+int(num_chains))]
            # these chains are pdb, author may have other, but pdb just list alphabetically.
            for chain in chains:
                chain_info = get_pdb_instance_data(
                    pdb, chain, desired_cols, cols_to_group)
                info = combine_dict_info(info, chain_info, old_info)

        if "assembly" not in info_not_needed:
            old_info = list(info.keys())
            assemblies = table_utils.get_list_from_table_entry(
                info['rcsb_entry_container_identifiers.assembly_ids'])
            for assembly in assemblies:
                assembly_info = get_assembly_data(
                    pdb, assembly, desired_cols, cols_to_group)
                info = combine_dict_info(info, assembly_info, old_info)

        if ("chem" not in info_not_needed or "chem: drugbank" not in info_not_needed) and\
                ("pdbx_entity_nonpoly.comp_id" in info.keys()):
            old_info = list(info.keys())
            if int(info['rcsb_entry_info.nonpolymer_entity_count']) > 0:
                chems = table_utils.get_list_from_table_entry(
                    info['pdbx_entity_nonpoly.comp_id'])
                for chem in chems:
                    if "chem" not in info_not_needed:
                        chem_info = get_chemcomp_data(
                            chem, desired_cols, cols_to_group)
                        info = combine_dict_info(info, chem_info, old_info)
                    if "chem: drugbank" not in info_not_needed:
                        drugbank_info = get_drugbank_chemcomp_data(
                            chem, desired_cols, cols_to_group)
                        info = combine_dict_info(info, drugbank_info, old_info)

        all_info[pdb] = info

        if (i != 0 and i % batches == 0) or (i == len(pdb_ids)-1):
            df = pd.DataFrame.from_dict(all_info, orient='index')
            df['Entry ID'] = df.index
            df = data.loc[data['Entry ID'].isin(df.index)].merge(
                df, how="outer", on="Entry ID")
            # if not output_dir:
            #     output_file = file_management.get_nonexistant_file(str(Path(ensemble_csv).parents[0])+'/summary_report_'+label+'_'+str(i)+'.csv')
            # else:
            output_file = output_dir + \
                '/summary_report_'+label+'_'+str(i)+'.csv'
            # print(f"\nSaving summary table {output_file}. Continue to filter.")
            # missing_cols = [col for col in desired_cols if col not in df.columns]
            # if len(missing_cols) > 0:
            #     print("WARNING: No data found for",missing_cols)
            df = df.applymap(table_utils.delete_all_one_value)
            df = df.rename(columns=desired_cols)

            df.to_csv(output_file, index=False)
            all_info = {}

    all_data = glob(output_dir + '/summary_report_'+label+'_'+'*.csv')
    # if len(all_data) > 1:
    to_concat = []
    for data in all_data:
        to_concat.append(pd.read_csv(data))
        os.remove(data)

    all_data_df = pd.concat(to_concat)
    all_data_df.to_csv(output_dir + '/summary_report_'+label+'.csv')
    return all_data_df

# https://data.rcsb.org/redoc/index.html
# https://data.rcsb.org/data-attributes.html
# https://search.rcsb.org/search-attributes.html

###############################################################################
# Helper functions
###############################################################################

# Get PDB data


def fetch_pdb_data(url, pdb, alt_url=None):
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
            print("Connection Error")
            if counter == 3:
                raise Exception("Too Many Connection Errors")
    if response is None or response.status_code != 200:
        if alt_url is not None:
            if len(alt_url) == 1:
                return fetch_pdb_data(alt_url[0], pdb)
            else:
                return fetch_pdb_data(alt_url[0], pdb, alt_url[1:])
        source = url.split("/")[6].split("_")[-1]
        # print ('Failed to retrieve data for:',source,pdb)
        return
    return json.loads(response.text)


def get_pdb_entry_data(pdb, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/entry/'
    return parse_dict_info(fetch_pdb_data(url, pdb), desired_cols, cols_to_group=cols_to_group)


def get_pubmed_entry_data(pdb, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
    return parse_dict_info(fetch_pdb_data(url, pdb), desired_cols, cols_to_group=cols_to_group)


def get_pdb_entity_data(pdb, entity, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/polymer_entity/'
    alt_urls = ['https://data.rcsb.org/rest/v1/core/nonpolymer_entity/',
                'https://data.rcsb.org/rest/v1/core/branched_entity/']
    return parse_dict_info(fetch_pdb_data(url, f"{pdb}/{entity}", alt_urls), desired_cols, cols_to_group=cols_to_group)


def get_uniprot_entity_data(pdb, entity, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/uniprot/'
    return parse_dict_info(fetch_pdb_data(url, f"{pdb}/{entity}"), desired_cols, cols_to_group=cols_to_group)


def get_pdb_instance_data(pdb, chain, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/polymer_entity_instance/'
    alt_urls = ['https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/',
                'https://data.rcsb.org/rest/v1/core/branched_entity_instance/']
    return parse_dict_info(fetch_pdb_data(url, f"{pdb}/{chain}", alt_urls), desired_cols, cols_to_group=cols_to_group)


def get_assembly_data(pdb, assembly, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/assembly/'
    return parse_dict_info(fetch_pdb_data(url, f"{pdb}/{assembly}"), desired_cols, cols_to_group=cols_to_group)


def get_chemcomp_data(comp_id, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/chemcomp/'
    return parse_dict_info(fetch_pdb_data(url, comp_id), desired_cols, cols_to_group=cols_to_group)


def get_drugbank_chemcomp_data(comp_id, desired_cols, cols_to_group=[]):
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
    url = 'https://data.rcsb.org/rest/v1/core/drugbank/'
    return parse_dict_info(fetch_pdb_data(url, comp_id), desired_cols, cols_to_group=cols_to_group)

# Intepret PDB data


def parse_dict_info(info, desired_cols, prefix="", cols_to_group=[], collapsed_dict=None, check_group=None):
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
    Output: 
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
    Output: 
        new_collapsed_dict with the information from temp added
    '''
    for key, value in temp.items():
        if key not in old_info:
            if key not in new_collapsed_dict:
                new_collapsed_dict[key] = value
            else:
                new_collapsed_dict[key] = table_utils.delete_all_one_value(
                    new_collapsed_dict[key]+" ~ "+str(value))
    return new_collapsed_dict
