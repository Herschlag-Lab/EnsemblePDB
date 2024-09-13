""" EnsemblePDB.utils.anlysis_utils

Basic functions in analysis atomic positions.

Author:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
from functools import reduce
from sklearn.utils import resample
from EnsemblePDB.utils.file_management import get_nonexistant_file


def calculate_msd(list_of_coords):
    '''
    Arguments:
        list/array-like object with x,y or z coordinates of one atom

    Out:
        array of mean square displacement calculated using each point 
        as reference(center)
    '''
    msd_list = []

    list_of_coords = list_of_coords.tolist()

    for pos in list_of_coords:
        displacement = []
        for coord in list_of_coords:
            displacement.append((coord - pos) ** 2)
        msd = np.mean(displacement)
        msd_list.append(msd)

    return np.array(msd_list)


def calculate_rmsd(group):
    '''
    Arguments:
        grouped atoms coordinates

    Returns:
        Dataframe of RMSD for each group
    '''
    xyz_spread = (calculate_msd(group['x_coord']) +
                  calculate_msd(group['y_coord']) +
                  calculate_msd(group['z_coord']))
    min_msd = xyz_spread.min()
    rmsd = np.sqrt(min_msd)

    return rmsd


def bootstrap_analysis(group, iter, apply_func=None):
    '''
    Apply on a group object (one atom) to perform bootstrap analysis 
    and returns a standard deviation from the bootstrap distribution.

    Arguments:
        group, a group in pandas groupby object
        iter, number of iterations to perform bootstrapping
        apply_func: function to apply on resampled data {default: calculate_rmsd}

    Returns:
        standard deviation of the bootstrap distribution
    '''
    data = []
    for i in range(iter):
        sample = resample(group)
        # for each sample calculate MDev
        if apply_func:
            sample = apply_func(sample)
        data.append(sample)
    return np.std(data)


# def calculate_MDev(df, bootstrap, iter):
#     '''
#     Arguments:
#         df, dataframe of atom coordinates

#     Returns:
#         MDevs, dataframe with MDev and count
#     '''
#     grouped = df.groupby(['residue_number', 'insertion',
#                           'residue_name', 'atom_name'])
#     MDev = grouped.progress_apply(lambda x: calculate_rmsd(x))
#     size = grouped['Entry ID'].count()
#     to_concat = [MDev, size]
#     if bootstrap:
#         tqdm.pandas(desc=f'Perform bootstrap analysis')
#         boot = grouped.progress_apply(lambda x: bootstrap_analysis(x, iter,apply_func=calculate_rmsd))
#         to_concat.append(boot)
#     MDevs = pd.concat(to_concat, axis=1)
#     MDevs = pd.DataFrame(MDevs).reset_index()\
#         .rename(columns={0: 'MDev',
#                          'Entry ID': 'n',
#                          1: 'std (bootstrap)'})
#     return MDevs


# def combine_save_MDev(df, chains, reference_PDB, save_to, bootstrap, iter):
#     '''
#     Arguments:
#         df, dataframe of atom coords
#         chains, list of strings (which chains to calculate MDev on)
#     Returns:
#         dictionary: key = chain ID, value = dataframe of MDev of each atom
#         saves a csv if save_to is not None
#     '''
#     MDevs = {}

#     for chain in chains:
#         df_temp = df.loc[df['chain_id'] == chain]
#         # print('Calculating MDev for chain {}...'.format(chain))
#         tqdm.pandas(desc=f'Calculating atomic MDevs for chain {chain}')
#         MDev = calculate_MDev(df_temp, bootstrap, iter)

#         # get rid of mutants
#         reference = df_temp.loc[df_temp['Entry ID'] == reference_PDB.lower()]
#         wt_ids = reference.set_index(
#             ['residue_number', 'insertion', 'residue_name']).index
#         MDev = MDev.set_index(['residue_number', 'insertion', 'residue_name'])
#         MDev = MDev[MDev.index.isin(wt_ids)].reset_index()
#         fname = get_nonexistant_file(
#             save_to + '/all_atom_MDev_chain_{}.csv'.format(chain))
#         MDev.to_csv(fname)
#         print(f'\nMDev output for chain {chain} saved to {fname}.')
#         MDevs[chain] = MDev

#     return MDevs


def get_distance(x, y, z, x0, y0, z0):
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)


# def get_outliers_report(atoms, output_directory):
#     '''
#     Counts number of outliers in each structure.
#     Arguments:
#         atoms (dictionary): output of get_distances_distributions
#         output_directory (str): directory to save output to.
#     '''
#     to_join = []
#     for chain, df in atoms.items():
#         df = df.reset_index()
#         df_center = df.loc[df['outlier_from_center'] == True]
#         number_center = df_center.value_counts('Entry ID').to_frame(
#         ).reset_index().rename(columns={0: 'outliers_from_center'})
#         df_average = df.loc[df['outlier_from_average'] == True]
#         number_average = df_average.value_counts('Entry ID').to_frame(
#         ).reset_index().rename(columns={0: 'outliers_from_average'})
#         count = number_center.merge(number_average, on='Entry ID', how='outer')
#         to_join.append(count)

#     if len(to_join) > 1:
#         final = reduce(lambda left, right: pd.merge(
#             left, right, how='outer', on='Entry ID'), to_join)
#     else:
#         final = to_join[0]
#     fname = get_nonexistant_file(
#         output_directory + '/atom_number_of outliers_report.csv')
#     final.to_csv(fname)
#     print(f'Outliers report saved to {fname}.')
#     return final
