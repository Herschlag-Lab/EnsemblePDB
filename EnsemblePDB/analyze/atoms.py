'''EnsemblePDB.analyze.atoms

Functions to analyze deviation of atom positions within the aligned ensembles.

Authors:
    Rachael Kretsch(rkretsch@stanford.edu), 
    Siyuan Du (dusiyuan@stanford.edu),
    Jacob Parres-Gold
'''

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from EnsemblePDB.utils.file_management import get_nonexistant_file
from EnsemblePDB.utils.biopandas_utils import get_PandasPDBs, get_xyz, clean_multiconformers
from EnsemblePDB.utils.analysis_utils import *


def get_MDev(directory, chains, reference_PDB, multiconformers=False,
             output_directory=None, bootstrap=True, iter=50,
             include_mutations=False):
    '''
    Takes .pdb files and calculates the mean deviation of atom 
    positions for all atoms. Note no alignment is perferformed.

    Arguments:
        directory (str): Folder of aligned .pdb files
        chains (list of str): List of (renamed) chains to analyze
        reference_PDB (str): Reference PDB to obtain wild-type sequence
        mutliconformers (bool): if True, count multiconformers as individual 
                        structures (with equal weight); if False, delete all 
                        multiconformer atoms {default: False}
        output_directory (str): Directory to save output MDev files to. If 
                        None, saves to the parent of given directory 
                        {default: None}
        bootstrap (bool): if bootstrap is True, perform bootstrap analysis and 
                        include standard deviation of the bootstrap 
                        distribution. Notice that this may take a long time to 
                        complete for large number of atoms {default: True}
        iter (int): number of iteration to resample data for bootstrap analysis 
                        {default: 50}
        include_mutations (bool): if True, include mutants in MDev calculates; 
                    if False, separate residues from mutants {default: False}

    Returns:
        DataFrames of MDev values for each atom.

    Output:
        Saves MDev files to given output directory or default.
    '''
    ppdbs = get_PandasPDBs(directory)
    atoms = get_xyz(ppdbs)
    if not multiconformers:
        atoms = clean_multiconformers(atoms)

    if output_directory is None:
        output_directory = str(Path(directory).parents[0])
    MDev = combine_save_MDev(atoms, chains, reference_PDB,
                             output_directory, bootstrap,
                             iter, include_mutations)
    return MDev


def get_distance_distributions(directory, chains, multiconformers=False,
                               quantile=0.95, report_outliers=True,
                               output_directory=None, include_mutations=False):
    '''
    Get all coordinates for an atom in the ensemble and calculate the distance 
    of each from the center atom of the ensemble and from the average position.
    Also determines whether a position is an outlier of the ensemble based on 
    given quantile. 

    Arguments:
        directory (str): folder of all aligned PDB file.
        chains (list): list of selected chain to get atoms 
        multiconformers (bool): if False will remove all multiconfomers
                                {defaults: False}
        quantile (float): the quantile of data used to identify outliers 
                            {default: 0.95} 
        report_outliers (bool): if True, calculates the number of outliers in 
                            each PDB structure and saves a separate file. 
                            {default: True}
        output_directory (str): Directory to save output files to. If None, 
                        saves to the parent of given directory {default: None}
        include_mutations (bool): if True, include mutants in MDev calculates; 
                    if False, separate residues from mutants {default: False}

    Returns:
        dataframe with new columns of distance from center atom and average position 
        and outlier

    Output:
        Saves atomic positions and their distances from center/average position
        as a csv. Saves outlier report if user chose to get it.
    '''
    ppdbs = get_PandasPDBs(directory)
    atoms = get_xyz(ppdbs)
    if not multiconformers:
        atoms = clean_multiconformers(atoms)

    if output_directory is None:
        output_directory = str(Path(directory).parents[0])
    final = {}

    for chain in chains:
        atoms = atoms.loc[atoms['chain_id'] == chain]
        if not include_mutations:
            atoms = atoms.set_index(
                ['residue_number', 'insertion', 'residue_name',
                 'atom_name']).sort_index()
        else:
            atoms = atoms.set_index(
                ['residue_number', 'insertion', 'atom_name']).sort_index()
        to_concat = []

        for i, index in tqdm(enumerate(atoms.index.unique()),
                             total=len(atoms.index.unique()),
                             desc='Calculating distances and ' +
                             f'outliers in chain {chain}'):
            atom = atoms.loc[index, :].copy()
            atom['xyz_spread'] = (calculate_msd(atom['x_coord']) +
                                  calculate_msd(atom['y_coord']) +
                                  calculate_msd(atom['z_coord']))

            # find center atom
            center = atom.iloc[atom['xyz_spread'].argmin()]
            # find average position
            average = (atom['x_coord'].mean(),
                       atom['y_coord'].mean(), atom['z_coord'].mean())

            atom['distance_from_center'] = atom.apply(lambda x: get_distance(
                x['x_coord'], x['y_coord'], x['z_coord'], center['x_coord'],
                center['y_coord'], center['z_coord']), axis=1)
            atom['outlier_from_center'] = atom['distance_from_center'].apply(
                lambda x: x > atom['distance_from_center'].quantile(quantile))

            atom['distance_from_average'] = atom.apply(lambda x: get_distance(
                x['x_coord'], x['y_coord'], x['z_coord'], average[0],
                average[1], average[2]), axis=1)
            atom['outlier_from_average'] = atom['distance_from_average'].apply(
                lambda x: x > atom['distance_from_average'].quantile(quantile))
            to_concat.append(atom)
        chain_out = pd.concat(to_concat)
        fname = get_nonexistant_file(
            f'{output_directory}/all_atom_positions_summary_chain_{chain}.csv')
        chain_out.to_csv(fname)
        print(f'\nAtomic positions and distances saved to {fname}')
        final[chain] = chain_out
    if report_outliers:
        number_of_outliers = get_outliers_report(final, output_directory)

    return final


def calculate_MDev(df, bootstrap, iter, include_mutations):
    '''
    Arguments:
        df, dataframe of atom coordinates

    Returns:
        MDevs, dataframe with MDev and count
    '''
    if not include_mutations:
        grouped = df.groupby(['residue_number', 'insertion',
                              'residue_name', 'atom_name'])
    else:
        grouped = df.groupby(['residue_number', 'insertion', 'atom_name'])
    MDev = grouped.progress_apply(lambda x: calculate_rmsd(x))
    size = grouped['Entry ID'].count()
    to_concat = [MDev, size]
    if bootstrap:
        tqdm.pandas(desc=f'Perform bootstrap analysis')
        boot = grouped.progress_apply(
            lambda x: bootstrap_analysis(x, iter, apply_func=calculate_rmsd))
        to_concat.append(boot)
    MDevs = pd.concat(to_concat, axis=1)
    MDevs = pd.DataFrame(MDevs).reset_index()\
        .rename(columns={0: 'MDev',
                         'Entry ID': 'n',
                         1: 'std (bootstrap)'})
    return MDevs


def combine_save_MDev(df, chains, reference_PDB, save_to, bootstrap,
                      iter, include_mutations):
    '''
    Arguments:
        df, dataframe of atom coords
        chains, list of strings (which chains to calculate MDev on)
    Returns:
        dictionary: key = chain ID, value = dataframe of MDev of each atom
        saves a csv if save_to is not None
    '''
    MDevs = {}

    for chain in chains:
        df_temp = df.loc[df['chain_id'] == chain]
        # print('Calculating MDev for chain {}...'.format(chain))
        tqdm.pandas(desc=f'Calculating atomic MDevs for chain {chain}')
        MDev = calculate_MDev(df_temp, bootstrap, iter, include_mutations)

        # get rid of mutants
        if not include_mutations:
            reference = df_temp.loc[(df_temp['Entry ID'] == reference_PDB) | (
                df_temp['Entry ID'] == reference_PDB.lower()) |
                (df_temp['Entry ID'] == reference_PDB.upper())]
            wt_ids = reference.set_index(
                ['residue_number', 'insertion', 'residue_name']).index
            MDev = MDev.set_index(
                ['residue_number', 'insertion', 'residue_name'])
            MDev = MDev[MDev.index.isin(wt_ids)].reset_index()
        fname = get_nonexistant_file(
            save_to + '/all_atom_MDev_chain_{}.csv'.format(chain))
        MDev.to_csv(fname)
        print(f'\nMDev output for chain {chain} saved to {fname}.')
        MDevs[chain] = MDev

    return MDevs


def get_outliers_report(atoms, output_directory):
    '''
    Counts number of outliers in each structure.
    Arguments:
        atoms (dictionary): output of get_distances_distributions
        output_directory (str): directory to save output to.
    '''
    to_join = []
    for chain, df in atoms.items():
        df = df.reset_index()
        df_center = df.loc[df['outlier_from_center'] == True]
        number_center = df_center.value_counts('Entry ID').to_frame(
        ).reset_index().rename(columns={0: 'outliers_from_center'})
        df_average = df.loc[df['outlier_from_average'] == True]
        number_average = df_average.value_counts('Entry ID').to_frame(
        ).reset_index().rename(columns={0: 'outliers_from_average'})
        count = number_center.merge(number_average, on='Entry ID', how='outer')
        to_join.append(count)

    if len(to_join) > 1:
        final = reduce(lambda left, right: pd.merge(
            left, right, how='outer', on='Entry ID'), to_join)
    else:
        final = to_join[0]
    fname = get_nonexistant_file(
        output_directory + '/atom_number_of outliers_report.csv')
    final.to_csv(fname)
    print(f'Outliers report saved to {fname}.')
    return final
