'''EnsemblePDB.analyze.energy

From a collection of geometric measurements, get statistical potential and assign energies.

Authors: 
    Siyuan Du (dusiyuan@stanford.edu)

'''
import pandas as pd
import numpy as np


def get_potential_from_dist(dist, nbins=300, temp=298, save_to=None):
    '''
    Arguments:
        dist (1D array-like): distribution of interaction parameter
        nbins (int): number of bins 
    Returns:
        pandas dataframe of potential energy in kcal/mol
    '''
    data = pd.DataFrame({'metric': dist})
    data['bins'] = pd.cut(data['metric'], nbins)
    f_random = 1/nbins
    f_bins_map = data.groupby('bins')['metric'].count()
    data['f_bin'] = data['bins'].apply(
        lambda x: f_bins_map[x]/f_bins_map.sum())
    data['E'] = data['f_bin'].apply(
        lambda x: -1.38*temp*6.02/4184*np.log(x/f_random))
    data['bin min'] = data['bins'].apply(lambda x: float(x.left))
    data['bin max'] = data['bins'].apply(lambda x: float(x.right))
    data['bin mid'] = data['bins'].apply(lambda x: float(x.mid))
    ret_data = data[['bin min', 'bin max',
                     'bin mid', 'f_bin', 'E']].astype('float').sort_values(by='bin mid').drop_duplicates()
    if save_to:
        ret_data.to_csv(save_to)
    return ret_data


def get_energy_from_stats(potential, measured, default_maximum=True,
                          set_maximum=None):
    '''
    Arguments:
        potential (pandas dataframe): potential energy dataframe wih columns: 
            bin min, bin max and E
        measured (float): measured geometric parameter to get energy of
        default_maximum (bool): if True, maximum eneregy will be assigned to 
            geometric parameters not found in any bins that the potential energy 
            function covers. If False, will use the value specified in set_maximum {default: True}
        set_maximum (float): set maximum energy value when the geometric 
            parameter is not found in any bins that the potential energy function covers.
    Returns
        energy value (kcal/mol). 
    '''
    if np.isnan(measured):
        return np.nan
    dfE = potential.loc[(potential['bin min'] < measured)
                        & (potential['bin max'] >= measured)]
    if len(dfE) == 0:
        if default_maximum:
            return potential['E'].max()
        elif (not default_maximum) and set_maximum:
            return set_maximum
        else:
            print("Need to specify set_maximum if not using default maximum energy")
    else:
        return dfE.iloc[0]['E']

# TODO: 3D - joint potentials
# def get_potential_from_smmol(data,nbins=300):
#     '''
#     Arguments:
#         data (array-like): distribution of parameter
#         nbins (int): number of bins
#     '''
#     # df = pd.DataFrame({'parameter_value': data})
#     df['bins'] = pd.cut(df['parameter_value'],nbins)
#     f_random = 1/nbins
#     f_bins_map = df.groupby('bins')['parameter_value'].count()
#     data['f_bin'] = data['bins'].apply(lambda x: f_bins_map[x]/f_bins_map.sum())
#     data['E'] = data['f_bin'].apply(lambda x: -1.38*298*6.02/4184*np.log(x/f_random))
#     return data

# def get_energy_from_stats(df, measured,step=0.005):
#     '''
#     TODO
#     '''
#     data = df.loc[df['bins'].apply(lambda x: measured in x)]
#     energy = data['E'].mean()
#     d=0
#     while np.isnan(energy):
#         d += step
#         data = df.loc[df['bins'].apply(lambda x: (measured-d in x)|(measured+d in x))]
#         energy = data['E'].mean()
#     return energy
