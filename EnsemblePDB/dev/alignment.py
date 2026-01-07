''' EnsemblePDB.build.alignment

Structural alignment of PDB files using Kabsch algorithm
'''

from shutil import copyfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.linalg import svd
import re

from EnsemblePDB.utils.biopandas_utils import get_PandasPDB
from EnsemblePDB.utils.file_management import get_dir, get_nonexistant_file


def parse_selection(selection_str, atoms_df):
    '''
    Parse PyMOL-style selection string and filter atoms dataframe.
    Supports simple selections like "chain A and name CA" or "chain A".
    
    Arguments:
        selection_str (str): Selection string (e.g., "chain A and name CA")
        atoms_df (pd.DataFrame): DataFrame with ATOM records from biopandas
        
    Returns:
        pd.DataFrame: Filtered atoms dataframe
    '''
    if not selection_str or selection_str.strip() == '':
        return atoms_df
    
    # Convert to lowercase for case-insensitive matching
    selection = selection_str.lower().strip()
    
    # Start with all atoms
    filtered = atoms_df.copy()
    
    # Parse "chain X" or "chain X and name Y" patterns
    chain_pattern = r'chain\s+([a-z0-9]+)'
    name_pattern = r'name\s+([a-z0-9\*]+)'
    residue_pattern = r'residue\s+(\d+)'
    
    # Extract chain selection
    chain_match = re.search(chain_pattern, selection)
    if chain_match:
        chain_id = chain_match.group(1).upper()
        filtered = filtered[filtered['chain_id'] == chain_id]
    
    # Extract atom name selection
    name_match = re.search(name_pattern, selection)
    if name_match:
        atom_name = name_match.group(1).upper()
        if atom_name == '*':
            # Keep all atoms
            pass
        else:
            filtered = filtered[filtered['atom_name'] == atom_name]
    
    # Extract residue number selection (if present)
    residue_match = re.search(residue_pattern, selection)
    if residue_match:
        residue_num = int(residue_match.group(1))
        filtered = filtered[filtered['residue_number'] == residue_num]
    
    return filtered


def kabsch_algorithm(P, Q):
    '''
    Kabsch algorithm for optimal rotation matrix between two sets of points.
    
    Arguments:
        P (np.array): Nx3 array of coordinates to align (mobile)
        Q (np.array): Nx3 array of reference coordinates (fixed)
        
    Returns:
        R (np.array): 3x3 rotation matrix
        t (np.array): 3x1 translation vector
    '''
    # Center the point sets
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_Q - R @ centroid_P
    
    return R, t


def calculate_rmsd(coords1, coords2):
    '''
    Calculate RMSD between two sets of coordinates.
    
    Arguments:
        coords1 (np.array): Nx3 array of coordinates
        coords2 (np.array): Nx3 array of coordinates
        
    Returns:
        float: RMSD value
    '''
    if len(coords1) == 0 or len(coords2) == 0:
        return np.nan
    
    diff = coords1 - coords2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd


def align_structures(mobile_atoms, ref_atoms, mobile_all_atoms, ref_all_atoms,
                     cutoff=2.0, cycles=5):
    '''
    Align mobile structure to reference structure using iterative refinement.
    
    Arguments:
        mobile_atoms (pd.DataFrame): Atoms to use for alignment (mobile)
        ref_atoms (pd.DataFrame): Atoms to use for alignment (reference)
        mobile_all_atoms (pd.DataFrame): All atoms in mobile structure
        ref_all_atoms (pd.DataFrame): All atoms in reference structure
        cutoff (float): Outlier rejection cutoff in RMSD {default: 2.0}
        cycles (int): Maximum number of refinement cycles {default: 5}
        
    Returns:
        tuple: (aligned_all_atoms, alignment_stats)
            alignment_stats: dict with RMSD values and atom counts
    '''
    # Need matching atoms - match by residue number, insertion, chain, and atom name
    # This assumes structures are already renumbered
    mobile_atoms = mobile_atoms.copy()
    ref_atoms = ref_atoms.copy()
    
    # Create matching keys as strings for easier handling
    mobile_atoms['match_key'] = (mobile_atoms['chain_id'].astype(str) + '_' + 
                                 mobile_atoms['residue_number'].astype(str) + '_' + 
                                 mobile_atoms['insertion'].fillna('').astype(str) + '_' + 
                                 mobile_atoms['atom_name'].astype(str))
    ref_atoms['match_key'] = (ref_atoms['chain_id'].astype(str) + '_' + 
                              ref_atoms['residue_number'].astype(str) + '_' + 
                              ref_atoms['insertion'].fillna('').astype(str) + '_' + 
                              ref_atoms['atom_name'].astype(str))
    
    # Find common atoms
    common_keys = set(mobile_atoms['match_key']) & set(ref_atoms['match_key'])
    
    if len(common_keys) == 0:
        raise ValueError("No common atoms found for alignment")
    
    # Filter to common atoms and sort by match_key to ensure matching order
    mobile_common = mobile_atoms[mobile_atoms['match_key'].isin(common_keys)].copy()
    ref_common = ref_atoms[ref_atoms['match_key'].isin(common_keys)].copy()
    
    # Sort by match_key to ensure matching order
    mobile_common = mobile_common.sort_values('match_key').reset_index(drop=True)
    ref_common = ref_common.sort_values('match_key').reset_index(drop=True)
    
    # Verify they have the same keys in the same order
    if not mobile_common['match_key'].equals(ref_common['match_key']):
        # If not equal, try matching by index after sorting by columns
        sort_cols = ['chain_id', 'residue_number', 'insertion', 'atom_name']
        mobile_common = mobile_common.sort_values(sort_cols).reset_index(drop=True)
        ref_common = ref_common.sort_values(sort_cols).reset_index(drop=True)
        # Recreate keys and match
        mobile_common['match_key'] = (mobile_common['chain_id'].astype(str) + '_' + 
                                     mobile_common['residue_number'].astype(str) + '_' + 
                                     mobile_common['insertion'].fillna('').astype(str) + '_' + 
                                     mobile_common['atom_name'].astype(str))
        ref_common['match_key'] = (ref_common['chain_id'].astype(str) + '_' + 
                                  ref_common['residue_number'].astype(str) + '_' + 
                                  ref_common['insertion'].fillna('').astype(str) + '_' + 
                                  ref_common['atom_name'].astype(str))
        # Keep only truly matching keys
        matching_keys = set(mobile_common['match_key']) & set(ref_common['match_key'])
        mobile_common = mobile_common[mobile_common['match_key'].isin(matching_keys)].sort_values('match_key').reset_index(drop=True)
        ref_common = ref_common[ref_common['match_key'].isin(matching_keys)].sort_values('match_key').reset_index(drop=True)
    
    # Extract coordinates
    mobile_coords = mobile_common[['x_coord', 'y_coord', 'z_coord']].values
    ref_coords = ref_common[['x_coord', 'y_coord', 'z_coord']].values
    
    # Initial alignment
    R, t = kabsch_algorithm(mobile_coords, ref_coords)
    
    # Initial RMSD
    mobile_aligned = (R @ mobile_coords.T).T + t
    rmsd_before = calculate_rmsd(mobile_aligned, ref_coords)
    
    # Iterative refinement with outlier rejection
    n_cycles = 0
    rmsd_before = calculate_rmsd(mobile_coords, ref_coords)  # Initial RMSD before alignment
    
    for cycle in range(cycles):
        # Calculate distances after current alignment
        mobile_aligned = (R @ mobile_coords.T).T + t
        distances = np.sqrt(np.sum((mobile_aligned - ref_coords)**2, axis=1))
        
        # Remove outliers (cutoff is in Angstroms)
        inlier_mask = distances <= cutoff
        
        if np.sum(inlier_mask) < 3:  # Need at least 3 points
            break
        
        # Re-align with inliers only
        mobile_inliers = mobile_coords[inlier_mask]
        ref_inliers = ref_coords[inlier_mask]
        
        if len(mobile_inliers) < 3:
            break
        
        R_new, t_new = kabsch_algorithm(mobile_inliers, ref_inliers)
        
        # Check convergence
        mobile_inliers_aligned = (R_new @ mobile_inliers.T).T + t_new
        new_rmsd = calculate_rmsd(mobile_inliers_aligned, ref_inliers)
        
        # Update transformation
        R, t = R_new, t_new
        n_cycles = cycle + 1
        
        # Check for convergence
        if cycle > 0 and abs(new_rmsd - rmsd_before) < 0.001:
            break
        rmsd_before = new_rmsd
    
    # Final RMSD after refinement (on all matched atoms)
    mobile_aligned = (R @ mobile_coords.T).T + t
    rmsd_after = calculate_rmsd(mobile_aligned, ref_coords)
    
    # Count inliers after final alignment
    distances = np.sqrt(np.sum((mobile_aligned - ref_coords)**2, axis=1))
    inlier_mask = distances <= cutoff
    n_aligned_after = np.sum(inlier_mask)
    
    # Apply transformation to all atoms
    mobile_all_coords = mobile_all_atoms[['x_coord', 'y_coord', 'z_coord']].values
    mobile_all_aligned = (R @ mobile_all_coords.T).T + t
    
    # Update coordinates in dataframe
    aligned_all_atoms = mobile_all_atoms.copy()
    aligned_all_atoms['x_coord'] = mobile_all_aligned[:, 0]
    aligned_all_atoms['y_coord'] = mobile_all_aligned[:, 1]
    aligned_all_atoms['z_coord'] = mobile_all_aligned[:, 2]
    
    # Calculate all-atom RMSD (before alignment, no transform)
    mobile_all_atoms_copy = mobile_all_atoms.copy()
    ref_all_atoms_copy = ref_all_atoms.copy()
    
    # Match all atoms for RMSD calculation using same key method
    mobile_all_atoms_copy['match_key'] = (mobile_all_atoms_copy['chain_id'].astype(str) + '_' + 
                                         mobile_all_atoms_copy['residue_number'].astype(str) + '_' + 
                                         mobile_all_atoms_copy['insertion'].fillna('').astype(str) + '_' + 
                                         mobile_all_atoms_copy['atom_name'].astype(str))
    ref_all_atoms_copy['match_key'] = (ref_all_atoms_copy['chain_id'].astype(str) + '_' + 
                                      ref_all_atoms_copy['residue_number'].astype(str) + '_' + 
                                      ref_all_atoms_copy['insertion'].fillna('').astype(str) + '_' + 
                                      ref_all_atoms_copy['atom_name'].astype(str))
    
    common_all_keys = set(mobile_all_atoms_copy['match_key']) & set(ref_all_atoms_copy['match_key'])
    
    if len(common_all_keys) > 0:
        mobile_all_common = mobile_all_atoms_copy[
            mobile_all_atoms_copy['match_key'].isin(common_all_keys)].copy()
        ref_all_common = ref_all_atoms_copy[
            ref_all_atoms_copy['match_key'].isin(common_all_keys)].copy()
        
        # Sort to ensure matching order
        mobile_all_common = mobile_all_common.sort_values('match_key').reset_index(drop=True)
        ref_all_common = ref_all_common.sort_values('match_key').reset_index(drop=True)
        
        mobile_all_common_coords = mobile_all_common[['x_coord', 'y_coord', 'z_coord']].values
        ref_all_common_coords = ref_all_common[['x_coord', 'y_coord', 'z_coord']].values
        
        # Align for RMSD calculation (no transform applied to structure)
        R_rmsd, t_rmsd = kabsch_algorithm(mobile_all_common_coords, ref_all_common_coords)
        mobile_all_aligned_rmsd = (R_rmsd @ mobile_all_common_coords.T).T + t_rmsd
        rmsd_all_atom = calculate_rmsd(mobile_all_aligned_rmsd, ref_all_common_coords)
        n_all_atom = len(common_all_keys)
    else:
        rmsd_all_atom = np.nan
        n_all_atom = 0
    
    # Prepare alignment statistics
    alignment_stats = {
        'RMSD after refinement': rmsd_after,
        'Number of aligned atoms after refinement': n_aligned_after,
        'Number of refinement cycles': n_cycles,
        'RMSD before refinement': rmsd_before if n_cycles == 0 else rmsd_before,
        'Number of aligned atoms before refinement': len(mobile_coords),
        'Raw alignment score': rmsd_after,  # Using RMSD as score
            'Number of residues aligned': len(set(mobile_common['residue_number'])),
        'RMSD all atoms (before alignment)': rmsd_all_atom,
        'Number of all atoms': n_all_atom
    }
    
    return aligned_all_atoms, alignment_stats


def align_all_pymol(directory, alignment, loaded=False, reference_pdb=None,
                    reference_pdb_file=None, cutoff=2.0, cycles=5, gap=-10.0,
                    max_gap=50, extend=-0.5, max_skip=0, matrix='BLOSUM62',
                    align_method="align", name=None, output_directory=None):
    '''
    Aligns all pdb in directory to the reference_pdb using the specified alignment;
    Saves alignment output file and an rmsd summary file

    Arguments: 
        directory (str): directory where renamed PDB files are saved
        reference_pdb (str): Entry ID of the reference pdb in the given directory, 
                            if not specified, assumes first object in the directory is reference pdb
        reference_pdb_file (str): path to the reference pdb file if desired reference pdb is not 
                            in the given directory, will use this only if reference_pdb is not given.
        alignment (str): Selection string on which to do final alignment. 
                        Use PyMOL-style selection language (e.g., "chain A and name CA").
                        Empty string means align all atoms.
        loaded (bool): whether the structures are already loaded (ignored, kept for compatibility)
        cutoff (float): outlier rejection cutoff in RMS {default: 2.0}
        cycles (int): maximum number of outlier rejection cycles {default: 5}
        gap (float): Not used, kept for compatibility {default: -10.0}
        max_gap (int): Not used, kept for compatibility {default: 50}
        extend (float): Not used, kept for compatibility {default: -0.5}
        max_skip (int): Not used, kept for compatibility {default: 0}
        matrix (str): Not used, kept for compatibility {default: 'BLOSUM62'}
        align_method (str): Must be "align" (other methods not yet implemented) {default: "align"}
        name (str): Name for this alignment {default: None}
        output_directory (str): Path to the folder in which these realigned pdbs and summary 
                            report will be saved {default: None}

    Returns: 
        tuple: (alignment_output, rmsd_output) - DataFrames with alignment statistics
    '''
    assert(align_method in ["align", "cealign", "super"]
           ), "align_method must be align, cealign, or super (only 'align' currently implemented)"
    
    if align_method != "align":
        print(f"Warning: {align_method} not yet implemented, using 'align' method")
    
    # Get list of PDB files
    files = list(Path(directory).glob("*.pdb"))
    assert len(files) > 0, f"No pdbs in {directory}"
    
    pdbs = [pdb.stem for pdb in files]

    # Determine reference PDB
    if reference_pdb is not None:
        reference_pdb = reference_pdb[:4].lower() + reference_pdb[4:]
        if reference_pdb not in pdbs and reference_pdb_file is None:
            raise ValueError(f"Reference PDB {reference_pdb} not found in directory")
    elif reference_pdb_file is not None:
        reference_pdb = Path(reference_pdb_file).stem
    else:
        reference_pdb = pdbs[0]

    # Remove reference from list of structures to align
    struct_list = [pdb for pdb in pdbs if pdb != reference_pdb]

    # Set up output directories
    if name is not None:
        alignment_named = name
    else:
        alignment_named = alignment.replace(
            '.', '_').replace('+', '_').replace(' ', '_') if alignment else 'all_atoms'
    
    if not output_directory:
        parent_directory = str(Path(directory).parent)
        output_directory = get_dir(
            f'{parent_directory}/Alignment_on_{alignment_named}_ref_{reference_pdb}')
    
    output_pdbs_directory = get_dir(
        f'{output_directory}/Renumbered_aligned_pdbs_{alignment_named}_ref_{reference_pdb}')

    # Load reference structure
    if reference_pdb_file is None:
        ref_file = Path(directory, f"{reference_pdb}.pdb")
    else:
        ref_file = Path(reference_pdb_file)
    
    copyfile(ref_file, Path(output_pdbs_directory, f"{reference_pdb}.pdb"))
    
    ref_struct = get_PandasPDB(ref_file)
    ref_atoms = ref_struct.df['ATOM']
    ref_all_atoms = ref_struct.df['ATOM'].copy()
    
    # Parse alignment selection for reference
    if alignment:
        ref_align_atoms = parse_selection(alignment, ref_atoms)
    else:
        ref_align_atoms = ref_atoms

    # Set up alignment tables
    columns = ('RMSD after refinement',
               'Number of aligned atoms after refinement',
               'Number of refinement cycles', 'RMSD before refinement',
               'Number of aligned atoms before refinement',
               'Raw alignment score', 'Number of residues aligned')
    
    alignment_output = []
    rmsd_output = []
    EntryIDs = []
    alignment_failed = []

    # Align structures
    for struct in tqdm(struct_list, total=len(struct_list),
                       desc=f'Aligning: {alignment_named}'):
        try:
            # Load mobile structure
            mobile_file = Path(directory, f"{struct}.pdb")
            mobile_struct = get_PandasPDB(mobile_file)
            mobile_atoms = mobile_struct.df['ATOM']
            mobile_all_atoms = mobile_struct.df['ATOM'].copy()
            
            # Parse alignment selection for mobile
            if alignment:
                mobile_align_atoms = parse_selection(alignment, mobile_atoms)
            else:
                mobile_align_atoms = mobile_atoms
            
            # Perform alignment
            aligned_atoms, stats = align_structures(
                mobile_align_atoms, ref_align_atoms,
                mobile_all_atoms, ref_all_atoms,
                cutoff=cutoff, cycles=cycles
            )
            
            # Update structure with aligned coordinates
            mobile_struct.df['ATOM'] = aligned_atoms
            
            # Save aligned PDB
            output_pdb_path = Path(output_pdbs_directory, f"{struct}.pdb")
            mobile_struct.to_pdb(path=str(output_pdb_path), gz=False)
            
            # Prepare alignment output (after alignment)
            align_output_row = {
                'RMSD after refinement': stats['RMSD after refinement'],
                'Number of aligned atoms after refinement': stats['Number of aligned atoms after refinement'],
                'Number of refinement cycles': stats['Number of refinement cycles'],
                'RMSD before refinement': stats['RMSD before refinement'],
                'Number of aligned atoms before refinement': stats['Number of aligned atoms before refinement'],
                'Raw alignment score': stats['Raw alignment score'],
                'Number of residues aligned': stats['Number of residues aligned']
            }
            alignment_output.append(align_output_row)
            
            # Prepare RMSD output (all atoms, before alignment transform)
            rmsd_output_row = {
                'RMSD after refinement': stats['RMSD all atoms (before alignment)'],
                'Number of aligned atoms after refinement': stats['Number of all atoms'],
                'Number of refinement cycles': 0,  # No cycles for all-atom RMSD
                'RMSD before refinement': stats['RMSD all atoms (before alignment)'],
                'Number of aligned atoms before refinement': stats['Number of all atoms'],
                'Raw alignment score': stats['RMSD all atoms (before alignment)'],
                'Number of residues aligned': stats.get('Number of residues aligned', 0)
            }
            rmsd_output.append(rmsd_output_row)
            
            EntryIDs.append(struct)
            
        except Exception as e:
            print(f"Alignment failed for {struct}: {str(e)}")
            alignment_failed.append(struct)
    
    # Convert to DataFrames
    if len(alignment_output) > 0:
        alignment_output = pd.DataFrame(alignment_output, columns=columns)
        alignment_output.insert(loc=0, column='Entry ID', value=EntryIDs)
        
        rmsd_output = pd.DataFrame(rmsd_output, columns=columns)
        rmsd_output.insert(loc=0, column="Entry ID", value=EntryIDs)
    else:
        alignment_output = pd.DataFrame(columns=['Entry ID'] + list(columns))
        rmsd_output = pd.DataFrame(columns=['Entry ID'] + list(columns))
    
    # Save output files
    all_atom_output_file = get_nonexistant_file(
        Path(output_directory, f'all_atom_rmsds_before_alignment.csv'))
    alignment_output_file = get_nonexistant_file(
        Path(output_directory, f'alignment_{alignment_named}.csv'))

    print("Saving alignment for alignment " +
          alignment + " in ", str(alignment_output_file))
    print("Saving rmsd information in ", str(all_atom_output_file))

    if len(alignment_output) > 0:
    rmsd_output.to_csv(all_atom_output_file, index=False)
    alignment_output.to_csv(alignment_output_file, index=False)
    
    if len(alignment_failed) > 0:
        print(f'\nAlignment failed for {alignment_failed}')
    
    return alignment_output, rmsd_output


def multiple_alignments(directory, alignments=[""], loaded=False,
                        reference_pdb=None, naive=False, cutoff=2.0, cycles=5,
                        gap=-10.0, max_gap=50, extend=-0.5, max_skip=0,
                        matrix='BLOSUM62', align_method="align",
                        output_directory=None):
    '''
    Given a list of alignments performs all alignments
    output the individual alignment summaries and save the aligned pdbs, and then 
    output a summary of the comparative performance of all alignments

    Arguments: 
        directory (str): renamed PDB folder
        output_directory (str): folder to save all files to.
        reference_pdb (str): the pdb id of reference, if not given assume it is
                        the first PDB in the directory
        naive (bool): if True do a naive alignment on all atoms {default: False}
        loaded (bool): Not used, kept for compatibility {default: False}
        cutoff (float): outlier rejection cutoff in RMS {default: 2.0}
        cycles (int): maximum number of outlier rejection cycles {default: 5}
        gap (float): Not used, kept for compatibility {default: -10.0}
        max_gap (int): Not used, kept for compatibility {default: 50}
        extend (float): Not used, kept for compatibility {default: -0.5}
        max_skip (int): Not used, kept for compatibility {default: 0}
        matrix (str): Not used, kept for compatibility {default: 'BLOSUM62'}
        align_method (str): Must be "align" {default: "align"}
        output_directory (str): Path to save output files {default: None}

    Returns: 
        pd.DataFrame: RMSD statistics summary

    Outputs: 
        Saves the alignment output and all-atom rmsd for all alignments, and
        pdbs for all alignment in labeled folders. Then saves a summary statistics comparing
        all alignments.
    '''
    # Get list of PDB files to determine reference
    files = list(Path(directory).glob("*.pdb"))
    assert len(files) > 0, f"No pdbs in {directory}"
    
    pdbs = [pdb.stem for pdb in files]
    
    if reference_pdb is None:
        reference_pdb = pdbs[0]
    else:
        reference_pdb = reference_pdb[:4].lower() + reference_pdb[4:]

    # Set up output directory
    if not output_directory:
        parent_directory = str(Path(directory).parent)
        all_output_directory = get_dir(f'{parent_directory}/Alignments')
    else:
        all_output_directory = output_directory

    # Do a naive alignment
    if naive:
        alignment_directory = get_dir(
            str(all_output_directory) + '/Naive_alignment')
        alignment_output, rmsd_output = align_all_pymol(
            directory, "", loaded=False, reference_pdb=reference_pdb,
            cutoff=cutoff, cycles=cycles, gap=gap, max_gap=max_gap,
            extend=extend, max_skip=max_skip, matrix=matrix,
            align_method=align_method, name="Naive",
            output_directory=str(alignment_directory))

        # Save naive to summary stats
        if len(rmsd_output) > 0:
        rmsd_stats = rmsd_output[
            'RMSD after refinement'].describe().to_frame().T
        rmsd_stats['Alignment'] = 'Naive'
        else:
            rmsd_stats = pd.DataFrame()
    else:
        rmsd_stats = pd.DataFrame()

    # Align all for every alignment
    for alignment in alignments:
        alignment_named = alignment.replace(
            '.', '_').replace('+', '_').replace(' ', '_') if alignment else 'all_atoms'
        alignment_directory = get_dir(
            f'{all_output_directory}/Alignment_on_{alignment_named}')
        alignment_output, rmsd_output = align_all_pymol(
            directory=directory, alignment=alignment, loaded=False,
            reference_pdb=reference_pdb, cutoff=cutoff, cycles=cycles, gap=gap,
            max_gap=max_gap, extend=extend, max_skip=max_skip, matrix=matrix,
            align_method=align_method,
            output_directory=str(alignment_directory))
        
        if len(alignment_output) > 0:
        align_rmsd_sum = alignment_output[
            'RMSD after refinement'].describe().to_frame().T
            align_rmsd_sum['Alignment'] = alignment + '_all' if alignment else 'all_atoms'
        rmsd_stats = pd.concat([rmsd_stats, align_rmsd_sum], ignore_index=True)

    # Save summary stats
    if len(rmsd_stats) > 0:
    stats_output_file = get_nonexistant_file(str(all_output_directory) +
                                             '/alignments_summary.csv')
    print("\nSummary stats saved in", str(stats_output_file))
    rmsd_stats.to_csv(stats_output_file, index=False)
        
        if 'mean' in rmsd_stats.columns:
            min_mean_idx = rmsd_stats['mean'].idxmin()
    print('Alignment with min mean rmsd: ')
            print(rmsd_stats.loc[min_mean_idx, 'Alignment'])
    
    return rmsd_stats
