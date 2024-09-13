''' EnsemblePDB.build.alignment

pymol function to align all pdbs

Authors:
    Rachael Kretsch (rkretsch@stanford.edu), 
    Siyuan Du, 
    Jacob Parres-Gold
'''

from shutil import copyfile
import glob
from pathlib import Path
from tqdm import tqdm
from pymol import cmd
import pandas as pd
import warnings

from EnsemblePDB.visualize.load_ensemble import load_ensemble
from EnsemblePDB.utils.file_management import get_dir, get_nonexistant_file

warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymol')


def align_all_pymol(directory, alignment, loaded=False, reference_pdb=None,
                    reference_pdb_file=None, cutoff=2.0, cycles=5, gap=-10.0,
                    max_gap=50, extend=-0.5, max_skip=0, matrix='BLOSUM62',
                    align_method="align", name=None, output_directory=None):
    '''
    Aligns all pdb in directory to the reference_pdb using the specifed alignment;
    Saves alignment output file and an rmsd summary file

    Arguments: 
        directory (str): directory where renamed PDB files are saved
        reference_pdb (str): Entry ID of the reference pdb in the given directory, if not specified, assumes first object in the directory is reference pdb
        reference_pdb_file (str): path to the reference pdb file if desired reference pdb is not in the given directory, will use this only if reference_pdb is not given.
        alignment (str): Pymol selection on which to do final alignment. Use only word selection language (e.g. chain A and name CA), not "/" selection.
                        (alignment parameters) https://pymolwiki.org/index.php/Align
        loaded (bool): whether the structures are already loaded in pymol. If not, will load everything in the given directory
        cutoff (float): outlier rejection cutoff in RMS {default: 2.0}
        cycles (int): maximum number of outlier rejection cycles
                        {default: 5}
        gap (float): {default: 10.0}
        max_gap (int): {default: 50}
        extend (float): {default: 0.5}
        max_skip (int): {default: 0}
        matrix (str):  name of substitution matrix for sequence alignment
                            {default: 'BLOSUM62'}
        align_methods (str): align, cealign or super. Unless low sequence alignment, align
                            is reccomended {default: "align"}
        output_directory (str): Path to the folder in which these realigned pdbs and summary report will be saved {default:None}

    Returns: 
        alignment_output and rmsd summary for the specified alignment
    '''
    assert(align_method in ["align", "cealign", "super"]
           ), "align_method must be align, cealign, or super"

    # load structure
    if not loaded:
        load_ensemble(directory)
        if reference_pdb_file is not None:
            cmd.load(reference_pdb_file)

    # get all the structures loaded
    struct_list = cmd.get_names()

    files = Path(directory).glob("*.pdb")
    assert (files != []), "No pdbs in " + directory
    pdbs = [pdb.stem for pdb in files]
    struct_list = [pdb for pdb in struct_list if pdb in pdbs]

    if reference_pdb is not None:
        reference_pdb = reference_pdb[:4]+reference_pdb[4:]
        struct_list.remove(reference_pdb)
    elif reference_pdb is None and reference_pdb_file is not None:
        reference_pdb = Path(reference_pdb_file).stem
        if reference_pdb in struct_list:
            struct_list.remove(reference_pdb)
    else:
        reference_pdb = struct_list.pop(0)

    if name is not None:
        alignment_named = name
    else:
        alignment_named = alignment.replace(
            '.', '_').replace('+', '_').replace(' ', '_')
    if not output_directory:
        parent_directory = str(Path(directory).parents[0])
        output_directory = get_dir(
            f'{parent_directory}/Alignment_on_{alignment_named}_ref_{reference_pdb}')
    output_pdbs_directory = get_dir(
        f'{output_directory}/Renumbered_aligned_pdbs_{alignment_named}_ref_{reference_pdb}')

    # make sure to save reference pdb to new folder
    if reference_pdb_file is None:
        copyfile(Path(directory, f"{reference_pdb}.pdb"),
                 Path(output_pdbs_directory, f"{reference_pdb}.pdb"))
    else:
        copyfile(reference_pdb_file,
                 Path(output_pdbs_directory, f"{reference_pdb}.pdb"))

    # Set-up algnment tables
    columns = ('RMSD after refinement',
               'Number of aligned atoms after refinement',
               'Number of refinement cycles', 'RMSD before refinement',
               'Number of aligned atoms before refinement',
               'Raw alignment score', 'Number of residues aligned')
    alignment_output = None
    rmsd_output = None
    EntryIDs = []

    # Align structures
    alignment_failed = []
    for struct in tqdm(struct_list, total=len(struct_list),
                       desc=f'Aligning: {alignment_named}'):

        # align based on the specified alignment
        try:
            if alignment != '':
                struct_align = f'{struct} and {alignment}'
                ref_align = f'{reference_pdb} and {alignment}'
            else:
                struct_align = struct
                ref_align = reference_pdb
            align_output = cmd.align(struct_align, ref_align, cutoff=cutoff,
                                     cycles=cycles, gap=gap, max_gap=max_gap,
                                     extend=extend, max_skip=max_skip,
                                     matrix=matrix)

            # Save aligned pdb in new output directory
            cmd.save(Path(output_pdbs_directory, f"{struct}.pdb"), struct)

            # do not align but get all-atom rmsd values
            output = cmd.align(f"/{struct}", f"/{reference_pdb}", cycles=0,
                               transform=0, cutoff=cutoff, gap=gap,
                               max_gap=max_gap, extend=extend,
                               max_skip=max_skip, matrix=matrix)

            # Save alignment details
            align_output = pd.DataFrame([align_output], columns=columns)
            if alignment_output is None:
                alignment_output = align_output
            else:
                alignment_output = pd.concat([alignment_output, align_output])

            # save rmsd details
            output = pd.DataFrame([output], columns=columns)
            if rmsd_output is None:
                rmsd_output = output
            else:
                rmsd_output = pd.concat([rmsd_output, output])
            EntryIDs.append(struct)
        except:
            # print("align did not work for",struct)
            alignment_failed.append(struct)
            struct_list.remove(struct)

    # return struct_list,alignment_output
    # Label the rows with the pdb ids
    alignment_output.insert(
        loc=0, column='Entry ID', value=EntryIDs)
    rmsd_output.insert(
        loc=0, column="Entry ID", value=EntryIDs)

    # save all alignment and rmsd information
    all_atom_output_file = get_nonexistant_file(
        Path(output_directory, f'all_atom_rmsds_before_alignment.csv'))
    alignment_output_file = get_nonexistant_file(
        Path(output_directory, f'alignment_{alignment_named}.csv'))

    print("Saving alignment for alignment " +
          alignment + " in ", str(alignment_output_file))
    print("Saving rmsd information in ", str(all_atom_output_file))

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
    output the indiviual alignment summaries and save the aligned pdbs, and then 
    output a summary of the comparitive performance of all alignments

    Arguments: 
        directory (str): renamed PDB folder
        output_directory (str): folder to save all files to.
        reference_pdb (str): the pdb id of reference, if not given assume it is
                        at the top of the objects
        naive (bool): if True do a naive alignment on all atoms {default: False}
        cutoff (float): outlier rejection cutoff in RMS {default: 2.0}
        cycles (int): maximum number of outlier rejection cycles
                        {default: 5}
        gap (float): {default: 10.0}
        max_gap (int): {default: 50}
        extend (float): {default: 0.5}
        max_skip (int): {default: 0}
        matrix (String):  name of substitution matrix for sequence alignment
                            {default: 'BLOSUM62'}

    Returns: 
        rmsd stats

    Outputs: 
        save the alignment output and all-atom rmsd for all alignments, and
        pdbs for all alignment in labeled folders. Then save a summary statistics comparing
        all alignments.
    '''
    if not loaded:
        load_ensemble(directory)
    # Get list of the structures
    structures_list = cmd.get_object_list()
    if reference_pdb is None:
        reference_pdb = structures_list.pop(0)
    else:
        reference_pdb = reference_pdb[:4].lower()+reference_pdb[4:]
        structures_list.remove(reference_pdb)

    if not output_directory:
        parent_directory = str(Path(directory).parents[0])
        all_output_directory = get_dir(f'{parent_directory}/Alignments')
    else:
        all_output_directory = output_directory

    # Do a naive alignment
    if naive:
        alignment_directory = get_dir(
            str(all_output_directory)+'/Naive_alignment')
        alignment_output, rmsd_output = align_all_pymol(
            directory, "", loaded=True, reference_pdb=reference_pdb,
            cutoff=cutoff, cycles=cycles, gap=gap, max_gap=max_gap,
            extend=extend, max_skip=max_skip, matrix=matrix,
            align_method=align_method, name="Naive",
            output_directory=str(alignment_directory))

        # Save naive to summary stats
        rmsd_stats = rmsd_output[
            'RMSD after refinement'].describe().to_frame().T
        rmsd_stats['Alignment'] = 'Naive'
    else:
        rmsd_stats = pd.DataFrame()

    # Align all for every alignement
    for alignment in alignments:
        alignment_named = alignment.replace(
            '.', '_').replace('+', '_').replace(' ', '_')
        alignment_directory = get_dir(
            f'{all_output_directory}/Alignment_on_{alignment_named}')
        alignment_output, rmsd_output = align_all_pymol(
            directory=directory, alignment=alignment, loaded=True,
            reference_pdb=reference_pdb, cutoff=cutoff, cycles=cycles, gap=gap,
            max_gap=max_gap, extend=extend, max_skip=max_skip, matrix=matrix,
            align_method=align_method,
            output_directory=str(alignment_directory))
        align_rmsd_sum = alignment_output[
            'RMSD after refinement'].describe().to_frame().T
        align_rmsd_sum['Alignment'] = alignment + '_all'
        rmsd_stats = pd.concat([rmsd_stats, align_rmsd_sum], ignore_index=True)

    # save summary stats
    stats_output_file = get_nonexistant_file(str(all_output_directory) +
                                             '/alignments_summary.csv')
    print("\nSummary stats saved in", str(stats_output_file))
    rmsd_stats.to_csv(stats_output_file, index=False)
    print('Alignment with min mean rmsd: ')
    print(rmsd_stats[rmsd_stats['mean'] ==
                     rmsd_stats['mean'].min()]['Alignment'].item())
    return rmsd_stats
