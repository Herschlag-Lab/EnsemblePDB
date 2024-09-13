import sys,os
from EnsemblePDB import search, build, refine, analyze
from EnsemblePDB.utils import file_management
import pandas as pd

dir = os.path.dirname(os.path.abspath(__file__))
output_dir = file_management.get_dir(os.path.join(dir,'test_ensembles_2'))

protease = 'chymotrypsin'
seq = 'CGVPAIQPVLSGLIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTTSDVVVAGEFDQGSSSEKIQKLKIAKVFKNSKYNSLTINNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYANTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGASGVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWVQQTLAAN'
ref_pdb = '1ACB'
ref_chains = ['E']

#search
summary = search.query_pdb.query_by_sequence(sequences=[seq], label=protease,macromolecule_name=None, seq_id=0.95,xray_only=True, evalue=10,pairwise_align_options = {'alignment_type':"local",'gap_open_score':-0.5, 'gap_extend_score':-0.1},output_dir=output_dir,min_length=20)
data = search.get_data.get_pdb_data(f'{output_dir}/seq_search_output_{protease}.csv', label=protease)
#filter
data = refine.filter.filter_pdb_data(summary_report_csv=f'{output_dir}/summary_report_{protease}.csv'
, protein_name='chymotrypsin', exclude_terms=['zymogen','radiation damage','proenzyme','chymotrypsinogen'], max_res=2.5, organism=['bos taurus'] , max_muts=5)
# check if chains need to be combined
data = refine.check_chains.check_multimer_chains(ensemble_csv=f'{output_dir}/summary_report_{protease}_filtered.csv', allowance=10)
# renumber
data = build.renumber.download_and_renumber(summary_report_csv=f'{output_dir}/summary_report_{protease}_filtered_checked.csv', reference_pdb = ref_pdb,reference_chains=ref_chains,multimer_save=True, combine_chains_by='all')
# align
align, rmsd = build.alignment.align_all_pymol(directory=f'{output_dir}/Renumbered_unaligned_pdbs', alignment='chain A and name CA', loaded=False,reference_pdb=ref_pdb, cutoff=2.0, cycles=5,gap=-10.0, max_gap=50, extend=-0.5, max_skip=0, matrix='BLOSUM62', align_method="align", name=None,output_directory=None)
# renumber
refine.rename.rename_ambiguous(directory=f'{output_dir}/Alignment_on_chain_A_and_name_CA_ref_1ACB/Renumbered_aligned_pdbs_chain_A_and_name_CA_ref_1ACB')
# get MDev
MDev = analyze.atoms.get_MDev(directory=f'{output_dir}/Alignment_on_chain_A_and_name_CA_ref_1ACB/Renumbered_aligned_pdbs_chain_A_and_name_CA_ref_1ACB_renamed', chains=['A'], reference_PDB=ref_pdb, multiconformers=False)
# get distances
distances = analyze.atoms.get_distance_distributions(directory=f'{output_dir}/Alignment_on_chain_A_and_name_CA_ref_1ACB/Renumbered_aligned_pdbs_chain_A_and_name_CA_ref_1ACB_renamed', chains=['A'], multiconformers =False, quantile = 0.95,report_outliers = True, output_directory = None)