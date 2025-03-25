# Environment Configuration

1.Create a new conda virtual environment with Python 3.9:

<BASH>
conda create -n myenv python=3.9


2.Activate the environment:

<BASH>
conda activate myenv


3.Install required dependencies:

<BASH>
pip install -r requirements.txt


# DAPG-TPHMM
DAPG-TPHMM, which is for FTO-DAPG construction and DAPG-TPHMM model training and alignment on the obtained FTO-DAPG.

DAPG-TPHMM usage:
<BASH>
python3 DAPG_TPHMM/DAG_Ali.py [Fastapath] [outputpath] [fragment length] [threads]

Fastapath is a directory for a single FASTA format sequence file to be processed, which implied that user need to place all sequences into a single FASTA file.
outputpath is a directory for saving the output specified by the user.
fragment length is the length of sequence fragment utilized for FTO-DAPG construction, the recommended value is in the range of 32~128
threads is the number of threads to be utilized by DAPG-TPHMM, 24 is recommended for parallel training of six different super parameter sets.
The sum of pairs score and entropy for each of 6 MSA is saved in the outputpath/report.txt 
Alignment results (FASTA format) are saved to "outputpath/bestAlign.fasta". The file size is larger than the initial FASTA sequence file due to gap insertions.

DAPG-TPHMM splits the input FASTA file into 1000-sequence subsets, and construct a basic FTO-DAPG for each of which. These FTO-DAPGs are subsequently merged into the full graph through iterative two-to-one merging operation. The DAPG-TPHMM training is carried out on the full graph and Viterbi performed on 4000-sequence subgraphs. Six sets of results from different super parameter combinations (described below) are saved in the outputpath/V_result/alizips/ directory. Each set includes graphs (for each sequence subsets, the full graph and intermediate ones ) and MSA for all sequences.

6 sets of super parameters are listed below (other parameter initialization is the same for all sets)

parameter set		tr1	tr2	tr3	tr4	tr5	tr6
global LW threshold	0.01	0.01	0.01	0.001	0.001	0.001
Tail LW threshold	0.01	0.01	0.01	0.01	0.01	0.01
Match Emission
smoothing constant	exp(-3)	exp(-5)	exp(-7)	exp(-3)	exp(-5)	exp(-7)

Genome ID for the total genome dataset and SLSW/LLSW accuracy test sets are presented in the Data.zip. Unzip and see sequence_names.txt for genome ID of the total set, see S/LLSW_?_5000.txt for genome ID included in each S/LLSW set, with ? in the range [1,24].
