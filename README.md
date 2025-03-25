# Environment Configuration

## 1. Create a new conda virtual environment with Python 3.9

```bash
conda create -n myenv python=3.9
```

## 2. Activate the environment

```bash
conda activate myenv
```

## 3. Install required dependencies

```bash
pip install -r requirements.txt
```

## 4. Unzip Application Files

```bash
unzip DAPG_TPHMM-main.zip -d .
```
# DAPG-TPHMM

DAPG-TPHMM is used for FTO-DAPG construction and model training/alignment on the obtained FTO-DAPG.

## Usage

```bash
python DAPG_TPHMM/DAG_Ali.py [Fastapath] [outputpath] [fragment length] [threads]
```

### Parameters

| Parameter         | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `Fastapath`       | Directory path of a single FASTA format sequence file (all sequences must be in one file) |
| `outputpath`      | Directory path for saving output files                       |
| `fragment length` | Length of sequence fragments for FTO-DAPG construction (recommended: 32~128) |
| `threads`         | Number of threads (recommended: 36 for parallel training of 6 super parameter sets) |

### Output Results
- **Scores**: Sum of pairs score and entropy for each of 6 MSAs are saved in `outputpath/report.txt`
- **Alignments**: Alignment results (FASTA format) saved to `outputpath/bestAlign.fasta` (contains inserted gaps)

---

## Workflow
1. **Graph Construction**:
   - Splits input FASTA file into **1000-sequence subsets**
   - Builds a basic FTO-DAPG for each subset
   - Merges these FTO-DAPGs into a full graph through iterative two-to-one merging

2. **Training & Alignment**:
   - DAPG-TPHMM training is performed on the full graph
   - Viterbi alignment is run on **4000-sequence subgraphs**
   - 6 sets of results from different super parameter combinations are saved in `outputpath/V_result/alizips/`, including:
     - Graphs for all sequence subsets (full graph + intermediates)
     - MSA results for all sequences

---

## Super Parameter Sets
| Parameter Set | Global LW Threshold | Tail LW Threshold | Match Emission Smoothing Constant |
| ------------- | ------------------- | ----------------- | --------------------------------- |
| tr1           | 0.01                | 0.01              | exp(-3)                           |
| tr2           | 0.01                | 0.01              | exp(-5)                           |
| tr3           | 0.01                | 0.01              | exp(-7)                           |
| tr4           | 0.001               | 0.01              | exp(-3)                           |
| tr5           | 0.001               | 0.01              | exp(-5)                           |
| tr6           | 0.001               | 0.01              | exp(-7)                           |

*Note: Other parameters are initialized identically across all sets.*

---

## Dataset Information
- Genome IDs for the total genome dataset and SLSW/LLSW test sets are available in `Data.zip`
- Files to check:
  - Total dataset: `sequence_names.txt`
  - S/LLSW test sets: `SLSW_?_5000.txt` or `LLSW_?_5000.txt` (where `?` ranges from 1 to 24)

