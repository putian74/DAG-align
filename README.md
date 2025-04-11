
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
python DAPG_TPHMM/DAG_Ali.py -i [INPUT_FASTA] -o [OUTPUT_DIR] -f [FRAGMENT_LENGTH] -t [THREADS] -c [CHUNK_SIZE]
```

### Parameters

| Parameter               | Description                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| `-i, --input`           | Path to input FASTA file containing all sequences (required)               |
| `-o, --output`          | Output directory path to save results (required)                           |
| `-f, --fragment_Length` | Fragment length for FTO-DAPG construction (default: 16, recommended:16~64) |
| `-t, --threads`         | Number of parallel threads (default: 36)                                   |
| `-c, --chunk_size`      | Sequence chunk size for splitting large datasets (default: 5000)           |

### Output Results
- **Scores**: Sum of pairs score and entropy for each of 6 MSAs are saved in `outputpath/report.txt`
- **Alignments**: Alignment results (FASTA format) saved to `outputpath/bestEntropy.fasta` and `outputpath/bestSP.fasta`  

---

## Workflow
1. **Graph Construction**:
   - Splits input FASTA file into **${chunk_size}-sequence subsets** using specified chunk size
   - Builds a basic FTO-DAPG for each subset
   - Merges these FTO-DAPGs into a full graph through iterative two-to-one merging

2. **Training & Alignment**:
   - DAPG-TPHMM training is performed on the full graph
   - Graphs for all sequence and subsets saved in `outputpath/subgraphs/1` and `outputpath/Merging_graphs/
   - Viterbi alignment is run on **20000-sequence subgraphs（default)**
   - 6 sets of results from different super parameter combinations are saved in `outputpath/V_result/alizips/tr*/`where `*` ranges from 1 to 6, including:
     - Column-compressed full MSA results saved in `alizip.npz`

---

## Super Parameter Sets
| Parameter Set | Parameter Initialization Methods | Global LW Threshold | Match Emission Smoothing Constant |
| ------------- | -------------------------------- | ------------------- | --------------------------------- |
| tr1           | Length First                     | 0.01                | exp(-3)                           |
| tr2           | Length First                     | 0.01                | exp(-5)                           |
| tr3           | Length First                     | 0.01                | exp(-7)                           |
| tr4           | Weight First                     | 0.01                | exp(-3)                           |
| tr5           | Weight First                     | 0.01                | exp(-5)                           |
| tr6           | Weight First                     | 0.01                | exp(-7)                           |

*Note: Other parameters are initialized identically across all sets.*

---

## Dataset Information
- Genome IDs for the total genome dataset and SLSW/LLSW test sets are available in `Data.zip`
- Files to check:
  - Total dataset: `sequence_names.txt`
  - S/LLSW test sets: `SLSW_?_5000.txt` or `LLSW_?_5000.txt` (where `?` ranges from 1 to 24)