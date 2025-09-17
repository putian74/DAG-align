# **DAPG-TPHMM: Accurate Multiple Sequence Alignment of Ultramassive Genome Sets**

**DAPG-TPHMM** is a high-performance, high-accuracy multiple sequence alignment (MSA) tool designed to handle ultramassive genomic datasets containing millions of sequences. It leverages a novel pangenome graph structure (FTO-DAPG) and a Tiled Profile Hidden Markov Model (TPHMM) to achieve state-of-the-art accuracy with unprecedented speed and scalability.

This software is the official implementation for the research paper: *"Accurate Multiple Sequence Alignment of Ultramassive Genome Sets"*.

## **Features**

* **High Accuracy**: Achieves alignment accuracy comparable to the gold-standard aligner, Muscle.  
* **Massive Scalability**: Efficiently aligns millions of sequences by eliminating redundancy through a graph-based structure.  
* **Parallel Processing**: Utilizes multi-threading for both graph construction and alignment stages to significantly accelerate computation.  
* **Adaptive Parameter Selection**: Automatically analyzes input data to select optimal parameters (fragment\_length) for different levels of sequence diversity.  
* **Large-Scale Mode**: A specialized mode (-l) with optimized presets for aligning massive datasets quickly.  
* **Compressed Output**: Implements a novel compressed sparse alignment format (.npz) for efficient storage of massive alignment results.

## **Installation**

### **1\. Create a new conda virtual environment with Python 3.9**

```
conda create \-n myenv python=3.9
```

### **2\. Activate the environment**

```
conda activate myenv
```

### **3\. Unzip Application Files**

```
unzip DAPG\_TPHMM-main.zip \-d . 
```

### **4\. Install required dependencies**

Navigate into the unzipped directory containing requirements.txt and run:

```
pip install \-r requirements.txt
```

## **Usage**

The main script is DAG\_Ali.py. Below are the command-line options and examples.

### **Basic Command**

```
python3 DAG_Ali.py -i <input.fasta> -o <output_directory> -t <threads\>
```



### **Command-Line Arguments**

| Argument | Short | Type | Default | Description |
| :---- | :---- | :---- | :---- | :---- |
| \--input | \-i | string | **Required** | Path to the input FASTA file. |
| \--output | \-o | string | **Required** | Path to the directory where results will be saved. |
| \--threads | \-t | integer | 36 | Number of parallel threads to use. |
| \--large\_scale | \-l | flag | False | Enables Large-Scale Mode for massive datasets. Presets chunk\_size to 5000 and fragment\_length to 32 (unless otherwise specified) and skips pre-analysis for speed. |
| \--fragment\_Length | \-f | integer | auto | Fragment length for building the DAPG. In standard mode, this is determined automatically based on k-mer diversity. In large-scale mode, it defaults to 32\. |
| \--chunk\_size | \-c | integer | auto | The number of sequences per chunk for parallel processing. In standard mode, this is determined adaptively based on average sequence length. In large-scale mode, it defaults to 5000\. |
| \--Onlybuild | \-b | flag | False | If specified, the program will only perform graph construction and will not proceed to the alignment stage. |

### **Modes of Operation**

DAPG-TPHMM has two main operating modes tailored for different dataset sizes.

#### **1\. Standard Mode (Default)**

This is the default mode for datasets of moderate size. It performs a pre-analysis of the input sequences to determine the optimal fragment\_length and chunk\_size automatically, balancing speed and accuracy.

* **fragment\_length**: Set to 16 for highly diverse sequences (k-mer diversity \>= 0.9) and 32 otherwise.  
* **chunk\_size**: Determined adaptively based on the average sequence length.

**Example:**

python3 DAG\_Ali.py \-i my\_sequences.fasta \-o ./results \-t 24

#### **2\. Large-Scale Mode (-l)**

This mode is specifically optimized for aligning massive datasets (e.g., \>100,000 sequences). When the \-l flag is used, the program skips the time-consuming pre-analysis and uses robust default parameters.

* **fragment\_length**: Defaults to 32 (can be overridden with \-f).  
* **chunk\_size**: Defaults to 5000 (can be overridden with \-c).

**Example for a very large dataset:**

python3 DAG\_Ali.py \-i viral\_sequences.fasta \-o ./align\_results \-t 36 \-l

## **Output Structure**

The program implements an adaptive output strategy based on the number of input sequences to balance convenience with storage efficiency.

* **For datasets with 40,000 or fewer sequences:**  
  * In addition to the compressed results, a standard FASTA file of the final alignment will be generated at \<output\_directory\>/align\_result.fasta.  
* **For datasets with more than 40,000 sequences:**  
  * To conserve disk space, the standard FASTA output is suppressed. The alignment results are saved *exclusively* in the compressed sparse format (.npz).  
* **Dual Alignment Results & Selection:**  
  * The software runs the alignment process twice with different initial parameters to find a better result. These two compressed alignments are saved in:  
    * \<output\_directory\>/V\_result/alizips/tr0  
    * \<output\_directory\>/V\_result/alizips/tr1  
  * A report file is generated at \<output\_directory\>/report.txt. This file contains the final likelihood scores for both alignment runs. **Users should consult this file to select the alignment with the higher score for downstream analysis.**
