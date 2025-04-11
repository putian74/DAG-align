#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from Bio import SeqIO
import sys
from multiprocessing import Queue, Lock, Manager, Process, Value
import os
import time
import subprocess
from tqdm import tqdm
from collections import Counter
def split_fasta(input_file, output_prefix, chunk_size=5000):

    seq_num = 0               
    seq_records = []              
    seqfileList = []             
    i = 0                      
    for record in SeqIO.parse(input_file, "fasta"):
        seq_num += 1
        seq_records.append(record)
        if len(seq_records) == chunk_size:
            output_file = os.path.join(output_prefix, f"{i + 1}.fasta")
            seqfileList.append(str(output_file))
            with open(output_file, "w") as out_handle:
                SeqIO.write(seq_records, out_handle, "fasta")
            seq_records = []  
            i += 1
    if seq_records:
        output_file = os.path.join(output_prefix, f"{i + 1}.fasta")
        seqfileList.append(str(output_file))
        with open(output_file, "w") as out_handle:
            SeqIO.write(seq_records, out_handle, "fasta")
    return seq_num, seqfileList
def split_single_fasta(rootdir, outdir):

    index = 0         
    for subdir, _, files in tqdm(os.walk(rootdir)):
        for file in files:
            if file.endswith('.fna'):                  
                file_path = os.path.join(subdir, file)
                for record in SeqIO.parse(file_path, 'fasta'):
                    outfile = outdir / '{}.fasta'.format(record.id)
                    SeqIO.write(record, outfile, 'fasta')
                    index += 1         
    print(f"\nTotal {index} single-sequence files generated in {outdir}")