#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import random
import statistics
import gc
import copy
import argparse
import time
from typing import List, Tuple, Optional, Set
from itertools import combinations
from multiprocessing import Process, Value, Manager, Lock, Queue, Pool, cpu_count

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import numpy as np
from DAG_operator import *
from DAG_Phmm import DAGPhmm
from sqlite_master import sql_master



def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    A simple and fast FASTA file parser.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        A list of tuples, where each tuple contains a header and its sequence.
    """
    sequences = []
    header, sequence_parts = None, []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header:
                    sequences.append((header, ''.join(sequence_parts)))
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header:
        sequences.append((header, ''.join(sequence_parts)))
    return sequences

def print_x(start, step, allstep, current_stage="Sequence alignment is in progress"):
    """Formats and prints a progress bar to the console."""
    runtime = time.time() - start
    percent = step / allstep
    bar = ('#' * int(percent * 20)).ljust(20)
    
    hours, remainder = divmod(runtime, 3600)
    mins, secs = divmod(remainder, 60)
    time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))

    sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ({time_format}) | Stage: {current_stage:<30}')
    sys.stdout.flush()



def _process_kmer_chunk(sequences_chunk: List[Tuple[str, str]], k_size: int) -> Tuple[int, Set[str]]:
    """
    Worker function for multiprocessing. Processes a chunk of sequences to find
    total and unique k-mers. Must be a top-level function.
    """
    local_unique_kmers = set()
    local_total_kmers = 0
    for header, sequence in sequences_chunk:
        if len(sequence) < k_size:
            continue
        
        num_kmers_in_seq = len(sequence) - k_size + 1
        local_total_kmers += num_kmers_in_seq
        
        for i in range(num_kmers_in_seq):
            local_unique_kmers.add(sequence[i:i+k_size])
            
    return (local_total_kmers, local_unique_kmers)

def calculate_kmer_diversity_ratio(fasta_path: str, sample_size: int = 10000, k_size: int = 21, threads: Optional[int] = None) -> Optional[float]:
    """
    Calculates the ratio of unique k-mers to total k-mers from a sample of sequences
    using multiple processes to speed up the calculation.
    """
    if not os.path.exists(fasta_path):
        print(f"Error: Input FASTA file not found at '{fasta_path}'", file=sys.stderr)
        return None
        
    try:
        all_sequences = parse_fasta(fasta_path)
    except Exception as e:
        print(f"Error parsing FASTA file: {e}", file=sys.stderr)
        return None

    if not all_sequences:
        print("Warning: FASTA file is empty or could not be parsed.", file=sys.stderr)
        return None

    if len(all_sequences) > sample_size:
        sampled_sequences = random.sample(all_sequences, sample_size)
    else:
        sampled_sequences = all_sequences

    num_threads = threads if threads is not None else cpu_count() or 1
    
    unique_kmers = set()
    total_kmers = 0

    if num_threads == 1 or len(sampled_sequences) < num_threads * 2:
        total_kmers, unique_kmers = _process_kmer_chunk(sampled_sequences, k_size)
    else:
        chunk_size = (len(sampled_sequences) + num_threads - 1) // num_threads
        chunks = [sampled_sequences[i:i + chunk_size] for i in range(0, len(sampled_sequences), chunk_size)]
        pool_args = [(chunk, k_size) for chunk in chunks]

        with Pool(processes=num_threads) as pool:
            results = pool.starmap(_process_kmer_chunk, pool_args)

        total_kmers = sum(res[0] for res in results)
        unique_kmers = set().union(*(res[1] for res in results))

    if total_kmers == 0:
        return 0.0

    ratio = len(unique_kmers) / total_kmers
    return ratio

def sequence_to_kmers(sequence: str, k: int) -> Set[str]:
    """Converts a DNA sequence into a set of its k-mers."""
    if len(sequence) < k:
        return set()
    return {sequence[i:i+k] for i in range(len(sequence) - k + 1)}

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculates the Jaccard similarity between two sets of k-mers."""
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def _calculate_jaccard_similarity_pair(pair: Tuple[Set[str], Set[str]]) -> float:
    """Helper function for multiprocessing.Pool.map."""
    return calculate_jaccard_similarity(pair[0], pair[1])

def estimate_average_similarity(
    fasta_path: str, num_samples: int = 5, sample_size: int = 1000,
    threads: Optional[int] = None, k_size: int = 21
) -> Optional[float]:
    """Estimates the average pairwise sequence similarity using a k-mer based Jaccard index."""
    if not os.path.exists(fasta_path):
        print(f"Error: Input FASTA file not found at '{fasta_path}'", file=sys.stderr)
        return None

    try: all_sequences = parse_fasta(fasta_path)
    except Exception as e:
        print(f"Error parsing FASTA file: {e}", file=sys.stderr)
        return None
        
    total_seq_count = len(all_sequences)
    if total_seq_count < 2: return 1.0
    
    actual_sample_size = min(sample_size, total_seq_count)
    sample_avg_similarities = []
    num_threads = threads if threads is not None else cpu_count() or 1
    print(f"Starting similarity estimation with {num_samples} samples of size {actual_sample_size} using {num_threads} threads...")

    for i in range(num_samples):
        print(f"  - Processing sample {i + 1}/{num_samples}...")
        sampled_sequences = random.sample(all_sequences, actual_sample_size) if actual_sample_size < total_seq_count else all_sequences
        kmer_sets = [sequence_to_kmers(seq, k_size) for _, seq in sampled_sequences]
        pairs = list(combinations(kmer_sets, 2))
        
        if not pairs: sample_avg = 1.0
        else:
            if num_threads > 1 and len(pairs) > 1000:
                with Pool(processes=num_threads) as pool:
                    pairwise_similarities = pool.map(_calculate_jaccard_similarity_pair, pairs)
            else:
                pairwise_similarities = [_calculate_jaccard_similarity_pair(p) for p in pairs]
            sample_avg = statistics.mean(pairwise_similarities)
        sample_avg_similarities.append(sample_avg)

    if not sample_avg_similarities: return None
    return statistics.mean(sample_avg_similarities)




def ini_paras(Ref_seq, emProbMatrix,insertRanges, ME, MD, MI, II, DM, pi_MID, outpath, parasName,perturbation=(0,0)):

    assert np.exp(MD) + np.exp(MI) < 1.0 - 1e-6, "The probability of MD+MI exceeds the valid range"
    assert np.exp(II) < 1.0 - 1e-6, "II probability invalid"
    assert np.exp(DM) < 1.0 - 1e-6, "DM probability invalid"
    assert emProbMatrix.shape[0] == 4, "Dimension error in emission probability matrix"
    assert emProbMatrix.shape[1] == len(Ref_seq), "The length of the emission probability does not match the reference sequence"
        
    n_positions = len(Ref_seq) + 1  
    pi_sum = np.sum(pi_MID)      

    mm_base = np.log(1 - np.exp(MD) - np.exp(MI))  
    im_base = np.log(1 - np.exp(II))           
    dd_base = np.log(1 - np.exp(DM)) 

    _mi = np.full(n_positions, MI, dtype=np.float64)    
    _md = np.full(n_positions, MD, dtype=np.float64)    
    _mm = np.full(n_positions, mm_base, dtype=np.float64) 
    hight_MI = np.logaddexp2(MI,np.log(0.1))
    for rg in insertRanges:
        for i in range(rg[0],rg[1]):
            _mi[i]=hight_MI

    _mi[0] = np.log(pi_MID[1]/pi_sum)  
    _md[0] = np.log(pi_MID[2]/pi_sum)  
    _mm[0] = np.log(pi_MID[0]/pi_sum) 

    _mm[-1] = ME           
    _mi[-1] = np.log(1 - np.exp(ME))  

    _ii = np.full(n_positions, II, dtype=np.float64)  
    _im = np.full(n_positions, im_base, dtype=np.float64)  
    _id = np.full(n_positions, -np.inf, dtype=np.float64)  

    _dm = np.full(n_positions, DM, dtype=np.float64)  
    _dd = np.full(n_positions, dd_base, dtype=np.float64)  
    _di = np.full(n_positions, -np.inf, dtype=np.float64)  

    _dm[0] = _dd[0] = _dd[-1] = -np.inf  
    _dm[-1] = 0  

    _em = np.log(emProbMatrix.T + 1e-16)  
    _ei = np.full((_em.shape[0]+1, _em.shape[1]), np.log(0.25), dtype=np.float64)  

    parameterDict = {
        "_mm": _mm, "_md": _md, "_mi": _mi,
        "_im": _im, "_id": _id, "_ii": _ii,
        "_dm": _dm, "_dd": _dd, "_di": _di,
        "match_emission": _em, "insert_emission": _ei
    }

    np.save(outpath/"ini/init_{}.npy".format(parasName), parameterDict)


def ref_graph_build(graph_path, thr=0.01, MissMatchScore=-5):

    ref_dict = np.load(graph_path/'thr_{}.npz'.format(thr))

    ref_seq = str(ref_dict['ref_seq'])
    ref_node_list = list(ref_dict['ref_node_list'])
    emProbMatrix = ref_dict['emProbMatrix']
    insertRanges = ref_dict['insert_range']
    emProbMatrix += np.exp(MissMatchScore)
    sum_of_emProbMatrix = np.sum(emProbMatrix, axis=0)
    emProbMatrix = emProbMatrix / sum_of_emProbMatrix
    sum_of_emProbMatrix = np.sum(emProbMatrix, axis=0)
    emProbMatrix = emProbMatrix / sum_of_emProbMatrix
    pi_MID = [1, 1, 1]
    return ref_seq, ref_node_list, emProbMatrix, pi_MID,insertRanges

def train(DAG,Viterbi_result_path, finalGraphPath, train_DAG_Path, hyperParameterDict,lock,start=None,taskNum=None,allstep=None, threads=3,fit=True):
    hyperParameterDict = dict(hyperParameterDict)
    for index in hyperParameterDict.keys():            
        parasName = 'tr{}'.format(index)               
        modifyDict = {}           
        modifyDict['emProbAdds_Match'] = hyperParameterDict[index][1]          
        modifyDict['emProbAdds_Match_head'] = modifyDict['emProbAdds_Match']            
        modifyDict['emProbAdds_Match_tail'] = modifyDict['emProbAdds_Match']            
        modifyDict['random'] = hyperParameterDict[index][2]
        modifyDict['init_M2D'] = -2                  
        modifyDict['init_M2I'] = -5                 
        modifyDict['init_I2I'] = np.log(1/2)           
        modifyDict['init_D2M'] = np.log(1/2)                 
        modifyDict['init_M2End'] = np.log(1/2)          
        modifyDict['weight_thr'] = hyperParameterDict[index][0]         
        modifyDict['head_length'] = 50             
        modifyDict['tail_length'] = 50        
        modifyDict['trProbAdds_mm'] = -3                   
        modifyDict['trProbAdds_md'] = -4                   
        modifyDict['trProbAdds_mi'] = -50                   
        modifyDict['trProbAdds_PiM'] = -1            
        modifyDict['trProbAdds_PiI'] = -1             
        modifyDict['trProbAdds_PiD'] = -1             
        modifyDict['trProbAdds_im'] = -1                   
        modifyDict['trProbAdds_ii'] = -1          
        modifyDict['trProbAdds_iend'] = -1           
        modifyDict['trProbAdds_ii_tail'] = -1             
        modifyDict['trProbAdds_mend'] = -10            
        modifyDict['trProbAdds_mi_tail'] = -10              
        np.save(train_DAG_Path/'ini/{}_modifydict.npy'.format(parasName), modifyDict)
        outlog = open(Viterbi_result_path/'train_{}.log'.format(parasName), 'w')
        sys.stdout = outlog               
        sys.stderr = outlog         
        windows_length = 100               
        ref_seq, ref_node_list, emProbMatrix, pi_MID,insertRanges = ref_graph_build(
            finalGraphPath,
            thr=modifyDict['weight_thr'],         
            MissMatchScore=modifyDict['emProbAdds_Match']         
        )
        ini_paras(
            ref_seq,        
            emProbMatrix,         
            insertRanges,
            modifyDict['init_M2End'],    
            modifyDict['init_M2D'],          
            modifyDict['init_M2I'],          
            modifyDict['init_I2I'],        
            modifyDict['init_D2M'],          
            pi_MID,         
            train_DAG_Path,         
            parasName,         
            modifyDict['random']
        )
        parameter_path = train_DAG_Path/"ini/init_{}.npy".format(parasName)
        ph = DAGPhmm(
            train_DAG_Path,               
            train_DAG_Path,                 
            parasName,          
            parameter_path=parameter_path         
        )
        if fit:
            ph.init_train_data_with_DAG(train_DAG_Path,ref_node_list,ref_seq,modifyDict,copy.deepcopy(DAG),True,windows_length, threads)
            
            ph.fit()
        sys.stdout = sys.__stdout__         
        sys.stderr = sys.__stderr__         
        if not taskNum is None:
            lock.acquire()
            taskNum.value+=1
            print_x(start,taskNum.value,allstep,'Training alignment model')
            lock.release()

def write_report(sp_and_entropy, outpath):
    sp_list = [item[0] for item in sp_and_entropy]
    entropy_list = [item[1] for item in sp_and_entropy]
    
    bestsp_idx = np.argmax(sp_list)
    besten_idx = np.argmin(entropy_list)
    
    header = [' ', 'sp score', 'entropy', 'best entropy', 'best sp score', 'alignment length']
    data_rows = [header]
    
    for i, item in enumerate(sp_and_entropy):
        is_best_sp = '*' if i == bestsp_idx else ''
        is_best_en = '*' if i == besten_idx else ''
        
        row = [
            f'parameter {i}',
            item[0],          
            item[1],          
            is_best_en,       
            is_best_sp,       
            item[-2]          
        ]
        data_rows.append(row)
        
    formatted_rows = []
    for row in data_rows:
        formatted_row = []
        for i, cell in enumerate(row):
            if isinstance(cell, float):
                formatted_row.append(f"{cell:.5f}")
            else:
                formatted_row.append(str(cell))
        formatted_rows.append(formatted_row)
        
    col_widths = [max(len(item) for item in col) for col in zip(*formatted_rows)]
    
    with open(outpath / 'report.txt', 'w', encoding='utf-8') as file:
        header_line = ' | '.join(cell.ljust(width) for cell, width in zip(formatted_rows[0], col_widths))
        file.write(header_line + '\n')
        
        separator_line = '-+-'.join('-' * width for width in col_widths)
        file.write(separator_line + '\n')
        
        for row in formatted_rows[1:]:
            data_line = ' | '.join(cell.ljust(width) for cell, width in zip(row, col_widths))
            file.write(data_line + '\n')
            
    return besten_idx, bestsp_idx

def zipAlign2Fasta(zipAliPath,save_path,ref:bool=False):
    draw_dict = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
    zipAliPath = sanitize_path(zipAliPath,'input')
    save_path = sanitize_path(save_path,'output')
    alignInfo = np.load(zipAliPath,allow_pickle=True)
    zipali = alignInfo['align'].tolist()[:-1]
    namelist = alignInfo['namelist'].tolist()
    if ref:
        add=1
    else:
        add=0
    ali_matrix = np.full((len(namelist)+add,len(zipali)),0,dtype=np.uint8)
    for index,ali in enumerate(zipali):
        ali_matrix[:,index] = ali[0]
        for base in ali[1:]:
            ali_matrix[np.array(base[1])+add,index] = base[0]
    vectorized_draw_dict = np.vectorize(draw_dict.get)
    string_matrix = vectorized_draw_dict(ali_matrix)
    xs = np.arange(len(string_matrix))
    if ref:
        namelist.insert(0,'ref')
    seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in zip(xs,string_matrix)]
    SeqIO.write(seqlist,save_path,'fasta')


def save_refseq(graphPath,graph, thrs):
    graph.fragmentReduce()
    no_degenerate_edgeDict,pureNodes = graph.edgeWeightDict,set(np.arange(graph.totalNodes)) 
    
    for thr in thrs:
        _ref_seq, _ref_node_list, _emProbMatrix,_insertrgs = graph.convertToAliReferenceDAG_new(no_degenerate_edgeDict,pureNodes,thr)
        np.savez(
            graphPath/'thr_{}.npz'.format(thr),
            ref_seq=_ref_seq,
            ref_node_list=_ref_node_list,
            emProbMatrix=_emProbMatrix,
            insert_range=_insertrgs
        )
        

def train_in_all(hierarchy, finalGraphPath, outpath, subgraph_num,threads=3, fit=True):
    thrs = [0.001]
    Matchs = [-3,-5]
    Randoms = [(0,0)]
    finalGraphPath = sanitize_path(finalGraphPath, 'input') 
    outpath = sanitize_path(outpath, 'output')  
    start = time.time()
    Viterbi_result_path = outpath / 'V_result'
    if not os.path.exists(Viterbi_result_path):
        os.mkdir(Viterbi_result_path)
    train_DAG_Path = finalGraphPath
    os.makedirs(train_DAG_Path / 'ini', exist_ok=True)
    taskNum = Value('i',0)
    v_hierarchy = min(hierarchy,2)
    a = subgraph_num
    b = 2 ** v_hierarchy
    min_num = (a // b) + (1 if a % b != 0 else 0) 
    subgraphList = list(range(
        (1-1) * (2 ** (hierarchy - v_hierarchy)) + 1,
        min(min_num+1, 1 * (2 ** (hierarchy - v_hierarchy)) + 1)
    ))

    allstep=1
    allstep+=len(thrs)*len(Matchs)*len(Randoms)

    print_x(start, taskNum.value, allstep,'Loading graph data')
    lock = Manager().Lock()
    
    paraNames=[]
    hyperParameterDict = {}
    idx=0
    
    graph = load_DAG(finalGraphPath)
    graph.SourceList = [ [nodeid] for nodeid in range(graph.totalNodes)]
    graph.fragmentReduce(16)
    graph.merge_sameCoorFraNodes_loop_zip_new()
    taskNum.value+=1
    print_x(start, taskNum.value, allstep,'Preparing model parameters')
    save_refseq(finalGraphPath,copy.deepcopy(graph), thrs)
    filtered_files = []
    for thr in thrs:
        filtered_files.append('thr_{}'.format(thr))
    for f in filtered_files:
        datas = f.split('_')
        thr = float(datas[1])
        for m in Matchs:
            for rdm in Randoms:
                hyperParameterDict[idx] = (thr,m,rdm)
                paraNames.append('tr{}'.format(idx))
                idx+=1
    hyperParameterList = list(hyperParameterDict.items())
    taskList = Queue()
    os.makedirs(Viterbi_result_path/'alizips/Logs',exist_ok=True)
    for subGid in subgraphList:
        taskList.put([subGid])
        for para in paraNames:
            os.makedirs(Viterbi_result_path/'alizips/{}'.format(para),exist_ok=True)
    if subgraph_num>400:
        train_threads = 2
    else:
        train_threads = threads
    processlist = []
    pool_num = min(6,train_threads)
    print_x(start, taskNum.value, allstep,'Training alignment model')
    for pid in range(pool_num):
        processlist.append(Process(target=train, args=(graph,
            Viterbi_result_path, finalGraphPath, train_DAG_Path,
            hyperParameterList[pid::pool_num],lock,start,taskNum,allstep, threads//pool_num,fit, )))
    [p.start() for p in processlist]
    [p.join() for p in processlist]
    del graph
    gc.collect()

def merge_zipali_new(iniPath,subZipaliPath, mergedZipaliPath, paraIndexs, sp_and_entropy, v_hierarchy):
    negative = {          
        1: {1:  0, 3: -1, 4: -1, 2: -1 ,0: -2},
        3: {1: -1, 3:  0, 4: -1, 2: -1 ,0: -2},
        4: {1: -1, 3: -1, 4:  0, 2: -1 ,0: -2},
        2: {1: -1, 3: -1, 4: -1, 2:  0 ,0: -2},
        0: {1: -2, 3: -2, 4: -2, 2: -2 ,0: 0},
        } 
    positive = {          
        1: {1:  1, 3:  0, 4:  0, 2:  0 ,0:  0},
        3: {1:  0, 3:  1, 4:  0, 2:  0 ,0:  0},
        4: {1:  0, 3:  0, 4:  1, 2:  0 ,0:  0},
        2: {1:  0, 3:  0, 4:  0, 2:  1 ,0:  0},
        0: {1:  0, 3:  0, 4:  0, 2:  0 ,0:  0},
        }
    def reformat_col_data(data):
        total_ids = data[-1]
        columns = data[:-1]
        processed_columns = []
        all_position = set(np.arange(total_ids))
        for col in columns:
            main_char = col[0]
            minor_components = col[1:]

            if minor_components:
                component_num_list = [len(n[1]) for n in minor_components]
                ori_main_num = total_ids-sum(component_num_list)
                if ori_main_num < max(component_num_list):

                    new_main_char = minor_components[np.argmax(component_num_list)][0]

                    ori_less_ids = set()
                    for comp in minor_components:
                        ori_less_ids|=set(comp[1])

                    ori_main_components = all_position-ori_less_ids

                    if ori_main_components:
                        minor_components.append([main_char,list(ori_main_components)])

                    minor_components = [n for n in minor_components if n[0]!=new_main_char]
                    main_char = new_main_char
                    sorted_minor_components = sorted(
                        minor_components,
                        key=lambda x: min(x[1])
                    )
                else:
                    sorted_minor_components = sorted(
                        minor_components,
                        key=lambda x: min(x[1])
                    )
            else:
                sorted_minor_components=[]
            reformed_col = [main_char] + sorted_minor_components

            processed_columns.append(reformed_col)
        return processed_columns + [total_ids]

    os.makedirs(mergedZipaliPath/'alizips', exist_ok=True) 
    for paramsIdx in paraIndexs:
        parasName = 'tr{}'.format(paramsIdx)
        if v_hierarchy==0:
            fileList = os.listdir(subZipaliPath/'subgraphs/')
            fileList = [file for file in fileList if 'alizips' not in file]
            fileList = sorted(fileList, key=int)
        if v_hierarchy == hierarchy:
            fileList = [1]
        else:
            fileList = os.listdir(subZipaliPath/'Merging_graphs/merge_{}/'.format(v_hierarchy))
            fileList = [file for file in fileList if 'alizips' not in file]
            fileList = sorted(fileList, key=int)               
        maxInsertLengthGlobal = np.load(mergedZipaliPath/'{}/{}/insert_length_dict.npy'.format(fileList[0],parasName),allow_pickle=True).item()
        for i in range(1, len(fileList)):
            maxInsertLengthSub = np.load(mergedZipaliPath/'{}/{}/insert_length_dict.npy'.format(fileList[i],parasName),allow_pickle=True).item()
            maxInsertLengthGlobal = {key: max(maxInsertLengthGlobal[key], maxInsertLengthSub[key]) for key in maxInsertLengthGlobal}

        sequenceNameList = []  
        zipali_global = []    
        for i in range(len(fileList)):
            maxInsertLengthSub = np.load(mergedZipaliPath/'{}/{}/insert_length_dict.npy'.format(fileList[i],parasName),allow_pickle=True).item()
            indexdict = np.load(mergedZipaliPath/'{}/{}/indexdict.npy'.format(fileList[i],parasName),allow_pickle=True).item() 
            alignInfo = np.load(mergedZipaliPath/'{}/{}/zipalign.npz'.format(fileList[i],parasName),allow_pickle=True)
            zipali = alignInfo['align'].tolist()[:-1]  
            namelist = alignInfo['namelist']
            insertList = []
            for x in maxInsertLengthGlobal.keys():
                if x in maxInsertLengthSub.keys():
                    insert_global = maxInsertLengthGlobal[x]
                    insert_local = maxInsertLengthSub[x]
                    if insert_global > insert_local:
                        insertList.append((indexdict[(2,x)], insert_global - insert_local))
            insertList.sort(key=lambda x: x[0], reverse=True)
            for j in insertList:
                for k in range(j[1]):
                    zipali.insert(j[0], [0])

            if not zipali_global:  
                global_ali_length = len(zipali)
                for columnCursor in range(global_ali_length):
                    componentNum = len(zipali[columnCursor])
                    for component in range(1, componentNum):
                        zipali[columnCursor][component][1] = zipali[columnCursor][component][1].tolist()
                zipali_global = zipali
            else:  
                addSequenceNum = len(sequenceNameList)
                global_ali_length = len(zipali_global)
                for columnCursor in range(global_ali_length):
                    componentIndicesDict = {}
                    componentLists = zipali[columnCursor]

                    for k in range(1, len(componentLists)):
                        componentAndIndices = componentLists[k]
                        component = componentAndIndices[0]
                        indicesArray = componentAndIndices[1]  
                        indicesArray += addSequenceNum
                        componentIndicesDict.setdefault(component, []).extend(indicesArray.tolist())
                    donebase = set()  
                    componentNum = len(zipali_global[columnCursor])
                    
                    for bindex in range(1, componentNum):
                        component = zipali_global[columnCursor][bindex][0]
                        if component in componentIndicesDict:
                            zipali_global[columnCursor][bindex][1] += componentIndicesDict[component]
                            donebase.add(component)
                    for component in componentIndicesDict.keys() - donebase:
                        zipali_global[columnCursor].append([component, componentIndicesDict[component]])

            sequenceNameList.extend(namelist)  
        zipali_global.append(len(sequenceNameList))  
        zipali_global = reformat_col_data(zipali_global)
        if not os.path.exists(mergedZipaliPath/'alizips'/parasName):
            os.mkdir(mergedZipaliPath/'alizips'/parasName)

        a, c,b = sp_entro_zip_loaded(zipali_global, positive, negative)
        sp_and_entropy.append((a, c,len(zipali_global), paramsIdx))
        zipali_global = np.array(zipali_global, dtype=object)
        np.savez(mergedZipaliPath/'alizips/{}/zipalign.npz'.format(parasName), 
                 namelist=sequenceNameList, 
                 align=zipali_global)
        
def viterbi_whithout_mapping(parameter_path, Viterbi_DAG_Path, Viterbi_result_path, parasName, seqiddb, ref_seq,DAG, threads=3):
    
    windows_length = 100
    
    ph = DAGPhmm(
        '',        
        Viterbi_DAG_Path,     
        parasName,             
        parameter_path=parameter_path  
    )
    ph.init_viterbi_data_with_refseq(
        Viterbi_DAG_Path,
        Viterbi_result_path,
        ref_seq,DAG,
        windows_length=windows_length,
        threads=threads
    )
    ph.Viterbi(seqiddb,threads=threads)
    ph.state_to_aligment(seqiddb)
    del ph
    gc.collect()

def viterbi_in_Graph(testGraphPath, iniPath,outpath,threads,chunk_size,seq_num):
    
    def subViterbi():
        while not taskList.empty():
            try:
                task = taskList.get(timeout=1)
            except:
                continue

            subGraphLevel = v_hierarchy
            index = task[0]
            oriGraphPath = testGraphPath / "subgraphs/"
            oriGraphIdlist = range(
                (index-1)*(2**subGraphLevel)+1, 
                min(suborigraph_num+1, index*(2**subGraphLevel)+1)
            )
            with open(Viterbi_result_path / f'alizips/Logs/{index}.log', 'w',buffering=1) as outlog:
                sys.stdout = outlog
                sys.stderr = outlog
                
                Viterbi_DAG_Path = testGraphPath/f'subgraphs/{index}' if v_hierarchy == 0 else \
                                 testGraphPath / f"Merging_graphs/merge_{subGraphLevel}/{index}/"
                if v_hierarchy == hierarchy:
                    Viterbi_DAG_Path = testGraphPath/'final_graph'

                with lock:                
                    datalist = []
                    for i in oriGraphIdlist:
                        array = np.load(oriGraphPath/'{}/osm.npy'.format(i),allow_pickle=True)
                        datalist.append([i,array])
                seqiddb = sql_master('',db=oriGraphPath,dbidList=datalist)
            
                for paraid in range(paraNum):
                    parasName = f'tr{paraid}'

                    sub_Viterbi_result_path = Viterbi_result_path / f'{index}/'
                    os.makedirs(sub_Viterbi_result_path, exist_ok=True)

                    parameter_path = iniPath/'final_graph/ini/{}_pc_1.npy'.format(parasName)
                    parameterDict=np.load(parameter_path,allow_pickle=True).item() 
                    Me_Matrix=parameterDict['match_emission']
                    baseDict = {0:'A',1:'T',2:'C',3:'G'}
                    ref_seq = ''
                    for i in range(Me_Matrix.shape[0]):
                        ref_seq += baseDict[np.argmax(Me_Matrix[i,:])]
                    
                    (Viterbi_result_path / 'alizips'/ parasName).mkdir(exist_ok=True)
                    inputDAG = load_DAG(Viterbi_DAG_Path, load_onmfile=True)

                    viterbi_whithout_mapping(parameter_path, Viterbi_DAG_Path, sub_Viterbi_result_path,
                            parasName, seqiddb, ref_seq,inputDAG,threads=threads)
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                    taskNum.value+=1
                    print_x(start, taskNum.value, allstep,'Performing Viterbi alignment')
                    sys.stdout = outlog
                    sys.stderr = outlog
                seqiddb.conn.close()
                
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                

    start = time.time()
    paraNum = int(len(os.listdir(iniPath/'final_graph/ini'))/3)
    hierarchy = len(os.listdir(testGraphPath/'Merging_graphs'))

    tmpv= hierarchy
    for i in range(hierarchy):
        if chunk_size*(2**i)>=80000:
            tmpv = i
            break
    v_hierarchy=min(tmpv,hierarchy)
    
    a = len(os.listdir(testGraphPath/'subgraphs'))
    b = 2 ** v_hierarchy
    min_num = a // b + (1 if a % b else 0)
    subgraphList = list(range(
        (1 - 1) * (2 ** (hierarchy - v_hierarchy)) + 1,
        min(min_num + 1, 1 * (2 ** (hierarchy - v_hierarchy)) + 1)
    ))

    suborigraph_num = len(os.listdir(testGraphPath/f'subgraphs'))
    lock = Lock()
    allstep = 1
    taskList = Queue()
    for subGid in subgraphList:
        taskList.put([subGid])
        allstep+=paraNum
    
    if seq_num<40000:
        allstep+=1
    taskNum = Value('i',0)

    print_x(start, taskNum.value, allstep,'Performing Viterbi alignment')
    
    Viterbi_result_path = outpath / 'V_result'
    Viterbi_result_path.mkdir(exist_ok=True)
    os.makedirs(Viterbi_result_path/'alizips',exist_ok=True)
    os.makedirs(Viterbi_result_path/'alizips'/'Logs',exist_ok=True)
    
    processlist = []
    pool_num = min([10, threads // 4,len(subgraphList)])
    if pool_num==0:
        pool_num=1
    for idx in range(pool_num):
        processlist.append(Process(
            target=subViterbi,
        ))

    [p.start() for p in processlist]
    [p.join() for p in processlist]

    print_x(start, taskNum.value, allstep,'Merging alignment results')
    paraList= list(range(paraNum))
    sp_and_entropy = Manager().list()
    pool_num = min(6, threads)
    processlist = []
    
    for pid in range(pool_num):
        processlist.append(Process(
            target=merge_zipali_new,
            args=(iniPath,outpath, Viterbi_result_path, paraList[pid::pool_num],
                  sp_and_entropy, v_hierarchy)
        ))

    [p.start() for p in processlist]
    [p.join() for p in processlist]
    taskNum.value+=1
    print_x(start, taskNum.value, allstep,'Evaluating alignment quality')
    sp_and_entropy = sorted(sp_and_entropy, key=lambda x: x[-1])
    besten, bestsp = write_report(sp_and_entropy, outpath)
    if seq_num<40000:
        print_x(start, taskNum.value, allstep,'Writing final FASTA output')
        zipAlign2Fasta(Viterbi_result_path/f'alizips/tr{besten}/zipalign.npz', outpath/'align_result.fasta')
        taskNum.value+=1
        print_x(start, taskNum.value, allstep,'Process finished successfully')

def adaptive_chunk_size(avg_seq_len: float) -> int:
    a = 1.7805e+18
    b = -3.2505
    raw = a * (avg_seq_len ** b)
    return max(100, min(10000, int(raw)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPG-PHMM: Nucleotide multi-sequence aligner using DAPG and Profile Hidden Markov Models",
        epilog="Example usage for large datasets:\n  python DAG_Ali.py -i viral_sequences.fasta -o ./align_results -t 36 -l",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input'          , '-i', required=True, help='Input FASTA file path (required)')
    parser.add_argument('--output'         , '-o', required=True, help='Output directory path (required)')
    parser.add_argument('--fragment_Length', '-f', type=int,     help='Fragment length for building DAPG. If not specified, enters automatic mode (default: auto)', default=None)
    parser.add_argument('--threads'        , '-t', type=int,     help='Number of parallel threads (default: 36)', default=36)
    parser.add_argument('--chunk_size'     , '-c', type=int,     help='Sequence chunk size for splitting (default: auto based on average sequence length)', default=None)
    parser.add_argument('--large_scale'    , '-l', action='store_true', help='Enable large-scale alignment mode. This presets chunk_size to 5000 and fragment_length to 32 (unless specified) and skips pre-analysis for speed.')
    parser.add_argument('--Onlybuild'      , '-b', action='store_true', help='Only build the graph, do not perform alignment (default: False)')

    args = parser.parse_args()

    inpath = sanitize_path(args.input, 'input')
    outpath = sanitize_path(args.output, 'output')
    threads = args.threads

    if args.large_scale:
        print("Large-scale alignment mode enabled.")
        chunk_size = args.chunk_size if args.chunk_size is not None else 5000
        fragment_length = args.fragment_Length if args.fragment_Length is not None else 32
        print(f"  - Using chunk size: {chunk_size}")
        print(f"  - Using fragment length: {fragment_length}")

    else:
        print("Standard alignment mode: Performing pre-analysis to select optimal parameters...")
        if args.chunk_size is not None:
            chunk_size = args.chunk_size
        else:
            avg_len = fast_avg_seq_length_noloop(inpath)
            chunk_size = adaptive_chunk_size(avg_len)
        
        if args.fragment_Length is not None:
            fragment_length = args.fragment_Length
        else:
            kmer_diversity_ratio = calculate_kmer_diversity_ratio(inpath, k_size=32, threads=threads)
            print(kmer_diversity_ratio,'dd')
            if kmer_diversity_ratio is not None and kmer_diversity_ratio >= 0.9:
                fragment_length = 16
            else:
                fragment_length = 32 
        
        print(f"  - Analysis complete. Using fragment length: {fragment_length}")

    print("\nStage 1: Preparing data for alignment...")
    os.makedirs(outpath, exist_ok=True)
    sub_fasta_path = os.path.join(outpath, 'subfastas')
    os.makedirs(sub_fasta_path, exist_ok=True)
    seq_num, seqfileList = split_fasta(inpath, sub_fasta_path, chunk_size=chunk_size)
    
    print("\nStage 2: Constructing sequence graph...")
    final_graph_path, hierarchy, subgraph_num = graph_construction(
        outpath, seqfileList, build_fragment_graph, merge_graph_new, Tracing_merge,
        threads=threads, fragmentLength=fragment_length
    )


    if not args.Onlybuild:
        print("\nStage 3: Aligning sequences...")
        final_graph_path = outpath/'final_graph'
        train_in_all(hierarchy, final_graph_path, outpath, subgraph_num, threads, fit=True)
        print()
        viterbi_in_Graph(outpath, outpath, outpath, threads, chunk_size,seq_num)
        
    print("\nAlignment complete.")

