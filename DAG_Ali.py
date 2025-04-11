#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
import numpy as np
from DAG_operator import *
from DAG_Phmm import DAGPhmm
from sqlite_master import sql_master
import gc
from queue import Empty
import copy
import argparse
def add_noise_and_normalize(*log_probs, noise_low=0, noise_high=0):
    if len(log_probs) == 0:
        raise ValueError("At least one vector must be provided")
    shapes = [p.shape for p in log_probs]
    if len(set(shapes)) != 1:
        raise ValueError("All vectors must be the same length")
    log_probs_matrix = np.stack(log_probs, axis=0)

    noise = np.random.uniform(noise_low, noise_high, log_probs_matrix.shape)
    perturbed = log_probs_matrix + noise
    
    max_vals = np.max(perturbed, axis=0, keepdims=True) 
    shifted = perturbed - max_vals    
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=0, keepdims=True)
    log_sum_exp = np.log(sum_exp) + max_vals  
    normalized = perturbed - log_sum_exp   
    
    return tuple(normalized[i] for i in range(len(log_probs)))
def add_noise_and_normalize_matrix(log_probs_matrix: np.ndarray,noise_low: float = 0,noise_high: float = 0,axis: int = 0) -> np.ndarray:

    if log_probs_matrix.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if axis not in (0, 1):
        raise ValueError("轴必须为 0（列）或 1（行）")
    
    noise = np.random.uniform(
        low=noise_low,
        high=noise_high,
        size=log_probs_matrix.shape
    )
    perturbed = log_probs_matrix + noise
    
    max_vals = np.max(perturbed, axis=axis, keepdims=True)
    shifted = perturbed - max_vals
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp) + max_vals
    normalized = perturbed - log_sum_exp
    
    return normalized

def ini_paras(Ref_seq, emProbMatrix,insertRanges, ME, MD, MI, II, DM, pi_MID, outpath, parasName,perturbation=(0,0)):

    assert np.exp(MD) + np.exp(MI) < 1.0 - 1e-6, 
    assert np.exp(II) < 1.0 - 1e-6, 
    assert np.exp(DM) < 1.0 - 1e-6, 
    assert emProbMatrix.shape[0] == 4, 
    assert emProbMatrix.shape[1] == len(Ref_seq), 
    assert np.sum(pi_MID) > 1e-6, 
    
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

    _mi,_md,_mm = add_noise_and_normalize(_mi,_md,_mm,noise_low=perturbation[0],noise_high=perturbation[1])
    _mi[0] = np.log(pi_MID[1]/pi_sum)
    _md[0] = np.log(pi_MID[2]/pi_sum) 
    _mm[0] = np.log(pi_MID[0]/pi_sum)

    _mm[-1] = ME      
    _mi[-1] = np.log(1 - np.exp(ME))  

    _mm[-1],_mi[-1] = add_noise_and_normalize(_mm[-1],_mi[-1],noise_low=perturbation[0],noise_high=perturbation[1])

    _ii = np.full(n_positions, II, dtype=np.float64) 
    _im = np.full(n_positions, im_base, dtype=np.float64)  
    _id = np.full(n_positions, -np.inf, dtype=np.float64)  

    _ii,_im,_id = add_noise_and_normalize(_ii,_im,_id,noise_low=perturbation[0],noise_high=perturbation[1])

    _dm = np.full(n_positions, DM, dtype=np.float64) 
    _dd = np.full(n_positions, dd_base, dtype=np.float64)  
    _di = np.full(n_positions, -np.inf, dtype=np.float64)  

    _dm,_dd = add_noise_and_normalize(_dm,_dd,noise_low=perturbation[0],noise_high=perturbation[1])

    _dm[0] = _dd[0] = _dd[-1] = -np.inf  
    _dm[-1] = 0  



    _em = np.log(emProbMatrix.T + 1e-16)  
    _em = add_noise_and_normalize_matrix(_em,axis=1)
    print(_em)
    _ei = np.full((_em.shape[0]+1, _em.shape[1]), np.log(0.25), dtype=np.float64) 

    parameterDict = {
        "_mm": _mm, "_md": _md, "_mi": _mi,
        "_im": _im, "_id": _id, "_ii": _ii,
        "_dm": _dm, "_dd": _dd, "_di": _di,
        "match_emission": _em, "insert_emission": _ei
    }

    np.save(outpath/"ini/init_{}.npy".format(parasName), parameterDict)


def ref_graph_build(graph_path, thr=0.01,type=True, MissMatchScore=-5):
    
    ref_dict = np.load(graph_path/'thr_{}_{}.npz'.format(thr,type))

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
def ali_local(train_DAG_Path, Viterbi_DAG_Path, Viterbi_result_path,  seqiddb, outpath, hyperParameterList, threads,fit=True):
    hyperParameterDict = dict(hyperParameterList)
    for index in hyperParameterDict.keys():            
        parasName = 'tr{}'.format(index)                  
        outlog = open(outpath/'train_and_viterbi_{}.log'.format(parasName), 'w')
        sys.stdout = outlog                
        sys.stderr = outlog                
        modifyDict = {}               
        modifyDict['init_type'] = hyperParameterDict[index][1]
        modifyDict['emProbAdds_Match'] = hyperParameterDict[index][2]           
        modifyDict['emProbAdds_Match_head'] = modifyDict['emProbAdds_Match']            
        modifyDict['emProbAdds_Match_tail'] = modifyDict['emProbAdds_Match']            
        modifyDict['random'] = hyperParameterDict[index][3]
        modifyDict['init_M2D'] = -2                        
        modifyDict['init_M2I'] = -5                        
        modifyDict['init_I2I'] = np.log(1/2)              
        modifyDict['init_D2M'] = np.log(1/2)                    
        modifyDict['init_M2End'] = np.log(1/2)            
        modifyDict['weight_thr'] = hyperParameterDict[index][0]           
        modifyDict['head_length'] = 50              
        modifyDict['tail_length'] = 50          
        modifyDict['trProbAdds_mm'] = -3                      
        modifyDict['trProbAdds_md'] = -5                       
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
        windows_length = 100            
        ref_seq, ref_node_list, emProbMatrix, pi_MID,insertRanges = ref_graph_build(
            outpath, thr=modifyDict['weight_thr'],
            MissMatchScore=modifyDict['emProbAdds_Match'],type=modifyDict['init_type']
        )
        ini_paras(ref_seq, emProbMatrix,insertRanges,
               modifyDict['init_M2End'], modifyDict['init_M2D'],
               modifyDict['init_M2I'], modifyDict['init_I2I'],
               modifyDict['init_D2M'], pi_MID,
               train_DAG_Path, parasName,modifyDict['random'])
        parameter_path = train_DAG_Path/"ini/init_{}.npy".format(parasName)
        ph = DAGPhmm(train_DAG_Path, train_DAG_Path, parasName,
                    parameter_path=parameter_path)
        if fit:
            ph.init_train_data(train_DAG_Path,ref_node_list,ref_seq,modifyDict,True,windows_length, threads//2)
            ph.fit()
        ph.init_viterbi_data(Viterbi_DAG_Path, Viterbi_result_path/'alizips/',
                            ref_node_list, ref_seq,
                            windows_length=windows_length, threads=threads)
        os.makedirs(Viterbi_result_path/'alizips'/'{}'.format(parasName),exist_ok=True)
        np.save(Viterbi_result_path/'alizips'/'{}/modifydict.npy'.format(parasName), modifyDict)
        ph.Viterbi(seqiddb)
        ph.state_to_aligment(seqiddb)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
def train(DAG,Viterbi_result_path, finalGraphPath, train_DAG_Path, hyperParameterDict,lock,start=None,taskNum=None,allstep=None, threads=3,fit=True):
    hyperParameterDict = dict(hyperParameterDict)
    for index in hyperParameterDict.keys():            
        parasName = 'tr{}'.format(index)                  
        modifyDict = {}               
        modifyDict['init_type'] = hyperParameterDict[index][1]
        modifyDict['emProbAdds_Match'] = hyperParameterDict[index][2]           
        modifyDict['emProbAdds_Match_head'] = modifyDict['emProbAdds_Match']            
        modifyDict['emProbAdds_Match_tail'] = modifyDict['emProbAdds_Match']            
        modifyDict['random'] = hyperParameterDict[index][3]
        modifyDict['init_M2D'] = -2                       
        modifyDict['init_M2I'] = -5                      
        modifyDict['init_I2I'] = np.log(1/2)              
        modifyDict['init_D2M'] = np.log(1/2)                    
        modifyDict['init_M2End'] = np.log(1/2)            
        modifyDict['weight_thr'] = hyperParameterDict[index][0]           
        modifyDict['head_length'] = 50              
        modifyDict['tail_length'] = 50          
        modifyDict['trProbAdds_mm'] = -3                      
        modifyDict['trProbAdds_md'] = -5                       
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
            MissMatchScore=modifyDict['emProbAdds_Match'],type=modifyDict['init_type']           
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
            ph.init_train_data_with_DAG(train_DAG_Path,ref_node_list,ref_seq,modifyDict,copy.copy(DAG),True,windows_length, threads)
            
            ph.fit()
        sys.stdout = sys.__stdout__          
        sys.stderr = sys.__stderr__          
        if not taskNum is None:
            lock.acquire()
            taskNum.value+=1
            print_x(start,taskNum.value,allstep,'Training ...')
            lock.release()
def viterbi(train_DAG_Path, Viterbi_DAG_Path, Viterbi_result_path, parasName, seqiddb, ref_seq, ref_node_list,DAG,coarseDAGinfo,coarseDAG, threads=3,fit=True):
    
    windows_length = 100                 
    train_times = 1               
    while True:             
        checkpoint_path = train_DAG_Path/'ini/{}_pc_{}.npy'.format(parasName, train_times)
        if not os.path.exists(checkpoint_path):
            break                   
        parameter_path = checkpoint_path
        train_times += 1            
    if train_times == 1 or not fit:
        checkpoint_path = train_DAG_Path/'ini/init_{}.npy'.format(parasName)
        parameter_path = checkpoint_path
    ph = DAGPhmm(
        train_DAG_Path,                    
        Viterbi_DAG_Path,                   
        parasName,                   
        parameter_path=parameter_path             
    )

    ph.init_viterbi_data_withDAG(
        Viterbi_DAG_Path,                    
        Viterbi_result_path,               
        ref_node_list,                       
        ref_seq,DAG=DAG,
        coarseDAGinfo=coarseDAGinfo,
        coarseDAG=coarseDAG,                         
        windows_length=windows_length,          
        threads=threads                
    )
    ph.Viterbi(seqiddb)             
    ph.state_to_aligment(seqiddb)
    del ph
    gc.collect()            
def save_ref_node(graphPath,graph, thrs, save_ref_onm=False):
    
    try:
        no_degenerate_edgeDict,pureNodes = graph.noDegenerateGraph()
    except:
        no_degenerate_edgeDict,pureNodes = graph.edgeWeightDict,set(np.arange(graph.totalNodes))
    if save_ref_onm:
        ori_node_list = np.load(graphPath/'onm.npy', allow_pickle=True)          
        ori_node_index = np.load(graphPath/'onm_index.npy', allow_pickle=True)      
    for thr in thrs:
        old_init_type = True
        _ref_seq, _ref_node_list, _emProbMatrix,_insertrgs = graph.convertToAliReferenceDAG(pureNodes,thr)
        np.savez(
            graphPath/'thr_{}_{}.npz'.format(thr,old_init_type),
            ref_seq=_ref_seq,
            ref_node_list=_ref_node_list,
            emProbMatrix=_emProbMatrix,
            insert_range=_insertrgs
        )
        if save_ref_onm:
            ref_onm_list = []
            for node in _ref_node_list:             
                start_idx, end_idx = ori_node_index[node]
                ref_onm_list.append(ori_node_list[start_idx:end_idx])
            np.save(
                graphPath/'ref_onm_list_{}_{}.npy'.format(thr,old_init_type),
                np.array(ref_onm_list, dtype=object)            
            )
            np.save(graphPath/'refseq_{}_{}.npy'.format(thr,old_init_type), _ref_seq)
    
        old_init_type = False
        _ref_seq, _ref_node_list, _emProbMatrix,_insertrgs = graph.convertToAliReferenceDAG_new(no_degenerate_edgeDict,pureNodes,thr)
        np.savez(
            graphPath/'thr_{}_{}.npz'.format(thr,old_init_type),
            ref_seq=_ref_seq,
            ref_node_list=_ref_node_list,
            emProbMatrix=_emProbMatrix,
            insert_range=_insertrgs
        )
        if save_ref_onm:      
            ref_onm_list = []
            for node in _ref_node_list:             
                start_idx, end_idx = ori_node_index[node]
                ref_onm_list.append(ori_node_list[start_idx:end_idx])
            np.save(
                graphPath/'ref_onm_list_{}_{}.npy'.format(thr,old_init_type),
                np.array(ref_onm_list, dtype=object)            
            )
            np.save(graphPath/'refseq_{}_{}.npy'.format(thr,old_init_type), _ref_seq)
    
    
def lookup_mapped_reference_nodes(Viterbi_DAG_Path, ref_nodelist, ref_seq):

    graph = load_DAG(Viterbi_DAG_Path, load_onmfile=True)
    viterbi_reflist = []
    w_length = graph.fragmentLength              
    w_num = len(ref_seq) - graph.fragmentLength + 1              
    coor_list = []                 
    lastcoor = 0                     
    check_flag = 0           
    for idx in range(w_num):
        m_mode = ref_nodelist[idx + graph.fragmentLength - 1]  
        seq = ref_seq[idx:idx + w_length]  
        ori_nodes = graph.fragmentNodeDict.get(seq, [])  
        refnode = -1         
        for node in ori_nodes:
            if set(graph.SourceList[node]) & set(m_mode):
                refnode = node
                break           
        if refnode != -1:
            viterbi_reflist.append(refnode)
            current_coor = graph.queryGraph.coordinateList[refnode]
            coor_list.append(current_coor)
            if current_coor <= lastcoor:
                check_flag = 1            
            lastcoor = current_coor          
        else:
            viterbi_reflist.append(-1)
            coor_list.append(-1)
    if check_flag == 1:
        Block_list, weights, Block_dif = array_to_block(coor_list)
        cprg = remove_points_to_increase(Block_list, weights)
        for rg in cprg:
            st, ed = rg[0][1], rg[1][1] + 1
            for i in range(st, ed):
                viterbi_reflist[i] = -1
                coor_list[i] = -1
    return viterbi_reflist
def lookup_mapped_reference_batch(hierarchy,set_name,pathlist,ref_onm_list,ref_seq,thr,old_init_type=True):

    for outpath in pathlist:
        new_reflist = lookup_mapped_reference_nodes(outpath,ref_onm_list,ref_seq)
        np.save(outpath/'local_ref_nodelist_{}_{}_{}_{}.npy'.format(hierarchy,set_name,thr,old_init_type),new_reflist)
def find_global_ref(outpath,hierarchy,set_name,train_ref_Path,thr,subgraph_num,v_hierarchy,threads,old_init_type=True):

    a = subgraph_num                
    b = 2**v_hierarchy                    
    min_num = (a // b) + (1 if a % b != 0 else 0)             
    subGraphidstart = (set_name-1)*(2**(hierarchy-v_hierarchy))+1          
    subGraphend = min(set_name*(2**(hierarchy-v_hierarchy))+1, min_num+1)          
    ref_onm_list = np.load(train_ref_Path/'ref_onm_list_{}_{}.npy'.format(thr,old_init_type), allow_pickle=True)              
    ref_seq = str(np.load(train_ref_Path/'refseq_{}_{}.npy'.format(thr,old_init_type)))
    if v_hierarchy==0:
        subGraphPathList = [train_ref_Path]
    else:
        subGraphPathList = []
        for graph_name in range(int(subGraphidstart), int(subGraphend)):
            subGraphPathList.append(outpath/"Merging_graphs/merge_{}/{}/".format(v_hierarchy, graph_name))
    processlist = []
    pool_num = min(threads, (subGraphend - subGraphidstart))
    for idx in range(pool_num):
        processlist.append(Process(
            target=lookup_mapped_reference_batch,
            args=(hierarchy, set_name, subGraphPathList[idx::pool_num], ref_onm_list, ref_seq, thr,old_init_type, )
        ))
    [p.start() for p in processlist]
    [p.join() for p in processlist]

def print_x(start, step, allstep, current_stage="Sequence alignment is in progress"):

    runtime = time.time() - start
    percent = step / allstep
    bar = ('#' * int(percent * 20)).ljust(20)
    
    hours, remainder = divmod(runtime, 3600)
    mins, secs = divmod(remainder, 60)
    time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))

    sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ({time_format}) | Stage: {current_stage:<30}')
    sys.stdout.flush()

def train_and_subViterbi(hierarchy, set_name, finalGraphPath, outpath, subgraph_num,threads=3, fit=True):

    thrs = [0.01]
    Matchs = [-3,-5,-7]  
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
        (set_name-1) * (2 ** (hierarchy - v_hierarchy)) + 1,
        min(min_num+1, set_name * (2 ** (hierarchy - v_hierarchy)) + 1)
    ))
    allstep=4
    allstep+=2*len(thrs)
    allstep+=2*len(thrs)*len(Matchs)*len(Randoms)
    allstep+=len(Randoms)*len(subgraphList)

    print_x(start, taskNum.value, allstep,'Extracting parameters(Length First) ...')
    lock = Manager().Lock()
    
    paraNames=[]
    hyperParameterDict = {}
    idx=0
    filtered_files = []

    graph = load_DAG(finalGraphPath)
    graph.SourceList = [ [nodeid] for nodeid in range(graph.totalNodes)]
    graph.fragmentReduce()

    save_ref_node(finalGraphPath,graph, thrs, save_ref_onm=True)
    for thr in thrs:
        
        find_global_ref(outpath, hierarchy, set_name, finalGraphPath, thr, 
                    subgraph_num, v_hierarchy, threads,old_init_type=True)
        filtered_files.append('thr_{}_{}'.format(thr,True))
        taskNum.value += 1
        print_x(start, taskNum.value, allstep,'Extracting parameters(Weight First) ...')

    for thr in thrs:
        find_global_ref(outpath, hierarchy, set_name, finalGraphPath, thr, 
                    subgraph_num, v_hierarchy, threads,old_init_type=False)
        filtered_files.append('thr_{}_{}'.format(thr,False))
        taskNum.value += 1
        print_x(start, taskNum.value, allstep,'Training ...                           ')


    for f in filtered_files:
        datas = f.split('_')
        thr = float(datas[1])
        type = str(datas[2])
        for m in Matchs:
            for rdm in Randoms:
                hyperParameterDict[idx] = (thr,type,m,rdm)
                paraNames.append('tr{}'.format(idx))
                idx+=1
    hyperParameterList = list(hyperParameterDict.items())
    taskList = Queue()
    os.makedirs(Viterbi_result_path/'alizips/Logs',exist_ok=True)
    for subGid in subgraphList:
        taskList.put([subGid])
        for para in paraNames:
            os.makedirs(Viterbi_result_path/'alizips/{}'.format(para),exist_ok=True)
    gc.collect()
    if subgraph_num>400:
        train_threads = 1
    else:
        train_threads = threads
    graph.SourceList = [ set(nodeids) for nodeids in graph.SourceList]
    processlist = []
    pool_num = min(6,train_threads)
    for pid in range(pool_num):
        processlist.append(Process(target=train, args=(graph,
            Viterbi_result_path, finalGraphPath, train_DAG_Path,
            hyperParameterList[pid::pool_num],lock,start,taskNum,allstep, threads//pool_num,fit, )))
    [p.start() for p in processlist]
    [p.join() for p in processlist]
    del graph
    gc.collect()

    taskNum.value += 1
    print_x(start, taskNum.value, allstep,'Viterbi in subgraph ...               ')
    processlist = []
    pool_num = min(20,threads//4)
    for idx in range(pool_num):
        processlist.append(Process(target=viterbi_subgraph, args=(
            v_hierarchy, subgraph_num, finalGraphPath, outpath, train_DAG_Path,
            hierarchy, set_name, Viterbi_result_path, taskList,paraNames,start,taskNum,allstep, lock, 4,fit,  
        )))                   
    [p.start() for p in processlist]
    [p.join() for p in processlist]
    gc.collect()
    taskNum.value += 1
    print_x(start, taskNum.value, allstep,'Mergeing results ...                 ')

    sp_and_entropy = Manager().list()                  
    pool_num = min(6,threads)                  
    processlist = []
    for pid in range(pool_num):
        processlist.append(Process(target=merge_zipali, args=(
            outpath, Viterbi_result_path, hyperParameterList[pid::pool_num],
            sp_and_entropy, v_hierarchy, )))              
    [p.start() for p in processlist]
    [p.join() for p in processlist]
    sp_and_entropy = sorted(sp_and_entropy,key = lambda x:x[-1])
    besten,bestsp = write_report(sp_and_entropy, outpath)
    
    if subgraph_num<400:
        taskNum.value += 1
        print_x(start, taskNum.value, allstep,'Writing Fasta file ...             ')
        zipAlign2Fasta(Viterbi_result_path/'alizips/tr{}/zipalign.npz'.format(besten),outpath/'bestEntropy.fasta')
        zipAlign2Fasta(Viterbi_result_path/'alizips/tr{}/zipalign.npz'.format(bestsp),outpath/'bestSP.fasta')
    taskNum.value += 1
    print_x(start, taskNum.value, allstep,'Done                                ')
    print()
def viterbi_subgraph(v_hierarchy, subgraph_num, finalGraphPath, outpath, train_DAG_Path, hierarchy, set_name, Viterbi_result_path,taskList,parasNameList,start,taskNum,allstep, lock, threads,fit=True):
    last_oriGraphIdlist=None
    while not taskList.empty():
        try:
            info = taskList.get(timeout=1)  
        except:
            continue
        index = info[0]
        subGraphLevel = v_hierarchy  
        if v_hierarchy!=0:
            Viterbi_DAG_Path = outpath / "Merging_graphs/merge_{}/{}/".format(subGraphLevel, index)
        else:
            Viterbi_DAG_Path = finalGraphPath
        sub_Viterbi_result_path = Viterbi_result_path / '{}/'.format(index)
        outlog = open(Viterbi_result_path/'alizips/Logs/{}.log'.format(index), 'w')
        sys.stdout = outlog
        sys.stderr = outlog
        os.makedirs(sub_Viterbi_result_path,exist_ok=True)
        oriGraphPath = outpath / "subgraphs/"
        oriGraphIdlist = range((index-1)*(2**subGraphLevel)+1, min(subgraph_num+1, index*(2**subGraphLevel)+1))

        DAG = load_DAG(Viterbi_DAG_Path, load_onmfile=True)
        DAG.fragmentReduce()
                   
        linearPath_list, linearPath_link, _ = build_coarse_grained_graph(
            DAG.queryGraph, DAG.edgeWeightDict)
        coarseDAGinfo = (linearPath_list, linearPath_link)
        coarseDAG = DAGStru(len(linearPath_list), linearPath_link)
        coarseDAG.calculateCoordinates()               

        for parasName in parasNameList:
            parasPath = train_DAG_Path/'ini/{}_modifydict.npy'.format(parasName)
            modifyDict = np.load(parasPath,allow_pickle=True).item()
            
            if last_oriGraphIdlist!=oriGraphIdlist:
                lock.acquire()
                seqiddb = sql_master('', db=oriGraphPath, mode='build', dbidList=oriGraphIdlist)  
                lock.release()
            last_oriGraphIdlist=oriGraphIdlist
            ref_seq = str(np.load(finalGraphPath/'refseq_{}_{}.npy'.format(modifyDict['weight_thr'],modifyDict['init_type'])))
            ref_node_list = np.load(
                Viterbi_DAG_Path/'local_ref_nodelist_{}_{}_{}_{}.npy'.format(
                    hierarchy, set_name, modifyDict['weight_thr'],modifyDict['init_type']
                )
            ).tolist()
            if not os.path.exists(sub_Viterbi_result_path/'ini/'):
                os.mkdir(sub_Viterbi_result_path/'ini/')
            np.save(Viterbi_result_path/'alizips/{}/modifydict.npy'.format(parasName), modifyDict)
            viterbi(train_DAG_Path, Viterbi_DAG_Path, sub_Viterbi_result_path,
                    parasName, seqiddb, ref_seq, ref_node_list,DAG,coarseDAGinfo,coarseDAG, threads,fit)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not taskNum is None and not allstep is None:
            with lock:
                taskNum.value+=1
                print_x(start,taskNum.value,allstep,'Viterbi in subgraph ...')


def evaluate(zipali_path,positive,negative,outpath,tofasta=True):
    sp_and_entropy=[]
    for i,path in enumerate(zipali_path):
        a,c = sp_entro_zip(path/'zipalign.npz',positive,negative)
        modifyDict = np.load(path/'modifydict.npy',allow_pickle=True).item()
        sp_and_entropy.append((modifyDict['weight_thr'],modifyDict['init_type'],modifyDict['emProbAdds_Match'],modifyDict['random'],a,c,i))
    besten,bestsp = write_report(sp_and_entropy,outpath)
    if tofasta:
        zipAlign2Fasta(zipali_path[besten]/'zipalign.npz',outpath/'bestEntropy.fasta')
        zipAlign2Fasta(zipali_path[bestsp]/'zipalign.npz',outpath/'bestSP.fasta')
    
def write_report(sp_and_entropy,outpath):
    def tuple_to_str(t):
        return '(' + ', '.join(tuple_to_str(x) if isinstance(x, tuple) else str(x) for x in t) + ')'
    sp_list = [0]*len(sp_and_entropy)
    entropy_list = [0]*len(sp_and_entropy)
    for idx,sande in enumerate(sp_and_entropy):
        sp_list[idx] = sande[4]
        entropy_list[idx] = sande[5]
    bestsp = np.argmax(sp_list)
    besten = np.argmin(entropy_list)
    data = [[' ','weight_thr','old_init','minEmProbAdds_Match','random','sp score','entropy','best entropy','best sp score']]
    for i in range(len(sp_and_entropy)):
        if i == bestsp:
            isbestsp = '✔'
        else:
            isbestsp = ' '
        if i == besten:
            isbesten = '✔'
        else:
            isbesten = ' '
        data.append(['tr{}'.format(i),sp_and_entropy[i][0],sp_and_entropy[i][1],sp_and_entropy[i][2],tuple_to_str(sp_and_entropy[i][3]),sp_list[i],entropy_list[i],isbesten,isbestsp])
    col_widths = [max(len(str(item)) for item in column) for column in zip(*data)]
    with open(outpath/'report.txt', 'w') as file:
        def format_row(row):
            return ' | '.join(f"{str(item).ljust(col_widths[i])}" if not isinstance(item, float) 
                            else f"{item:.5f}".ljust(col_widths[i]) for i, item in enumerate(row))
        header = format_row(data[0])
        file.write(header + '\n')
        file.write('-' * (sum(col_widths) + 4 * (len(col_widths) - 1)) + '\n')
        for row in data[1:]:
            file.write(format_row(row) + '\n')
    np.save(outpath/'bestEntropyScore.npy',[sp_list[besten],entropy_list[besten]])
    np.save(outpath/'bestSPScore.npy',[sp_list[bestsp],entropy_list[bestsp]])
    return besten,bestsp
def train_and_viterbi(hierarchy,finalGraphPath, outpath, threads=3,fit=True):
    
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

    finalGraphPath = sanitize_path(finalGraphPath, 'input')          
    outpath = sanitize_path(outpath, 'output')                        
    train_DAG_Path = finalGraphPath
    Viterbi_result_path = outpath / 'V_result/'                  
    os.makedirs(Viterbi_result_path, exist_ok=True)
    os.makedirs(train_DAG_Path/'ini', exist_ok=True)          
    os.makedirs(Viterbi_result_path/'alizips', exist_ok=True)
    start = time.time()
    step, allstep = 0, 5
    print_x(start, step, allstep,'Initializing')
    oriGraphIdlist = os.listdir(outpath/'subgraphs/')            
    seqiddb = sql_master('', db=outpath/'subgraphs/', mode='build', dbidList=oriGraphIdlist)
    step += 1
    print_x(start, step, allstep,'Extracting parameters(Length First) ...')
    filtered_files = []
    thrs = [0.01]
    graph = load_DAG(finalGraphPath)
    graph.SourceList = [ [nodeid] for nodeid in range(graph.totalNodes)]
    graph.fragmentReduce()
    save_ref_node(finalGraphPath,graph, thrs, save_ref_onm=True)
    del graph
    gc.collect()
    for thr in thrs:
        filtered_files.append('thr_{}_{}'.format(thr,True))
        step += 1
        print_x(start, step, allstep,'Extracting parameters(Weight First) ...')
    for thr in thrs:
        filtered_files.append('thr_{}_{}'.format(thr,False))
        step += 1
        print_x(start, step, allstep,'Train and Viterbi ...                  ')
    
    gc.collect()  
    processlist = []
    pool_num = 6
    Matchs = [-3,-5,-7]
    Randoms = [(0,0)] 
    hyperParameterDict = {}
    idx=0
    for f in filtered_files:
        datas = f.split('_')
        thr = float(datas[1])
        type = str(datas[2])
        for m in Matchs:
            for rdm in Randoms:
                hyperParameterDict[idx] = (thr,type,m,rdm)
                idx+=1
    hyperParameterList = list(hyperParameterDict.items())
    processlist = []  
    for pid in range(pool_num):
        processlist.append(
            Process(target=ali_local, 
                    args=(train_DAG_Path, train_DAG_Path, Viterbi_result_path,
                          seqiddb, finalGraphPath, 
                          hyperParameterList[pid::pool_num], threads//6,fit, ))          
        )
    [p.start() for p in processlist]        
    [p.join() for p in processlist]         
    step +=1
    print_x(start, step, allstep,'Result evaluating ...                      ')
    zipali_path = []
    for i in range(len(hyperParameterList)):
        zipali_path.append(Viterbi_result_path/f'alizips/tr{i}')
    if hierarchy>8:
        tofasta=False
    else:
        tofasta=True
    evaluate(zipali_path, positive, negative, outpath,tofasta)
    step +=1
    print_x(start, step, allstep,'Done                                       ')
    print()

def merge_zipali(outpath, path, hyperParameterList, sp_and_entropy, v_hierarchy):

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
    os.makedirs(path/'alizips', exist_ok=True)            
    hyperParameterDict = dict(hyperParameterList)
    for paramsIdx in hyperParameterDict.keys():
        parasName = 'tr{}'.format(paramsIdx)
        if v_hierarchy==0:
            fileList = [1]
        else:
            fileList = os.listdir(outpath/'Merging_graphs/merge_{}/'.format(v_hierarchy))
            fileList = [file for file in fileList if 'alizips' not in file]
            fileList = sorted(fileList, key=int)                 
        maxInsertLengthGlobal = np.load(path/'{}/{}/insert_length_dict.npy'.format(fileList[0],parasName),allow_pickle=True).item()
        for i in range(1, len(fileList)):
            maxInsertLengthSub = np.load(path/'{}/{}/insert_length_dict.npy'.format(fileList[i],parasName),allow_pickle=True).item()
            maxInsertLengthGlobal = {key: max(maxInsertLengthGlobal[key], maxInsertLengthSub[key]) for key in maxInsertLengthGlobal}
        sequenceNameList = []           
        zipali_global = []               
        for i in range(len(fileList)):
            maxInsertLengthSub = np.load(path/'{}/{}/insert_length_dict.npy'.format(fileList[i],parasName),allow_pickle=True).item()
            indexdict = np.load(path/'{}/{}/indexdict.npy'.format(fileList[i],parasName),allow_pickle=True).item()           
            alignInfo = np.load(path/'{}/{}/zipalign.npz'.format(fileList[i],parasName),allow_pickle=True)
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
        if not os.path.exists(path/'alizips'/parasName):
            os.mkdir(path/'alizips'/parasName)
        modifyDict = np.load(path/'alizips'/parasName/'modifydict.npy',allow_pickle=True).item()
        a, c = sp_entro_zip_loaded(zipali_global, positive, negative)
        sp_and_entropy.append((modifyDict['weight_thr'],modifyDict['init_type'],modifyDict['emProbAdds_Match'],modifyDict['random'],a, c, paramsIdx))                 
        zipali_global = np.array(zipali_global, dtype=object)
        np.savez(path/'alizips/{}/zipalign.npz'.format(parasName), 
                namelist=sequenceNameList, 
                align=zipali_global)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DAPG-PHMM: Nucleotide multi-sequence aligner using DAPG  and Profile Hidden Markov Models",
        epilog="Example usage for large datasets:\n  python DAG_Ali.py -i viral_sequences.fasta -o ./align_results -f 200 -t 36 --c 10000",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True, 
                      help='Input FASTA file path (required)')
    parser.add_argument('--output', '-o', required=True, 
                      help='Output directory path (required)')
    parser.add_argument('--fragment_Length', '-f', type=int, default=16,
                      help='Fragment length for building DAPG (default: 16)')
    parser.add_argument('--threads', '-t', type=int, default=36,
                      help='Number of parallel threads (default: 36)')
    parser.add_argument('--chunk_size', '-c', type=int, default=5000,
                      help='Sequence chunk size for splitting (default: 5000)')
    
    args = parser.parse_args()
    inpath = args.input
    outpath = args.output
    fra = args.fragment_Length
    threads = args.threads
    chunk_size = args.chunk_size  
    inpath = sanitize_path(inpath, 'input')
    outpath = sanitize_path(outpath, 'output')
    os.makedirs(outpath, exist_ok=True)
    sub_fasta_path = sanitize_path(os.path.join(outpath, 'subfastas'), 'output')
    os.makedirs(sub_fasta_path, exist_ok=True)
    print('Preparing data')
    seq_num, seqfileList = split_fasta(inpath, sub_fasta_path, chunk_size=chunk_size)
    print('Building graph')
    final_graph, hierarchy, subgraph_num = graph_construction(
        outpath, seqfileList, build_fragment_graph, merge_graph, Tracing_merge,
        threads=threads, fragmentLength=int(fra)
    )
    print('Sequence alignment')
    if seq_num > 40000:
        train_and_subViterbi(hierarchy, 1, final_graph, outpath, subgraph_num, threads, fit=True)
    else:
        train_and_viterbi(hierarchy, final_graph, outpath, threads, fit=True)
