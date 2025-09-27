#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Manager,Process,Queue,shared_memory,Value,Lock,Array
from DAG_info import *
from DAG_operator import *
from datetime import datetime
import time
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from numba import jit
import sys
from queue import Empty

class DAGPhmm(object):

    def __init__(self,train_DAG_Path,Viterbi_DAG_Path=None,partmeter_name='',parameter_path = ''):
        self.commonBaseDict = {"A":0,"T":1,"C":2,"G":3}               
        self.allBaseDict = {"A":0,"T":1,"C":2,"G":3,'R': 4, 'Y': 5, 'M': 6, 'K': 7, 'S': 8, 'W': 9, 'H': 10, 'B': 11, 'V': 12, 'D': 13, 'N': 14}         
        self.allBaseDictReverse = {0: 'A', 1: 'T', 2: 'C', 3: 'G', 4: 'R', 5: 'Y', 6: 'M', 7: 'K', 8: 'S', 9: 'W', 10: 'H', 11: 'B', 12: 'V', 13: 'D', 14: 'N'}                    
        self.alignmentBaseDictionary = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}          
        self.degenerateBaseDictionary = {'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}          
        self.normalBaseCount = len(self.commonBaseDict)          
        self.train_DAG_Path = train_DAG_Path                     
        self.parameterName = partmeter_name          
        if Viterbi_DAG_Path is None:  
            self.Viterbi_DAG_Path = train_DAG_Path                
        else:
            self.Viterbi_DAG_Path = Viterbi_DAG_Path
        parameterDict=np.load(parameter_path,allow_pickle=True).item()            
        for k in ['_mm','_mi','_md','_dm','_dd','_di','_im','_id','_ii','insert_emission','match_emission']:
            parameterDict[k] = np.array(parameterDict[k],dtype=np.float64)                 
        self.train_times = 0                     
        self.M2M_array=parameterDict['_mm'][1:-1]                         
        self.Match_num = len(self.M2M_array)+1                     
        self.M2I_array=parameterDict['_mi'][1:]                
        self.M2D_array=parameterDict['_md'][1:-1]            
        self.D2M_array=parameterDict['_dm'][1:-1]
        self.D2D_array=parameterDict['_dd'][1:-1]
        self.D2I_array=parameterDict['_di'][1:]
        self.I2M_array=parameterDict['_im'][:-1]
        self.I2I_array=parameterDict['_ii']
        self.I2D_array=parameterDict['_id'][:-1]
        self.D2E=parameterDict['_dm'][-1]
        self.I2E=parameterDict['_im'][-1]
        self.M2E=parameterDict['_mm'][-1]
        self.Ie_Matrix=parameterDict['insert_emission']                          
        self.Me_Matrix=parameterDict['match_emission']                       
        self.pi_M=parameterDict['_mm'][0]
        self.pi_D=parameterDict['_md'][0]
        self.pi_I=parameterDict['_mi'][0]
        self.Me_Matrix_degenerate_base=[]                
        self.Ie_Matrix_degenerate_base=[]                 
        for i in ['A','T','C','G']:                
            self.Me_Matrix_degenerate_base.append(self.Me_Matrix[:,self.commonBaseDict[i]])               
            self.Ie_Matrix_degenerate_base.append(self.Ie_Matrix[:,self.commonBaseDict[i]])
        for base in ["R","Y","M","K","S","W","H","B","V","D","N"]:             
            degenerate_base = self.degenerateBaseDictionary[base]
            self.Ie_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Ie_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))  
            self.Me_Matrix_degenerate_base.append(np.logaddexp.reduce(self.Me_Matrix.T[degenerate_base]-np.log(len(degenerate_base)),axis=0))  
        self.Me_Matrix_degenerate_base = np.array(self.Me_Matrix_degenerate_base)              
        self.Ie_Matrix_degenerate_base = np.array(self.Ie_Matrix_degenerate_base)

    @jit(nopython=True)
    def calculate_alpha_values(last_alpha_M, last_alpha_I, last_alpha_D, Ie, Me, stateRangeStart, stateRangeEnd, D2D_array, I2D_array, M2D_array, D2I_array, I2I_array, M2I_array, D2M_array, I2M_array, M2M_array, arrayLength):

        alpha_M = np.full(arrayLength, -np.inf)                      
        alpha_I = np.full(arrayLength + 1, -np.inf)                        
        alpha_D = np.full(arrayLength, -np.inf)                
        alpha_I[0] = last_alpha_I[0] + I2D_array[stateRangeStart] + Ie[stateRangeStart]
        alpha_M[0] = last_alpha_I[0] + I2M_array[stateRangeStart] + Me[stateRangeStart]
        alpha_I[1:stateRangeEnd+1] = np.logaddexp(
            last_alpha_M + M2I_array[stateRangeStart:stateRangeEnd],           
            last_alpha_I[1:] + I2I_array[stateRangeStart+1:stateRangeEnd+1]        
        )
        alpha_I[1:stateRangeEnd+1] = np.logaddexp(
            alpha_I[1:stateRangeEnd+1],
            last_alpha_D + D2I_array[stateRangeStart:stateRangeEnd]           
        ) + Ie[stateRangeStart+1:stateRangeEnd+1]             
        alpha_M[1:stateRangeEnd] = np.logaddexp(
            last_alpha_M[:-1] + M2M_array[stateRangeStart:stateRangeEnd-1],        
            last_alpha_I[1:-1] + I2M_array[stateRangeStart+1:stateRangeEnd]         
        )
        alpha_M[1:stateRangeEnd] = np.logaddexp(
            alpha_M[1:stateRangeEnd],
            last_alpha_D[:-1] + D2M_array[stateRangeStart:stateRangeEnd-1]         
        ) + Me[stateRangeStart+1:stateRangeEnd]          
        alpha_D[0] = alpha_I[0] + I2D_array[stateRangeStart]
        tm = np.logaddexp(
            alpha_M[:-1] + M2D_array[stateRangeStart:stateRangeEnd-1],           
            alpha_I[1:-1] + I2D_array[stateRangeStart+1:stateRangeEnd]           
        )
        for i in range(1, arrayLength):
            alpha_D[i] = np.logaddexp(
                tm[i-1],               
                alpha_D[i-1] + D2D_array[stateRangeStart+i-1]        
            )
        return alpha_I, alpha_M, alpha_D
    
    @jit(nopython=True)
    def calculate_beta_values(nextBeta_M, nextBeta_I, beta_D, stateRangeStart, stateRangeEnd, Match_num, D2I_array, D2M_array, D2D_array, I2I_array, I2M_array, I2D_array, M2M_array, M2I_array, M2D_array, arrayLength):

        beta_D[-1] = nextBeta_I[-1] + D2I_array[stateRangeEnd-1]            
        tm = np.logaddexp(
            nextBeta_M[1:] + D2M_array[stateRangeStart:stateRangeEnd-1],            
            nextBeta_I[1:-1] + D2I_array[stateRangeStart:stateRangeEnd-1]            
        )
        for i in range(arrayLength - 2, -1, -1):              
            beta_D[i] = np.logaddexp(
                tm[i],            
                beta_D[i + 1] + D2D_array[i]          
            )
        beta_I = np.logaddexp(
            nextBeta_I + I2I_array[stateRangeStart:stateRangeEnd+1],         
            np.append(nextBeta_M + I2M_array[stateRangeStart:stateRangeEnd], -np.inf)          
        )
        beta_I = np.logaddexp(
            beta_I,
            np.append(beta_D + I2D_array[stateRangeStart:stateRangeEnd], -np.inf)          
        )
        beta_M = np.logaddexp(
            np.append(nextBeta_M[1:] + M2M_array[stateRangeStart:stateRangeEnd-1], -np.inf),          
            nextBeta_I[1:] + M2I_array[stateRangeStart:stateRangeEnd]        
        )
        beta_M = np.logaddexp(
            beta_M,
            np.append(beta_D[1:] + M2D_array[stateRangeStart:stateRangeEnd-1], -np.inf)          
        )
        return beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I            

    @jit(nopython=True)
    def update_state(laststate_local, range_length, nowstate, nowstateindex, delta_index_start, decrease_list):
        while nowstate == 1:            
            state_slot = nowstate * range_length + nowstateindex - delta_index_start
            nowstate = laststate_local[state_slot]              
            nowstateindex += decrease_list[nowstate]                 
        return nowstate, nowstateindex  
                    
    @jit(nopython=True)
    def calculate_delta_values(stateRangeStart, stateRangeEnd, arrayLength,
                    last_delta_I, last_delta_M, last_delta_D,
                    I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                    Ie, Me, M2D_array):

        delta_I = np.full(arrayLength+1, -np.inf, dtype=np.float64)                       
        delta_M = np.full(arrayLength, -np.inf, dtype=np.float64)              
        delta_D = np.full(arrayLength, -np.inf, dtype=np.float64)              
        delta_I[0] = last_delta_I[0] + I2I_array[stateRangeStart] + Ie[stateRangeStart]            
        delta_M[0] = last_delta_I[0] + I2M_array[stateRangeStart] + Me[stateRangeStart]            
        I_array = np.vstack((
            last_delta_M + M2I_array[stateRangeStart:stateRangeEnd],                                               
            last_delta_D + D2I_array[stateRangeStart:stateRangeEnd],             
            last_delta_I[1:] + I2I_array[stateRangeStart+1:stateRangeEnd+1]                
        ))
        M_array = np.vstack((
            last_delta_M[:-1] + M2M_array[stateRangeStart:stateRangeEnd-1],               
            last_delta_D[:-1] + D2M_array[stateRangeStart:stateRangeEnd-1],         
            last_delta_I[1:-1] + I2M_array[stateRangeStart+1:stateRangeEnd]                
        ))
        maxProbOrigin_I = np.full(arrayLength+1, 2)                  
        maxProbOrigin_M = np.full(arrayLength, 2)                   
        maxProbOrigin_D = np.full(arrayLength, 2)                   
        maxProbOrigin_I[1:] = np.argmax(I_array, axis=0)             
        maxProbOrigin_M[1:] = np.argmax(M_array, axis=0)              
        for i in range(1, arrayLength+1):
            delta_I[i] = I_array[maxProbOrigin_I[i], i-1]            
        for i in range(1, arrayLength):
            delta_M[i] = M_array[maxProbOrigin_M[i], i-1]
        delta_I[1:] += Ie[stateRangeStart+1:stateRangeEnd+1]           
        delta_M[1:] += Me[stateRangeStart+1:stateRangeEnd]             
        delta_D[0] = delta_I[0] + I2D_array[stateRangeStart]             
        for i in range(1, arrayLength):
            D_arg = np.array([
                delta_M[i-1] + M2D_array[stateRangeStart+i-1],         
                delta_D[i-1] + D2D_array[stateRangeStart+i-1],         
                delta_I[i] + I2D_array[stateRangeStart+i]             
            ])
            maxProbOrigin_D[i] = np.argmax(D_arg)          
            delta_D[i] = D_arg[maxProbOrigin_D[i]]          
        return delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D
    
    def forward(self,alpha_Matrix_M,alpha_Matrix_D,alpha_Matrix_I):

        def write_alpha_head(indexlist,alpha_head_dict):

            for index in indexlist:
                linearPath = self.linearPath_list[index]             
                node = linearPath[0]             
                arrayRangeStart, arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.fragments[node][-1]]
                alpha_M, alpha_I, alpha_D = alpha_head_dict[baseID]
                stateRangeStart, stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = alpha_M[stateRangeStart:stateRangeEnd]
                alpha_Matrix_I[arrayRangeStart:arrayRangeEnd] = alpha_I[stateRangeStart:stateRangeEnd+1]               
                alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = alpha_D[stateRangeStart:stateRangeEnd]
                doneList[node] = 0              
                write_alpha(linearPath[1:])            
                childenlinearPaths = self.DAG.CG_DAG.findChildNodes(index)
                for i in childenlinearPaths:
                    with lock:
                        indegree_dict[i] -= 1
                        if indegree_dict[i] == 0:
                            q.put(i)
        def init_alpha_head():
            alpha_head_dict = {}
            for baseID in range(self.Me_Matrix_degenerate_base.shape[0]):
                Me = self.Me_Matrix_degenerate_base[baseID]            
                Ie = self.Ie_Matrix_degenerate_base[baseID]            
                alpha_M = np.full(self.Match_num, -np.inf)
                alpha_I = np.full(self.Match_num+1, -np.inf)
                alpha_D = np.full(self.Match_num, -np.inf)
                alpha_M[0] = self.pi_M + Me[0]           
                alpha_I[0] = self.pi_I + Ie[0]           
                last_alpha_D = np.full(self.Match_num, -np.inf, dtype=np.float64)
                last_alpha_D[:self.maxrange] = First_alpha_M_D
                alpha_M[1:] = last_alpha_D[:-1] + self.D2M_array + Me[1:]         
                alpha_I[1:] = last_alpha_D + self.D2I_array + Ie[1:]              
                alpha_D[0] = alpha_I[0] + self.I2D_array[0]         
                tm = np.logaddexp(alpha_M[:-1] + self.M2D_array, alpha_I[1:-1] + self.I2D_array[1:])
                for i in range(1, self.Match_num):
                    alpha_D[i] = np.logaddexp(tm[i-1], alpha_D[i-1] + self.D2D_array[i-1])
                alpha_head_dict[baseID] = [alpha_M, alpha_I, alpha_D]
            return alpha_head_dict
        def write_alpha(nodelist):

            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            Match_num = self.Match_num
            for node in nodelist:
                arrayRangeStart,arrayRangeEnd=self.arrayRangeDict[node]
                baseID = self.allBaseDict[self.DAG.fragments[node][-1]]
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                partennodes = self.DAG.queryGraph.findParentNodes(node)
                alist = [self.DAG.edgeWeightDict[(lnode, node)] for lnode in partennodes]
                b = np.sum(alist)
                parentNodeWeightList = [
                    [np.log((self.DAG.edgeWeightDict[(lnode, node)] + 0 / len(partennodes)) / (b + 0)), lnode]
                    for lnode in partennodes
                ]
                parentNodeCount = len(partennodes)
                last_alpha_M_list = np.full((parentNodeCount,Match_num), -np.inf)
                last_alpha_I_list = np.full((parentNodeCount,Match_num + 1), -np.inf)
                last_alpha_D_list = np.full((parentNodeCount,Match_num), -np.inf)
                for i, fnode in enumerate(parentNodeWeightList):
                    _arrayRangeStart,_arrayRangeEnd=self.arrayRangeDict[fnode[1]]
                    weight = fnode[0]
                    _stateRangeStart,_stateRangeEnd = self.stateRangeDict[2*fnode[1]:2*fnode[1]+2]
                    last_alpha_M_list[i][_stateRangeStart:_stateRangeEnd] = alpha_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_alpha_I_list[i][_stateRangeStart:_stateRangeEnd+1] = alpha_Matrix_I[_arrayRangeStart:_arrayRangeEnd] + weight
                    last_alpha_D_list[i][_stateRangeStart:_stateRangeEnd] = alpha_Matrix_D[_arrayRangeStart:_arrayRangeEnd-1] + weight
                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                last_alpha_M_list = last_alpha_M_list[:,stateRangeStart:stateRangeEnd]
                last_alpha_I_list = last_alpha_I_list[:,stateRangeStart:stateRangeEnd+1]
                last_alpha_D_list = last_alpha_D_list[:,stateRangeStart:stateRangeEnd]
                arrayLength = stateRangeEnd-stateRangeStart
                last_alpha_M = np.logaddexp.reduce(last_alpha_M_list,axis=0)
                last_alpha_I = np.logaddexp.reduce(last_alpha_I_list,axis=0)
                last_alpha_D = np.logaddexp.reduce(last_alpha_D_list,axis=0)
                alpha_I, alpha_M, alpha_D = DAGPhmm.calculate_alpha_values(last_alpha_M, last_alpha_I, last_alpha_D,  Ie, Me, stateRangeStart, stateRangeEnd,D2D_array,I2D_array,M2D_array,D2I_array,I2I_array,M2I_array,D2M_array,I2M_array,M2M_array,arrayLength)
                alpha_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = alpha_M
                alpha_Matrix_I[arrayRangeStart:arrayRangeEnd] = alpha_I
                alpha_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = alpha_D
                doneList[node]=0
        def calculate_alpha(lock, nodelist, alpha_head_dict):

            linearPath_list = self.linearPath_list
            write_alpha_head(nodelist, alpha_head_dict)                 
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)                
                except Empty:
                    continue                 
                while start_node is not None:
                    linearPath = linearPath_list[start_node]
                    write_alpha(linearPath)               
                    checklist = self.DAG.CG_DAG.findChildNodes(start_node)
                    todolist = []
                    for i in checklist:
                        with lock:              
                            indegree_dict[i] -= 1
                            if indegree_dict[i] == 0:               
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()                    
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node = None            
        def is_end(goon_flag):

            start = time.time()
            while True:
                time.sleep(1)             
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes - np.count_nonzero(doneList)) / self.DAG.totalNodes, 5)
                bar = ('#' * int(percent * 20)).ljust(20)            
                hours, remainder = divmod(runtime, 3600)
                mins, secs = divmod(remainder, 60)
                time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ( {time_format} )')
                sys.stdout.flush()                   
                if 0 == np.count_nonzero(doneList > 0):                          
                    goon_flag.value = 0                 
                    sys.stdout.write('\n')                 
                    break
        lock = Lock()                   
        goon_flag = Value('i',1)                       
        arrayRangeStart=self.arrayRangeDict[-1][0]
        alpha_Matrix_D[arrayRangeStart] = self.pi_D


        for i in range(1,self.maxrange):
            alpha_Matrix_D[arrayRangeStart+i] = alpha_Matrix_D[arrayRangeStart+i-1]+self.D2D_array[i-1]          
        First_alpha_M_D = alpha_Matrix_D[arrayRangeStart:arrayRangeStart+self.maxrange]           
        q = Queue()
        alpha_head_dict=init_alpha_head()
        startnodelist = list(self.DAG.CG_DAG.startNodeSet)
        indegree_dict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint16).itemsize)
        indegree_dict = np.ndarray((self.DAG.totalNodes,), dtype=np.int16, buffer=indegree_dict_shm.buf)
        link_keys=self.linearPath_link
        for link in link_keys:
            indegree_dict[link[1]] += 1               
        doneList_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint8).itemsize)
        doneList = np.ndarray((self.DAG.totalNodes,), dtype=np.int8, buffer=doneList_shm.buf)
        doneList[:]=[1]*self.DAG.totalNodes             
        processlist=[]
        pool_num = self.pool_num              
        for idx in range(pool_num):
            processlist.append(Process(
                target=calculate_alpha, 
                args=(lock, startnodelist[idx::pool_num], alpha_head_dict, )
            ))
        processlist.append(Process(target=is_end, args=(goon_flag, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        last_alpha_M_list = []
        last_alpha_I_list = []
        last_alpha_D_list = []
        for fnode in self.graphEndNodes:
            _arrayRangeStart, _arrayRangeEnd = self.arrayRangeDict[fnode]
            weight = np.log(self.DAG.weights[fnode]/self.sequenceNum)
            _stateRangeStart = self.stateRangeDict[2*fnode]
            _stateRangeEnd = self.stateRangeDict[2*fnode+1]
            temp_alpha_M = np.full(self.Match_num, -np.inf)
            temp_alpha_I = np.full(self.Match_num+1, -np.inf)            
            temp_alpha_D = np.full(self.Match_num, -np.inf)
            temp_alpha_M[_stateRangeStart:_stateRangeEnd] = alpha_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1]
            temp_alpha_I[_stateRangeStart:_stateRangeEnd+1] = alpha_Matrix_I[_arrayRangeStart:_arrayRangeEnd]
            temp_alpha_D[_stateRangeStart:_stateRangeEnd] = alpha_Matrix_D[_arrayRangeStart:_arrayRangeEnd-1]
            last_alpha_M_list.append(temp_alpha_M + weight)
            last_alpha_I_list.append(temp_alpha_I + weight)
            last_alpha_D_list.append(temp_alpha_D + weight) 
        last_alpha_M = np.logaddexp.reduce(last_alpha_M_list, axis=0)
        last_alpha_I = np.logaddexp.reduce(last_alpha_I_list, axis=0)
        last_alpha_D = np.logaddexp.reduce(last_alpha_D_list, axis=0)
        prob = np.logaddexp.reduce([
            last_alpha_M[-1]+self.M2E,           
            last_alpha_I[-1]+self.I2E,             
            last_alpha_D[-1]+self.D2E            
        ], axis=0)
        indegree_dict_shm.close()
        indegree_dict_shm.unlink()
        doneList_shm.close()
        doneList_shm.unlink()

    def backward(self, beta_Matrix_M, beta_Matrix_D, beta_Matrix_I, left_beta_Matrix_M, left_beta_Matrix_I, problist):

        def write_beta_head(indexlist, beta_head):
            for index in indexlist:
                linearPath = self.linearPath_list[index]
                node = linearPath.pop()               
                arrayRangeStart,arrayRangeEnd=self.arrayRangeDict[node]
                weight = np.log(self.DAG.weights[node]/self.sequenceNum)
                beta_M, beta_I, beta_D = beta_head
                stateRangeStart, stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                arrayLength = stateRangeEnd-stateRangeStart
                nextBeta_M = np.full(arrayLength, -np.inf)
                nextBeta_I = np.full(arrayLength+1, -np.inf)             
                beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = beta_M[stateRangeStart:stateRangeEnd] + weight
                beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = beta_I[stateRangeStart:stateRangeEnd+1] + weight
                beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = beta_D[stateRangeStart:stateRangeEnd] + weight
                left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = nextBeta_M
                left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = nextBeta_I
                doneList[node] = 0             
                write_beta(linearPath)            
                checklist = self.DAG.CG_DAG.findParentNodes(index)
                for i in checklist:
                    with lock:             
                        outdegreeDict[i] -= 1
                        if outdegreeDict[i] == 0:              
                            q.put(i)          
        def init_beta_head():

            Match_num = self.Match_num
            beta_M = np.full(Match_num, -np.inf)
            beta_I = np.full(Match_num + 1, -np.inf)
            beta_D = np.full(Match_num, -np.inf)
            beta_M[-1] = self.M2E                  
            beta_I[-1] = self.I2E              
            beta_D[-1] = self.D2E              
            for i in range(Match_num-1)[::-1]:
                beta_D[i] = beta_D[i+1] + self.D2D_array[i]           
            beta_M[:-1] = beta_D[1:] + self.M2D_array         
            beta_I[:-1] = beta_D + self.I2D_array             
            return [beta_M, beta_I, beta_D]
        def write_beta(nodes):
            Match_num = self.Match_num
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            for node in nodes[::-1]:
                sonnodes = self.DAG.queryGraph.findChildNodes(node)
                son_nodelist = [
                    [
                        np.log(
                            self.DAG.edgeWeightDict[(node, lnode)] / 
                            np.sum([
                                self.DAG.edgeWeightDict[(fnode, lnode)] 
                                for fnode in self.DAG.queryGraph.findParentNodes(lnode)
                            ])
                        ), 
                        lnode
                    ]
                    for lnode in sonnodes
                ]
                arrayRangeStart, arrayRangeEnd = self.arrayRangeDict[node]
                sonnde_num = len(son_nodelist)
                nextBeta_M_List = np.full((sonnde_num, Match_num), -np.inf, dtype=np.float64)
                nextBeta_I_List = np.full((sonnde_num, Match_num + 1), -np.inf, dtype=np.float64)
                for i, fnode in enumerate(son_nodelist):
                    _arrayRangeStart, _arrayRangeEnd = self.arrayRangeDict[fnode[1]]
                    weight = fnode[0]            
                    fathernode_leftlimit, fathernode_rightlimit = self.stateRangeDict[2*fnode[1]:2*fnode[1]+2]
                    lo = self.allBaseDict[self.DAG.fragments[fnode[1]]]        
                    nextBeta_M_List[i][fathernode_leftlimit:fathernode_rightlimit] = (
                        beta_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1] + 
                        self.Me_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit] + 
                        weight
                    )
                    nextBeta_I_List[i][fathernode_leftlimit:fathernode_rightlimit+1] = (
                        beta_Matrix_I[_arrayRangeStart:_arrayRangeEnd] + 
                        self.Ie_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit+1] + 
                        weight
                    )
                stateRangeStart, stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                nextBeta_M_List = nextBeta_M_List[:, stateRangeStart:stateRangeEnd]
                nextBeta_I_List = nextBeta_I_List[:, stateRangeStart:stateRangeEnd+1]
                nextBeta_M = np.logaddexp.reduce(nextBeta_M_List, axis=0)          
                nextBeta_I = np.logaddexp.reduce(nextBeta_I_List, axis=0)
                arrayLength = stateRangeEnd - stateRangeStart
                beta_D = np.full(arrayLength, -np.inf)
                beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I = DAGPhmm.calculate_beta_values(
                    nextBeta_M, nextBeta_I, beta_D, stateRangeStart, stateRangeEnd, 
                    Match_num, D2I_array, D2M_array, D2D_array, I2I_array, 
                    I2M_array, I2D_array, M2M_array, M2I_array, M2D_array, arrayLength
                )
                beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = beta_M
                beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = beta_I
                beta_Matrix_D[arrayRangeStart:arrayRangeEnd-1] = beta_D
                left_beta_Matrix_M[arrayRangeStart:arrayRangeEnd-1] = nextBeta_M
                left_beta_Matrix_I[arrayRangeStart:arrayRangeEnd] = nextBeta_I
                doneList[node] = 0          
        def calculate_beta(lock, nodelist, beta_head):

            linearPath_list = self.linearPath_list
            write_beta_head(nodelist, beta_head)           
            while goon_flag.value:            
                try:
                    start_node = q.get(timeout=1)            
                except Empty:
                    continue          
                while start_node is not None:
                    linearPath = linearPath_list[start_node]
                    write_beta(linearPath)            
                    checklist = self.DAG.CG_DAG.findParentNodes(start_node)
                    todolist = []
                    for i in checklist:
                        with lock:           
                            outdegreeDict[i] -= 1
                            if outdegreeDict[i] == 0:           
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()          
                        for i in todolist:
                            q.put(i)             
                    else:
                        start_node = None           
        def is_end(goon_flag):
            while True:
                if 0 == np.count_nonzero(doneList > 0):           
                    goon_flag.value = 0
                    break

        outdegreeDict_shm = shared_memory.SharedMemory(
            create=True, 
            size=self.DAG.CG_DAG.totalNodes * np.dtype(np.uint16).itemsize            
        )
        outdegreeDict = np.ndarray(
            (self.DAG.CG_DAG.totalNodes,), 
            dtype=np.int16, 
            buffer=outdegreeDict_shm.buf            
        )
        link_keys = self.linearPath_link
        for link in link_keys:
            outdegreeDict[link[0]] += 1                            

        doneList_shm = shared_memory.SharedMemory(
            create=True, 
            size=self.DAG.totalNodes * np.dtype(np.uint8).itemsize
        )
        doneList = np.ndarray(
            (self.DAG.totalNodes,), 
            dtype=np.int8, 
            buffer=doneList_shm.buf
        )
        doneList[:] = [1] * self.DAG.totalNodes              
        lock = Lock()                                          
        goon_flag = Value('i', 1)                   
        q = Queue()                       
        beta_head = init_beta_head()
        endnodelist = list(self.DAG.CG_DAG.endNodeSet)                    
        processlist = []
        pool_num = self.pool_num               
        for idx in range(pool_num):
            processlist.append(Process(
                target=calculate_beta,
                args=(lock, endnodelist[idx::pool_num], beta_head)
            ))
        processlist.append(Process(target=is_end, args=(goon_flag, )))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        nextBeta_M_List = np.full(               
            (len(self.graphStartNodes), self.Match_num), 
            -np.inf, 
            dtype=np.float64             
        )
        nextBeta_I_List = np.full(                
            (len(self.graphStartNodes), self.Match_num + 1), 
            -np.inf, 
            dtype=np.float64
        )
        maxright = 0                       
        graphStartNodes = list(self.graphStartNodes)
        for i, fnode in enumerate(graphStartNodes):
            _arrayRangeStart, _arrayRangeEnd = self.arrayRangeDict[fnode]
            fathernode_leftlimit = self.stateRangeDict[2*fnode]
            fathernode_rightlimit = self.stateRangeDict[2*fnode+1]
            lo = self.allBaseDict[self.DAG.fragments[fnode]]          
            if fathernode_rightlimit > maxright:
                maxright = fathernode_rightlimit
            nextBeta_M_List[i][fathernode_leftlimit:fathernode_rightlimit] = (
                beta_Matrix_M[_arrayRangeStart:_arrayRangeEnd-1] 
                + self.Me_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit] 
                + 0                 
            )
            nextBeta_I_List[i][fathernode_leftlimit:fathernode_rightlimit+1] = (
                beta_Matrix_I[_arrayRangeStart:_arrayRangeEnd] 
                + self.Ie_Matrix_degenerate_base[lo][fathernode_leftlimit:fathernode_rightlimit+1] 
                + 0
            )
        stateRangeStart = 0               
        stateRangeEnd = self.maxrange               
        arrayLength = stateRangeEnd - stateRangeStart
        beta_D = np.full(arrayLength, -np.inf)
        nextBeta_M_List = nextBeta_M_List[:, stateRangeStart:stateRangeEnd]
        nextBeta_I_List = nextBeta_I_List[:, stateRangeStart:stateRangeEnd+1]
        nextBeta_M = np.logaddexp.reduce(nextBeta_M_List, axis=0)           
        nextBeta_I = np.logaddexp.reduce(nextBeta_I_List, axis=0)
        beta_D, beta_I, beta_M, nextBeta_M, nextBeta_I = DAGPhmm.calculate_beta_values(
            nextBeta_M, nextBeta_I, beta_D,
            stateRangeStart, stateRangeEnd,
            self.Match_num,  
            self.D2I_array, self.D2M_array, self.D2D_array,           
            self.I2I_array, self.I2M_array, self.I2D_array,             
            self.M2M_array, self.M2I_array, self.M2D_array,           
            arrayLength
        )
        arrayRangeStart = self.arrayRangeDict[-1][0]                 
        beta_Matrix_M[arrayRangeStart:arrayRangeStart+arrayLength] = beta_M
        beta_Matrix_I[arrayRangeStart:arrayRangeStart+arrayLength+1] = beta_I          
        beta_Matrix_D[arrayRangeStart:arrayRangeStart+arrayLength] = beta_D
        left_beta_Matrix_M[arrayRangeStart:arrayRangeStart+arrayLength] = nextBeta_M
        left_beta_Matrix_I[arrayRangeStart:arrayRangeStart+arrayLength+1] = nextBeta_I
        prob = np.logaddexp.reduce([
            nextBeta_M[0] + self.pi_M,             
            nextBeta_I[0] + self.pi_I,              
            beta_D[0] + self.pi_D                   
        ], axis=0)
        outdegreeDict_shm.close()                 
        outdegreeDict_shm.unlink()                           
        doneList_shm.close()
        doneList_shm.unlink()
        problist.append(prob)

    def estep(self,alpha_Matrix_M,alpha_Matrix_I,alpha_Matrix_D,beta_Matrix_M,beta_Matrix_D,beta_Matrix_I,left_beta_Matrix_M,left_beta_Matrix_I):
        def calculate_gamma(nodes,gamma_o_M_list,gamma_o_I_list,E_MM_list,E_MD_list,E_MI_list,E_II_list,E_IM_list,E_DM_list,E_DD_list,MEi_list,IEi_list,DEi_list):

            Match_num = self.Match_num
            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array

            degenerate_base_dict = self.degenerateBaseDictionary.copy()
            degenerate_base_dict.update({'A':[0], 'T':[1], 'C':[2], 'G':[3]})
            allBaseDictReverse = self.allBaseDictReverse

            gamma_o_M = np.full((Match_num, self.normalBaseCount), -np.inf, dtype=np.float64)
            gamma_o_I = np.full((Match_num+1, self.normalBaseCount), -np.inf, dtype=np.float64)
            E_MM = np.full(Match_num-1, -np.inf)
            E_MD = np.full(Match_num-1, -np.inf)
            E_MI = np.full(Match_num, -np.inf)
            E_II = np.full(Match_num+1, -np.inf)
            E_IM = np.full(Match_num, -np.inf)
            E_ID = np.full(Match_num, -np.inf)
            E_DM = np.full(Match_num-1, -np.inf)
            E_DD = np.full(Match_num-1, -np.inf)
            E_DI = np.full(Match_num, -np.inf)
            MEi = np.full(Match_num, -np.inf)
            IEi = np.full(Match_num+1, -np.inf)
            DEi = np.full(Match_num, -np.inf)

            nodes = set(nodes)
            stnodes = self.graphStartNodes | {-1}
            spnodes = nodes & stnodes
            nodes -= spnodes

            def process_node(node, is_start_node):
                if node == -1:
                    stateRangeStart = 0
                    stateRangeEnd = self.maxrange
                else:
                    stateRangeStart = 0 if is_start_node else self.stateRangeDict[2 * node]
                    stateRangeEnd = self.stateRangeDict[2 * node + 1]

                arrayRangeStart, arrayRangeEnd = self.arrayRangeDict[node]
                stateRange = slice(stateRangeStart, stateRangeEnd)
                stateRangeM = slice(stateRangeStart, stateRangeEnd - 1)
                arrayRange = slice(arrayRangeStart, arrayRangeEnd)
                arrayRangeM = slice(arrayRangeStart, arrayRangeEnd - 1)

                baseID = self.allBaseDict[self.DAG.fragments[node]]
                alpha_M = alpha_Matrix_M[arrayRangeM]
                alpha_I = alpha_Matrix_I[arrayRange]
                alpha_D = alpha_Matrix_D[arrayRangeM]
                beta_M = beta_Matrix_M[arrayRangeM]
                beta_I = beta_Matrix_I[arrayRange]
                beta_D = beta_Matrix_D[arrayRangeM]
                nextBeta_M = left_beta_Matrix_M[arrayRangeM]
                nextBeta_I = left_beta_Matrix_I[arrayRange]

                gamma_M = alpha_M + beta_M
                gamma_I = alpha_I + beta_I
                gamma_D = alpha_D + beta_D

                E_MM[stateRangeM] = np.logaddexp(alpha_M[:-1] + M2M_array[stateRangeM] + nextBeta_M[1:], E_MM[stateRangeM])
                E_MD[stateRangeM] = np.logaddexp(alpha_M[:-1] + M2D_array[stateRangeM] + beta_D[1:], E_MD[stateRangeM])
                E_MI[stateRange]  = np.logaddexp(alpha_M + M2I_array[stateRange] + nextBeta_I[1:], E_MI[stateRange])
                E_II[stateRange.start:stateRange.stop+1] = np.logaddexp(alpha_I + I2I_array[stateRange.start:stateRange.stop+1] + nextBeta_I, E_II[stateRange.start:stateRange.stop+1])
                E_IM[stateRange] = np.logaddexp(alpha_I[:-1] + I2M_array[stateRange] + nextBeta_M, E_IM[stateRange])
                E_ID[stateRange] = np.logaddexp(alpha_I[:-1] + I2D_array[stateRange] + beta_D, E_ID[stateRange])
                E_DM[stateRangeM] = np.logaddexp(alpha_D[:-1] + D2M_array[stateRangeM] + nextBeta_M[1:], E_DM[stateRangeM])
                E_DD[stateRangeM] = np.logaddexp(alpha_D[:-1] + D2D_array[stateRangeM] + beta_D[1:], E_DD[stateRangeM])
                E_DI[stateRange] = np.logaddexp(alpha_D + D2I_array[stateRange] + nextBeta_I[1:], E_DI[stateRange])

                MEi[stateRange] = np.logaddexp(gamma_M, MEi[stateRange])
                IEi[stateRange.start:stateRange.stop+1] = np.logaddexp(gamma_I, IEi[stateRange.start:stateRange.stop+1])
                DEi[stateRange] = np.logaddexp(gamma_D, DEi[stateRange])

                if is_start_node:
                    self.gamma_M[node] = [gamma_M[0]]
                    self.gamma_I[node] = [gamma_I[0]]
                    self.gamma_D[node] = [gamma_D[0]]
                else:
                    bases = degenerate_base_dict[allBaseDictReverse[baseID]]
                    gamma_o_M[stateRange, bases] = np.logaddexp(gamma_o_M[stateRange, bases],
                                                                gamma_M[:, None] - np.log(len(bases)))
                    gamma_o_I[stateRange.start:stateRange.stop+1, bases] = np.logaddexp(
                        gamma_o_I[stateRange.start:stateRange.stop+1, bases],
                        gamma_I[:, None] - np.log(len(bases))
                    )

            for node in spnodes:
                process_node(node, is_start_node=True)
            for node in nodes:
                process_node(node, is_start_node=False)

            gamma_o_M_list.append(gamma_o_M)
            gamma_o_I_list.append(gamma_o_I)
            E_MM_list.append(E_MM)
            E_MD_list.append(E_MD)
            E_MI_list.append(E_MI)
            E_II_list.append(E_II)
            E_IM_list.append(E_IM)
            E_DM_list.append(E_DM)
            E_DD_list.append(E_DD)
            MEi_list.append(MEi)
            DEi_list.append(DEi)
            IEi_list.append(IEi)
        
        problist = Manager().list()                
        problist.append([])                        
        allprocesslist = []
        allprocesslist.append(Process(
            target=self.forward,
            args=(alpha_Matrix_M, alpha_Matrix_D, alpha_Matrix_I)
        ))  
        allprocesslist.append(Process(
            target=self.backward,
            args=(beta_Matrix_M, beta_Matrix_D, beta_Matrix_I, 
                 left_beta_Matrix_M, left_beta_Matrix_I, problist)
        ))  
        [p.start() for p in allprocesslist]
        [p.join() for p in allprocesslist]
        prob = problist[-1]            
        self.gamma_M = Manager().dict()                
        self.gamma_I = Manager().dict()                   
        self.gamma_D = Manager().dict()                 
        self.nodelist = list(range(self.DAG.totalNodes))        
        self.nodelist.append(-1)                  
        pool_num = self.pool_num * 2                     
        gamma_o_M_list = Manager().list()                  
        gamma_o_I_list = Manager().list()                   
        E_MM_list = Manager().list()            
        E_MD_list = Manager().list()            
        E_MI_list = Manager().list()            
        E_II_list = Manager().list()              
        E_IM_list = Manager().list()            
        E_ID_list = Manager().list()            
        E_DM_list = Manager().list()            
        E_DD_list = Manager().list()              
        E_DI_list = Manager().list()            
        MEi_list = Manager().list()              
        IEi_list = Manager().list()               
        DEi_list = Manager().list()               
        processlist = []
        for index in range(pool_num):
            processlist.append(Process(
                target=calculate_gamma,
                args=(self.nodelist[index::pool_num],          
                      gamma_o_M_list, gamma_o_I_list,
                      E_MM_list, E_MD_list, E_MI_list,
                      E_II_list, E_IM_list, E_DM_list,
                      E_DD_list, MEi_list, IEi_list, DEi_list)
            ))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        self.gamma_o_M = np.logaddexp.reduce(gamma_o_M_list, axis=0) - prob
        self.gamma_o_I = np.logaddexp.reduce(gamma_o_I_list, axis=0) - prob
        self.E_MM = np.logaddexp.reduce(E_MM_list, axis=0) - prob        
        self.E_MD = np.logaddexp.reduce(E_MD_list, axis=0) - prob        
        self.E_MI = np.logaddexp.reduce(E_MI_list, axis=0) - prob        
        self.E_II = np.logaddexp.reduce(E_II_list, axis=0) - prob        
        self.E_IM = np.logaddexp.reduce(E_IM_list, axis=0) - prob        
        self.E_ID = np.logaddexp.reduce(E_ID_list, axis=0) - prob          
        self.E_DM = np.logaddexp.reduce(E_DM_list, axis=0) - prob        
        self.E_DD = np.logaddexp.reduce(E_DD_list, axis=0) - prob        
        self.E_DI = np.logaddexp.reduce(E_DI_list, axis=0) - prob        
        self.MEi = np.logaddexp.reduce(MEi_list, axis=0) - prob             
        self.DEi = np.logaddexp.reduce(DEi_list, axis=0) - prob              
        self.IEi = np.logaddexp.reduce(IEi_list, axis=0) - prob              
        print()
        print('P(x): ',prob)
        return prob
    
    def mstep(self, perturbation=False, m_global_random_low=0.2, m_global_random_up=0.4):
        
        self.Ie_Matrix = self.gamma_o_I - self.IEi.reshape(-1, 1)
        self.Me_Matrix = self.gamma_o_M - self.MEi.reshape(-1, 1)
        adds = np.array([self.emProbAdds_Match] * (self.normalBaseCount), dtype=np.float64)
        head_adds = np.array([self.emProbAdds_Match_head] * (self.normalBaseCount), dtype=np.float64)
        tail_adds = np.array([self.emProbAdds_Match_tail] * (self.normalBaseCount), dtype=np.float64)
        head_length = self.head_length
        tail_length = self.tail_length
        for i in range(self.Match_num):
            if i <= head_length:
                Madds = head_adds
            elif i >= self.Match_num - tail_length:
                Madds = tail_adds
            else:
                Madds = adds
            if perturbation:
                random_array = np.log(np.random.rand(self.normalBaseCount))
                tmp = np.logaddexp(self.Me_Matrix[i], random_array)
            tmp = np.logaddexp(self.Me_Matrix[i], Madds)
            Psum = np.logaddexp.reduce(tmp, axis=0)          
            self.Me_Matrix[i] = tmp - Psum           
        self.Ie_Matrix = np.full_like(self.Ie_Matrix, np.log(1/4))
        Adds_m_ = [self.trProbAdds_mm, self.trProbAdds_mi, self.trProbAdds_md]
        self.M2M_array = self.E_MM - self.MEi[:-1]
        self.M2I_array = self.E_MI - self.MEi
        self.M2D_array = self.E_MD - self.MEi[:-1]
        for i in range(self.Match_num-1):
            Adds = Adds_m_
            tmp = np.zeros(3, dtype=np.float64)
            tmp[0] = np.logaddexp(self.M2M_array[i], Adds[0])
            tmp[1] = np.logaddexp(self.M2I_array[i], Adds[1])
            tmp[2] = np.logaddexp(self.M2D_array[i], Adds[2])
            if perturbation:
                random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
                random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
                random_3 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
                tmp[0] = np.logaddexp(tmp[0], random_1)
                tmp[1] = np.logaddexp(tmp[1], random_2)
                tmp[2] = np.logaddexp(tmp[2], random_3)
            Psum = np.logaddexp.reduce(tmp, axis=0)
            self.M2M_array[i] = tmp[0] - Psum
            self.M2I_array[i] = tmp[1] - Psum
            self.M2D_array[i] = tmp[2] - Psum
        tmp = np.zeros(2, dtype=np.float64)
        tmp[0] = np.logaddexp(self.M2E, self.trProbAdds_mend)
        tmp[1] = np.logaddexp(self.M2I_array[-1], self.trProbAdds_mi_tail)
        if perturbation:
            random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            tmp[0] = np.logaddexp(tmp[0], random_1)
            tmp[1] = np.logaddexp(tmp[1], random_2)
        Psum = np.logaddexp.reduce(tmp, axis=0)
        self.M2E = tmp[0] - Psum
        self.M2I_array[-1] = tmp[1] - Psum
        Adds_im = self.trProbAdds_im
        Adds_ii = self.trProbAdds_ii
        Adds_IE = self.trProbAdds_iend 
        Adds_iitail = self.trProbAdds_ii_tail 
        self.I2D_array = self.E_ID - self.IEi[:-1]
        self.I2I_array[1:-1] = self.E_II[1:-1] - self.IEi[1:-1]
        self.I2M_array[1:] = self.E_IM[1:] - self.IEi[1:-1]
        i = 0
        tmp = np.zeros(2, dtype=np.float64)
        tmp[0] = np.logaddexp(self.I2M_array[i], Adds_im)
        tmp[1] = np.logaddexp(self.I2I_array[i], Adds_ii)
        Psum = np.logaddexp.reduce(tmp, axis=0)
        self.I2M_array[0] = tmp[0] - Psum
        self.I2I_array[0] = tmp[1] - Psum
        for i in range(1, self.Match_num):
            tmp = np.zeros(2, dtype=np.float64)
            tmp[0] = np.logaddexp(self.I2M_array[i], Adds_im)
            tmp[1] = np.logaddexp(self.I2I_array[i], Adds_ii)
            if perturbation:
                random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
                random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
                tmp[0] = np.logaddexp(tmp[0], random_1)
                tmp[1] = np.logaddexp(tmp[1], random_2)
            Psum = np.logaddexp.reduce(tmp, axis=0)
            self.I2M_array[i] = tmp[0] - Psum
            self.I2I_array[i] = tmp[1] - Psum
        tmp = np.zeros(2, dtype=np.float64)
        tmp[0] = np.logaddexp(self.I2E, Adds_IE)
        tmp[1] = np.logaddexp(self.I2I_array[-1], Adds_iitail)
        if perturbation:
            random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            tmp[0] = np.logaddexp(tmp[0], random_1)
            tmp[1] = np.logaddexp(tmp[1], random_2)
        Psum = np.logaddexp.reduce(tmp, axis=0)
        self.I2E = tmp[0] - Psum
        self.I2I_array[-1] = tmp[1] - Psum
        self.D2M_array = np.full_like(self.D2M_array, np.log(1/2))
        self.D2D_array = np.full_like(self.D2D_array, np.log(1/2))
        Adds_Pi_M = self.trProbAdds_PiM
        Adds_Pi_I = self.trProbAdds_PiI
        Adds_Pi_D = self.trProbAdds_PiD
        gamma_start_M_list = []
        gamma_start_I_list = []
        for i in self.graphStartNodes:
            gamma_start_M_list.append(self.gamma_M[i])
            gamma_start_I_list.append(self.gamma_I[i])
        gamma_start_M = np.logaddexp.reduce(gamma_start_M_list, axis=0)
        gamma_start_I = np.logaddexp.reduce(gamma_start_I_list, axis=0)
        self.pi_M = gamma_start_M[0] - np.logaddexp.reduce(
            [gamma_start_M[0], gamma_start_I[0], self.gamma_D[-1][0]], axis=0)
        self.pi_I = gamma_start_I[0] - np.logaddexp.reduce(
            [gamma_start_M[0], gamma_start_I[0], self.gamma_D[-1][0]], axis=0)
        self.pi_D = self.gamma_D[-1][0] - np.logaddexp.reduce(
            [gamma_start_M[0], gamma_start_I[0], self.gamma_D[-1][0]], axis=0)
        tmp = np.zeros(3, dtype=np.float64)
        tmp[0] = np.logaddexp(self.pi_M, Adds_Pi_M)
        tmp[1] = np.logaddexp(self.pi_D, Adds_Pi_D)
        tmp[2] = np.logaddexp(self.pi_I, Adds_Pi_I)
        if perturbation:
            random_1 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_2 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            random_3 = np.log(np.random.uniform(m_global_random_low, m_global_random_up, 1))
            tmp[0] = np.logaddexp(tmp[0], random_1)
            tmp[1] = np.logaddexp(tmp[1], random_2)
            tmp[2] = np.logaddexp(tmp[2], random_3)
        Psum = np.logaddexp.reduce(tmp, axis=0)
        self.pi_M = tmp[0] - Psum
        self.pi_D = tmp[1] - Psum
        self.pi_I = tmp[2] - Psum
        self.train_times += 1
        self.Me_Matrix_degenerate_base = []
        self.Ie_Matrix_degenerate_base = []
        for i in ['A', 'T', 'C', 'G']:
            self.Me_Matrix_degenerate_base.append(self.Me_Matrix[:, self.commonBaseDict[i]])
            self.Ie_Matrix_degenerate_base.append(self.Ie_Matrix[:, self.commonBaseDict[i]])
        for base in ["R", "Y", "M", "K", "S", "W", "H", "B", "V", "D", "N"]:
            degenerate_base = self.degenerateBaseDictionary[base]
            self.Ie_Matrix_degenerate_base.append(
                np.logaddexp.reduce(self.Ie_Matrix.T[degenerate_base] - np.log(len(degenerate_base)), axis=0))
            self.Me_Matrix_degenerate_base.append(
                np.logaddexp.reduce(self.Me_Matrix.T[degenerate_base] - np.log(len(degenerate_base)), axis=0))
        self.Me_Matrix_degenerate_base = np.array(self.Me_Matrix_degenerate_base)
        self.Ie_Matrix_degenerate_base = np.array(self.Ie_Matrix_degenerate_base)
        np.nan_to_num(self.Me_Matrix_degenerate_base)
        np.nan_to_num(self.Ie_Matrix_degenerate_base)
        parameterDict = {}
        parameterDict['_mm'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_mm'][1:-1] = self.M2M_array
        parameterDict['_mm'][0] = self.pi_M
        parameterDict['_mm'][-1] = self.M2E
        parameterDict['_dm'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_dm'][1:-1] = self.D2M_array
        parameterDict['_dm'][-1] = self.D2E
        parameterDict['_im'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_im'][:-1] = self.I2M_array
        parameterDict['_im'][-1]=self.I2E
        parameterDict['_mi'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_mi'][1:] = self.M2I_array
        parameterDict['_mi'][0] = self.pi_I
        parameterDict['_md'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_md'][1:-1] = self.M2D_array
        parameterDict['_md'][0] = self.pi_D
        parameterDict['_dd'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_dd'][1:-1] = self.D2D_array
        parameterDict['_di'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_di'][1:] = self.D2I_array
        parameterDict['_ii'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_ii'] = self.I2I_array
        parameterDict['_id'] = np.full(self.Match_num+1,-np.inf)
        parameterDict['_id'][:-1]=self.I2D_array
        parameterDict['match_emission'] = np.full((self.Match_num,4),-np.inf)
        parameterDict['match_emission'] = self.Me_Matrix
        parameterDict['insert_emission'] = np.full((self.Match_num+1,4),-np.inf)
        parameterDict['insert_emission'] = self.Ie_Matrix
        np.save(self.train_DAG_Path/'ini/{}_pc_{}.npy'.format(self.parameterName,self.train_times),parameterDict)


    def init_train_data_with_DAG(self, Viterbi_DAG_Path, ref_node_list, ref_seq, modify_dict,DAG, kill_degenerate_base=True, windows_length=100, threads=3):
        self.windows_length = windows_length
        self.head_length = modify_dict['head_length']             
        self.tail_length = modify_dict['tail_length']             
        self.emProbAdds_Match = modify_dict['emProbAdds_Match']                    
        self.emProbAdds_Match_head = modify_dict['emProbAdds_Match_head']              
        self.emProbAdds_Match_tail = modify_dict['emProbAdds_Match_tail']              
        self.trProbAdds_mm = modify_dict['trProbAdds_mm']               
        self.trProbAdds_md = modify_dict['trProbAdds_md']                
        self.trProbAdds_mi = modify_dict['trProbAdds_mi']               
        self.trProbAdds_PiM = modify_dict['trProbAdds_PiM']               
        self.trProbAdds_PiI = modify_dict['trProbAdds_PiI']               
        self.trProbAdds_PiD = modify_dict['trProbAdds_PiD']               
        self.trProbAdds_im = modify_dict['trProbAdds_im']               
        self.trProbAdds_ii = modify_dict['trProbAdds_ii']               
        self.trProbAdds_iend = modify_dict['trProbAdds_iend']           
        self.trProbAdds_ii_tail = modify_dict['trProbAdds_ii_tail']                
        self.trProbAdds_mend = modify_dict['trProbAdds_mend']            
        self.trProbAdds_mi_tail = modify_dict['trProbAdds_mi_tail']

        self.DAG = DAG

        ref_node_list = self.DAG.map_ref_seq_to_graph(ref_seq).tolist()
        ref_node_list = ['x']*(len(ref_seq)-len(ref_node_list)) + ['x' if i==-1 else i  for i in ref_node_list]
        self.DAG.fragmentReduce()

        
        range_List = self.DAG.queryGraph.calculateStateRange(ref_node_list)
        

        if kill_degenerate_base:
            try:
                idmapping = self.DAG.removeDegenerateBasePaths()
                idmapping = {value: key for key, value in idmapping.items()}
            except:
                idmapping = {i: i for i in range(self.DAG.totalNodes)}
        else:
            idmapping = {i: i for i in range(self.DAG.totalNodes)}

        self.maxrange = 0                          
        self.stateRangeDict = []                              
        self.range_length = 0                   
        self.arrayRangeDict = []                    
        for index in range(self.DAG.totalNodes):
            node = idmapping[index]

            start = max(0, range_List[node][0] - self.windows_length)
            end = min(len(ref_node_list), range_List[node][1] + self.windows_length)
            self.stateRangeDict.extend([start, end])
            sted = [self.range_length]
            self.range_length += (end - start + 1)
            sted.append(self.range_length)
            self.arrayRangeDict.append(sted)
            if (end - start + 1) > self.maxrange:
                self.maxrange = end - start + 1
        self.maxrange = min(self.Match_num,self.maxrange)
        sted = [self.range_length]
        self.range_length += (self.maxrange + 1)
        sted.append(self.range_length)
        self.arrayRangeDict.append(sted)
        self.vnum = self.DAG.sequenceNum
        vtuple = np.load(Viterbi_DAG_Path / 'v_id.npy').tolist()
        self.all_source = {tu[1] for tu in vtuple}            
        self.vlist = [tu[1] for tu in vtuple]               
        self.v2id_dict = dict(vtuple)                         
        self.id2v_dict = {value: key for key, value in vtuple}          
        self.pool_num = max(threads // 2, 1)                
        self.graphStartNodes = set()
        self.graphEndNodes = set()
        for nodeid in range(self.DAG.totalNodes):
            if self.DAG.queryGraph.findParentNodes(nodeid) == []:
                self.graphStartNodes.add(nodeid)
            if self.DAG.queryGraph.findChildNodes(nodeid) == []:
                self.graphEndNodes.add(nodeid)
        ed = datetime.now()
        self.DAG.startNodeSet = self.graphStartNodes
        self.DAG.endNodeSet = self.graphEndNodes
        self.linearPath_list, self.linearPath_link, nodeID_linearPathID_Dict = build_coarse_grained_graph(
            self.DAG.queryGraph, self.DAG.edgeWeightDict)
        self.DAG.CG_DAG = DAGStru(len(self.linearPath_list), self.linearPath_link)
        self.DAG.CG_DAG.calculateCoordinates()            
        self.sequenceNum = self.DAG.sequenceNum

    def fit(self):


        self.range_length = int(self.range_length)
        shared_array_alphaM = Array('d', self.range_length)               
        alpha_Matrix_M = np.frombuffer(shared_array_alphaM.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_alphaI = Array('d', self.range_length)                
        alpha_Matrix_I = np.frombuffer(shared_array_alphaI.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_alphaD = Array('d', self.range_length)                
        alpha_Matrix_D = np.frombuffer(shared_array_alphaD.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_betaM = Array('d', self.range_length)                  
        beta_Matrix_M = np.frombuffer(shared_array_betaM.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_betaI = Array('d', self.range_length)                 
        beta_Matrix_I = np.frombuffer(shared_array_betaI.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_betaD = Array('d', self.range_length)                 
        beta_Matrix_D = np.frombuffer(shared_array_betaD.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_leftbetaM = Array('d', self.range_length)              
        left_beta_Matrix_M = np.frombuffer(shared_array_leftbetaM.get_obj(), dtype=np.float64).reshape(self.range_length)
        shared_array_leftbetaI = Array('d', self.range_length)              
        left_beta_Matrix_I = np.frombuffer(shared_array_leftbetaI.get_obj(), dtype=np.float64).reshape(self.range_length)
        while True:
            alpha_Matrix_M[:] = alpha_Matrix_D[:] = alpha_Matrix_I[:] = -np.inf
            beta_Matrix_M[:] = beta_Matrix_D[:] = beta_Matrix_I[:] = -np.inf  
            left_beta_Matrix_M[:] = left_beta_Matrix_I[:] = -np.inf
            prob = self.estep(alpha_Matrix_M, alpha_Matrix_I, alpha_Matrix_D,
                            beta_Matrix_M, beta_Matrix_D, beta_Matrix_I,
                            left_beta_Matrix_M, left_beta_Matrix_I)

            self.mstep()  
            break

    def init_viterbi_data_with_refseq(self, Viterbi_DAG_Path, Viterbi_result_path,ref_seq,DAG, polyA=False, windows_length=100, threads=3):
        self.Viterbi_result_path = Viterbi_result_path
        self.windows_length = windows_length
        self.Viterbi_DAG_Path = Viterbi_DAG_Path
        
        self.DAG = DAG
        ref_node_list = self.DAG.map_ref_seq_to_graph(ref_seq)
        ref_node_list = ['x']*(len(ref_seq)-len(ref_node_list)) + [anchor if anchor!=-1 else 'x' for anchor in  ref_node_list]

        self.DAG.fragmentReduce()

        self.linearPath_list, self.linearPath_link, nodeID_linearPathID_Dict = build_coarse_grained_graph(
            self.DAG.queryGraph, self.DAG.edgeWeightDict)
        self.DAG.CG_DAG = DAGStru(len(self.linearPath_list), self.linearPath_link)
        self.DAG.CG_DAG.calculateCoordinates() 

        self.DAG.fragmentReduce() 

        self.DAG.queryGraph.calculateStateRange(ref_node_list)
        self.vnum = self.DAG.sequenceNum  
        vtuple = np.load(Viterbi_DAG_Path/'v_id.npy').tolist()
        self.all_source = {tu[1] for tu in vtuple}  
        self.vlist = [tu[1] for tu in vtuple]  
        self.id2v_dict = {tu[1]:tu[0] for tu in vtuple}
        self.v2id_dict = dict(vtuple)

        self.pool_num = max(threads, 1) 
        self.DAG.totalNodes = self.DAG.totalNodes  
        
        self.graphStartNodes = self.DAG.startNodeSet  
        self.graphEndNodes = self.DAG.endNodeSet  
        self.sequenceNum = self.DAG.sequenceNum  
        
        self.maxrange = 0  
        self.stateRangeDict = []  
        self.range_length = 0    
        self.arrayRangeDict = [] 
        
        for node in range(self.DAG.totalNodes):
            min_pos = self.DAG.queryGraph.ref_coor[node][0] - self.windows_length
            max_pos = self.DAG.queryGraph.ref_coor[node][1] + self.windows_length
            self.stateRangeDict.extend([
                max(0, min_pos), 
                min(len(ref_node_list), max_pos)
            ])
            
            sted = [self.range_length]
            self.range_length += (self.stateRangeDict[-1] - self.stateRangeDict[-2] + 1)
            sted.append(self.range_length)
            self.arrayRangeDict.append(sted)
            
            current_range = self.stateRangeDict[-1] - self.stateRangeDict[-2]
            if current_range > self.maxrange:
                self.maxrange = current_range

        self.maxrange = min(self.Match_num,self.maxrange)
        sted = [self.range_length]
        self.range_length += self.maxrange  
        sted.append(self.range_length)
        self.arrayRangeDict.append(sted)
        
        if polyA == True:
            self.M2I_array[-1] = np.log(0.1)                      
            self.M2E = np.log(0.9)                            
                      
        self.ref_seq = ref_seq
    

    def Viterbi(self,seqiddb,threads=4):

        def write_hiddensates_head(indexlist):

            decrease_list=np.array([-1,-1,0])
            for index in indexlist:
                linearPath = self.linearPath_list[index]
                node = linearPath.pop()
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                delta_all = [delta_M_mem[arrayRangeEnd-2],delta_D_mem[arrayRangeEnd-2],delta_I_mem[arrayRangeEnd-1]]
                delta_index_start,delta_index_end = self.stateRangeDict[2*node:2*node+2]
                matrix_length=delta_index_end-delta_index_start
                nowstate = np.argmax([delta_all[0]+self.M2E,delta_all[1]+self.D2E,delta_all[2]+self.I2E])
                nowstateindex = self.Match_num+decrease_list[nowstate]
                laststate=laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]
                nowstate, nowstateindex = DAGPhmm.update_state(laststate,matrix_length,nowstate, nowstateindex, delta_index_start, decrease_list)
                if nowstate==2:
                    hidden_states[node] = {(nowstate,nowstateindex):['',1]}
                    ali[nowstateindex] = max([1,ali[nowstateindex]])
                else:
                    hidden_states[node] = {(nowstate,nowstateindex):['',0]}
                write_hiddensates(linearPath)
                doneList[node]=0
                checklist=self.DAG.CG_DAG.findParentNodes(index)
                for i in checklist:
                    with lock:
                        outdegreeDict[i]-=1
                        if outdegreeDict[i]==0:
                            q.put(i)
        def write_hiddensates(nodes):

            is_head=True
            for node in nodes[::-1]:
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                children_nodes = self.DAG.queryGraph.findChildNodes(node)
                decrease_list=np.array([-1,-1,0])
                delta_index_start,delta_index_end = self.stateRangeDict[2*node:2*node+2]
                range_length = delta_index_end-delta_index_start
                tmp_state_dict={}
                laststate_local = laststate_mem[3*arrayRangeStart:3*arrayRangeEnd]
                delta_M,delta_D,delta_I = delta_M_mem[arrayRangeStart:arrayRangeEnd-1],delta_D_mem[arrayRangeStart:arrayRangeEnd-1],delta_I_mem[arrayRangeStart:arrayRangeEnd]
                for fnode in children_nodes:
                    _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode]
                    delta_index_start_f,delta_index_end_f = self.stateRangeDict[2*fnode:2*fnode+2]
                    range_length_f = delta_index_end_f - delta_index_start_f
                    orinode = orinode_mem[3*_arrayRangeStart:3*_arrayRangeEnd]
                    laststate = laststate_mem[3*_arrayRangeStart:3*_arrayRangeEnd]
                    children_node_hiddenstate = hidden_states[fnode]
                    for oristate in children_node_hiddenstate.keys():
                        nowstate = oristate[0]
                        nowstateindex = oristate[1]
                        fm_index=nowstate*range_length_f+nowstateindex-delta_index_start_f
                        try:
                            ori_node = orinode[fm_index]
                        except:
                            ori_node=-1
                        from_state = [(fnode,oristate)]
                        ori_step = children_node_hiddenstate[oristate][1]
                        if ori_node==node:
                            nowstate = laststate[fm_index]
                            nowstateindex += decrease_list[nowstate]
                            nowstate, nowstateindex = DAGPhmm.update_state(laststate_local,range_length,nowstate, nowstateindex, delta_index_start, decrease_list)
                        else:
                            if nowstateindex-delta_index_start == 0:
                                nowstate=2
                            else:
                                indexs=nowstateindex-delta_index_start+decrease_list
                                try:
                                    delta_all = np.array([delta_M[indexs[0]],delta_D[indexs[1]],delta_I[indexs[2]]])
                                except:
                                    delta_all = np.array([-1,-1,-1])
                                P_T =  np.fromiter((Adict[(laststate,nowstate)][nowstateindex+decrease_list[laststate]] for laststate in range(3)),dtype=np.float64)
                                nowstate = np.argmax(delta_all+P_T)
                            nowstateindex+=decrease_list[nowstate]
                            nowstate, nowstateindex = DAGPhmm.update_state(laststate_local,range_length,nowstate, nowstateindex, delta_index_start, decrease_list)
                        state = (nowstate,nowstateindex)
                        tmp_state_dict[state] = tmp_state_dict.get(state,[[],[]])
                        tmp_state_dict[state][0].extend(from_state)
                        tmp_state_dict[state][1].append(int((ori_step+1)*(nowstate/2)))

                if len(tmp_state_dict.keys())==1:
                    new_tmp_state_dict={}
                    state = next(iter(tmp_state_dict))
                    if state[0]==2:
                        new_tmp_state_dict[state] = ['', max(tmp_state_dict[state][1])]
                        lock.acquire()
                        ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                        lock.release()
                    else:
                        new_tmp_state_dict[state] = ['', 0]
                else:
                    if is_head:
                        tmp_hiddenstates_dict={}
                        for childernnode in children_nodes:
                            tmp_hiddenstates_dict[childernnode] = hidden_states[childernnode]
                        content_source = query_contend_sequence_id(node)
                        content_source-=self.DAG.offsetArray[node]
                        new_tmp_state_dict = {}
                        for state in tmp_state_dict.keys():
                            sourcelist = []
                            allsize=0
                            for f in tmp_state_dict[state][0]:
                                fn = f[0]
                                fstate=f[1]
                                if isinstance(tmp_hiddenstates_dict[fn][fstate][0],str):
                                    source = query_contend_sequence_id(fn)
                                    if source.size!=0:
                                        source-=1
                                        sourcelist.append(source)
                                        allsize+=source.shape[0]
                                else:
                                    source = tmp_hiddenstates_dict[fn][fstate][0]
                                    source-=1
                                    sourcelist.append(source)
                                    allsize+=source.shape[0]
                            source_matrix = np.full((allsize,),0,dtype=np.uint64)
                            size_now = 0
                            for v in sourcelist:
                                col_size = v.shape[0]             
                                source_matrix[size_now:size_now+col_size] = v            
                                size_now += col_size
                            if sourcelist:
                                nct = np.intersect1d(content_source, source_matrix)
                                if nct.size!=0:
                                    if state[0]==2:
                                        new_tmp_state_dict[state] = [nct,max(tmp_state_dict[state][1])]
                                        lock.acquire()
                                        ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                                        lock.release()
                                    else:
                                        new_tmp_state_dict[state] = [nct,0]
                    else:
                        tmp_hiddenstates_dict={}
                        for childernnode in children_nodes:
                            tmp_hiddenstates_dict[childernnode] = hidden_states[childernnode]
                        new_tmp_state_dict = {}
                        for state in tmp_state_dict.keys():
                            sourcelist = []
                            allsize=0
                            for f in tmp_state_dict[state][0]:
                                fn = f[0]
                                fstate=f[1]
                                source = tmp_hiddenstates_dict[fn][fstate][0]
                                source-=1
                                sourcelist.append(source)
                                allsize+=source.shape[0]
                            source_matrix = np.full((allsize,),0,dtype=np.uint64)
                            size_now = 0
                            for v in sourcelist:
                                col_size = v.shape[0]             
                                source_matrix[size_now:size_now+col_size] = v            
                                size_now += col_size
                            if state[0]==2:
                                new_tmp_state_dict[state] = [source_matrix,max(tmp_state_dict[state][1])]
                                lock.acquire()
                                ali[state[1]] = max([new_tmp_state_dict[state][1],ali[state[1]]])
                                lock.release()
                            else:
                                new_tmp_state_dict[state] = [source_matrix,0]
                    if len(new_tmp_state_dict) == 1:
                        state = next(iter(new_tmp_state_dict))
                        new_tmp_state_dict[state] = ('', new_tmp_state_dict[state][1])
                hidden_states[node] = new_tmp_state_dict

                doneList[node]=0
                is_head = False

                
        def init_delta_head():

            delta_head_dict={}
            for baseID in range(self.Me_Matrix_degenerate_base.shape[0]):
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                delta_M = np.full(self.Match_num, -np.inf, dtype=np.float64)
                delta_I = np.full(self.Match_num + 1, -np.inf, dtype=np.float64)
                delta_D = np.full(self.Match_num, -np.inf, dtype=np.float64)
                delta_M[0]=self.pi_M+Me[0]
                delta_I[0]=self.pi_I+Ie[0]
                last_delta_D = np.full(self.Match_num, -np.inf, dtype=np.float64)
                last_delta_D[:self.maxrange] = First_deata_D
                delta_M[1:] = last_delta_D[:-1]+self.D2M_array+Me[1:]
                delta_I[1:] = last_delta_D+self.D2I_array+Ie[1:]
                maxProbOrigin_D = np.array([0]*(self.Match_num),dtype='int')
                maxProbOrigin_D[0]=2
                delta_D[0]=delta_I[0]+self.I2D_array[0]
                for i in range(1,self.Match_num):
                    D_arg = [delta_M[i-1]+self.M2D_array[i-1],delta_D[i-1]+self.D2D_array[i-1],delta_I[i]+self.I2D_array[i]]
                    delta_D[i] = np.max(D_arg)
                    maxProbOrigin_D[i] = np.argmax(D_arg)
                maxProbOrigin_M = np.full(self.Match_num, 2, dtype='int')
                maxProbOrigin_M[0]=3
                maxProbOrigin_I = np.full(self.Match_num+1, 2, dtype='int')
                maxProbOrigin_I[0]=3
                delta_head_dict[baseID]=[delta_M,delta_I,delta_D,maxProbOrigin_M,maxProbOrigin_I,maxProbOrigin_D]
            return delta_head_dict
        def write_delta_head(indexlist,delta_head_dict):

            for index in indexlist:
                
                linearPath = self.linearPath_list[index]
                node = linearPath[0]
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict.get(self.DAG.fragments[node][-1],14)
                values = delta_head_dict[baseID]
                stateRangeStart=self.stateRangeDict[2*node]
                stateRangeEnd=self.stateRangeDict[2*node+1]
                maxProbOrigin_M = values[3][stateRangeStart:stateRangeEnd]
                maxProbOrigin_I = values[4][stateRangeStart:stateRangeEnd+1]
                maxProbOrigin_D = values[5][stateRangeStart:stateRangeEnd]
                delta_M_mem[arrayRangeStart:arrayRangeEnd-1] = values[0][stateRangeStart:stateRangeEnd]
                delta_D_mem[arrayRangeStart:arrayRangeEnd-1] = values[2][stateRangeStart:stateRangeEnd]
                delta_I_mem[arrayRangeStart:arrayRangeEnd] = values[1][stateRangeStart:stateRangeEnd+1]
                laststate_mem[3*arrayRangeStart:3*arrayRangeEnd-2]= np.concatenate((maxProbOrigin_M,maxProbOrigin_D,maxProbOrigin_I))
                write_delta(linearPath[1:])
                doneList[node]=0
                checklist=self.DAG.CG_DAG.findChildNodes(index)
                for i in checklist:
                    with lock:
                        indegree_dict[i]-=1
                        if indegree_dict[i]==0:
                            q.put(i)
        def write_delta(nodes):

            D2D_array = self.D2D_array
            I2D_array = self.I2D_array
            M2D_array = self.M2D_array
            D2I_array = self.D2I_array
            I2I_array = self.I2I_array
            M2I_array = self.M2I_array
            D2M_array = self.D2M_array
            I2M_array = self.I2M_array
            M2M_array = self.M2M_array
            st=0
            for node in nodes:
                arrayRangeStart,arrayRangeEnd = self.arrayRangeDict[node]
                baseID = self.allBaseDict.get(self.DAG.fragments[node][-1],14)
                partennodes = self.DAG.queryGraph.findParentNodes(node)
                parentNodeWeightList=[]
                if st:
                    for lnode in partennodes:
                        parentNodeWeightList.append([1,lnode])
                else:
                    alist = [self.DAG.edgeWeightDict[(lnode,node)] for lnode in partennodes]
                    b = np.sum(alist)
                    for lnode in partennodes:
                        a = self.DAG.edgeWeightDict[(lnode,node)]
                        adds=0
                        ab = (a+adds/len(partennodes))/(b+adds)
                        parentNodeWeightList.append([ab,lnode])

                st+=1
                Me = self.Me_Matrix_degenerate_base[baseID]
                Ie = self.Ie_Matrix_degenerate_base[baseID]
                last_delta_M_list = np.full((len(parentNodeWeightList), self.Match_num), -np.inf, dtype=np.float64)
                last_delta_I_list = np.full((len(parentNodeWeightList), self.Match_num+1), -np.inf, dtype=np.float64)
                last_delta_D_list = np.full((len(parentNodeWeightList), self.Match_num), -np.inf, dtype=np.float64)
                for i, fnode in enumerate(parentNodeWeightList):
                    _arrayRangeStart,_arrayRangeEnd = self.arrayRangeDict[fnode[1]]
                    weight = np.log(fnode[0])
                    fathernode_left_limit = self.stateRangeDict[2*fnode[1]]
                    fathernode_right_limit = self.stateRangeDict[2*fnode[1]+1]
                    last_delta_M_list[i][fathernode_left_limit:fathernode_right_limit] = delta_M_mem[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_delta_D_list[i][fathernode_left_limit:fathernode_right_limit] = delta_D_mem[_arrayRangeStart:_arrayRangeEnd-1] + weight
                    last_delta_I_list[i][fathernode_left_limit:fathernode_right_limit+1] = delta_I_mem[_arrayRangeStart:_arrayRangeEnd] + weight
                stateRangeStart,stateRangeEnd = self.stateRangeDict[2*node:2*node+2]
                arrayLength =stateRangeEnd-stateRangeStart
                last_delta_M_list = last_delta_M_list[:,stateRangeStart:stateRangeEnd]
                last_delta_I_list = last_delta_I_list[:,stateRangeStart:stateRangeEnd+1]
                last_delta_D_list = last_delta_D_list[:,stateRangeStart:stateRangeEnd]
                last_delta_M = np.max(last_delta_M_list,axis=0) 
                last_delta_I = np.max(last_delta_I_list,axis=0) 
                last_delta_D = np.max(last_delta_D_list,axis=0)
                lm=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_M_list,axis=0)]
                li=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_I_list,axis=0)]
                ld=[parentNodeWeightList[idx][1] for idx in  np.argmax(last_delta_D_list,axis=0)]
                delta_I, delta_M, delta_D, maxProbOrigin_I, maxProbOrigin_M, maxProbOrigin_D = DAGPhmm.calculate_delta_values(stateRangeStart, stateRangeEnd, arrayLength,
                                                                            last_delta_I, last_delta_M, last_delta_D,
                                                                            I2I_array, I2M_array, M2I_array, M2M_array, D2I_array, D2M_array, I2D_array, D2D_array,
                                                                            Ie, Me,M2D_array)
                delta_M_mem[arrayRangeStart:arrayRangeEnd-1] = delta_M
                delta_D_mem[arrayRangeStart:arrayRangeEnd-1] = delta_D
                delta_I_mem[arrayRangeStart:arrayRangeEnd] = delta_I
                st=3*arrayRangeStart
                laststate_mem[st:st+arrayLength]=maxProbOrigin_M
                st+=arrayLength
                laststate_mem[st:st+arrayLength]=maxProbOrigin_D
                st+=arrayLength
                laststate_mem[st:st+arrayLength+1]=maxProbOrigin_I
                st=3*arrayRangeStart
                orinode_mem[st:st+arrayLength]=lm
                st+=arrayLength
                orinode_mem[st:st+arrayLength]=ld
                st+=arrayLength
                orinode_mem[st:st+arrayLength+1]=li
                doneList[node]=0
        def calculate_delta(goon_flag,lock,nodelist,delta_head_dict):
            linearPath_list = self.linearPath_list
            write_delta_head(nodelist,delta_head_dict)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)            
                except:
                    continue
                while start_node!=None:
                    linearPath = linearPath_list[start_node]
                    write_delta(linearPath)
                    checklist=self.DAG.CG_DAG.findChildNodes(start_node)
                    todolist=[]
                    for i in checklist:
                        with lock:
                            indegree_dict[i]-=1
                            if indegree_dict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node=None
        def calculate_state(goon_flag,lock,nodelist):
            linearPath_list = self.linearPath_list
            write_hiddensates_head(nodelist)
            while goon_flag.value:
                try:
                    start_node = q.get(timeout=1)
                except:
                    continue
                while start_node!=None:
                    linearPath = linearPath_list[start_node]
                    write_hiddensates(linearPath)
                    checklist=self.DAG.CG_DAG.findParentNodes(start_node)
                    todolist=[]
                    for i in checklist:
                        with lock:
                            outdegreeDict[i]-=1
                            if outdegreeDict[i]==0:
                                todolist.append(i)
                    if todolist:
                        start_node = todolist.pop()
                        for i in todolist:
                            q.put(i)
                    else:
                        start_node=None
        def is_end_f(goon_flag):
            start = time.time()
            while True:
                time.sleep(1)
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes-np.count_nonzero(doneList))/self.DAG.totalNodes,5)
                bar = ('#' * int(percent * 20)).ljust(20)
                hours, remainder = divmod(runtime, 3600)
                mins, secs = divmod(remainder, 60)
                time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ( {time_format} )')
                sys.stdout.flush()
                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    break
        def is_end_b(goon_flag):
            start = time.time()
            while True:
                time.sleep(1)
                runtime = time.time() - start
                percent = np.round((self.DAG.totalNodes-np.count_nonzero(doneList))/self.DAG.totalNodes,5)
                bar = ('#' * int(percent * 20)).ljust(20)
                hours, remainder = divmod(runtime, 3600)
                mins, secs = divmod(remainder, 60)
                time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))
                sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ( {time_format} )')
                sys.stdout.flush()
                if 0==np.count_nonzero(doneList>0):
                    goon_flag.value=0
                    break
        def query_contend_sequence_id(node):
            return seqiddb.findSequenceSource(self.DAG.SourceList[node],self.DAG.firstBitofONM,self.DAG.allBitofONM,self.DAG.firstBitofOSM,self.DAG.allBitofOSM)
        
        tmp_delta_M_mem = np.full(self.range_length, -np.inf)
        delta_M_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
        delta_M_mem = np.ndarray(self.range_length, dtype=np.float64, buffer=delta_M_mem_shm.buf)
        delta_M_mem[:] = tmp_delta_M_mem             
        tmp_delta_I_mem = np.full(self.range_length, -np.inf)
        delta_I_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
        delta_I_mem = np.ndarray(self.range_length, dtype=np.float64, buffer=delta_I_mem_shm.buf)
        delta_I_mem[:] = tmp_delta_I_mem
        tmp_delta_D_mem = np.full(self.range_length, -np.inf)
        delta_D_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*np.dtype(np.float64).itemsize)
        delta_D_mem = np.ndarray(self.range_length, dtype=np.float64, buffer=delta_D_mem_shm.buf)
        delta_D_mem[:] = tmp_delta_D_mem
        laststate_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*3*np.dtype(np.uint8).itemsize)
        laststate_mem = np.ndarray(self.range_length*3, dtype=np.uint8, buffer=laststate_mem_shm.buf)
        laststate_mem[:] = 2                 
        orinode_mem_shm = shared_memory.SharedMemory(create=True, size=self.range_length*3*np.dtype('int').itemsize)
        orinode_mem = np.ndarray(self.range_length*3, dtype=int, buffer=orinode_mem_shm.buf)
        Adict = {  
            (0,0):self.M2M_array,                  
            (0,2):self.M2I_array,                   
            (0,1):self.M2D_array,                   
            (2,0):self.I2M_array,                   
            (1,2):self.D2I_array,                    
            (2,1):self.I2D_array,                    
            (2,2):self.I2I_array,                    
            (1,0):self.D2M_array,                   
            (1,1):self.D2D_array                     
        }
        arrayRangeStart = self.arrayRangeDict[-1][0]                
        delta_D_mem[arrayRangeStart] = self.pi_D               
        for i in range(1, self.maxrange):
            delta_D_mem[arrayRangeStart+i] = delta_D_mem[arrayRangeStart+i-1] + self.D2D_array[i-1]
        First_deata_D = delta_D_mem[arrayRangeStart:]              
        goon_flag = Value('i',1)                    
        indegree_dict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.CG_DAG.totalNodes*np.dtype(np.uint16).itemsize)
        indegree_dict = np.ndarray((self.DAG.CG_DAG.totalNodes,), dtype=np.int16, buffer=indegree_dict_shm.buf)
        indegree_dict[:] = np.zeros(self.DAG.CG_DAG.totalNodes)         
        for link in self.linearPath_link:
            indegree_dict[link[1]] += 1              
        doneList_shm = shared_memory.SharedMemory(create=True, size=self.DAG.totalNodes*np.dtype(np.uint8).itemsize)
        doneList = np.ndarray((self.DAG.totalNodes,), dtype=np.uint8, buffer=doneList_shm.buf)
        doneList[:] = np.full(self.DAG.totalNodes, 1)                
        lock = Lock()         
        v_dict = {}
        all_v = sorted(self.all_source, key=lambda x: (int(x.split("_")[0]), int(x.split("_")[-1])))
        namelist = []            
        namedict = {}              
        v_num = len(all_v)

        for i in range(v_num):            
            gid = int(all_v[i].split('_')[0])
            seqid = int(all_v[i].split('_')[1])
            v_dict[save_numbers(gid, seqid, 16, 32)] = i  
            namedict[i] = self.id2v_dict[all_v[i]]        
            namelist.append(self.id2v_dict[all_v[i]])          
        ali = Manager().dict()                    
        for v in range(self.Match_num+1):
            ali[v] = 0                  
        q = Queue()            
        delta_head_dict = init_delta_head()                
        startnodelist = list(self.DAG.CG_DAG.startNodeSet)              
        endnodes = []
        for node in self.graphEndNodes:
            endnodes.append(node)              
        
        processlist = []
        pool_num = threads           
        for idx in range(pool_num):
            processlist.append(Process(
                target=calculate_delta,
                args=(goon_flag, lock, startnodelist[idx::pool_num], delta_head_dict)
            ))
        processlist.append(Process(target=is_end_f, args=(goon_flag,)))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        indegree_dict_shm.close()
        indegree_dict_shm.unlink()
        if not os.path.exists(self.Viterbi_result_path/'{}'.format(self.parameterName)):
            os.mkdir(self.Viterbi_result_path/'{}'.format(self.parameterName))          
        hidden_states = Manager().dict() 
        outdegreeDict_shm = shared_memory.SharedMemory(create=True, size=self.DAG.CG_DAG.totalNodes*np.dtype(np.uint16).itemsize)
        outdegreeDict = np.ndarray((self.DAG.CG_DAG.totalNodes,), dtype=np.int16, buffer=outdegreeDict_shm.buf)
        outdegreeDict[:] = np.zeros(self.DAG.CG_DAG.totalNodes)
        doneList[:] = np.full(self.DAG.totalNodes, 1)          
        for link in self.linearPath_link:
            outdegreeDict[link[0]] += 1
        q = Queue()               
        endnldelist = list(self.DAG.CG_DAG.endNodeSet)             
        goon_flag = Value('i', 1)            
        processlist = []
        for idx in range(pool_num):
            processlist.append(Process(
                target=calculate_state,
                args=(goon_flag, lock, endnldelist[idx::pool_num])
            ))
        processlist.append(Process(target=is_end_b, args=(goon_flag,)))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        np.save(self.Viterbi_result_path/'{}/hiddenstate.npy'.format(self.parameterName), dict(hidden_states))          
        np.save(self.Viterbi_result_path/'{}/insert_length_dict.npy'.format(self.parameterName), dict(ali))            
        np.save(self.Viterbi_result_path/'{}/namelist.npy'.format(self.parameterName), namelist)           
        delta_M_mem_shm.close()
        delta_M_mem_shm.unlink()
        delta_I_mem_shm.close()
        delta_I_mem_shm.unlink()
        delta_D_mem_shm.close()
        delta_D_mem_shm.unlink()
        laststate_mem_shm.close()
        laststate_mem_shm.unlink()
        orinode_mem_shm.close()
        orinode_mem_shm.unlink()
        outdegreeDict_shm.close()
        outdegreeDict_shm.unlink()
        doneList_shm.close()
        doneList_shm.unlink()


    def state_to_aligment(self, seqiddb, mode='local', save_fasta=False, matrix=False):

        def query_contend_sequence_id(node):
            return seqiddb.findSequenceSource(
                self.DAG.SourceList[node],
                self.DAG.firstBitofONM,
                self.DAG.allBitofONM,
                self.DAG.firstBitofOSM,
                self.DAG.allBitofOSM
            )
        def draw_ali(nodelist):
            vectorized_v_dict = np.vectorize(v_dict.get)
            loacal_xdict = xdict.copy()
            loacal_ali = ali.copy()
            spnode_statedict = {}
            for node in nodelist:
                aa = self.DAG.fragments[node]
                node_hidden_states = hidden_states[node]
                for state in node_hidden_states.keys():
                    if state[0] == 2:
                        if (2, state[1]) in loacal_xdict.keys():
                            sx = loacal_xdict[(2, state[1])] + (loacal_ali[state[1]] - node_hidden_states[state][1])
                        else:
                            spnode_statedict[node] = spnode_statedict.get(node, {})
                            spnode_statedict[node][state] = hidden_states[node][state]
                            continue
                    else:
                        sx = loacal_xdict[(0, state[1])]
                    if isinstance(node_hidden_states[state][0], str):
                        contentsource = query_contend_sequence_id(node)
                    else:
                        contentsource = node_hidden_states[state][0]
                    
                    if contentsource.size != 0:

                        content = vectorized_v_dict(get_first_number(contentsource))
                        lock.acquire()
                        ali_matrix[sx, content] = self.allBaseDict.get(aa, 14) + 1
                        lock.release()
        ali = np.load(self.Viterbi_result_path/'{}/insert_length_dict.npy'.format(self.parameterName), allow_pickle=True).item()
        namelist = np.load(self.Viterbi_result_path/'{}/namelist.npy'.format(self.parameterName)).tolist()
        v_dict = {}
        all_v = sorted(self.all_source, key=lambda x: (int(x.split("_")[0]), int(x.split("_")[-1])))
        v_num = len(all_v)
        for i in range(v_num):
            gid = int(all_v[i].split('_')[0])
            seqid = int(all_v[i].split('_')[1])
            v_dict[save_numbers(gid, seqid, 16, 32)] = i            
        lock = Lock()         
        sdict = {0:'A', 1:'T', 2:'C', 3:'G'}           
        seqdict = {'x': [], 'ref': []}            
        xdict = {}          
        alilength = 0            
        ss = 0
        for v in range(self.Match_num + 1):
            ss += ali[v]          
            xdict[(2, v)] = alilength          
            for i in range(alilength, alilength + ali[v]):
                seqdict['x'].append([2, v])             
                seqdict['ref'].append(' ')                
            alilength += ali[v]
            if v != self.Match_num:
                xdict[(0, v)] = alilength          
                seqdict['x'].append([0, v])
                seqdict['ref'].append(sdict[np.argmax(self.Me_Matrix[v])])
                alilength += 1
        ali_matrix_shm = shared_memory.SharedMemory(
            create=True, 
            size=alilength * self.vnum * np.dtype(np.uint8).itemsize
        )
        ali_matrix = np.ndarray(
            (alilength, self.vnum), 
            dtype=np.uint8, 
            buffer=ali_matrix_shm.buf
        )
        pool_num = 20
        hidden_states = np.load(
            self.Viterbi_result_path/'{}/hiddenstate.npy'.format(self.parameterName),
            allow_pickle=True
        ).item()
        processlist = []
        klist = list(hidden_states.keys())
        for idx in range(pool_num):
            processlist.append(Process(
                target=draw_ali,
                args=(klist[idx::pool_num],)
            ))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        zip_ali = []
        for i in range(alilength):
            ali_array = ali_matrix[i]
            unique_elements, counts = np.unique(ali_array, return_counts=True)
            ref_index = 0 if seqdict['ref'][i] == ' ' else self.allBaseDict[seqdict['ref'][i]] + 1
            elements_indices = [
                [element, np.where(ali_array == element)[0]] 
                for element in unique_elements 
                if element != ref_index
            ]
            elements_indices.insert(0, ref_index)
            zip_ali.append(elements_indices)
        zip_ali.append(self.vnum)            
        zip_ali = np.array(zip_ali,dtype=object)
        np.savez(
            self.Viterbi_result_path/'{}/zipalign.npz'.format(self.parameterName),
            namelist=namelist,
            align=zip_ali
        )
        ali_matrix = ali_matrix.T
        if matrix:
            np.save(
                self.Viterbi_result_path/'{}/ali_matrix.npy'.format(self.parameterName),
                ali_matrix
            )
        np.save(self.Viterbi_result_path/'{}/indexdict.npy'.format(self.parameterName), xdict)
        np.save(self.Viterbi_result_path/'{}/seqdict.npy'.format(self.parameterName), seqdict)
        if save_fasta:
            ssdict = {0: 'm', 2: 'i'}          
            MorI = ''             
            ref_seq = ''       
            seqlist = []         
            for s in range(alilength):
                MorI += ssdict[seqdict['x'][s][0]]
                if ssdict[seqdict['x'][s][0]] == 'm':
                    ref_seq += seqdict['ref'][s]
                else:
                    ref_seq += '-'
            vectorized_draw_dict = np.vectorize(self.alignmentBaseDictionary.get)
            string_matrix = vectorized_draw_dict(ali_matrix)
            seqlist = [
                SeqRecord(Seq(''.join(i)), id=namelist[idx], description='') 
                for idx, i in enumerate(string_matrix)
            ]
            SeqIO.write(
                seqlist,
                self.Viterbi_result_path/'{}'.format(self.parameterName)/'aliresult.fasta',
                'fasta'
            )

        ali_matrix_shm.close()
        ali_matrix_shm.unlink()


