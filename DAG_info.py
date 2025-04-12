#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
from DAG_stru import DAGStru
import numpy as np
from multiprocessing import Lock,Manager,Process
from DAG_tools import *

class DAGInfo:
    def __init__(self, inpath: str, id: str, savePath: str,
             allBitofOSM: int = 64, firstBitofOSM: int = 32,
             allBitofONM: int = 64, firstBitofONM: int = 32,
             fragmentLength = None, main_fragment:set = set(),graph_type='fragment') -> None:
        if graph_type=='fragment' and fragmentLength is None:
            raise ValueError("When the graph type is 'fragment', the fragmentLength must not be set to its default value.")
        safe_inpath = sanitize_path(inpath, 'input')
        safe_savepath = sanitize_path(savePath, 'output')
        self.savePath = safe_savepath            
        self.fastaFilePathList = [safe_inpath]          
        self.graphID = id
        self.startFragmentNodeDict = {}
        self.endFragmentNodeDict = {}
        self.fragmentNodeDict = {}
        self.edgeWeightDict = dict()
        self.startNodeSet = set()
        self.endNodeSet = set()
        self.sequenceIDList = []
        self.currentEdgeSet = set()
        self.sequenceNum = 0
        self.totalNodes = 0
        self.main_fragment = main_fragment
        self.fragmentLength = fragmentLength
        self.originalfragmentLength = self.fragmentLength
        self.graph_type = graph_type
        self.firstBitofOSM = firstBitofOSM             
        self.allBitofOSM = allBitofOSM                
        self.firstBitofONM = firstBitofONM             
        self.allBitofONM = allBitofONM
        self.nodenumthr = 200000
        self.fragments = np.full(self.nodenumthr,'',f'U{fragmentLength}')
        self.weights = np.full(self.nodenumthr,0)
        self.headtails = np.full(self.nodenumthr,0,dtype=np.uint8)

    def addnewNodes(self,fragment,headandtail = 0):
        if self.totalNodes>=self.nodenumthr:
            self.nodenumthr+=50000
            self.fragments.resize(self.nodenumthr)
            self.weights.resize(self.nodenumthr)
            self.headtails.resize(self.nodenumthr)
        self.fragments[self.totalNodes] = fragment
        self.weights[self.totalNodes] = 1
        self.headtails[self.totalNodes] = headandtail
        self.totalNodes+=1

    def add_first_sequence(self, firstSequenceId: int, sequence: str) -> None:
        if self.graph_type != 'fragment':
            raise TypeError("The method can only be used when the graph type is 'fragment'.")
        if self.fragmentLength>len(sequence):
            raise ValueError("The length of the input sequence is less than fragmentLength.")
        w_num = len(sequence) - self.fragmentLength + 1
        row = 0          
        sequenceFragment = sequence[row:row+self.fragmentLength]           
        self.addnewNodes(sequenceFragment,headandtail=1)            
        self.startFragmentNodeDict.setdefault(sequenceFragment, []).append(self.totalNodes-1)
        for row in range(1, w_num-1):              
            sequenceFragment = sequence[row:row+self.fragmentLength]
            self.addnewNodes(sequenceFragment,headandtail=0)
            self.fragmentNodeDict.setdefault(sequenceFragment, []).append(self.totalNodes-1)
            self.edgeWeightDict[(row-1, row)] = 1
        row = w_num-1              
        sequenceFragment = sequence[row:row+self.fragmentLength]

        self.addnewNodes(sequenceFragment,headandtail=2) 
        self.endFragmentNodeDict.setdefault(sequenceFragment, []).append(self.totalNodes-1)

        self.edgeWeightDict[(row-1, row)] = 1              
        self.startNodeSet.add(0)
        self.endNodeSet.add(row)                     
        self.queryGraph = DAGStru(self.totalNodes, self.edgeWeightDict.keys())
        self.queryGraph.coordinateList = list(np.arange(row+1)+1)                       
        self.queryGraph.maxlength = row+1                
        self.region_list = [0]*(row+1)                 
        self.region_num = 1                       
        self.queryGraph.longestPathNodeSet = set(np.arange(row))
        self.sequenceIDList.append(firstSequenceId)          
                                   
        # self.sequencePathNodeMap = []
        self.seqnumthr = 6000   
        self.sequencePathNodeMap = np.array([], dtype=object)
        self.sequencePathNodeMap.resize(6000,refcheck=True)        
        self.sequencePathNodeMap[self.sequenceNum] = np.array(list(range(self.totalNodes)), dtype=np.uint32)
        self.sequenceNum += 1
    def lookup_fragment_main(self, fragment, empty_value=None):

        if empty_value is None:
            empty_value = []
        if fragment in self.main_fragment:                           
            return self.fragmentNodeDict.get(fragment, empty_value)
        return empty_value.copy()          
    def check_anchor(self,anchors: list[int]) -> int:

        if len(anchors) == 1:
            return anchors[0]              
        else:
            return -1       
    def lookup_fragment(self, fragment, empty_value=None):
        if empty_value is None:
            empty_value = []
        return self.fragmentNodeDict.get(fragment, empty_value)
            
    def check_anchor_maxweight(self, anchors: list[int], fast=False) -> int:

        max_weight = -np.inf                 
        max_index = -1                         
        count = 0                            
        for i, anchor in enumerate(anchors):
            w = self.weights[anchor] 
            if w > max_weight:
                max_weight = w            
                max_index = i                      
                count = 1                  
            elif w == max_weight and fast:
                count += 1                
        return anchors[max_index] if count == 1 else -1
    
    def find_max_id(self,ids):
        if not ids:
            return -1
        # ids = np.array(ids)
        subweights = self.weights[np.array(ids)]
        maxweightindex = np.argmax(subweights)
        if len(np.where(subweights==subweights[maxweightindex])[0])==1:
            anchor = ids[maxweightindex]
            return anchor
        else:
            return -1

    
    def anchor_coor(self, anchor: int) -> int:
        if anchor == -1:
            return -1
        else:
            return self.queryGraph.coordinateList[anchor]


    def sequence_to_path_fragment(self, sequence: str, fast: bool = False):

        seq_length = len(sequence)
        fragmentLength = self.fragmentLength
        w_num = seq_length - fragmentLength + 1          
        fragments = [sequence[i:i + fragmentLength] for i in range(w_num)]
        anchors_list = list(map(self.lookup_fragment, fragments))               
        anchors_list[0] = self.startFragmentNodeDict.get(fragments[0], [])            
        anchors_list[-1] = self.endFragmentNodeDict.get(fragments[-1], [])            
        sequencePathNodeArray = np.fromiter(
            map(lambda x: self.check_anchor_maxweight(x, fast), anchors_list),
            dtype=int
        )

        anchorCoordinateArray = np.fromiter(
            map(self.anchor_coor, sequencePathNodeArray),
            dtype=int
        )
        return sequencePathNodeArray, anchorCoordinateArray, anchors_list
    
    def map_to_OSM(self):

        SourceList = np.array([], dtype=object)
        SourceList.resize(self.totalNodes, refcheck=False)
        for nodeid in range(self.totalNodes):
            SourceList[nodeid] = np.zeros(self.weights[nodeid], dtype=int)
        index_array = np.zeros(self.totalNodes, dtype=int)
        seq_idx = 0               
        fragmentLength = self.fragmentLength
        startPointList = [fragmentLength - 1] * len(self.sequencePathNodeMap)
        for seq_nodes in self.sequencePathNodeMap:
            pathlength = len(seq_nodes)          
            seq_index = seq_idx            
            seqs_num_array = np.full(pathlength, seq_index)                 
            startPoint = startPointList[seq_idx]
            site_array = np.arange(startPoint, startPoint + pathlength)
            seqid_and_site_array = save_numbers(
                seqs_num_array, 
                site_array, 
                self.firstBitofOSM,
                self.allBitofOSM
            )
            for idx, node in enumerate(seq_nodes):
                SourceList[node][index_array[node]] = seqid_and_site_array[idx]
            index_array[seq_nodes] += 1
            seq_idx += 1              
        self.SourceList = SourceList

    def join_into_fragmentDAG(self, sequence, seq_id, sequencePathNodeArray, anchorCoordinateArray, anchor_list):
        sequencePathLength = len(sequencePathNodeArray)
        sequencePathCoordinates = np.full(sequencePathLength, 0)
        coordinateList = self.queryGraph.coordinateList
        fragmentLength = self.fragmentLength


        # unanchor_block = find_consecutive_negatives(anchorCoordinateArray)
        # for block in unanchor_block:
        #     subanchor_list = anchor_list[block[0]:block[1]+1]
        #     candidateNodesList = [[(a, self.queryGraph.coordinateList[a], self.weights[a]) 
        #                 for a in alist if block[2] < self.queryGraph.coordinateList[a] < block[3]] 
        #                 for alist in subanchor_list]
        #     _, update_sequencePathNodeList, updateanchorCoordinateList = find_max_weight_combination(candidateNodesList)
        #     sequencePathNodeArray[block[0]:block[1]+1] = np.array(update_sequencePathNodeList)
        #     anchorCoordinateArray[block[0]:block[1]+1] = np.array(updateanchorCoordinateList)

        sequencePathCoordinates = np.full(sequencePathLength, 0)
        cursor_coor = 0          
        coordinateUpdateList = []               
        diff = 0
        for index, node in enumerate(sequencePathNodeArray):
            if node != -1:
                node_coor = anchorCoordinateArray[index]
                diff = cursor_coor + 1 - node_coor
                if diff > 0:
                    coordinateUpdateList.append(node)
                cursor_coor = node_coor
            else:
                cursor_coor += 1
            sequencePathCoordinates[index] = cursor_coor

        if coordinateUpdateList==[]:
            for i in np.where(sequencePathNodeArray==-1)[0]:
                nid = [nodeid for nodeid in anchor_list[i] if coordinateList[nodeid] == sequencePathCoordinates[i]]
                if nid:
                    sequencePathNodeArray[i] = nid[0]
        index = 0
        node = sequencePathNodeArray[index]
        if node == -1:
            fragment = sequence[index:index+fragmentLength]
            sequencePathNodeArray[index] = self.totalNodes
            coordinateList.append(sequencePathCoordinates[index])
            self.startFragmentNodeDict.setdefault(fragment, []).append(self.totalNodes)
            self.startNodeSet.add(self.totalNodes)
            self.addnewNodes(fragment,1)
        else:
            self.weights[node] += 1
        index += 1
        for node in sequencePathNodeArray[1:-1]:
            if node == -1:
                fragment = sequence[index:index+fragmentLength]
                sequencePathNodeArray[index] = self.totalNodes
                coordinateList.append(sequencePathCoordinates[index])
                self.fragmentNodeDict.setdefault(fragment, []).append(self.totalNodes)
                self.addnewNodes(fragment,0)
            else:
                self.weights[node] += 1
            newlink = (sequencePathNodeArray[index-1], sequencePathNodeArray[index])
            linkWeight = self.edgeWeightDict.get(newlink, 0) + 1
            self.edgeWeightDict[newlink] = linkWeight
            if linkWeight == 1:
                self.queryGraph.newEdges.add(newlink)
            index += 1
        node = sequencePathNodeArray[index]
        if node == -1:
            fragment = sequence[index:index+fragmentLength]
            sequencePathNodeArray[index] = self.totalNodes
            coordinateList.append(sequencePathCoordinates[index])
            self.endFragmentNodeDict.setdefault(fragment, []).append(self.totalNodes)
            self.endNodeSet.add(self.totalNodes)
            self.addnewNodes(fragment,2)

        else:
            self.weights[node] += 1
        newlink = (sequencePathNodeArray[index-1], sequencePathNodeArray[index])
        linkWeight = self.edgeWeightDict.get(newlink, 0) + 1
        self.edgeWeightDict[newlink] = linkWeight
        if linkWeight == 1:
            self.queryGraph.newEdges.add(newlink)

        if coordinateUpdateList != []:
            self.queryGraph.update(self.totalNodes)
            self.updateCoordiante(update_stem=False, error_nodes=coordinateUpdateList)
        self.sequenceIDList.append(seq_id)

        if self.sequenceNum>=self.seqnumthr:
            self.seqnumthr+=10000
            self.sequencePathNodeMap.resize(self.seqnumthr, refcheck=True)
        self.sequencePathNodeMap[self.sequenceNum] = np.array(sequencePathNodeArray, dtype=np.uint64)
        self.sequenceNum += 1
    def add_fragment_seq(self, seq, seq_id,allow_gap=True):

        sequencePathNodeArray, anchorCoordinateArray, anchors_list = self.sequence_to_path_fragment(seq)
        sequencePathNodeArray, anchorCoordinateArray = self.Repair_topology(
            sequencePathNodeArray, 
            anchorCoordinateArray,
            allow_gap=allow_gap                 
        )
        self.join_into_fragmentDAG(seq, seq_id, sequencePathNodeArray, anchorCoordinateArray, anchors_list)
    def merge_Source(self, nodeA, nodeB):
        oldsize = self.SourceList[nodeA].size
        self.SourceList[nodeA].resize(oldsize + self.SourceList[nodeB].size)
        self.SourceList[nodeA][oldsize:] = self.SourceList[nodeB]
        self.SourceList[nodeB] = 0
    def merge_Source_batch(self,merge_list):

        for tup in merge_list:
            nodeA = tup[0]                               
            fra_list = []                  
            size_list = []                
            for nodeB in tup[1:]:
                size_list.append(self.SourceList[nodeB].size)          
                fra_list.append(self.SourceList[nodeB])                  
                self.SourceList[nodeB] = 0

            oldsize = self.SourceList[nodeA].size              
            self.SourceList[nodeA].resize(oldsize + np.sum(size_list))
            cursor = oldsize                           
            for idx, fras in enumerate(fra_list):
                self.SourceList[nodeA][cursor:cursor + size_list[idx]] = fras
                cursor += size_list[idx]

    def merge_listSource(self, nodeA, nodeB):
        self.SourceList[nodeA]+=self.SourceList[nodeB]

    def merge_listSource_batch(self,merge_list):
        for tup in merge_list:
            nodeA = tup[0]                               
            for nodeB in tup[1:]:
                self.SourceList[nodeA]+=self.SourceList[nodeB]
                self.SourceList[nodeB]=[]



    def merge_node(self,nodeA,nodeB):
        self.weights[nodeA]+=self.weights[nodeB]
        self.headtails[nodeA]|=self.headtails[nodeB]
        self.weights[nodeB]=0
    def merge_node_info_batch(self, merge_list):
        delList = []                
        for tup in merge_list:
            nodeA = tup[0]                   
            delList.extend(tup[1:])                  
            for node in tup[1:]:
                self.merge_node(nodeA,node)
        return delList                  
    def merge_sameCoorFraNodes(self, ArraySource=False, forward=True):
        def check_merge_list(subFragmentList, merge_list, lock, forward=True):

            coordinateList = self.queryGraph.coordinateList if forward else self.queryGraph.backwardCoordinateList
            
            tmp_merge_list = []
            nodeList = [i for i in subFragmentList if len(i) > 1]
            for i in nodeList:
                tlist = {}
                for j in [j for j in i if  self.headtails[j]==0]:
                    tlist.setdefault(coordinateList[j], []).append(j)
                for tup in tlist.values():
                    if len(tup) > 1:
                        tmp_merge_list.append(tup)
            with lock:
                merge_list.extend(tmp_merge_list)
        merge_list = Manager().list()
        lock = Lock()
        processlist = []
        fragmentList = list(self.fragmentNodeDict.values())+list(self.startFragmentNodeDict.values())+list(self.endFragmentNodeDict.values())
        pool_num = 10
        for idx in range(pool_num):
            processlist.append(Process(target=check_merge_list, 
                                    args=(fragmentList[idx::pool_num], merge_list, lock, forward)))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        replaceDict = np.arange(self.totalNodes)
        indices = [tup[1:] for tup in merge_list]              
        values = [tup[0] for tup in merge_list]                
        for idx, v in zip(indices, values):
            replaceDict[idx] = v          
        delList = self.merge_node_info_batch(merge_list)
        if len(merge_list) != 0:
            newlinkset = {}
            for i in self.edgeWeightDict.keys():
                k = (int(replaceDict[i[0]]), int(replaceDict[i[1]]))
                newlinkset[k] = newlinkset.get(k, 0) + self.edgeWeightDict[i]
            self.edgeWeightDict = newlinkset

        if ArraySource:
            self.merge_Source_batch(merge_list)
        else:
            self.merge_listSource_batch(merge_list)
        self.fast_remove(delList)
        self.reorderNodelist(ArraySource)
        return len(delList)
    
    def removeDegenerateBasePaths(self):

        if self.graph_type != 'fragment':
            raise TypeError("The method can only be used when the graph type is 'fragment'.")
        bannedSet = set()                    
        startNode = set()                      
        endNode = set()                        
        for nodeid in range(self.totalNodes):
            fragment = self.fragments[nodeid]
            ht = self.headtails[nodeid]
            if fragment[-1] not in {'A', 'T', 'C', 'G'}:
                bannedSet.add(nodeid)
                startNode.discard(nodeid)
                endNode.discard(nodeid)
            else:
                if ht == 1:
                    startNode.add(nodeid)
                elif ht == 2:
                    endNode.add(nodeid)
                elif ht ==3:
                    startNode.add(nodeid)
                    endNode.add(nodeid)
        setA, _ = self.queryGraph.BFS_DAG(list(startNode), bannedSet=bannedSet)
        setB, _ = self.queryGraph.BFS_DAG(list(endNode), forward=False, bannedSet=bannedSet)
        allset = setA & setB
        delset = set(range(self.totalNodes)) - allset
        if not allset:
            raise RuntimeError("Failed to find a complete non-degenerate nucleotide path.")
        self.edgeWeightDict = {link: value for link, value in self.edgeWeightDict.items() 
                            if not set(link) & delset}
        self.fast_remove(delset)
        id_mapping = self.reorderNodelist()
        return id_mapping
    def noDegenerateGraph(self):
        if self.graph_type != 'fragment':
            raise TypeError("The method can only be used when the graph type is 'fragment'.")
        if self.fragmentLength != 1:
            raise TypeError("The method can only be used when the fragmentLength is 1.")
        bannedSet = set()                
        startNode = set()            
        endNode = set()             

        for nodeid  in range(self.totalNodes):
            fragment = self.fragments[nodeid]
            ht = self.headtails[nodeid]
            if fragment[-1] not in {'A', 'T', 'C', 'G'}:
                bannedSet.add(nodeid)
                startNode.discard(nodeid)
                endNode.discard(nodeid)
            else:
                if ht == 1:
                    startNode.add(nodeid)
                elif ht == 2:
                    endNode.add(nodeid)
                elif ht ==3:
                    startNode.add(nodeid)
                    endNode.add(nodeid)
        setA, _ = self.queryGraph.BFS_DAG(list(startNode), bannedSet=bannedSet)
        setB, _ = self.queryGraph.BFS_DAG(list(endNode), forward=False, bannedSet=bannedSet)
        allset = setA & setB             
        delset = set(range(self.totalNodes)) - allset             
        if not allset:
            raise RuntimeError("Failed to find a complete non-degenerate nucleotide path.")
        no_degenerate_edgeWeightDict = {link: value for link, value in self.edgeWeightDict.items() 
                            if not set(link) & delset}
        return no_degenerate_edgeWeightDict,allset
    def search_same_fragment_nodes(self, newnode_id: int, minthr: float = 0.01, maxdistant: int = 2000) -> set:
        graphcoordinateList = np.array(self.queryGraph.coordinateList)
        tupset = set()                
        useableSameFraNodesList = [v for v in self.fragmentNodeDict.values() if len(v) > 1]
        weightArray = self.weights
        if newnode_id == 0:              
            for i in useableSameFraNodesList:              
                i_num = len(i)
                w_sum = sum([weightArray[x] for x in i])             
                if w_sum > minthr * self.sequenceNum:
                    for j in range(i_num):
                        for k in range(j+1, i_num):
                            if (abs(graphcoordinateList[i[j]] - graphcoordinateList[i[k]]) < maxdistant ):                                    
                                if weightArray[i[j]] + weightArray[i[k]] > 0.5 * w_sum:
                                    tupset.add((i[j], i[k]))
        else:               
            for i in useableSameFraNodesList:
                i_num = len(i)
                w_sum = sum([weightArray[x] for x in i])
                if w_sum > minthr * self.sequenceNum:
                    setA = [x for x in i if x <= newnode_id]
                    setB = [x for x in i if x > newnode_id]
                    for j in setA:
                        for k in setB:
                            if (abs(graphcoordinateList[j] - graphcoordinateList[k]) < maxdistant ):                                    
                                if weightArray[j] + weightArray[k] > 0.5 * w_sum:
                                    tupset.add((j, k))
        return tupset
    def merge_sameFraNodes(self,newnode_id=0,ArraySource=False,maxdistant=2000):

        def merge_in_subgraphs(indexList, lock):

            tmp_merge_list = []                 
            for si in indexList:
                subGraphStru, nodeIdDict = build_subgraphStru(subGraphLinkDictList[si])
                subGraphLinkDict = {k:1 for k in subGraphStru.currentEdgeSet}
                nodeIdDictReverse = {v: k for k, v in nodeIdDict.items()}                     
                subGraphStru.calculateCoordinates()
                chooseNodeSet = set()                            
                oriNodeTuples = tupleLists[si]           
                inforTuples = []
                for t in oriNodeTuples:
                    weight_sum = self.weights[t[0]] + self.weights[t[1]]
                    inforTuples.append((
                        t[0], t[1],
                        self.queryGraph.coordinateList[t[0]],
                        self.queryGraph.coordinateList[t[1]],
                        weight_sum
                    ))
                subGraphNodeTuples = []
                for i in inforTuples:
                    new_id1 = nodeIdDict[i[0]]
                    new_id2 = nodeIdDict[i[1]]
                    chooseNodeSet.add(new_id1)
                    chooseNodeSet.add(new_id2)
                    coord_diff = abs(i[3] - i[2])
                    subGraphNodeTuples.append([
                        (new_id1, new_id2), 
                        coord_diff, 
                        -i[4]                
                    ])
                subGraphNodeTuples = sorted(subGraphNodeTuples, key=lambda item: (item[1], item[2]))
                subGraphStru.coordinateList = np.array(subGraphStru.coordinateList)
                for tup in subGraphNodeTuples:
                    j, k = tup[0]             
                    if j in chooseNodeSet and k in chooseNodeSet:
                        if subGraphStru.coordinateList[j] == subGraphStru.coordinateList[k]:
                            chooseNodeSet.remove(k)
                            subGraphLinkDict = subGraphStru.merge_node_in_stru(j, k, subGraphLinkDict)
                            tmp_merge_list.append([nodeIdDictReverse[j], nodeIdDictReverse[k]])            
                        else:
                            if subGraphStru.nonCyclic([j, k]):
                                chooseNodeSet.remove(k)
                                subGraphLinkDict = subGraphStru.merge_node_in_stru(j, k, subGraphLinkDict)
                                tmp_merge_list.append([nodeIdDictReverse[j], nodeIdDictReverse[k]])
                                coord_diff = max(subGraphStru.coordinateList[j], subGraphStru.coordinateList[k]) - min(subGraphStru.coordinateList[j], subGraphStru.coordinateList[k])
                                if coord_diff == 1:          
                                    subGraphStru.coordinateList[j] = min(subGraphStru.coordinateList[j], subGraphStru.coordinateList[k])
                                    pushnodes, _ = subGraphStru.BFS_DAG([j], filter=subGraphStru.isConsecutive)
                                    subGraphStru.coordinateList[np.array(list(pushnodes))] += coord_diff
                                else:           
                                    subGraphStru.coordinateList[j] = max(subGraphStru.coordinateList[j], subGraphStru.coordinateList[k])
                                    stem_push = 0
                                    step = 0
                                    index = subGraphStru.coordinateList[j] + 1
                                    now = [node for node in subGraphStru.findChildNodes([j]) 
                                        if subGraphStru.coordinateList[node] < index]
                                    while len(now) != 0:
                                        if step > 1000:                       
                                            subGraphStru.coordinateList[now] = index
                                            stem_push = 1
                                            break
                                        subGraphStru.coordinateList[now] = index
                                        index += 1
                                        step += 1
                                        now = [node for node in subGraphStru.findChildNodes(now) 
                                            if subGraphStru.coordinateList[node] < index]
                                    if stem_push == 1:
                                        subGraphStru.local_update_coordinate(now)
                                        subGraphStru.coordinateList = np.array(subGraphStru.coordinateList)
            lock.acquire()
            try:
                merge_list.extend(tmp_merge_list)
            finally:
                lock.release()
        graphcoordinateList = self.queryGraph.coordinateList
        tupset = self.search_same_fragment_nodes(newnode_id=newnode_id,maxdistant=maxdistant)
        tupdict = []
        for i in tupset:
            tupdict.append([
                i[0], i[1],
                min(graphcoordinateList[i[0]], graphcoordinateList[i[1]]) - 1,
                max(graphcoordinateList[i[0]], graphcoordinateList[i[1]]) + 1
            ])
        subRanges = merge_intervals(tupdict)
        grap_index_array = np.full(self.queryGraph.maxlength, -1)
        tupleLists = []
        blockindex = 0
        for block in subRanges:
            rg = block[0]
            tups = block[1]
            for i in range(max(rg[0]-1, 0), min(self.queryGraph.maxlength, rg[1])):
                grap_index_array[i] = blockindex
            tupleLists.append(tups)
            blockindex += 1
        subGraphLinkDictList = []
        for i in subRanges:
            subGraphLinkDictList.append({})
        for linksAndWight in self.edgeWeightDict.items():
            links = linksAndWight[0]
            gindex = set([
                grap_index_array[self.queryGraph.coordinateList[links[0]] - 1],
                grap_index_array[self.queryGraph.coordinateList[links[1]] - 1]
            ])
            if gindex != {-1}:
                subGraphLinkDictList[gindex.pop()][links]=linksAndWight[1]
        merge_list = Manager().list()
        lock = Manager().Lock()
        processlist = []
        pool_num = min(len(subGraphLinkDictList), 40)
        gList = range(len(subGraphLinkDictList))
        for idx in range(pool_num):
            processlist.append(Process(
                target=merge_in_subgraphs, 
                args=(gList[idx::pool_num], lock,)
            ))
        [p.start() for p in processlist]
        [p.join() for p in processlist]
        self.queryGraph.calculateCoordinates()
        delList = []
        for t in merge_list:
            Anode = t[0]
            Bnode = t[1]
            self.edgeWeightDict = self.queryGraph.merge_node_in_stru(
                Anode, Bnode, self.edgeWeightDict
            )

            self.merge_node(Anode,Bnode)
                       
            delList.append(Bnode)

        if ArraySource:
            self.merge_Source_batch(merge_list)
        else:
            self.merge_listSource_batch(merge_list)
        self.fast_remove(delList)
        self.reorderNodelist(ArraySource)
        self.queryGraph.calculateCoordinates()
        return delList != []
    def merge_check(self, ArraySource: bool = False, maxdistant: int = 2000) -> None:
        merge_flag = True          
        self.merge_sameCoorFraNodes_loop(ArraySource)
        while merge_flag:
            merge_flag = self.merge_sameFraNodes(newnode_id=0,ArraySource=ArraySource, maxdistant=maxdistant)
            # merge_flag = False
            self.merge_sameCoorFraNodes_loop(ArraySource)
    def reorderNodelist(self, ArraySource=False):
        if not isinstance(ArraySource, bool):
            raise ValueError(f"无效模式参数：{ArraySource}，仅支持True/False")
                                         
        self.fragmentNodeDict = {}                                    
        self.startFragmentNodeDict = {}
        self.endFragmentNodeDict = {}
        self.startNodeSet=set()
        self.endNodeSet=set()
        valid_indices = np.nonzero(self.weights)[0].tolist() 
        
        self.fragments = self.fragments[valid_indices]
        self.weights = self.weights[valid_indices]
        self.headtails = self.headtails[valid_indices]
        self.totalNodes = len(valid_indices)
        for idx in range(self.totalNodes):
            ht = self.headtails[idx]
            if ht==0:
               self.fragmentNodeDict.setdefault(self.fragments[idx], []).append(idx)  
            elif ht==1:
                self.startFragmentNodeDict.setdefault(self.fragments[idx], []).append(idx) 
                self.startNodeSet.add(idx)
            elif ht ==2:
                self.endFragmentNodeDict.setdefault(self.fragments[idx], []).append(idx)
                self.endNodeSet.add(idx)
            else:
                self.startFragmentNodeDict.setdefault(self.fragments[idx], []).append(idx)
                self.startNodeSet.add(idx)
                self.endFragmentNodeDict.setdefault(self.fragments[idx], []).append(idx)
                self.endNodeSet.add(idx)

        if ArraySource:
            self.SourceList = self.SourceList[valid_indices]
        else:
            newSourceList = []
            for s in self.SourceList:
                if s:
                    newSourceList.append(s)
            self.SourceList = newSourceList
        id_mapping = {oriidx:idx for idx,oriidx in enumerate(valid_indices)} 

        self.edgeWeightDict = {
            (id_mapping[u], id_mapping[v]): w
            for (u, v), w in self.edgeWeightDict.items()
            if u in id_mapping and v in id_mapping             
        }

        self.nodenumthr = self.totalNodes

        self.queryGraph = DAGStru(self.totalNodes, self.edgeWeightDict.keys())
        self.queryGraph.calculateCoordinates()  
        return id_mapping
    
    def reorder_ref_graph_nodes(self, node_set, ref_edgeWeightDict):

        id_mapping = {}                       
        index = 0
        for node in node_set:
            id_mapping[node] = index                                      
            index += 1
        ref_edgeWeightDict = {
            (id_mapping[link[0]], id_mapping[link[1]]): value 
            for (link, value) in ref_edgeWeightDict.items()
            if link[0] in id_mapping and link[1] in id_mapping            
        }
        ref_totalNodes = len(node_set)          
        ref_queryGraph = DAGStru(ref_totalNodes, ref_edgeWeightDict.keys())           
        ref_queryGraph.startNodeSet = set()              
        ref_queryGraph.endNodeSet = set()                
        for node in range(ref_totalNodes):
            if not ref_queryGraph.findParentNodes(node):             
                ref_queryGraph.startNodeSet.add(node)
            elif not ref_queryGraph.findChildNodes(node):             
                ref_queryGraph.endNodeSet.add(node)
        ref_queryGraph.calculateCoordinates()                   
        subregions = ref_queryGraph.split_island()                  
        ref_queryGraph.findLongestPath()                     
        return {v: k for k, v in id_mapping.items()}, subregions,ref_queryGraph
    
    def save_graph(self, savePath: str = None, mode: str = 'build', maxdistant: int = 2000, ArraySource: bool = False) -> None:

        if not savePath is None:
            self.savePath = sanitize_path(savePath,'output')
            if not os.path.exists(self.savePath):
                os.mkdir(self.savePath)          
        else:
            savePath = self.savePath    
        self.merge_check(ArraySource, maxdistant=maxdistant)      

        if self.graph_type == 'fragment' and mode=='build':
            ori_node_list = np.full((self.totalNodes), 0)
            onmindex = np.full((self.totalNodes, 2), 0)

            for nodeid in range(self.totalNodes):
                ori_node_list[nodeid] = save_numbers(
                    int(self.graphID),
                    nodeid,
                    self.firstBitofONM,
                    self.allBitofONM
                )
                onmindex[nodeid] = np.array([nodeid, nodeid + 1])
            ori_node_list = np.array(ori_node_list, dtype=object)
            new_vid=[]
            for idx,v in enumerate(self.sequenceIDList):
                key = f"{self.graphID}_{idx}"
                new_vid.append([v,key])

            for nodeid in range(self.totalNodes):
                if len(self.SourceList[nodeid])!=self.weights[nodeid]:
                    raise ValueError(nodeid,self.weights[nodeid],len(self.SourceList[nodeid]))
            np.save(self.savePath/'osm.npy',self.SourceList)
            np.save(self.savePath/'onm.npy',ori_node_list)
            np.save(self.savePath/'onm_index.npy',onmindex)
            np.save(self.savePath/'v_id.npy',new_vid)

        elif self.graph_type == 'fragment' and mode!='build':
            ori_node_list = []                                                        
            for nodeid in range(self.totalNodes):
                ori_node_list.append(self.SourceList[nodeid])
            ori_node_list = np.array(ori_node_list,dtype=object)
            np.save(self.savePath/'Traceability_path.npy',ori_node_list)

        self.reflist = self.findMainPathNodes()           
        self.queryGraph.calculateStateRange(self.reflist, mode='build')          
        self.queryGraph.findLongestPath()          
        state_range_dict = {}
        for node in range(self.totalNodes):
            range_length = self.queryGraph.ref_coor[node][1] - self.queryGraph.ref_coor[node][0]
            state_range_dict.setdefault(range_length, []).append(node)
        print('\n//////////////Basic information of graphs//////////////////')
        print('The number of nodes in the graph : ', self.totalNodes)
        print('The number of links in the graph: ', len(self.edgeWeightDict))
        print('The longest path length in the graph : ', self.queryGraph.maxlength)
        print('The number of longestpath_nodes in the graph: ', len(self.queryGraph.longestPathNodeSet))
        print('ref_sequence_length: ', len(self.reflist))
        print('mean_weight of ref_seq', np.mean([self.weights[node] for node in self.reflist]))
        print('max_range of nodes in graph:', max(list(state_range_dict.keys())))
        print('graph file save in ', self.savePath)

        self.edgeWeightDict = np.array([
            [link[0], link[1], value] 
            for (link, value) in self.edgeWeightDict.items()
        ], dtype=object)
        self.startNodeSet = np.array(list(self.startNodeSet), dtype=np.uint32)
        self.endNodeSet = np.array(list(self.endNodeSet), dtype=np.uint32)
        npz_path = os.path.join(self.savePath,'data.npz')
        np.savez(
            npz_path,
            edgeWeightDict=self.edgeWeightDict,
            fragments=self.fragments,
            weights = self.weights,
            # headtails = self.headtails,
            startNodeSet=self.startNodeSet,
            endNodeSet=self.endNodeSet
        )
        all_attributes = self.__dict__.keys()
        whitelist = [
            "graphID", "originalfragmentLength", "fragmentLength",
            "savePath", "sequenceNum", "fastaFilePathList",
            "firstBitofOSM", "allBitofOSM", "firstBitofONM",
            "allBitofONM", "totalNodes", "sequenceIDList", "graph_type",
            "offsetArray"
        ]
        for attr in list(all_attributes):
            if attr not in whitelist:
                delattr(self, attr)
        with open(os.path.join(self.savePath,'graph.pkl'), 'wb') as graphfile:
            pickle.dump(self, graphfile)
    def Cyclic_Anchor_Combination_Exclusion(self, copyrglist, sequencePathNodeArray, anchorCoordinateArray):

        for i in copyrglist:
            start, end = i[0][1], i[1][1] + 1
            sequencePathNodeArray[start:end] = [
                -1 if anchorCoordinateArray[j] != -1 else sequencePathNodeArray[j] 
                for j in range(start, end)
            ]
            anchorCoordinateArray[start:end] = [
                -1 if anchorCoordinateArray[j] != -1 else anchorCoordinateArray[j] 
                for j in range(start, end)
            ]
        return sequencePathNodeArray, anchorCoordinateArray
    def Repair_topology(self, sequencePathNodeArray, anchorCoordinateArray, allow_gap=False):

        Coordinate_block_list, Coordinate_block_weight, Coordinate_block_dif = array_to_block(anchorCoordinateArray, allow_gap)
        block_weight = []
        for block in Coordinate_block_list:
            st = block[0][1]          
            ed = block[1][1]          
            w = sum([self.weights[i] for i in sequencePathNodeArray[st:ed+1] if i != -1])
            block_weight.append(w)
        copyrglist = Cyclic_Anchor_Combination_Detection(Coordinate_block_list, Coordinate_block_dif, block_weight)
        while copyrglist != []:
            sequencePathNodeArray, anchorCoordinateArray = self.Cyclic_Anchor_Combination_Exclusion(
                copyrglist, sequencePathNodeArray, anchorCoordinateArray)
            Coordinate_block_list, Coordinate_block_weight, Coordinate_block_dif = array_to_block(anchorCoordinateArray, allow_gap)
            block_weight = []
            for block in Coordinate_block_list:
                st = block[0][1]
                ed = block[1][1]
                w = sum([self.weights[i] for i in sequencePathNodeArray[st:ed+1] if i != -1])
                block_weight.append(w)
            copyrglist = Cyclic_Anchor_Combination_Detection(Coordinate_block_list, Coordinate_block_dif, Coordinate_block_weight)
        return sequencePathNodeArray, anchorCoordinateArray
    def fragmentReduce(self,newLength=1):
        offsetDict = {}
        if self.fragmentLength>newLength:
            
            self.fragmentLength = newLength
            nodes = range(self.totalNodes)
            oldfragmentsArraay = self.fragments.copy()
            self.fragments = np.full(self.totalNodes+10000,'',f'U{newLength}')
            self.weights.resize(self.totalNodes+10000)
            self.headtails.resize(self.totalNodes+10000)

            for nodeid in nodes:
                if nodeid in self.startNodeSet:
                    sequence = oldfragmentsArraay[nodeid]
                    self.queryGraph.coordinateList[nodeid] +=len(sequence)-newLength
                    self.fragments[nodeid] = sequence[-newLength:]
                    nextnode = nodeid
                    w_num =len(sequence)-newLength+1
                    for i in range(1,w_num):
                        seq = sequence[-newLength-i:-i]
                        # self.SourceList[self.totalNodes] = self.SourceList[nodeid]
                        self.SourceList.append(self.SourceList[nodeid])
                        newnodeid = self.totalNodes
                        self.addnewNodes(seq,0)
                        self.weights[newnodeid] = self.weights[nodeid]
                        self.queryGraph.coordinateList.append(self.queryGraph.coordinateList[nodeid]-i)
                        self.edgeWeightDict[(newnodeid,nextnode)]=self.weights[nodeid]
                        nextnode = newnodeid
                        offsetDict[newnodeid]=i
                    self.headtails[newnodeid]=self.headtails[nodeid]
                    self.headtails[nodeid] = 0
                    self.startNodeSet.remove(nodeid)
                    self.startNodeSet.add(newnodeid)

                else:
                    
                    sequence = oldfragmentsArraay[nodeid]
                    self.fragments[nodeid] = sequence[-newLength:]
                    self.queryGraph.coordinateList[nodeid] +=len(sequence)-newLength

            self.fragmentNodeDict = {}
            self.startFragmentNodeDict = {}
            self.endFragmentNodeDict = {}

            
            for nodeid in range(self.totalNodes):
                if self.headtails[nodeid] == 1:
                    self.startFragmentNodeDict.setdefault(self.fragments[nodeid], []).append(nodeid)
                elif self.headtails[nodeid] == 2:
                    self.endFragmentNodeDict.setdefault(self.fragments[nodeid], []).append(nodeid)
                elif self.headtails[nodeid] == 3:
                    self.startFragmentNodeDict.setdefault(self.fragments[nodeid], []).append(nodeid)
                    self.endFragmentNodeDict.setdefault(self.fragments[nodeid], []).append(nodeid)
                else:
                    self.fragmentNodeDict.setdefault(self.fragments[nodeid], []).append(nodeid)
        self.fragments = self.fragments[:self.totalNodes]
        self.weights = self.weights[:self.totalNodes]
        self.headtails = self.headtails[:self.totalNodes]
        self.nodenumthr = self.totalNodes
        self.queryGraph=DAGStru(self.totalNodes,self.edgeWeightDict)
        self.queryGraph.calculateCoordinates()
        self.offsetArray = np.full(self.totalNodes,0,dtype=np.uint8)
        for k,v in offsetDict.items():
            self.offsetArray[k]=v

    def fast_remove(self, delList):
        for node in delList:
            self.weights[node] = 0
        self.startNodeSet -= set(delList)
        self.endNodeSet -= set(delList)
    def merge_sameCoorFraNodes_loop(self, ArraySource=False):

        def search_and_merge():
            merge_nodes_f = self.merge_sameCoorFraNodes(ArraySource)
            self.queryGraph.calculateCoordinates(self.endNodeSet, forward=False)  
            merge_nodes_b = self.merge_sameCoorFraNodes(ArraySource, forward=False)  
            self.queryGraph.calculateCoordinates()  
            return merge_nodes_f + merge_nodes_b            
        self.queryGraph.calculateCoordinates(self.endNodeSet, forward=False)
        merge_nodes = 1  
        while merge_nodes != 0:
            merge_nodes = search_and_merge()              
    def updateCoordiante(self, update_stem=True, error_nodes=[]):
        
        linksource = self.edgeWeightDict.keys()
        if error_nodes == []:
            error_nodes = [i[1] for i in linksource if self.queryGraph.coordinateList[i[0]] >= self.queryGraph.coordinateList[i[1]]]
        self.queryGraph.local_update_coordinate(error_nodes)
        maxlength_update = False
        current_max = max(self.queryGraph.coordinateList)
        if self.queryGraph.maxlength != current_max:
            self.queryGraph.maxlength = current_max
            maxlength_update = True
        if update_stem == True:
            self.queryGraph.findLongestPath()
        return maxlength_update
    def calculateReferenceCoordinates(ref_queryGraph):
        ref_queryGraph.calculateCoordinates(ref_queryGraph.endNodeSet,forward=False)
        head_nodes = ref_queryGraph.BFS_DAG(ref_queryGraph.startNodeSet,bannedSet=ref_queryGraph.longestPathNodeSet)
        ref_queryGraph.maxlength= max(ref_queryGraph.coordinateList)
        coor_node_dict={}
        for node in ref_queryGraph.longestPathNodeSet:

            coor = ref_queryGraph.coordinateList[node]
            coor_node_dict[coor] = coor_node_dict.get(coor,[])
            coor_node_dict[coor].append(node)
        return coor_node_dict

    def convertToAliReferenceDAG(self,pureNodes,thr):

            
        stemGraphStru,stemNodes,main_edge = self.find_stemGraph_node(pureNodes,thr=thr)

        stemGraphStru.findLongestPath()

        startnodes=stemGraphStru.startNodeSet&stemGraphStru.longestPathNodeSet
        startNode=-1
        max_num=0
        for node in startnodes:
            if self.weights[node]>max_num:
                startNode=node
                max_num=self.weights[node]
        coor_cursor=1

        ref_nodes = [startNode]
        tmp_refnodelist = [startNode]
        now = set(stemGraphStru.findChildNodes(startNode))&stemGraphStru.longestPathNodeSet
        while now!=set():
            coor_cursor+=1
            max_num=0
            max_node = -1
            for node in now:
                if self.weights[node]>max_num and stemGraphStru.coordinateList[node]==coor_cursor:
                    max_node=node
                    max_num=self.weights[node]
            if max_node!=-1:
                tmp_refnodelist.append(max_node)
                ref_nodes.append(max_node)
                now = set(stemGraphStru.findChildNodes(max_node))&stemGraphStru.longestPathNodeSet
            else:
                now=set()

        coordinateNodeDict={}
        for node in stemGraphStru.longestPathNodeSet:
            coor = stemGraphStru.coordinateList[node]-1
            coordinateNodeDict[coor] = coordinateNodeDict.get(coor,set())
            coordinateNodeDict[coor].add(node)

        emProbMatrix=np.full((4,len(ref_nodes)),0,dtype=np.float64)
        baseDict={'A':[0],'T':[1],'C':[2],'G':[3],'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}
        for coor,nodes in coordinateNodeDict.items():
            for node in nodes:
                weight = self.weights[node]
                bases = baseDict[self.fragments[node][-1]]
                for base in bases:
                    emProbMatrix[base][coor]+=weight/len(bases)
        sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
        emProbMatrix = emProbMatrix/sum_of_emProbMatrix

        ori_ref_nodes =[self.SourceList[ref_nodes[0]][0]]
        ref_seq=self.fragments[ref_nodes[0]]

        for node in ref_nodes[1:]:
            ref_seq+=self.fragments[node][-1]
            ori_ref_nodes.append(self.SourceList[node][0])

        ranges = []
        return ref_seq,ori_ref_nodes,emProbMatrix,ranges
    def convertToAliReferenceDAG_new(self,no_degenerate_edgeDict,pureNodes,thr):

        stemGraphStru,stemNodes,main_edge = self.find_stemGraph(edgeWeightDict=no_degenerate_edgeDict,pureNodes=pureNodes,thr=thr)

        linearPath_list,linearPath_link,_ = build_coarse_grained_graph(stemGraphStru,main_edge)
        stemCoarse_grained_garph = DAGStru(len(linearPath_list),linearPath_link)
        _, max_path = stemCoarse_grained_garph.find_max_weight_path()
        ref_nodes = []
        for c in max_path:
            ref_nodes.extend(linearPath_list[c])
        

        start=-1
        for i in range(len(ref_nodes)):
            node = ref_nodes[i]
            if self.weights[node]/self.sequenceNum<thr:
                start=i
            else:
                break
        end = len(ref_nodes)
        for i in range(len(ref_nodes)):
            index = len(ref_nodes)-i-1
            node = ref_nodes[index]
            if self.weights[node]/self.sequenceNum<thr:
                end = index
            else:
                break
        ref_nodes = ref_nodes[start+1:end]


        coordinateNodeDict={}
        for idx,node in enumerate(ref_nodes):
            coordinateNodeDict[idx]=[node]
        ref_nodes_set = set(ref_nodes)
        self.queryGraph.calculateStateRange(ref_nodes,mode='build')

        forkGraphLinkDictList = {edge:weight  for edge,weight in self.edgeWeightDict.items() if set(edge)&ref_nodes_set==set()}
        forkGraphStru,idDict = build_subgraphStru(forkGraphLinkDictList)
        reverseIdDict = {v:k for k,v in idDict.items()}
        regionsArray = forkGraphStru.split_island()
        coorArray = np.array(forkGraphStru.coordinateList)
        insertRanges=[]
        for region in range(1+len(set(regionsArray))+1):
            indices = np.where(regionsArray==region)[0]
            if indices.size:
                mincoor = min(coorArray[indices])
                maxcoor = max(coorArray[indices])
                left = []
                right = []
                for node in indices:
                    rg = self.queryGraph.ref_coor[reverseIdDict[node]]
                    left.append(rg[0])
                    right.append(rg[1])
                ref_length = max(right)-min(left)-1
                real_length = maxcoor-mincoor+1
                startINdex = min(left)
                if real_length==ref_length:
                    for subnode in indices:
                        coor = startINdex+coorArray[subnode]
                        node = reverseIdDict[subnode]
                        coordinateNodeDict[coor].append(node)
                elif real_length>ref_length:
                    insertRanges.append([min(left),max(right),min(left),max(right)])

        emProbMatrix=np.full((4,len(ref_nodes)),0,dtype=np.float64)
        baseDict={'A':[0],'T':[1],'C':[2],'G':[3],'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}
        for coor,nodes in coordinateNodeDict.items():
            for node in nodes:
                weight = self.weights[node]
                bases = baseDict[self.fragments[node][-1]]
                for base in bases:
                    emProbMatrix[base][coor]+=weight/len(bases)
        sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
        emProbMatrix = emProbMatrix/sum_of_emProbMatrix

        ori_ref_nodes =[self.SourceList[ref_nodes[0]][0]]
        ref_seq=self.fragments[ref_nodes[0]]

        for node in ref_nodes[1:]:
            ref_seq+=self.fragments[node][-1]
            ori_ref_nodes.append(self.SourceList[node][0])
            
        ranges  = merge_intervals(insertRanges)
        ranges = [rg[0] for rg in ranges]
        return ref_seq,ori_ref_nodes,emProbMatrix,ranges
    
    
    def findMainPathNodes(self, wthr=0.5):
        node_weight_list = []
        for nodeid in range(self.totalNodes):
            threshold = self.sequenceNum * wthr
            if self.weights[nodeid] > threshold:
                node_weight_list.append([nodeid, self.queryGraph.coordinateList[nodeid]])
        sorted_list = sorted(node_weight_list, key=lambda x: x[1])
        reflist = [x[0] for x in sorted_list]
        return reflist
    def find_stemGraph(self, edgeWeightDict=None,pureNodes=None, thr=0.05):
        if edgeWeightDict is None:
            edgeWeightDict = self.edgeWeightDict
        weightArray = self.weights
        main_edge = []
        for edge, weight in edgeWeightDict.items():
            u, v = edge  
            a = weight / weightArray[u]
            b = weight / weightArray[v]

            if a > thr and b > thr:
                main_edge.append([edge, weight])
        
        main_edge = dict(main_edge)
        stemGraphStru = DAGStru(self.totalNodes, main_edge)
        stemGraphStru.calculateCoordinates() 
        startNode = set()
        endNode = set()
        valid_indices = np.nonzero(self.headtails)[0]
        for index in valid_indices:
            if self.fragments[index][-1] in {'A', 'T', 'C', 'G'}:
                if self.headtails[index] == 1:
                    startNode.add(index)
                elif self.headtails[index] == 2:
                    endNode.add(index)
                elif self.headtails[index] == 3:
                    startNode.add(index)
                    endNode.add(index)
        setA, _ = stemGraphStru.BFS_DAG(list(startNode))
        setB, _ = stemGraphStru.BFS_DAG(list(endNode), forward=False)
        stemNodeSet = setA&setB

        newedges = {edge:weight for edge,weight in self.edgeWeightDict.items() if set(edge)-stemNodeSet==set()}
        stemGraphStru = DAGStru(self.totalNodes, newedges)
        stemGraphStru.calculateCoordinates()  
        return stemGraphStru, stemNodeSet, newedges
    
    def find_stemGraph_node(self,pureNodes, thr=0.05):
        for nodeid  in list(pureNodes):
            if self.weights[nodeid]/self.sequenceNum <thr:
                pureNodes-=set([nodeid])
        newedges = {link:value for link,value in self.edgeWeightDict.items() if set(link)-pureNodes==set()}
        stemGraphStru = DAGStru(self.totalNodes, newedges)
        stemGraphStru.calculateCoordinates()
        return stemGraphStru, pureNodes, newedges
    
