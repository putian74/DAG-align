# -*- coding: utf-8 -*-
import os
import pickle
from DAG_stru import DAGStru
import numpy as np
from multiprocessing import Lock,Manager,Process
from DAG_tools import *
from DAG_Node import DAGNode
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
        self.nodeList = []
        self.totalNodes = 0
        self.main_fragment = main_fragment
        self.fragmentLength = fragmentLength
        self.originalfragmentLength = self.fragmentLength
        self.graph_type = graph_type
        self.firstBitofOSM = firstBitofOSM             
        self.allBitofOSM = allBitofOSM                
        self.firstBitofONM = firstBitofONM             
        self.allBitofONM = allBitofONM                
    def add_first_sequence(self, firstSequenceId: int, sequence: str) -> None:

        if self.graph_type != 'fragment':
            raise TypeError("The method can only be used when the graph type is 'fragment'.")
        if self.fragmentLength>len(sequence):
            raise ValueError("The length of the input sequence is less than fragmentLength.")
        w_num = len(sequence) - self.fragmentLength + 1
        row = 0          
        sequenceFragment = sequence[row:row+self.fragmentLength]           
        newNode = DAGNode(sequenceFragment)          
        newNode.id = row               
        self.totalNodes += 1          
        self.nodeList.append(newNode)                
        self.startFragmentNodeDict.setdefault(newNode.fragment, []).append(newNode.id)
        for row in range(1, w_num-1):              
            sequenceFragment = sequence[row:row+self.fragmentLength]
            newNode = DAGNode(sequenceFragment)
            newNode.id = row
            self.totalNodes += 1
            self.nodeList.append(newNode)
            self.fragmentNodeDict.setdefault(newNode.fragment, []).append(newNode.id)
            self.edgeWeightDict[(row-1, row)] = 1
        row = w_num-1              
        sequenceFragment = sequence[row:row+self.fragmentLength]
        newNode = DAGNode(sequenceFragment)
        newNode.id = row
        self.totalNodes += 1
        self.nodeList.append(newNode)
        self.endFragmentNodeDict.setdefault(newNode.fragment, []).append(newNode.id)
        self.edgeWeightDict[(row-1, row)] = 1              
        self.nodeList[0].ishead = 1          
        self.startNodeSet.add(0)                    
        self.nodeList[row].istail = 1          
        self.endNodeSet.add(row)                     
        self.queryGraph = DAGStru(self.totalNodes, self.edgeWeightDict.keys())
        self.queryGraph.coordinateList = list(np.arange(row+1)+1)                       
        self.queryGraph.maxlength = row+1                
        self.region_list = [0]*(row+1)                 
        self.region_num = 1                       
        self.queryGraph.longestPathNodeSet = set(np.arange(row))
        self.sequenceIDList.append(firstSequenceId)          
        self.sequenceNum += 1                                 
        self.sequencePathNodeMap = np.array([], dtype=object)
        self.sequencePathNodeMap.resize(self.sequencePathNodeMap.size+1, refcheck=True)
        self.sequencePathNodeMap[-1] = np.array(list(range(self.totalNodes)), dtype=np.uint32)
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
            w = self.nodeList[anchor].weight  
            if w > max_weight:
                max_weight = w            
                max_index = i                      
                count = 1                  
            elif w == max_weight and fast:
                count += 1                
        return anchors[max_index] if count == 1 else -1
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
        # sequencePathNodeArray = np.fromiter(
        #     map(lambda x: self.check_anchor(x), anchors_list),
        #     dtype=int
        # )
        anchorCoordinateArray = np.fromiter(
            map(self.anchor_coor, sequencePathNodeArray),
            dtype=int
        )
        return sequencePathNodeArray, anchorCoordinateArray, anchors_list
    
    def map_to_OSM(self):

        SourceList = np.array([], dtype=object)
        SourceList.resize(self.totalNodes, refcheck=False)
        for node in self.nodeList:
            SourceList[node.id] = np.zeros(node.weight, dtype=int)
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
        newnodes = []
        newstarts = []
        newends = []
        index = 0
        node = sequencePathNodeArray[index]
        if node == -1:
            newnode = DAGNode(sequence[index:index+fragmentLength], id=self.totalNodes, ishead=1)
            self.totalNodes += 1
            sequencePathNodeArray[index] = newnode.id
            coordinateList.append(sequencePathCoordinates[index])
            newstarts.append(newnode)
            self.startFragmentNodeDict.setdefault(newnode.fragment, []).append(newnode.id)
            self.startNodeSet.add(newnode.id)
        else:
            self.nodeList[node].weight += 1
        index += 1
        for node in sequencePathNodeArray[1:-1]:
            if node == -1:
                newnode = DAGNode(sequence[index:index+fragmentLength], id=self.totalNodes)
                newnodes.append(newnode)
                self.totalNodes += 1
                sequencePathNodeArray[index] = newnode.id
                coordinateList.append(sequencePathCoordinates[index])
                self.fragmentNodeDict.setdefault(newnode.fragment, []).append(newnode.id)
            else:
                self.nodeList[node].weight += 1
            newlink = (sequencePathNodeArray[index-1], sequencePathNodeArray[index])
            linkWeight = self.edgeWeightDict.get(newlink, 0) + 1
            self.edgeWeightDict[newlink] = linkWeight
            if linkWeight == 1:
                self.queryGraph.newEdges.add(newlink)
            index += 1
        node = sequencePathNodeArray[index]
        if node == -1:
            newnode = DAGNode(sequence[index:index+fragmentLength], id=self.totalNodes, istail=1) 
            self.totalNodes += 1
            sequencePathNodeArray[index] = newnode.id
            coordinateList.append(sequencePathCoordinates[index])
            self.endFragmentNodeDict.setdefault(newnode.fragment, []).append(newnode.id)
            self.endNodeSet.add(newnode.id)
            newends.append(newnode)
        else:
            self.nodeList[node].weight += 1
        newlink = (sequencePathNodeArray[index-1], sequencePathNodeArray[index])
        linkWeight = self.edgeWeightDict.get(newlink, 0) + 1
        self.edgeWeightDict[newlink] = linkWeight
        if linkWeight == 1:
            self.queryGraph.newEdges.add(newlink)
        self.nodeList.extend(newstarts)
        self.nodeList.extend(newnodes)
        self.nodeList.extend(newends)
        if coordinateUpdateList != []:
            self.queryGraph.update(self.totalNodes)
            self.updateCoordiante(update_stem=False, error_nodes=coordinateUpdateList)
        self.sequenceIDList.append(seq_id)
        self.sequencePathNodeMap.resize(self.sequencePathNodeMap.size+1, refcheck=True)
        self.sequencePathNodeMap[-1] = np.array(sequencePathNodeArray, dtype=np.uint64)
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
    def merge_node_info_batch(self, merge_list):

        delList = []                
        for tup in merge_list:
            nodeA = tup[0]                   
            delList.extend(tup[1:])                  
            for node in tup[1:]:
                self.nodeList[nodeA].merge(self.nodeList[node])
                self.nodeList[node] = []
        return delList                  
    def merge_sameCoorFraNodes(self, external_sources=False, forward=True):

        def check_merge_list(subFragmentList, merge_list, lock, forward=True):

            coordinateList = self.queryGraph.coordinateList if forward else self.queryGraph.backwardCoordinateList
            DAG_nodeList = self.nodeList
            tmp_merge_list = []
            nodeList = [i for i in subFragmentList if len(i) > 1]
            for i in nodeList:
                tlist = {}
                for j in [j for j in i if DAG_nodeList[j].ishead + DAG_nodeList[j].istail == 0]:
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
        if external_sources:
            self.merge_Source_batch(merge_list)
        self.fast_remove(delList)
        self.reorderNodelist(external_sources)
        return len(delList)
    def removeDegenerateBasePaths(self):

        if self.graph_type != 'fragment':
            raise TypeError("The method can only be used when the graph type is 'fragment'.")
        bannedSet = set()                    
        startNode = set()                      
        endNode = set()                        
        for node in self.nodeList:
            if node.fragment[-1] not in {'A', 'T', 'C', 'G'}:
                bannedSet.add(node.id)
                startNode.discard(node.id)
                endNode.discard(node.id)
            else:
                if node.ishead > 0:
                    startNode.add(node.id)
                if node.istail > 0:
                    endNode.add(node.id)
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
        for node in self.nodeList:
            if node.fragment[-1] not in {'A', 'T', 'C', 'G'}:
                bannedSet.add(node.id)
                startNode.discard(node.id)
                endNode.discard(node.id)
            else:
                if node.ishead > 0:
                    startNode.add(node.id)
                if node.istail > 0:
                    endNode.add(node.id)
        setA, _ = self.queryGraph.BFS_DAG(list(startNode), bannedSet=bannedSet)
        setB, _ = self.queryGraph.BFS_DAG(list(endNode), forward=False, bannedSet=bannedSet)
        allset = setA & setB             
        delset = set(range(self.totalNodes)) - allset             
        if not allset:
            raise RuntimeError("Failed to find a complete non-degenerate nucleotide path.")
        no_degenerate_edgeWeightDict = {link: value for link, value in self.edgeWeightDict.items() 
                            if not set(link) & delset}
        return no_degenerate_edgeWeightDict
    def search_same_fragment_nodes(self, newnode_id: int, minthr: float = 0.01, maxdistant: int = 2000) -> set:
        graphcoordinateList = np.array(self.queryGraph.coordinateList)
        tupset = set()                
        useableSameFraNodesList = [v for v in self.fragmentNodeDict.values() if len(v) > 1]
        weightArray = np.array([n.weight for n in self.nodeList])
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
    def merge_sameFraNodes(self,newnode_id=0,external_sources=False,maxdistant=2000):

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
                    weight_sum = self.nodeList[t[0]].weight + self.nodeList[t[1]].weight
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
            self.nodeList[Anode].merge(self.nodeList[Bnode])
                       
            delList.append(Bnode)
        if external_sources:
            self.merge_Source_batch(merge_list)
        self.fast_remove(delList)
        self.reorderNodelist(external_sources)
        self.queryGraph.calculateCoordinates()
        return delList != []
    def merge_check(self, external_sources: bool = False, maxdistant: int = 2000) -> None:
        merge_flag = True          
        self.merge_sameCoorFraNodes_loop(external_sources)
        while merge_flag:
            merge_flag = self.merge_sameFraNodes(newnode_id=0,external_sources=external_sources, maxdistant=maxdistant)
            self.merge_sameCoorFraNodes_loop(external_sources)
    def reorderNodelist(self, external_sources=False):

        if not isinstance(external_sources, bool):
            raise ValueError(f"无效模式参数：{external_sources}，仅支持True/False")
        new_nodelist = []                            
        id_mapping = {}                                  
        self.fragmentNodeDict = {}                                    
        self.startFragmentNodeDict = {}                      
        self.endFragmentNodeDict = {}                        
        valid_nodes = [node for node in self.nodeList if node]
        if external_sources:
            newSourceList = np.empty(len(valid_nodes), dtype=object)
        iter_nodes = valid_nodes  
        for index, node in enumerate(iter_nodes):                        
            old_id = node.id
            id_mapping[old_id] = index  
            if external_sources:
                newSourceList[index] = self.SourceList[old_id].copy()
            node.id = index                   
            if node.ishead + node.istail == 0:                  
                self.fragmentNodeDict.setdefault(node.fragment, []).append(index)
            else:          
                if node.ishead > 0:
                    self.startFragmentNodeDict.setdefault(node.fragment, []).append(index)
                if node.istail > 0:
                    self.endFragmentNodeDict.setdefault(node.fragment, []).append(index)
            new_nodelist.append(node)  
        if external_sources:
            self.SourceList = newSourceList
        self.nodeList = new_nodelist
        self.edgeWeightDict = {
            (id_mapping[u], id_mapping[v]): w
            for (u, v), w in self.edgeWeightDict.items()
            if u in id_mapping and v in id_mapping             
        }
        self.startNodeSet = {id_mapping[n] for n in self.startNodeSet if n in id_mapping}
        self.endNodeSet = {id_mapping[n] for n in self.endNodeSet if n in id_mapping}
        self.totalNodes = len(self.nodeList)
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
    def save_graph(self, savePath: str = None, mode: str = 'build', maxdistant: int = 2000, external_sources: bool = False) -> None:

        if not savePath is None:
            self.savePath = sanitize_path(savePath,'output')
            if not os.path.exists(self.savePath):
                os.mkdir(self.savePath)          
        else:
            savePath = self.savePath    
        self.merge_check(external_sources, maxdistant=maxdistant)          
        if self.graph_type == 'fragment' and mode=='build':
            ori_node_list = np.full((self.totalNodes), 0)
            onmindex = np.full((self.totalNodes, 2), 0)
            for node in self.nodeList:
                ori_node_list[node.id] = save_numbers(
                    int(self.graphID),
                    node.id,
                    self.firstBitofONM,
                    self.allBitofONM
                )
                onmindex[node.id] = np.array([node.id, node.id + 1])
            ori_node_list = np.array(ori_node_list, dtype=object)
            new_vid=[]
            for idx,v in enumerate(self.sequenceIDList):
                key = f"{self.graphID}_{idx}"
                new_vid.append([v,key])

            for node in range(self.totalNodes):
                if len(self.SourceList[node])!=self.nodeList[node].weight:
                    raise ValueError(vars(self.nodeList[node]),len(self.SourceList[node]))
            np.save(self.savePath/'osm.npy',self.SourceList)
            np.save(self.savePath/'onm.npy',ori_node_list)
            np.save(self.savePath/'onm_index.npy',onmindex)
            np.save(self.savePath/'v_id.npy',new_vid)
        elif self.graph_type == 'fragment' and mode!='build':
            ori_node_list = []                                                        
            for node in self.nodeList:
                ori_node_list.append(node.Source)
                node.Source = []
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
        print('mean_weight of ref_seq', np.mean([self.nodeList[node].weight for node in self.reflist]))
        print('max_range of nodes in graph:', max(list(state_range_dict.keys())))
        print('graph file save in ', self.savePath)
        self.nodeList = np.array(self.nodeList, dtype=object)           
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
            nodeList=self.nodeList,
            startNodeSet=self.startNodeSet,
            endNodeSet=self.endNodeSet
        )
        all_attributes = self.__dict__.keys()
        whitelist = [
            "graphID", "originalfragmentLength", "fragmentLength",
            "savePath", "sequenceNum", "fastaFilePathList",
            "firstBitofOSM", "allBitofOSM", "firstBitofONM",
            "allBitofONM", "totalNodes", "sequenceIDList", "graph_type",
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
            w = sum([self.nodeList[i].weight for i in sequencePathNodeArray[st:ed+1] if i != -1])
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
                w = sum([self.nodeList[i].weight for i in sequencePathNodeArray[st:ed+1] if i != -1])
                block_weight.append(w)
            copyrglist = Cyclic_Anchor_Combination_Detection(Coordinate_block_list, Coordinate_block_dif, Coordinate_block_weight)
        return sequencePathNodeArray, anchorCoordinateArray
    def fragmentReduce(self,newLength=1):
        if self.fragmentLength>newLength:
            self.fragmentLength = newLength
            nodes = range(self.totalNodes)
            for node_id in nodes:
                node = self.nodeList[node_id]
                if node.id in self.startNodeSet:
                    sequence = node.fragment
                    node.fragment = sequence[-newLength:]
                    self.queryGraph.coordinateList[node.id] +=len(sequence)-newLength
                    nextnode = node.id
                    w_num =len(sequence)-newLength+1
                    for i in range(1,w_num):
                        seq = sequence[-newLength-i:-i]
                        newNode = DAGNode(seq)
                        newNode.id =self.totalNodes
                        self.totalNodes+=1
                        self.nodeList.append(newNode)
                        newNode.Source = node.Source
                        self.queryGraph.coordinateList.append(self.queryGraph.coordinateList[node.id]-i)
                        newNode.weight = node.weight
                        self.edgeWeightDict[(newNode.id,nextnode)]=node.weight
                        nextnode = newNode.id
                    newNode.ishead=node.ishead
                    node.ishead = 0
                    self.startNodeSet.remove(node.id)
                    self.startNodeSet.add(newNode.id)
                else:
                    sequence = node.fragment
                    node.fragment = sequence[-newLength:]
                    self.queryGraph.coordinateList[node.id] +=len(sequence)-newLength
            self.fragmentNodeDict = {}
            self.startFragmentNodeDict = {}
            self.endFragmentNodeDict = {}
            for node in self.nodeList:
                if node.ishead:
                    self.startFragmentNodeDict.setdefault(node.fragment, []).append(node.id)
                if node.istail:
                    self.endFragmentNodeDict.setdefault(node.fragment, []).append(node.id)
                if node.ishead+node.istail==0:
                    self.fragmentNodeDict.setdefault(node.fragment, []).append(node.id)
        self.queryGraph=DAGStru(self.totalNodes,self.edgeWeightDict)
        self.queryGraph.calculateCoordinates()
    def fast_remove(self, delList):

        for node in delList:
            self.nodeList[node] = []
        self.startNodeSet -= set(delList)
        self.endNodeSet -= set(delList)
    def merge_sameCoorFraNodes_loop(self, external_sources=False):

        def search_and_merge():
            merge_nodes_f = self.merge_sameCoorFraNodes(external_sources)
            self.queryGraph.calculateCoordinates(self.endNodeSet, forward=False)  
            merge_nodes_b = self.merge_sameCoorFraNodes(external_sources, forward=False)  
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
            if node in head_nodes:
                coor = ref_queryGraph.maxlength - ref_queryGraph.backwardCoordinateList[node]+1
            else:
                coor = ref_queryGraph.coordinateList[node]
            coor_node_dict[coor] = coor_node_dict.get(coor,[])
            coor_node_dict[coor].append(node)
        return coor_node_dict
    def convertToAliReferenceDAG(self,thr):

        for node in self.nodeList:
            node.Source=[node.id]
        self.fragmentReduce(20)
        self.merge_sameCoorFraNodes_loop()
        self.fragmentReduce()
        no_degenerate_edgeDict = self.noDegenerateGraph()
        ndDAGStru = DAGStru(self.totalNodes,no_degenerate_edgeDict)
        min_seq_num = max(thr*self.sequenceNum,1)
        node_set=set()
        for node in self.nodeList:
            if node.weight>min_seq_num:
                node_set.add(node.id)
        ndDAGStru.calculateCoordinates()
        ndDAGStru.findLongestPath()
        now = self.endNodeSet
        doneset=set()
        while now !=set():
            next_set=set()
            for node in now:
                if self.nodeList[node].weight<0.01*self.sequenceNum and node not in doneset:
                    node_set-=set([node])
                    next_set.add(node)
                    doneset.add(node)
            now = set(ndDAGStru.findParentNodes(next_set))
        delset = set(range(self.totalNodes))-node_set
        edgeWeightDict = {link:value for link,value in self.edgeWeightDict.items() if not set(link) & delset}
        ndDAGStru = DAGStru(self.totalNodes,edgeWeightDict)
        ndDAGStru.calculateCoordinates()
        ndDAGStru.findLongestPath()
        startnodes=ndDAGStru.startNodeSet&ndDAGStru.longestPathNodeSet
        startNode=-1
        max_num=0
        for node in startnodes:
            if self.nodeList[node].weight>max_num:
                startNode=node
                max_num=self.nodeList[node].weight
        coor_cursor=1
        ref_seq = self.nodeList[startNode].fragment
        ref_nodelist = [self.nodeList[startNode].Source[0]]
        tmp_refnodelist = [startNode]
        now = set(ndDAGStru.findChildNodes(startNode))&ndDAGStru.longestPathNodeSet
        while now!=set():
            coor_cursor+=1
            max_num=0
            max_node = -1
            for node in now:
                if self.nodeList[node].weight>max_num and ndDAGStru.coordinateList[node]==coor_cursor:
                    max_node=node
                    max_num=self.nodeList[node].weight
            if max_node!=-1:
                ref_seq+=self.nodeList[max_node].fragment[-1]
                tmp_refnodelist.append(max_node)
                ref_nodelist.append(self.nodeList[max_node].Source[0])
                now = set(ndDAGStru.findChildNodes(max_node))&ndDAGStru.longestPathNodeSet
            else:
                now=set()
        ndDAGStru.findLongestPath()
        coor_node_dict = DAGInfo.calculateReferenceCoordinates(ndDAGStru)
        emProbMatrix=np.full((4,ndDAGStru.maxlength),0,dtype=np.float64)
        Adict={'A':[0],'T':[1],'C':[2],'G':[3],'R': [0, 3], 'Y': [2, 1], 'M': [0, 2], 'K': [3, 1], 'S': [3, 2], 'W': [0, 1], 'H': [0, 1, 2], 'B': [3, 1, 2], 'V': [3, 0, 2], 'D': [3, 0, 1], 'N': [0, 1, 2, 3]}
        for i in range(1,ndDAGStru.maxlength+1):
            for node in coor_node_dict[i]:
                bases = Adict[self.nodeList[node].fragment[-1]]
                for base in bases:
                    emProbMatrix[base][i-1]+=self.nodeList[node].weight/len(bases)
        add_emProbMatrix = np.full((4,self.fragmentLength-1),0,dtype=np.float64)
        addseq = ref_seq[:self.fragmentLength-1]
        for i in range(self.fragmentLength-1):
            bases = Adict[addseq[i]]
            for base in bases:
                add_emProbMatrix[base][i]+= 1/len(bases)
        emProbMatrix = np.hstack((add_emProbMatrix,emProbMatrix))
        sum_of_emProbMatrix = np.sum(emProbMatrix,axis=0)
        emProbMatrix = emProbMatrix/sum_of_emProbMatrix
        return ref_seq,ref_nodelist,emProbMatrix,[]


    
    def findMainPathNodes(self, wthr=0.5):

        node_weight_list = []
        for node in self.nodeList:
            threshold = self.sequenceNum * wthr
            if node.weight > threshold:
                node_weight_list.append([node.id, self.queryGraph.coordinateList[node.id]])
        sorted_list = sorted(node_weight_list, key=lambda x: x[1])
        reflist = [x[0] for x in sorted_list]
        return reflist