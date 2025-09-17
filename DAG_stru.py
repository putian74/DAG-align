#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, Dict, Set, KeysView, List
from collections import deque
import numba
from typing import Dict, Set, Union, KeysView

@numba.jit(nopython=True, cache=True)
def numba_forward_pass(
        start_nodes, total_nodes, initial_coordinates,
        forwardNodeList, forwardEdgeList,
        backwardNodeList, backwardEdgeList
):
    degreeList = np.zeros(total_nodes, dtype=np.int32)
    reachable_nodes = np.full(total_nodes, False, dtype=np.bool_)
    q_buffer = np.empty(total_nodes, dtype=np.int32)
    q_head, q_tail = 0, 0
    for node in start_nodes:
        if not reachable_nodes[node]:
            q_buffer[q_tail], q_tail = node, q_tail + 1
            reachable_nodes[node] = True
    bfs_head = 0
    while bfs_head < q_tail:
        cursorNode = q_buffer[bfs_head]
        bfs_head += 1
        edge_ptr = forwardNodeList[cursorNode]
        while edge_ptr != -1:
            target_node = forwardEdgeList[edge_ptr][0]
            degreeList[target_node] += 1
            if not reachable_nodes[target_node]:
                q_buffer[q_tail], q_tail = target_node, q_tail + 1
                reachable_nodes[target_node] = True
            edge_ptr = forwardEdgeList[edge_ptr][1]
    coordinate_list = initial_coordinates.copy()
    q_head, q_tail = 0, 0
    for i in range(total_nodes):
        if reachable_nodes[i] and degreeList[i] == 0:
            q_buffer[q_tail], q_tail = i, q_tail + 1
    while q_head < q_tail:
        cursorNode = q_buffer[q_head]
        q_head += 1
        max_coord = -1
        parent_edge_ptr = backwardNodeList[cursorNode]
        while parent_edge_ptr != -1:
            parent_node = backwardEdgeList[parent_edge_ptr][0]
            if coordinate_list[parent_node] > max_coord:
                max_coord = coordinate_list[parent_node]
            parent_edge_ptr = backwardEdgeList[parent_edge_ptr][1]
        coordinate_list[cursorNode] = max_coord + 1
        edge_ptr = forwardNodeList[cursorNode]
        while edge_ptr != -1:
            target_node = forwardEdgeList[edge_ptr][0]
            if reachable_nodes[target_node]:
                degreeList[target_node] -= 1
                if degreeList[target_node] == 0:
                    q_buffer[q_tail], q_tail = target_node, q_tail + 1
            edge_ptr = forwardEdgeList[edge_ptr][1]
    return coordinate_list

@numba.jit(nopython=True, cache=True)
def numba_backward_pass_with_bfs(
    longest_path_nodes, total_nodes, current_coords,
    forwardNodeList, forwardEdgeList,
    backwardNodeList, backwardEdgeList,
    all_edges
):
    reachable_mask = np.full(total_nodes, False, dtype=np.bool_)
    q_buffer = np.empty(total_nodes, dtype=np.int32)
    q_head, q_tail = 0, 0

    for node in longest_path_nodes:
        if not reachable_mask[node]:
            q_buffer[q_tail], q_tail = node, q_tail + 1
            reachable_mask[node] = True

    bfs_head = 0
    while bfs_head < q_tail:
        cursorNode = q_buffer[bfs_head]
        bfs_head += 1
        edge_ptr = forwardNodeList[cursorNode]
        while edge_ptr != -1:
            target_node = forwardEdgeList[edge_ptr][0]
            if not reachable_mask[target_node]:
                q_buffer[q_tail], q_tail = target_node, q_tail + 1
                reachable_mask[target_node] = True
            edge_ptr = forwardEdgeList[edge_ptr][1]
    
    headnodes_mask = np.logical_not(reachable_mask)
    
    if not np.any(headnodes_mask):
        return current_coords

    indegreeList = np.zeros(total_nodes, dtype=np.int32)
    outdegree_list = np.zeros(total_nodes, dtype=np.int32)
    for i in range(all_edges.shape[0]):
        u, v = all_edges[i, 0], all_edges[i, 1]
        if headnodes_mask[u] and headnodes_mask[v]:
            indegreeList[v] += 1
            outdegree_list[u] += 1

    tight_coords = current_coords.copy()
    max_len = 0
    for x in current_coords:
        if x > max_len:
            max_len = x
            
            
    q_head, q_tail = 0, 0
    for i in range(total_nodes):
        if headnodes_mask[i] and outdegree_list[i] == 0:
            q_buffer[q_tail], q_tail = i, q_tail + 1
            
    while q_head < q_tail:
        cursorNode = q_buffer[q_head]
        q_head += 1


        min_coord = max_len + 1 
        has_any_child = False

        child_edge_ptr = forwardNodeList[cursorNode]
        while child_edge_ptr != -1:
            has_any_child = True
            child_node = forwardEdgeList[child_edge_ptr][0]
            
            child_coord = 0
            if headnodes_mask[child_node]:

                child_coord = tight_coords[child_node]
            else:
                child_coord = current_coords[child_node]
            
            if child_coord < min_coord:
                min_coord = child_coord
                
            child_edge_ptr = forwardEdgeList[child_edge_ptr][1]
        
        if has_any_child:
            tight_coords[cursorNode] = min_coord - 1
        else:
            tight_coords[cursorNode] = max_len

        parent_edge_ptr = backwardNodeList[cursorNode]
        while parent_edge_ptr != -1:
            parent_node = backwardEdgeList[parent_edge_ptr][0]
            if headnodes_mask[parent_node]:
                outdegree_list[parent_node] -= 1
                if outdegree_list[parent_node] == 0:
                    q_buffer[q_tail], q_tail = parent_node, q_tail + 1
            parent_edge_ptr = backwardEdgeList[parent_edge_ptr][1]

    final_coords = current_coords.copy()
    for i in range(total_nodes):
        if headnodes_mask[i]:
            final_coords[i] = tight_coords[i]
            
    return final_coords

@numba.jit(nopython=True, cache=True)
def numba_calculate_coordinates_engine(
    total_nodes, all_edges,
    forwardNodeList, forwardEdgeList,
    backwardNodeList, backwardEdgeList,
    forward
):
    indegreeList = np.zeros(total_nodes, dtype=np.int32)
    outdegree_list = np.zeros(total_nodes, dtype=np.int32)
    for i in range(all_edges.shape[0]):
        u, v = all_edges[i, 0], all_edges[i, 1]
        indegreeList[v] += 1
        outdegree_list[u] += 1

    coordinate_list = np.zeros(total_nodes, dtype=np.int32)
    q_buffer = np.empty(total_nodes, dtype=np.int32)
    q_head, q_tail = 0, 0
    
    degreeList_copy = None
    if forward:
        degreeList_copy = indegreeList.copy()
        for i in range(total_nodes):
            if degreeList_copy[i] == 0:
                q_buffer[q_tail], q_tail = i, q_tail + 1
                coordinate_list[i] = 1 
    else: 
        degreeList_copy = outdegree_list.copy()
        for i in range(total_nodes):
            if degreeList_copy[i] == 0:
                q_buffer[q_tail], q_tail = i, q_tail + 1
                coordinate_list[i] = 1 
                
    while q_head < q_tail:
        cursorNode = q_buffer[q_head]
        q_head += 1

        if forward:
            max_coord = 0 
            parent_edge_ptr = backwardNodeList[cursorNode]
            while parent_edge_ptr != -1:
                parent_node = backwardEdgeList[parent_edge_ptr][0]
                if coordinate_list[parent_node] > max_coord:
                    max_coord = coordinate_list[parent_node]
                parent_edge_ptr = backwardEdgeList[parent_edge_ptr][1]
            if max_coord > 0: 
                 coordinate_list[cursorNode] = max_coord + 1
        else: 
            max_coord = 0
            child_edge_ptr = forwardNodeList[cursorNode]
            while child_edge_ptr != -1:
                child_node = forwardEdgeList[child_edge_ptr][0]
                if coordinate_list[child_node] > max_coord:
                    max_coord = coordinate_list[child_node]
                child_edge_ptr = forwardEdgeList[child_edge_ptr][1]
            if max_coord > 0:
                coordinate_list[cursorNode] = max_coord + 1

        if forward:
            edge_ptr = forwardNodeList[cursorNode]
            while edge_ptr != -1:
                targetNode = forwardEdgeList[edge_ptr][0]
                degreeList_copy[targetNode] -= 1
                if degreeList_copy[targetNode] == 0:
                    q_buffer[q_tail], q_tail = targetNode, q_tail + 1
                edge_ptr = forwardEdgeList[edge_ptr][1]
        else: 
            edge_ptr = backwardNodeList[cursorNode]
            while edge_ptr != -1:
                targetNode = backwardEdgeList[edge_ptr][0]
                degreeList_copy[targetNode] -= 1
                if degreeList_copy[targetNode] == 0:
                    q_buffer[q_tail], q_tail = targetNode, q_tail + 1
                edge_ptr = backwardEdgeList[edge_ptr][1]
                
    return coordinate_list, indegreeList, outdegree_list

class DAGStru:

    def __init__(self, node_num: int, edgeWeightDict: Union[Dict, Set, KeysView]):

        self.totalNodes = node_num
        self.newEdges = set()
        forwardNodeList = [-1] * self.totalNodes                    
        backwardNodeList = [-1] * self.totalNodes                      
        forwardEdgeList = []                                        
        backwardEdgeList = []                                        
        if type(edgeWeightDict) == dict:
            for idx, i in enumerate(edgeWeightDict):
                forwardEdgeList.append([i[1], forwardNodeList[i[0]], edgeWeightDict[i]])
                forwardNodeList[i[0]] = idx               
                backwardEdgeList.append([i[0], backwardNodeList[i[1]], edgeWeightDict[i]])
                backwardNodeList[i[1]] = idx           
        elif isinstance(edgeWeightDict, (KeysView, set)):
            for idx, i in enumerate(edgeWeightDict):
                forwardEdgeList.append([i[1], forwardNodeList[i[0]], 0])
                forwardNodeList[i[0]] = idx
                backwardEdgeList.append([i[0], backwardNodeList[i[1]], 0])
                backwardNodeList[i[1]] = idx
        else:
            raise TypeError("edgeWeightDict must a dict or set")
        self.forwardNodeList = forwardNodeList
        self.backwardNodeList = backwardNodeList
        self.forwardEdgeList = forwardEdgeList
        self.backwardEdgeList = backwardEdgeList
        self.currentEdgeSet = set(edgeWeightDict)
    def add_edge_to_queryGraph(self, startNode: int, targetNode: int) -> None:

        self.forwardEdgeList.append([targetNode, self.forwardNodeList[startNode], 0])
        self.forwardNodeList[startNode] = len(self.forwardEdgeList) - 1
        self.backwardEdgeList.append([startNode, self.backwardNodeList[targetNode], 0])
        self.backwardNodeList[targetNode] = len(self.backwardEdgeList) - 1
        self.currentEdgeSet.add((startNode, targetNode))

    def update_old(self, totalNodes: int,edges) -> None:
        self.totalNodes = totalNodes

        forward_len = len(self.forwardNodeList)
        if forward_len < totalNodes:
            self.forwardNodeList.extend([-1] * (totalNodes - forward_len))

        backward_len = len(self.backwardNodeList)
        if backward_len < totalNodes:
            self.backwardNodeList.extend([-1] * (totalNodes - backward_len))
        nw = edges.keys()-self.currentEdgeSet
        if nw:
            add_edge = self.add_edge_to_queryGraph  
            for src, dst in nw:
                add_edge(src, dst)
            self.currentEdgeSet.update(self.newEdges)

    def merge_node_in_stru(self, j, k, edgeWeightDict):

        fathernodes = self.findParentNodes(k)           
        sonnodes = self.findChildNodes(k)               
        for child in sonnodes:
            edgeWeightDict[(j, child)] = edgeWeightDict.get((j, child), 0) + edgeWeightDict[(k, child)]
            del edgeWeightDict[(k, child)]                
        for parent in fathernodes:
            edgeWeightDict[(parent, j)] = edgeWeightDict.get((parent, j), 0) + edgeWeightDict[(parent, k)]
            del edgeWeightDict[(parent, k)]                 
        for node in fathernodes:
            i = self.forwardNodeList[node]                  
            while i != -1:            
                if self.forwardEdgeList[i][0] == k:
                    self.forwardEdgeList[i][0] = j
                i = self.forwardEdgeList[i][1]             
        for node in sonnodes:
            i = self.backwardNodeList[node]                  
            while i != -1:            
                if self.backwardEdgeList[i][0] == k:
                    self.backwardEdgeList[i][0] = j
                i = self.backwardEdgeList[i][1]             
        if self.forwardNodeList[k] != -1:          
            last_edge_index = self.forwardNodeList[j]                  
            if last_edge_index == -1:             
                self.forwardNodeList[j] = self.forwardNodeList[k]               
            else:
                while self.forwardEdgeList[last_edge_index][1] != -1:
                    last_edge_index = self.forwardEdgeList[last_edge_index][1]
                self.forwardEdgeList[last_edge_index][1] = self.forwardNodeList[k]
            self.forwardNodeList[k] = -1             
        if self.backwardNodeList[k] != -1:          
            last_edge_index = self.backwardNodeList[j]                  
            if last_edge_index == -1:
                self.backwardNodeList[j] = self.backwardNodeList[k]
            else:
                while self.backwardEdgeList[last_edge_index][1] != -1:
                    last_edge_index = self.backwardEdgeList[last_edge_index][1]
                self.backwardEdgeList[last_edge_index][1] = self.backwardNodeList[k]
            self.backwardNodeList[k] = -1             
        self.currentEdgeSet = edgeWeightDict.keys()
        return edgeWeightDict               
    def findParentNodes(self, nodes: List[int]) -> List[int]:

        if isinstance(nodes, (int, np.integer)):
            nodes=[nodes]
        parent = set()              
        for node in nodes:
            i = self.backwardNodeList[node]
            while i != -1:
                parent.add(self.backwardEdgeList[i][0])
                i = self.backwardEdgeList[i][1]
        return list(parent)               
    def findChildNodes(self, nodes: List[int]) -> List[int]:

        if isinstance(nodes, (int, np.integer)):
            nodes=[nodes]
        children = set()                
        for node in nodes:
            i = self.forwardNodeList[node]
            while i != -1:
                children.add(self.forwardEdgeList[i][0])
                i = self.forwardEdgeList[i][1]
        return list(children)               
    


    def isConsecutive(self,nodeA,nodeB,thr):
        if abs(self.coordinateList[nodeA]-self.coordinateList[nodeB]) ==1:
            return True
        else:
            False
    def less_than(self,nodeA,nodeB,thr):
        if self.coordinateList[nodeA] <=thr:
            return True
        else:
            False
    def more_than(self,nodeA,nodeB,thr):
        if self.coordinateList[nodeA] >=thr:
            return True
        else:
            False
    def BFS_DAG(self, startNode, forward=True, filter=lambda x, y, z: True,operate= lambda x:True , bannedSet=None, tagList=None,stopline=0):

        if bannedSet is None:
            bannedSet = set()
        if tagList is None:
            tagList = np.full(self.totalNodes, 0)
        else:
            if len(tagList) != self.totalNodes:
                raise ValueError("tagList length must match totalNodes")
        direction_suffix = "forward" if forward else "backward"
        nodeList = getattr(self, f"{direction_suffix}NodeList")            
        edgeList = getattr(self, f"{direction_suffix}EdgeList")           
        q = deque(startNode if isinstance(startNode, list) else list(startNode))
        touchedNodes = set()            
        touchedTags = set()            
        for node in bannedSet:
            tagList[node] = -1              
        while q:
            current = q.popleft()
            operate(current)
            if tagList[current] == 0:         
                edge_ptr = nodeList[current]
                while edge_ptr != -1:         
                    neighbor = edgeList[edge_ptr][0]
                    if filter(current, neighbor, stopline):
                        q.append(neighbor)
                    edge_ptr = edgeList[edge_ptr][1]           
                tagList[current] = -1          
                touchedNodes.add(current)
            else:
                if tagList[current] > 0:          
                    touchedTags.add(tagList[current])
        return touchedNodes, touchedTags

    
    def findLongestPath(self):

        if not self.endNodeSet:
            raise ValueError("终止节点集合为空，无法计算最长路径")
        if len(self.coordinateList) < self.totalNodes:
            raise ValueError("坐标列表不完整")
        indexdict = {}
        for node in self.endNodeSet:
            indexdict.setdefault(self.coordinateList[node], []).append(node)
        maxindex = max(indexdict.keys())
        self.maxlength = maxindex  
        mainend = indexdict[maxindex]
        self.longestPathNodeSet,_ = self.BFS_DAG(mainend, forward=False,filter=self.isConsecutive)
        
    def findParentNode_with_weight(self, node: int):
        parent = set()  
        i = self.backwardNodeList[node]
        while i != -1:
            parent.add((self.backwardEdgeList[i][0],self.backwardEdgeList[i][2]))
            i = self.backwardEdgeList[i][1]
        return list(parent) 
    
    def find_max_weight_path(self):
        self.calculateCoordinates()
        idx = 1  
        search = self.findParentNode_with_weight 

        nodeList = self.forwardNodeList  
        edgeList = self.forwardEdgeList  

            

        degreeList = [0] * self.totalNodes
        for link in self.currentEdgeSet:
            degreeList[link[idx]] += 1

        q = {i for i, x in enumerate(degreeList) if x == 0}

        Path_record = np.full(self.totalNodes,-1)
        WeightSum_record = np.full(self.totalNodes,0)
        q = deque(q)  
        while q:
            cursorNode = q.popleft()
            neighbourNodesAndWeight = search(cursorNode)
            if neighbourNodesAndWeight:
                Weights = np.array([n[1] for n in neighbourNodesAndWeight])+np.array([WeightSum_record[n[0]] for n in neighbourNodesAndWeight])
                nodes = [n[0] for n in neighbourNodesAndWeight]
                maxindex = np.argmax(Weights)
                maxnodes = nodes[maxindex]
                maxWeight = Weights[maxindex]
                WeightSum_record[cursorNode] = maxWeight
                Path_record[cursorNode]=maxnodes

            i = nodeList[cursorNode]  
            while i != -1:  
                targetNode = edgeList[i][0]  
                degreeList[targetNode] -= 1
                if degreeList[targetNode] == 0:
                    q.append(targetNode)
                
                i = edgeList[i][1]  

        maxWeight =0
        endNode = -1
        for node in self.endNodeSet:
            if WeightSum_record[node] >maxWeight:
                maxWeight = WeightSum_record[node]
                endNode = node
        path = [endNode]
        pnode = Path_record[endNode]
        while pnode!=-1:
            path.insert(0,pnode)
            pnode = Path_record[pnode]
        return maxWeight,path
    

    def split_island(self):
        islandID = np.full(self.totalNodes, 0)
        idx = 1 
        for stnode in self.startNodeSet:
            sonnodes, andset = self.BFS_DAG([stnode], tagList=islandID)
            sonnodes = np.array(list(sonnodes))  

            andset -= {-1}

            if len(andset) == 1:  
                to_idx = andset.pop()
                islandID[sonnodes] = to_idx 
            elif len(andset) > 1:  
                to_idx = min(andset)  
                islandID[sonnodes] = to_idx 
                andset -= {to_idx}
                for i in andset:
                    islandID[islandID == i] = to_idx
            else:
                islandID[sonnodes] = idx
                idx += 1 
                
        return islandID
    def nonCyclic(self, nodes: list) -> bool:

        coors = [self.coordinateList[node] for node in nodes]
        maxindex = np.argmax(coors)
        minindex = np.argmin(coors)
        q = deque([nodes[minindex]])
        endindex = coors[maxindex]
        check_node = nodes[maxindex]
        without_loop = True
        doneList = [0] * self.totalNodes
        forwardNodeList = self.forwardNodeList
        forwardEdgeList = self.forwardEdgeList
        while q:
            cursorNode = q.popleft()
            if doneList[cursorNode] == 0:
                i = forwardNodeList[cursorNode]
                while i != -1:
                    target_node = forwardEdgeList[i][0]
                    if self.coordinateList[target_node] < endindex:
                        q.append(target_node)
                    elif target_node == check_node:
                        without_loop = False
                        return without_loop            
                    i = forwardEdgeList[i][1]
                doneList[cursorNode] = 1
        return without_loop
    
    def local_update_coordinate(self, startnodes):
        if not startnodes:
            return

        np_start_nodes = np.array(startnodes, dtype=np.int32)
        np_initial_coords = np.array(self.coordinateList, dtype=np.int32)
        np_f_node_list = np.array(self.forwardNodeList, dtype=np.int32)
        np_f_edge_list = np.array(self.forwardEdgeList, dtype=np.int32) if self.forwardEdgeList else np.empty((0,3), dtype=np.int32)
        np_b_node_list = np.array(self.backwardNodeList, dtype=np.int32)
        np_b_edge_list = np.array(self.backwardEdgeList, dtype=np.int32) if self.backwardEdgeList else np.empty((0,3), dtype=np.int32)
        
        loose_coords = numba_forward_pass(
            np_start_nodes, self.totalNodes, np_initial_coords,
            np_f_node_list, np_f_edge_list, np_b_node_list, np_b_edge_list
        )

        np_longest_path = np.array(list(self.longestPathNodeSet), dtype=np.int32)
        np_all_edges = np.array(list(self.currentEdgeSet), dtype=np.int32) if self.currentEdgeSet else np.empty((0,2), dtype=np.int32)
        
        final_coords = numba_backward_pass_with_bfs(
            np_longest_path, self.totalNodes, loose_coords,
            np_f_node_list, np_f_edge_list, np_b_node_list, np_b_edge_list,
            np_all_edges
        )
        
        self.coordinateList = list(final_coords)
            

    def topological_traverse(self, operate=lambda x,y:True, forward=True, startNodes=None, degreeList=None):

        if forward:
            idx = 1                         
            search = self.findParentNodes            
            nodeList = self.forwardNodeList              
            edgeList = self.forwardEdgeList           
        else:
            idx = 0                        
            search = self.findChildNodes            
            nodeList = self.backwardNodeList              
            edgeList = self.backwardEdgeList           
        if degreeList is None:
            degreeList = [0] * self.totalNodes
            for link in self.currentEdgeSet:
                degreeList[link[idx]] += 1
        if startNodes is not None:
            q = set()
            for node in startNodes:
                if degreeList[node] == 0:
                    q.add(node)
        else:
            q = {i for i, x in enumerate(degreeList) if x == 0}
        q = deque(q)                          
        while q:
            cursorNode = q.popleft()
            neighbourNodes = search(cursorNode)
            operate(cursorNode, neighbourNodes)
            i = nodeList[cursorNode]            
            while i != -1:            
                targetNode = edgeList[i][0]          
                degreeList[targetNode] -= 1
                if degreeList[targetNode] == 0:
                    q.append(targetNode)
                i = edgeList[i][1]           


    def calculateCoordinates(self, check_loop=True, forward=True):
        np_all_edges = np.array(list(self.currentEdgeSet), dtype=np.int32) if self.currentEdgeSet else np.empty((0,2), dtype=np.int32)
        np_f_node_list = np.array(self.forwardNodeList, dtype=np.int32)
        np_f_edge_list = np.array(self.forwardEdgeList, dtype=np.int32) if self.forwardEdgeList else np.empty((0,3), dtype=np.int32)
        np_b_node_list = np.array(self.backwardNodeList, dtype=np.int32)
        np_b_edge_list = np.array(self.backwardEdgeList, dtype=np.int32) if self.backwardEdgeList else np.empty((0,3), dtype=np.int32)
        final_coords, indegree, outdegree = numba_calculate_coordinates_engine(
            self.totalNodes, np_all_edges,
            np_f_node_list, np_f_edge_list,
            np_b_node_list, np_b_edge_list,
            forward
        )
        if forward:
            self.coordinateList = list(final_coords)
        else:
            self.backwardCoordinateList = list(final_coords)
        
        self.startNodeSet = set(np.where(indegree == 0)[0])
        self.endNodeSet = set(np.where(outdegree == 0)[0])

        if check_loop:
            has_cycle = False
            for x in final_coords:
                if x == 0:
                    has_cycle = True
                    break
            if has_cycle:
                 raise RuntimeError('Cycle detected in graph structure')

        max_val = 0
        if len(final_coords) > 0:
            max_val = np.max(final_coords)
        self.maxlength = max_val

    def get_coordinate_forward(self, node: int,neighborNodes:List[int]) -> None:

        coordinates = [self.coordinateList[n] for n in neighborNodes]
        coordinates.append(0)           
        self.coordinateList[node] = max(coordinates) + 1
    def get_coordinate_backward(self, node: int,neighborNodes:List[int]) -> None:

        coordinates = [self.backwardCoordinateList[n] for n in neighborNodes]
        coordinates.append(0)         
        self.backwardCoordinateList[node] = max(coordinates) + 1

    
        

    def get_state_range_forward(self, node, neighborNodes):

        coordinates = [self.ref_coor[n][0] for n in neighborNodes]
        coordinates.append(self.ref_coor[node][0])
        self.ref_coor[node] = [max(coordinates), self.ref_coor[node][1]]
    def get_state_range_backward(self, node, childrennodes):

        coordinates = [self.ref_coor[n][1] for n in childrennodes]
        coordinates.append(self.ref_coor[node][1])
        self.ref_coor[node] = [self.ref_coor[node][0], min(coordinates)]

    def calculateStateRange(self, reflist, mode='hmm'):

        degreeList = [0] * self.totalNodes        
        outdegree_list = [0] * self.totalNodes        
        s = set(reflist)
        for link in self.currentEdgeSet:
            degreeList[link[1]] += 1
            outdegree_list[link[0]] += 1
        length = len(reflist)
        self.ref_coor = [[0, length]] * self.totalNodes         
        for index, node in enumerate(reflist):
            if node != 'x':
                self.ref_coor[node] = [index, index]               
        startNodeSet = [index for index, value in enumerate(degreeList) if value == 0]
        endNodeSet = [index for index, value in enumerate(outdegree_list) if value == 0]
        self.topological_traverse(startNodes=startNodeSet, operate=self.get_state_range_forward, degreeList=degreeList)
        self.topological_traverse(startNodes=endNodeSet, operate=self.get_state_range_backward, degreeList=outdegree_list,forward=False)

        if mode == 'hmm':
            new_range = []
            for node in range(self.totalNodes):
                childrens = self.findChildNodes(node)
                rights = [self.ref_coor[n][1] for n in childrens]
                rights.append(self.ref_coor[node][1])             
                new_range.append([self.ref_coor[node][0], max(rights)])
            self.ref_coor = new_range
        return self.ref_coor
    

    
    
    def Tight_coordinate_system_update(self, headnodes,forward=True):
        substru = DAGStru(self.totalNodes,{link for link in self.currentEdgeSet if set(link)-headnodes==set()})
        if forward:
            tmpcoors = self.coordinateList.copy()  
        else:
            tmpcoors = self.backwardCoordinateList.copy()  
        indegreeList = np.full(substru.totalNodes, 0)  
        outdegree_list = np.full(substru.totalNodes, 0)  
        for link_start, link_target in substru.currentEdgeSet:
            indegreeList[link_target] += 1  
            outdegree_list[link_start] += 1  

        substru.endNodeSet = set(np.where(outdegree_list == 0)[0])&headnodes  
        substru.startNodeSet = set(np.where(indegreeList == 0)[0])&headnodes  

        for node in headnodes:
            tmpcoors[node]=0


        if forward:
            search = self.findChildNodes  
            nodeList = substru.backwardNodeList 
            edgeList = substru.backwardEdgeList 
            degreeList = outdegree_list
            st = substru.endNodeSet
        else:
            search = self.findParentNodes  
            nodeList = substru.forwardNodeList  
            edgeList = substru.forwardEdgeList 
            degreeList = indegreeList
            st = substru.startNodeSet 


        q = deque(st)  
        while q:
            cursorNode = q.popleft()
            if not tmpcoors[cursorNode]:
                neighbourNodes = search(cursorNode)
                if neighbourNodes:
                    tmpcoors[cursorNode] = min([tmpcoors[n] for n in neighbourNodes])-1
                else:
                    tmpcoors[cursorNode] = self.maxlength

            i = nodeList[cursorNode]  
            while i != -1:  
                targetNode = edgeList[i][0]  
                degreeList[targetNode] -= 1
                if degreeList[targetNode] == 0:
                    q.append(targetNode)
                i = edgeList[i][1]  

        return tmpcoors
