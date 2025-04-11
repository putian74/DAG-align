#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, Dict, Set, KeysView, List
from collections import deque
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
    def update(self, totalNodes: int) -> None:

        self.totalNodes = totalNodes
        if len(self.forwardNodeList) != self.totalNodes:
            self.forwardNodeList.extend([-1] * (self.totalNodes - len(self.forwardNodeList)))
        if len(self.backwardNodeList) != self.totalNodes:
            self.backwardNodeList.extend([-1] * (self.totalNodes - len(self.backwardNodeList)))
        if self.newEdges != set():
            for link in self.newEdges:
                self.add_edge_to_queryGraph(link[0], link[1])
            self.currentEdgeSet |= self.newEdges
            self.newEdges = set()
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
    def calculate_degreeList(self, startNode, forward=True):

        direction_suffix = "forward" if forward else "backward"
        nodeList = getattr(self, f"{direction_suffix}NodeList")            
        edgeList = getattr(self, f"{direction_suffix}EdgeList")           
        degreeList = np.full(self.totalNodes, 0, dtype=np.int32)
        doneList = np.full(self.totalNodes, 0, dtype=np.uint8)
        q = deque(startNode if isinstance(startNode, list) else [startNode])
        while q:
            cursorNode = q.popleft()
            if doneList[cursorNode] == 0:
                edge_ptr = nodeList[cursorNode]
                while edge_ptr != -1:
                    target_node = edgeList[edge_ptr][0]            
                    degreeList[target_node] += 1             
                    q.append(target_node)
                    edge_ptr = edgeList[edge_ptr][1]
                doneList[cursorNode] = 1
        return degreeList
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
    def find_max_weight_path(self):
        max_weight = [float('-inf')] * self.totalNodes
        max_path = {i: [] for i in range(self.totalNodes)}

        degreeList = np.full(self.totalNodes, 0)
        for _, link_target in self.currentEdgeSet:
            degreeList[link_target] += 1
        self.startNodeSet = set(np.where(degreeList == 0)[0])

        for node in self.startNodeSet:
            max_weight[node] = 0

        def topological_sort():

            in_degree = degreeList[:] 
            stack = list(self.startNodeSet)  
            topo_order = []  
            
            while stack:
                node = stack.pop()
                topo_order.append(node)
                idx = self.forwardNodeList[node]
                while idx != -1:
                    neighbor = self.forwardEdgeList[idx][0]
                    in_degree[neighbor] -= 1 
                    if in_degree[neighbor] == 0:
                        stack.append(neighbor)
                    idx = self.forwardEdgeList[idx][1]
            return topo_order

        topo_order = topological_sort()

        for node in topo_order:
            idx = self.forwardNodeList[node]
            while idx != -1:
                neighbor, _, weight = self.forwardEdgeList[idx]
                if max_weight[node] + weight > max_weight[neighbor]:
                    max_weight[neighbor] = max_weight[node] + weight
                    max_path[neighbor] = max_path[node].copy() + [node]
                idx = self.forwardEdgeList[idx][1]

        max_weight_value = max(max_weight)
        max_node = max_weight.index(max_weight_value)
        max_path_result = max_path[max_node] + [max_node]

        return max_weight_value, max_path_result

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
    def push_coordinate(self, start_nodes, forward=True, degreeList=None):

        direction_suffix = "forward" if forward else "backward"
        nodeList = getattr(self, f"{direction_suffix}NodeList")          
        edgeList = getattr(self, f"{direction_suffix}EdgeList")         
        operate = self.findParentNodes if forward else self.findChildNodes              
        idx = 1 if forward else 0          
        if degreeList is None:
            degreeList = [0] * self.totalNodes
            for link in self.currentEdgeSet:
                degreeList[link[idx]] += 1  
        else:
            degreeList = list(degreeList)         
        q = deque({n for n in start_nodes if degreeList[n] == 0})  
        coordinate_list = self.coordinateList  
        while q:
            cursorNode = q.popleft()
            neighbours = operate(cursorNode)          
            max_coord = max([coordinate_list[n] for n in neighbours]) if neighbours else -1
            coordinate_list[cursorNode] = max_coord + 1        
            edge_ptr = nodeList[cursorNode]
            while edge_ptr != -1:
                target_node = edgeList[edge_ptr][0]           
                degreeList[target_node] -= 1        
                if degreeList[target_node] == 0:
                    q.append(target_node)            
                edge_ptr = edgeList[edge_ptr][1]           
    def calculateCoordinates(self, check_loop: bool = True, forward: bool = True) -> None:

        if forward:
            self.coordinateList = [0] * self.totalNodes          
            coordinateList = self.coordinateList
        else:
            self.backwardCoordinateList = [0] * self.totalNodes          
            coordinateList = self.backwardCoordinateList
        indegreeList = np.full(self.totalNodes, 0)          
        outdegree_list = np.full(self.totalNodes, 0)          
        for link_start, link_target in self.currentEdgeSet:
            indegreeList[link_target] += 1            
            outdegree_list[link_start] += 1            
        self.endNodeSet = set(np.where(outdegree_list == 0)[0])             
        self.startNodeSet = set(np.where(indegreeList == 0)[0])             
        choose_degreeList = indegreeList if forward else outdegree_list               
        stnode = self.startNodeSet if forward else self.endNodeSet         
        operate = self.get_coordinate_forward if forward else self.get_coordinate_backward          
        for node in self.startNodeSet:
            coordinateList[node] = 1              
        self.topological_traverse(
            operate=operate,
            startNodes=stnode, 
            forward=forward,
            degreeList=choose_degreeList                
        )
        if check_loop and (0 in coordinateList):
            raise RuntimeError('Cycle detected in graph structure')
        self.maxlength = max(self.coordinateList)
    def get_coordinate_forward(self, node: int,neighborNodes:List[int]) -> None:

        coordinates = [self.coordinateList[n] for n in neighborNodes]
        coordinates.append(0)           
        self.coordinateList[node] = max(coordinates) + 1
    def get_coordinate_backward(self, node: int,neighborNodes:List[int]) -> None:

        coordinates = [self.backwardCoordinateList[n] for n in neighborNodes]
        coordinates.append(0)         
        self.backwardCoordinateList[node] = max(coordinates) + 1
    def local_update_coordinate(self, startnodes):

        degreeList = self.calculate_degreeList(startnodes)
        self.push_coordinate(set(startnodes), degreeList=degreeList)
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