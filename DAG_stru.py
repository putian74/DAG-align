#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
from DAG_stru import DAGStru
from pathlib import Path


degenerate_base_dict = {0:[0],1:[1],2:[2],3:[3],4:[4],5: [1, 4], 6: [3, 2], 7: [1, 3], 8: [4, 2], 9: [4, 3], 10: [1, 2], 11: [1, 2, 3], 12: [4, 2, 3], 13: [4, 1, 3], 14: [4, 1, 2], 15: [1, 2, 3, 4]}
reverse_base_dict = {'-': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4, 'R': 5, 'Y': 6, 'M': 7, 'K': 8, 'S': 9, 'W': 10, 'H': 11, 'B': 12, 'V': 13, 'D': 14, 'N': 15}
basedict ={0: '-', 1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}

def get_first_number(num, firstBit=32, allBit=64):

    if not (isinstance(firstBit, int) and isinstance(allBit, int)):
        raise TypeError("位参数必须为整数")
    if allBit < firstBit:
        raise ValueError(f"总位数{allBit}不能小于需提取的位数{firstBit}")
    mask1 = (1 << firstBit) - 1
    return np.bitwise_and(
        np.right_shift(num, allBit - firstBit),          
        mask1                                              
    )

def get_second_number(num, firstBit=32, allBit=64):

    if not (isinstance(firstBit, int) and isinstance(allBit, int)):
        raise TypeError("位参数必须为整数")
    if allBit < firstBit:
        raise ValueError(f"总位数{allBit}不能小于需提取的位数{firstBit}")
    mask2 = (1 << (allBit - firstBit)) - 1
    return np.bitwise_and(num, mask2)

def save_numbers(num1, num2, firstBit=32, allBit=64):

    if not (isinstance(firstBit, int) and isinstance(allBit, int)):
        raise TypeError("位参数必须为整数")
    if allBit < firstBit:
        raise ValueError(f"总位数{allBit}不能小于需提取的位数{firstBit}")
    shift_bits = allBit - firstBit
    return np.bitwise_or(
        np.left_shift(num1, shift_bits),          
        num2                                        
    )

def array_to_block(arr, allow_gap=False):
    def _handle_gap_mode(valid):
        blocks = []
        weights = []
        differences = []
        start_idx, start_val = valid[0]
        prev_idx, prev_val = start_idx, start_val
        count = 1
        for i in range(1, len(valid)):
            curr_idx, curr_val = valid[i]
            if curr_val == prev_val + (curr_idx - prev_idx):
                count += 1
                prev_idx, prev_val = curr_idx, curr_val
            else:
                blocks.append([[start_val, start_idx], [prev_val, prev_idx]])
                weights.append(count)
                differences.append(start_val - start_idx)
                start_idx, start_val = curr_idx, curr_val
                prev_idx, prev_val = curr_idx, curr_val
                count = 1
        blocks.append([[start_val, start_idx], [prev_val, prev_idx]])
        weights.append(count)
        differences.append(start_val - start_idx)
        return blocks, weights, differences
    
    def _handle_strict_mode(valid):
        blocks = []
        weights = []
        differences = []
        start_idx, start_val = valid[0]
        prev_idx, prev_val = start_idx, start_val
        for i in range(1, len(valid)):
            curr_idx, curr_val = valid[i]
            if curr_val == prev_val + 1 and curr_idx == prev_idx + 1:
                prev_idx, prev_val = curr_idx, curr_val
            else:
                blocks.append([[start_val, start_idx], [prev_val, prev_idx]])
                weights.append(prev_val - start_val)
                differences.append(start_val - start_idx)
                start_idx, start_val = curr_idx, curr_val
                prev_idx, prev_val = curr_idx, curr_val
        blocks.append([[start_val, start_idx], [prev_val, prev_idx]])
        weights.append(prev_val - start_val)
        differences.append(start_val - start_idx)
        return blocks, weights, differences
    valid = [(i, v) for i, v in enumerate(arr) if v != -1]
    if not valid:
        return [], [], []
    if allow_gap:
        return _handle_gap_mode(valid)
    else:
        return _handle_strict_mode(valid)

def remove_points_to_increase(lst, weights):

    n = len(lst)
    dp = [0] * n
    prev = [-1] * n
    for i in range(n):
        dp[i] = weights[i]
        for j in range(i):
            if lst[j][1][0] < lst[i][0][0] and dp[j] + weights[i] > dp[i]:
                dp[i] = dp[j] + weights[i]           
                prev[i] = j          
    max_weight = max(dp)
    index = dp.index(max_weight)
    result = []
    while index != -1:
        result.append(index)
        index = prev[index]
    removed_indices = [i for i in range(n) if i not in result]
    cp_rg = []
    for i in removed_indices:
        cp_rg.append(lst[i])
    return cp_rg


def Cyclic_Anchor_Combination_Detection(Block_list, Block_dif, Block_weight):
    cp_rg = []
    fixflag = 0
    block_num = len(Block_dif)                  
    for i in range(1, block_num):
        if Block_list[i-1][1][0] >= Block_list[i][0][0]:           
            fixflag = 1
            break               
    if fixflag == 1:
        cp_rg = remove_points_to_increase(Block_list, Block_weight)
    return cp_rg

_DNA_PATTERN = re.compile(r'^[ATCG]*$', re.IGNORECASE)

def no_degenerate(sequence):

    if not isinstance(sequence, str):
        return False
    return bool(_DNA_PATTERN.fullmatch(sequence.upper()))

def find_consecutive_negatives(lst):
    arr = np.asarray(lst)
    mask = arr == -1
    if not np.any(mask):
        return []

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if mask[0]: 
        starts = np.insert(starts, 0, 0)
    if mask[-1]: 
        ends = np.append(ends, len(arr) - 1)

    results = []
    for s, e in zip(starts, ends):
        prev_val = arr[s - 1] if s > 0 else 0
        next_val = arr[e + 1] if e + 1 < len(arr) else float('inf')
        results.append([s, e, prev_val, next_val])

    return results

def merge_intervals(intervals):

    if not intervals:
        return []
    intervals.sort(key=lambda x: x[2])
    merged = []              
    tuples = []              
    for interval in intervals:
        if not merged or merged[-1][1] < interval[2]:
            merged.append(interval[2:])
            tuples.append([interval[:2]])
        else:
            merged[-1][1] = max(merged[-1][1], interval[3])
            tuples[-1].append(interval[:2])
    return list(zip(merged, tuples))


def build_coarse_grained_graph(DAGStru: DAGStru, edgeWeightDict):

    indegree_list = [0] * DAGStru.totalNodes                             
    outdegree_list_static = [0] * DAGStru.totalNodes                   
    for outdegree, indegree in DAGStru.currentEdgeSet:
        indegree_list[indegree] += 1
        outdegree_list_static[outdegree] += 1
    indegree_list_static = indegree_list.copy()                  
    new_start = set()
    for node in DAGStru.startNodeSet:
        if node != 'x':                           
            if indegree_list[node] == 0:                
                new_start.add(node)
    from collections import deque
    q = deque(new_start)    
    nodeID_linearPathID_Dict = np.full(DAGStru.totalNodes, -1)                              
    linearPath_list = []                            
    tmp_lst = []                                   
    cid = 0                           
    while q:
        start_node = q.pop()
        tmp_lst = [start_node]           
        while outdegree_list_static[start_node] == 1:
            start_node = DAGStru.findChildNodes(start_node)[0]  
            if indegree_list_static[start_node] == 1:  
                tmp_lst.append(start_node)
            else:
                break                
        linearPath_list.append(tmp_lst)
        nodeID_linearPathID_Dict[np.array(tmp_lst)] = cid                       
        cid += 1
        for child_node in DAGStru.findChildNodes(tmp_lst[-1]):
            indegree_list[child_node] -= 1             
            if indegree_list[child_node] == 0:                
                q.append(child_node)
    tail_dict = {}                      
    for path_id, path_nodes in enumerate(linearPath_list):
        tail_dict[path_nodes[-1]] = path_id
    linearPath_link = {}
    for path_id, path_nodes in enumerate(linearPath_list):
        for father_node in DAGStru.findParentNodes(path_nodes[0]):
            parent_path_id = tail_dict[father_node]
            linearPath_link[(parent_path_id, path_id)] = edgeWeightDict[(father_node, path_nodes[0])]
    return linearPath_list, linearPath_link, nodeID_linearPathID_Dict


def build_subgraphStru(subLinkDict: dict[(int, int): int]):

    idDict = {}                                     
    links = {}                                
    nodeset = set()                
    newid = 0                         
    for link in subLinkDict.items():
        newlink = []                  
        for oriNode in link[0]:
            if oriNode in nodeset:
                newlink.append(idDict[oriNode])
            else:
                nodeset.add(oriNode)
                idDict[oriNode] = newid
                newlink.append(idDict[oriNode])
                newid += 1           
        links[tuple(newlink)] = link[1]                 
    subGraphStru = DAGStru(len(idDict), links.keys())
    subGraphStru.calculateCoordinates()
    return subGraphStru, idDict

def sanitize_path(path: str, path_type: str='output') -> Path:

    try:
        clean_path = Path(path).resolve().absolute()
        if path_type == 'input' and not clean_path.exists():
            raise FileNotFoundError(f"输入路径不存在: {path}")
        if path_type == 'output':
            parent = clean_path.parent
            if not parent.exists():
                parent.mkdir(parents=True, mode=0o755, exist_ok=True)
        forbidden_patterns = [
            '..',        
            '//',        
            '~',        
            '\x00'      
        ]
        if any(p in str(clean_path) for p in forbidden_patterns):
            raise ValueError(f"路径包含非法特征: {path}")
        return clean_path
    except Exception as e:
        raise ValueError(f"路径处理失败 ({path_type}): {str(e)}") from e
    

def calculate_column(column_data, length, positive, negative):
    num_dict = {}
    sp_num = 0
    for i in column_data[1:]:
        num_dict[i[0]] = len(i[1])
        sp_num += num_dict[i[0]]
    if length - sp_num != 0:
        num_dict[column_data[0]] = length - sp_num

    if set(num_dict.keys()) == {0}:
        return 0, 0, 0, length 

    for n in [k for k in num_dict.keys() if k > 4]:
        deg_base = degenerate_base_dict[n]
        count_to_distribute = num_dict[n]
        for s in deg_base:
            num_dict[s] = num_dict.get(s, 0) + count_to_distribute / len(deg_base)
        del num_dict[n]

    n_score, p_score = 0, 0
    Klist = list(num_dict.keys())
    
    for i in range(len(Klist)):
        n1 = Klist[i]
        num1 = num_dict[n1]
        p_score += num1 * (num1 - 1) / 2 * positive[n1][n1]
        n_score += num1 * (num1 - 1) / 2 * negative[n1][n1]
        for j in range(i + 1, len(Klist)):
            n2 = Klist[j]
            num2 = num_dict[n2]
            p_score += num1 * num2 * positive[n1][n2]
            n_score += num1 * num2 * negative[n1][n2]

    entropy = 0
    for num in num_dict.values():
        if num > 0:
            P = num / length
            entropy += -P * np.log(P)
            
    gap_count = num_dict.get(0, 0)
    return n_score, p_score, entropy, gap_count

def sp_entro_zip_loaded(zip_ali, positive, negative, gap_threshold=0.7):
    total_entropy = 0
    negative_score = 0
    positive_score = 0
    total_core_entropy = 0
    num_core_columns = 0
    
    sumnum = zip_ali[-1]
    length = len(zip_ali) - 1
    if length == 0 or sumnum < 2:
        return 0.0, 0.0, 0.0

    for i in range(length):
        column = zip_ali[i]
        n_score, p_score, entropy, gap_count = calculate_column(column, sumnum, positive, negative)
        
        negative_score += n_score
        positive_score += p_score
        total_entropy += entropy
        
        if (gap_count / sumnum) < gap_threshold:
            num_core_columns += 1
            total_core_entropy += entropy

    scaled_sp = 2 * (positive_score + negative_score) / (sumnum * (sumnum - 1) * length)
    average_core_entropy = total_core_entropy / num_core_columns if num_core_columns > 0 else 0.0
    
    return scaled_sp, total_entropy, average_core_entropy

def sp_entro_zip(inpath,positive,negative, gap_threshold=0.7):
    Binpath = inpath
    zipaliandnamelist = np.load(Binpath,mmap_mode='r',allow_pickle=True)
    zip_ali = zipaliandnamelist['align'].tolist()
    total_entropy = 0
    negative_score = 0
    positive_score = 0
    total_core_entropy = 0
    num_core_columns = 0
    
    sumnum = zip_ali[-1]
    length = len(zip_ali) - 1
    if length == 0 or sumnum < 2:
        return 0.0, 0.0, 0.0

    for i in range(length):
        column = zip_ali[i]
        n_score, p_score, entropy, gap_count = calculate_column(column, sumnum, positive, negative)
        
        negative_score += n_score
        positive_score += p_score
        total_entropy += entropy
        
        if (gap_count / sumnum) < gap_threshold:
            num_core_columns += 1
            total_core_entropy += entropy

    scaled_sp = 2 * (positive_score + negative_score) / (sumnum * (sumnum - 1) * length)
    average_core_entropy = total_core_entropy / num_core_columns if num_core_columns > 0 else 0.0
    
    return scaled_sp, total_entropy, average_core_entropy


def find_max_weight_combination(nested_list):

    n = len(nested_list)
    dp = [(-1, [-1] * n, [-1] * n) for _ in range(n)] 

    for i in range(n):
        best_weight = 0
        best_ids = [-1] * n
        best_coords = [-1] * n
        for current in nested_list[i]:
            node_id, coord, weight = current


            max_prev_weight = 0
            best_prev_ids = [-1] * n
            best_prev_coords = [-1] * n
            for j in range(i):
                prev_weight, prev_ids, prev_coords = dp[j]
                if prev_weight == -1:
                    continue
                last_coord = prev_coords[j]
                if last_coord == -1 or last_coord < coord:
                    if prev_weight > max_prev_weight:
                        max_prev_weight = prev_weight
                        best_prev_ids = prev_ids
                        best_prev_coords = prev_coords

            new_ids = best_prev_ids[:]
            new_coords = best_prev_coords[:]
            new_ids[i] = node_id
            new_coords[i] = coord
            total_weight = max_prev_weight + weight

            if total_weight > best_weight:
                best_weight = total_weight
                best_ids = new_ids
                best_coords = new_coords

        dp[i] = (best_weight, best_ids, best_coords)

    best_result = max(dp, key=lambda x: x[0])
    return best_result[1], best_result[2]
