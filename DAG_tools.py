#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
from collections import deque,Counter
from DAG_stru import DAGStru
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm
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

    results = []
    i = 0
    n = len(lst)
    while i < n:
        if lst[i] == -1:
            start = i
            while i < n and lst[i] == -1:
                i += 1
            end = i - 1               
            prev_val = lst[start - 1] if start > 0 else 0
            next_val = lst[end + 1] if end < n - 1 else float('inf')
            results.append((start, end, prev_val, next_val))
        else:
            i += 1           
    return results
def replace_duplicates(lst: list[int], n: int = 3) -> list[int]:

    count = Counter(lst)             
    replace_set = {k for k, v in count.items() if v >= n}  
    return [-1 if x in replace_set else x for x in lst]  
def reduce_consecutive(lst: list[int], start_index=1):

    if not lst:
        return [], []
    result = [lst[0]]              
    indices = [[start_index]]                   
    for i, curr in enumerate(lst[1:], start=start_index):
        if curr != result[-1]:          
            result.append(curr)           
            indices.append([i + 1])              
        else:           
            indices[-1].append(i + 1)                     
    return result, indices
def find_long_negative_segments(data, threshold=-1, min_length=120):

    data = np.array(data)
    is_negative = data == threshold
    diff = np.diff(np.concatenate(([0], is_negative, [0])))
    start_indices = np.where(diff == 1)[0]            
    end_indices = np.where(diff == -1)[0] - 1               
    segments = [(start, end) 
               for start, end in zip(start_indices, end_indices) 
               if end - start + 1 > min_length]
    return segments
def search_increasing_blocks(numbers, min_length=10, max_dif=10):

    valid_indices = [(i, num) for i, num in enumerate(numbers) if num != -1]
    if not len(valid_indices) > 1:
        return []
    blocks = []
    start_idx, start_num = valid_indices[0]
    prev_num = start_num
    prev_idx = start_idx
    for curr_idx, curr_num in valid_indices[1:]:
        idx_diff = curr_idx - prev_idx
        num_diff = curr_num - prev_num
        if curr_num >= prev_num and abs(idx_diff - num_diff) < max_dif:
            pass
        else:
            block_length = prev_idx - start_idx + 1
            if block_length >= min_length:
                blocks.append((start_num, prev_num))
            start_idx, start_num = curr_idx, curr_num
        prev_num = curr_num
        prev_idx = curr_idx
    block_length = curr_idx - start_idx + 1                
    if block_length >= min_length:
        blocks.append((start_num, curr_num))
    return blocks
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
def calculate_column(column,length,positive,negative):
    num_dict={}
    sp_num=0
    for i in column[1:]:
        num_dict[i[0]]=len(i[1])
        sp_num+=num_dict[i[0]]
    if length-sp_num!=0:
        num_dict[column[0]]=length-sp_num
    if set(num_dict.keys())==set([0]):
        return 0,0,0
    for n in [n for n in num_dict.keys() if n > 4]:
        deg_base = degenerate_base_dict[n]
        for s in deg_base:
            num_dict[s] = num_dict.get(s, 0) + num_dict[n] / len(deg_base)
        del num_dict[n]
    n_score = 0
    p_score = 0
    Klist = num_dict.keys()
    doneset=set()
    for n in Klist:
        num = num_dict[n]
        n_score += (num - 1) * num / 2 * negative[n][n]
        p_score += (num - 1) * num / 2 * positive[n][n]
        doneset.add(n)
        for b in Klist-doneset:
            bnum = num_dict[b]
            n_score += num * bnum * negative[n][b]
            p_score += num * bnum * positive[n][b]
    entropy = 0
    for n in Klist:
        num = num_dict[n]
        if num != 0:
            P = num / length
            entropy += -P * np.log(P)
    return n_score,p_score,entropy
def sp_entro_zip_loaded(zip_ali,positive,negative):
    all_entropy = 0
    negative_score = 0
    positive_score=0
    sumnum = zip_ali[-1]
    length = len(zip_ali)-1
    for i in range(length):
        column = zip_ali[i]
        n_score,p_score,entropy = calculate_column(column,sumnum,positive,negative)
        negative_score += n_score
        positive_score += p_score
        all_entropy += entropy
    scaled_sp  = 2*(positive_score+negative_score)/(sumnum*(sumnum-1)*length)
    return scaled_sp,all_entropy
def sp_entro_zip(inpath,positive,negative):
    Binpath = inpath
    zipaliandnamelist = np.load(Binpath,mmap_mode='r',allow_pickle=True)
    zip_ali = zipaliandnamelist['align'].tolist()
    all_entropy = 0
    negative_score = 0
    positive_score=0
    sumnum = zip_ali[-1]
    length = len(zip_ali)-1
    for i in range(length):
        column = zip_ali[i]
        n_score,p_score,entropy = calculate_column(column,sumnum,positive,negative)
        negative_score += n_score
        positive_score += p_score
        all_entropy += entropy
    scaled_sp  = 2*(positive_score+negative_score)/(sumnum*(sumnum-1)*length)
    return scaled_sp,all_entropy
def save_fasta(gp_path,pc_name,save_path):
    print(save_path+'result.fastaa')
    draw_dict = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
    zipaliandnamelist = np.load(gp_path+'V_result/alizips/{}/zipalign.npz'.format(pc_name),allow_pickle=True)
    zipali = zipaliandnamelist['align'].tolist()[:-1]
    namelist = zipaliandnamelist['namelist']
    ali_matrix = np.full((len(namelist),len(zipali)),0)
    for index,ali in enumerate(zipali):
        ali_matrix[:,index] = ali[0]
        for base in ali[1:]:
            ali_matrix[base[1],index] = base[0]
    vectorized_draw_dict = np.vectorize(draw_dict.get)
    string_matrix = vectorized_draw_dict(ali_matrix)
    xs = np.arange(len(string_matrix))
    seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in tqdm(zip(xs,string_matrix))]
    SeqIO.write(seqlist,save_path,'fasta')
def save_fasta_with_ref(gp_path,pc_name,save_path):
    draw_dict = {0:'-',1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'R', 6: 'Y', 7: 'M', 8: 'K', 9: 'S', 10: 'W', 11: 'H', 12: 'B', 13: 'V', 14: 'D', 15: 'N'}
    zipaliandnamelist = np.load(gp_path+'V_result/alizips/{}/zipalign.npz'.format(pc_name),allow_pickle=True)
    zipali = zipaliandnamelist['align'].tolist()[:-1]
    namelist = zipaliandnamelist['namelist'].tolist()
    ali_matrix = np.full((len(namelist)+1,len(zipali)),0)
    for index,ali in enumerate(zipali):
        ali_matrix[:,index] = ali[0]
        for base in ali[1:]:
            ali_matrix[base[1]+1,index] = base[0]
    vectorized_draw_dict = np.vectorize(draw_dict.get)
    string_matrix = vectorized_draw_dict(ali_matrix)
    xs = np.arange(len(string_matrix))
    namelist.insert(0,'ref')
    seqlist = [SeqRecord(Seq(''.join(i)),id=namelist[idx],description='') for idx,i in tqdm(zip(xs,string_matrix))]
    SeqIO.write(seqlist,save_path,'fasta')


def find_max_weight_combination(nested_list):

    n = len(nested_list)
    if n == 0:
        return (-1, [], [])

    dp = {i: [] for i in range(n)}
    
    for i in range(n):
        if not nested_list[i]:
            continue
        candidates = []
        max_local = max(item[2] for item in nested_list[i]) if nested_list[i] else -1
        for item in nested_list[i]:
            if item[2] == max_local:
                candidates.append((item[2], item[1], -1, item))
        if candidates:
            best = max(candidates, key=lambda x: (x[0], x[1]))
            dp[i].append(best)

    for i in range(n):
        if not dp.get(i):
            continue
        for j in range(i):
            if not dp.get(j):
                continue
            for state_j in dp[j]:
                weight_j, last_coord_j, _, elem_j = state_j
                for item in nested_list[i]:
                    item_id, item_coord, item_weight = item
                    if item_coord > last_coord_j:
                        new_weight = weight_j + item_weight
                        existing = [s for s in dp[i] if s[0] >= new_weight and s[1] >= item_coord]
                        if not existing:
                            dp[i].append((new_weight, item_coord, j, item))
        if dp[i]:
            dp[i].sort(key=lambda x: (-x[0], -x[1]))
            filtered = []
            max_weight = dp[i][0][0]
            max_coord = dp[i][0][1]
            for s in dp[i]:
                if s[0] == max_weight and s[1] <= max_coord:
                    continue
                if s[0] < max_weight and s[1] <= max_coord:
                    continue
                filtered.append(s)
                if s[0] > max_weight:
                    max_weight = s[0]
                    max_coord = s[1]
                elif s[0] == max_weight and s[1] > max_coord:
                    max_coord = s[1]
            dp[i] = filtered

    max_weight = -1
    best_path = []
    for i in range(n):
        for state in dp.get(i, []):
            if state[0] > max_weight or (state[0] == max_weight and state[1] > best_path[-1][1] if best_path else False):
                max_weight = state[0]
                path = []
                current = state
                while current is not None:
                    path.append((i if current[2] == -1 else current[2], current[3]))
                    prev_idx = current[2]
                    current = dp[prev_idx][0] if prev_idx != -1 and dp[prev_idx] else None
                best_path = list(reversed(path))

    result_ids = [-1] * n
    result_coords = [-1] * n
    for entry in best_path:
        sub_idx, item = entry
        result_ids[sub_idx] = item[0]
        result_coords[sub_idx] = item[1]

    return (max_weight if max_weight != -1 else -1, result_ids, result_coords)