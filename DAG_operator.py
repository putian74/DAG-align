# -*- coding: utf-8 -*-
from DAG_info import *
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from File_operateor import *
from multiprocessing import Manager,Process,Queue,Value,Lock
import time
import os
import sys
import numpy as np
import shutil
def load_DAG(graphpath, load_onmfile=False):

    graphpath = sanitize_path(graphpath,'input')
    graph_file = open(graphpath / 'graph.pkl', 'rb')               
    graph = pickle.load(graph_file)           
    graph_data = np.load(graphpath / 'data.npz', allow_pickle=True)
    graph.nodeList = graph_data['nodeList'].tolist()                     
    graph.totalNodes = len(graph.nodeList)          
    graph.edgeWeightDict = graph_data['edgeWeightDict']           
    graph.fragmentNodeDict = {}
    graph.startFragmentNodeDict = {}
    graph.endFragmentNodeDict = {}
    for node in graph.nodeList:
        if node.ishead:
            graph.startFragmentNodeDict.setdefault(node.fragment, []).append(node.id)
        if node.istail:
            graph.endFragmentNodeDict.setdefault(node.fragment, []).append(node.id)
        if node.ishead+node.istail==0:
            graph.fragmentNodeDict.setdefault(node.fragment, []).append(node.id)
    graph.startNodeSet = set(graph_data['startNodeSet'].tolist())               
    graph.endNodeSet = set(graph_data['endNodeSet'].tolist())               
    graph.edgeWeightDict = { 
        (link_weight[0], link_weight[1]): link_weight[2] 
        for link_weight in graph.edgeWeightDict
    }
    graph.queryGraph = DAGStru(graph.totalNodes, graph.edgeWeightDict)
    graph.queryGraph.calculateCoordinates()           
    if load_onmfile:
        ori_node_list = np.load(graphpath / 'onm.npy', allow_pickle=True)
        ori_node_index = np.load(graphpath / 'onm_index.npy')
        for index, node in enumerate(graph.nodeList):
            oindex = ori_node_index[index]                 
            node.Source = ori_node_list[oindex[0]:oindex[1]]             
    graph_file.close()         
    return graph
def build_fragment_graph(inPath, savePath, fragmentLength,placeholderA=None,placeholderB=None, graphId='1'):
    os.makedirs(savePath,exist_ok=True)
    Sequence_record_list = SeqIO.parse(inPath, "fasta")
    seqs_list = []
    for sequence_record in tqdm(Sequence_record_list, desc="load"):
        seqs_list.append(sequence_record)                       
    def build(seqs_list):
        graph = None
        for idx, sequence_record in enumerate(tqdm(seqs_list, desc="construction")):
            sequence = str(sequence_record.seq).upper()          
            seq_id = sequence_record.id
            if idx == 0:
                graph = DAGInfo(inPath, graphId, 
                              savePath=savePath,
                              fragmentLength=fragmentLength)
                graph.add_first_sequence(seq_id, sequence)         
            else:
                graph.add_fragment_seq(sequence, seq_id)  
        if graph:         
            graph.map_to_OSM()          
            graph.queryGraph.update(graph.totalNodes)   
            graph.save_graph(mode='build', 
                           maxdistant=100,            
                           external_sources=True)            
    build(seqs_list)          
def build_and_merge(outpath, fasta_files, lock, todo, merge_dict, goon_flag, q, buildFunction, mergeFunction,tracingFunction=lambda x,y,z:True, fragmentLength=16, compassGenes=None, primaryGenes=None):
    
    outpath = sanitize_path(outpath,'output')
    subgraphs_path = outpath.joinpath('subgraphs')
    mergegraphs_path = outpath.joinpath('Merging_graphs')
    while goon_flag.value:
        try:
            target_graph = q.get_nowait()
        except:
            time.sleep(5)
            target_graph = None
        while target_graph != None:
            if target_graph[0] == 0:
                new_gidx = target_graph[1]
                if new_gidx < 80:
                    time.sleep(10 * new_gidx)
                graph_dir = subgraphs_path / '{}'.format(new_gidx)
                if not os.path.exists(graph_dir):
                    os.mkdir(graph_dir)
                outlog = open(graph_dir / 'out.log', 'w')
                sys.stdout = outlog
                sys.stderr = outlog
                buildFunction(
                    fasta_files[new_gidx-1],             
                    graph_dir,
                    fragmentLength,
                    compassGenes,
                    primaryGenes,
                    graphId = str(new_gidx)
                )
                outlog.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                next_gidx = (new_gidx // 2) if (new_gidx % 2 == 0) else ((new_gidx + 1) // 2)
                with lock:
                    key = (1, next_gidx)
                    if key in merge_dict.keys():
                        newvalue = merge_dict[key][:-1]+[merge_dict[key][-1]-1]
                        merge_dict[key] = newvalue         
                        if merge_dict[key][-1] == 0:
                            q.put(key)            
                    todo.value -= 1           
                    target_graph = None
            elif target_graph[0] != 'x':
                stage = target_graph[0]
                current_gidx = target_graph[1]
                ori_path = subgraphs_path if stage == 1 else mergegraphs_path / 'merge_{}'.format(stage-1)
                new_path = mergegraphs_path / 'merge_{}'.format(stage)
                with lock:
                    if not os.path.exists(new_path):
                        os.mkdir(new_path)
                ori_gidx_list = merge_dict[(stage, current_gidx)]
                new_graph_path = new_path / str(current_gidx)
                if len(ori_gidx_list) == 2:               
                    if os.path.exists(new_graph_path):
                        shutil.rmtree(new_graph_path)
                    shutil.copytree(ori_path / str(ori_gidx_list[0]), new_graph_path)
                else:
                    Agraph_path = ori_path/str(ori_gidx_list[0])
                    Bgraph_path = ori_path/str(ori_gidx_list[1])
                    if not os.path.exists(new_graph_path):
                        os.mkdir(new_graph_path)
                    outlog = open(new_graph_path / 'out.log', 'w')
                    sys.stdout = outlog
                    sys.stderr = outlog
                    mergeFunction(Agraph_path, Bgraph_path, new_graph_path)
                    tracingFunction(Agraph_path, Bgraph_path, new_graph_path)
                    outlog.close()
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__
                next_gidx = (current_gidx // 2) if (current_gidx % 2 == 0) else ((current_gidx + 1) // 2)
                with lock:
                    key = (stage+1, next_gidx)
                    if key in merge_dict:
                        newvalue = merge_dict[key][:-1]+[merge_dict[key][-1]-1]
                        merge_dict[key] = newvalue         
                        if merge_dict[key][-1] == 0:
                            q.put(key)
                    todo.value -= 1
                    target_graph = None
def is_end(goon_flag, todo, all_task):

    start = time.time()           
    while True:
        time.sleep(1)            
        runtime = time.time() - start
        hours, remainder = divmod(runtime, 3600)
        mins, secs = divmod(remainder, 60)
        time_format = '{:02d}:{:02d}:{:02d}'.format(int(hours), int(mins), int(secs))
        completed = all_task - todo.value
        percent = np.round(completed / all_task, 5) if all_task != 0 else 0.0
        bar = ('#' * int(percent * 20)).ljust(20)              
        sys.stdout.write(f'\r[{bar}] {percent * 100:.2f}%  ( {time_format} )')
        sys.stdout.flush()             
        if todo.value == 0:
            goon_flag.value = 0            
            break
def graph_construction(outpath, fasta_files, buildFunction, mergeFunction,tracingFunction=lambda x,y,z:True ,fragmentLength=16, compassGenes=None, primaryGenes=None, threads=20):

    outpath = sanitize_path(outpath,'output')
    if not os.path.exists(outpath.joinpath('subgraphs')):
        os.mkdir(outpath.joinpath('subgraphs'))
    if not os.path.exists(outpath.joinpath('Merging_graphs')):                             
        os.mkdir(outpath.joinpath('Merging_graphs'))
    max_range = len(fasta_files)
    todo = Value('i',0)                 
    x = max_range
    hi = 1            
    merge_dict = {}                                
    while x != 1:              
        next_idx = 0              
        tmplst = []                  
        for i in range(1, x+1):
            tmplst.append(i)
            if len(tmplst) == 2:                   
                next_idx += 1
                tmplst.append(len(tmplst))                       
                merge_dict[(hi, next_idx)] = tmplst          
                last_hi = hi            
                last_id = next_idx            
                todo.value += 1           
                tmplst = []          
        if tmplst != []:
            next_idx += 1
            tmplst.append(len(tmplst))            
            merge_dict[(hi, next_idx)] = tmplst
            last_hi = hi
            last_id = next_idx
            todo.value += 1
        hi += 1            
        x = next_idx               
    if merge_dict != {}:
        final_graph = outpath / "Merging_graphs" / f"merge_{last_hi}" / str(last_id)
    else:                   
        final_graph = outpath / 'subgraphs/1'
        last_hi = 0
        last_id = 1
    merge_dict = Manager().dict(merge_dict)
    lock = Lock()       
    goon_flag = Value('i',1)                     
    q = Queue()        
    for i in range(1, max_range+1):
        q.put([0,i])                           
        todo.value += 1           
    all_task = todo.value          
    processlist = []
    pool_num = threads
    processlist.append(Process(target=is_end, args=(goon_flag, todo, all_task)))
    for idx in range(pool_num):
        processlist.append(Process(target=build_and_merge, 
                                 args=(outpath, fasta_files, lock, todo, merge_dict, goon_flag, q, buildFunction, mergeFunction,tracingFunction, fragmentLength, compassGenes, primaryGenes, )))
    [p.start() for p in processlist]
    [p.join() for p in processlist]
    return final_graph, last_hi, max_range
def find_anchor_target(gp_base, gp_add, base_main, add_main):
    tupset = set()
    base_seqs_dict = {}
    for node in base_main:
        fragment = gp_base.nodeList[node[0]].fragment
        base_seqs_dict[fragment] = base_seqs_dict.get(fragment, [])
        base_seqs_dict[fragment].append(node[0])
    Coordinate_list = []                
    anchor_list = []                   
    anchors = []                     
    for node in add_main:
        seq = gp_add.nodeList[node[0]].fragment
        optional_anchors = base_seqs_dict.get(seq, [])
        if optional_anchors:
            anchor = optional_anchors[0]               
            Coordinate_list.append(gp_base.queryGraph.coordinateList[anchor])
            anchor_list.append((node[0], anchor, seq))
            anchors.append(anchor)
        else:
            Coordinate_list.append(-1)
            anchors.append(-1)              
            anchor_list.append(-1)
    blocklist, weights, blockdif = array_to_block(Coordinate_list)

    block_weight = []
    for block in blocklist:
        st = block[0][1]          
        ed = block[1][1]          
        w = sum([gp_base.nodeList[i].weight for i in anchors[st:ed+1] if i != -1])
        block_weight.append(w)
    copy_rg = Cyclic_Anchor_Combination_Detection(blocklist, block_weight, blockdif)

    while copy_rg:
        for rg in copy_rg:
            start_idx = rg[0][1]
            end_idx = rg[1][1]
            for j in range(start_idx, end_idx + 1):
                Coordinate_list[j] = -1
                anchor_list[j] = -1

        blocklist, weights, blockdif = array_to_block(Coordinate_list)
        copy_rg = Cyclic_Anchor_Combination_Detection(blocklist, weights, blockdif)

    for anchor_tuple in anchor_list:
        if anchor_tuple != -1:
            tupset.add(anchor_tuple)
    return tupset
def anchor_into_base_graph(gp_base,gp_add,newpath,anchor_tuple_list):
    gp_base.dellist = []                         
    newnodeset = set()
    for id in range(gp_add.totalNodes):
        if anchor_tuple_list[id]==-1:
            addNode = gp_add.nodeList[id]
            new_base_node = DAGNode(addNode.fragment)        
            new_base_node.id = gp_base.totalNodes
            new_base_node.weight = 0
            newnodeset.add(new_base_node.id)
            gp_base.nodeList.append(new_base_node)
            gp_base.totalNodes+=1
            gp_base.queryGraph.coordinateList.append('')
            anchor_tuple_list[id] = new_base_node.id                              
            if addNode.ishead:
                gp_base.startFragmentNodeDict.setdefault(new_base_node.fragment,[]).append(new_base_node.id)
            if addNode.istail:
                gp_base.endFragmentNodeDict.setdefault(new_base_node.fragment,[]).append(new_base_node.id)
            if addNode.ishead+addNode.istail==0:
                gp_base.fragmentNodeDict.setdefault(new_base_node.fragment,[]).append(new_base_node.id)

    for node in tqdm(gp_add.nodeList):
        tgnode = anchor_tuple_list[node.id]           
        gp_base.nodeList[tgnode].ishead |= node.ishead
        if gp_base.nodeList[tgnode].ishead:
            gp_base.startNodeSet.add(tgnode)
        gp_base.nodeList[tgnode].istail += node.istail           
        if gp_base.nodeList[tgnode].istail>0:
            gp_base.endNodeSet.add(tgnode)
        gp_base.nodeList[tgnode].weight += node.weight
        gp_base.nodeList[tgnode].Source += node.Source
    
    for link in tqdm(gp_add.edgeWeightDict):         
        gp_base.edgeWeightDict[(anchor_tuple_list[link[0]],anchor_tuple_list[link[1]])]  =  gp_base.edgeWeightDict.get((anchor_tuple_list[link[0]],anchor_tuple_list[link[1]]),0)+link[2]

    gp_base.queryGraph = DAGStru(gp_base.totalNodes,gp_base.edgeWeightDict.keys())
    del gp_add.edgeWeightDict
    gp_base.sequenceNum += gp_add.sequenceNum
    gp_base.savepath = newpath
    return newnodeset
def init_graph_merge(graphfile, tmp_gindex):

    graph = load_DAG(graphfile)                     
    graph.reflist = graph.findMainPathNodes(0.5)                   
    for node in graph.nodeList:
        node.Source = [(tmp_gindex, node.id)]             
    main_nodes = graph.findMainPathNodes()                    
    main_list = sorted([
        [node_id, graph.queryGraph.coordinateList[node_id]] 
        for node_id in main_nodes 
        if graph.nodeList[node_id].ishead + graph.nodeList[node_id].istail == 0          
    ], key=lambda x: x[1])              
    if tmp_gindex == 1:                   
        newlinkset = []
        for edge_tuple, weight in graph.edgeWeightDict.items():
            newlinkset.append([edge_tuple[0], edge_tuple[1], weight])
        graph.edgeWeightDict = newlinkset           
    return graph, main_list
def Tracing_merge(graphfileA, graphfileB, newpath):

    Trace_path = np.load(newpath / 'Traceability_path.npy', allow_pickle=True)
    source_A = np.load(graphfileA / 'onm.npy', allow_pickle=True)
    index_A = np.load(graphfileA / 'onm_index.npy')
    source_B = np.load(graphfileB / 'onm.npy', allow_pickle=True)
    index_B = np.load(graphfileB / 'onm_index.npy')
    new_onm = np.full(source_A.size + source_B.size, 0, dtype=object)
    new_index = []          
    sources = [source_A, source_B]            
    index_sources = [index_A, index_B]           
    index_cursor = 0            
    for cource_nodeids in tqdm(Trace_path):               
        source_gid, source_nid = cource_nodeids[0]
        nid_index = index_sources[source_gid][source_nid]
        onms_list = [sources[source_gid][nid_index[0]:nid_index[1]]]          
        for source_nodeid in cource_nodeids[1:]:
            source_gid, source_nid = source_nodeid
            nid_index = index_sources[source_gid][source_nid]
            onms_list.append(sources[source_gid][nid_index[0]:nid_index[1]])
        onm = np.hstack(onms_list) 
        index = [index_cursor]         
        index_cursor += onm.size        
        index.append(index_cursor)                       
        new_index.append(index)
        new_onm[index[0]:index[1]] = onm        
    np.save(newpath / 'onm.npy', new_onm)
    np.save(newpath / 'onm_index.npy', np.array(new_index))
    print('onm saved')
    print('finished')              
    v_idA = np.load(graphfileA / 'v_id.npy')
    v_idB = np.load(graphfileB / 'v_id.npy')
    np.save(newpath / 'v_id.npy', list(v_idA) + list(v_idB))
    print('finished')              
def Trace_zip(inPath, zippath):

    Trace_path = np.load(zippath / 'Traceability_path.npy', allow_pickle=True)
    source = np.load(inPath / 'onm.npy', allow_pickle=True)           
    index_list = np.load(inPath / 'onm_index.npy')                           
    new_onm = []                     
    new_index = []           
    index_cursor = 0            
    for cource_nodeids in tqdm(Trace_path):               
        source_gid, source_nid = cource_nodeids[0]
        source_index = index_list[source_nid]           
        onms_list = list(source[source_index[0]:source_index[1]])            
        for source_nodeid in cource_nodeids[1:]:
            source_gid, source_nid = source_nodeid
            source_index = index_list[source_nid]
            onms_list.extend(list(source[source_index[0]:source_index[1]]))          
        index = [index_cursor]
        index_cursor += len(onms_list)              
        index.append(index_cursor)
        new_index.append(index)
        new_onm.extend(onms_list) 
    print('saving')
    np.save(zippath / 'onm.npy', np.array(new_onm))                
    np.save(zippath / 'onm_index.npy', np.array(new_index))
    print('onm saved')
    print('finished')          
def merge_graph(graphfileA, graphfileB, newpath):

    if not os.path.exists(newpath):
        os.mkdir(newpath)
    graphA, main_list_A = init_graph_merge(graphfileA, 0)
    graphB, main_list_B = init_graph_merge(graphfileB, 1)
    print(graphfileA)            
    print(graphfileB)
    anchored_pairs_A = find_anchor_target(graphA, graphB, main_list_A, main_list_B)
    anchored_pairs_B = find_anchor_target(graphB, graphA, main_list_B, main_list_A)            
    anchor_tuple_list = [-1 for node in range(graphB.totalNodes)]  
    k = 0           
    for tup in tqdm(anchored_pairs_A):
        if (tup[1], tup[0], tup[2]) in anchored_pairs_B:                       
            anchor_tuple_list[tup[0]] = tup[1]                
            k += 1
    ori_total_nodes = graphA.totalNodes                           
    anchor_into_base_graph(graphA, graphB, newpath, anchor_tuple_list)  
    graphA.graphID = graphA.graphID + '&' + graphB.graphID         
    graphA.sequenceIDList += graphB.sequenceIDList           
    graphA.fastaFilePathList.extend(graphB.fastaFilePathList)            

    graphA.queryGraph.calculateCoordinates()                      
    graphA.merge_sameFraNodes(newnode_id=ori_total_nodes,maxdistant=100)                     
    graphA.save_graph(mode='merge',maxdistant=100, savePath=newpath)                 
def fragmentDAG_mutibuild(inpath,outpath,fra,threads,chunk_size):
    inpath = sanitize_path(inpath,'input')
    outpath = sanitize_path(outpath,'output')
    os.makedirs(outpath,exist_ok=True)
    sub_fasta_path = os.path.join(outpath, 'subfastas')
    os.makedirs(sub_fasta_path,exist_ok=True)
    if getattr(sys, 'frozen', False):  
        bundle_dir = sys._MEIPASS  
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))  
    print(bundle_dir)
    print('Preparing data')
    seq_num,seqfileList = split_fasta(inpath, sub_fasta_path, chunk_size=chunk_size)
    print("The number of sequences involved in the alignment is:",seq_num)
    final_graph, hierarchy, subgraph_num = graph_construction(outpath,seqfileList,build_fragment_graph,merge_graph,Tracing_merge,threads=min(80,threads),fragmentLength=int(fra))
    return final_graph, hierarchy, subgraph_num 

