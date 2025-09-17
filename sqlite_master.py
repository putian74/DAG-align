#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import pandas
from DAG_tools import get_first_number,get_second_number,save_numbers
class sql_master():
    def __init__(self,graph,db="",mode = 'build',dbidList=''):
        self.conn = sqlite3.connect(":memory:")
        cursor = self.conn.cursor()
        for setname,arrays in dbidList:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS vset{} (
                node_id INTEGER PRIMARY KEY,
                virus BLOB NOT NULL
            )
            '''.format(setname))
            self.conn.commit()
            binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]
            df = pandas.DataFrame(binary_arrays, columns=['node_id','virus'])
            df.to_sql('vset'+str(setname), con=self.conn, if_exists='append', index=False)
            self.conn.commit()
            sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(setname)
            cursor.execute(sql)
            self.conn.commit()
        
    def __enter__(self):
        return self
    def __exit__(self):
        self.conn.close
    def find_virus_sub(self,gid,node_id,first_number_bite_of_ONM=32,all_bite_of_ONM=64,first_number_bite_of_OSM=16,all_bite_of_OSM=32):
        cursor = self.conn.cursor()
        query = 'SELECT virus FROM vset'+str(gid)+' WHERE node_id IN ({})'.format(','.join(['?']*len(node_id)))
        nids = get_second_number(node_id,first_number_bite_of_ONM,all_bite_of_ONM)
        cursor.execute(query, [int(item) for item in nids])
        results=cursor.fetchall()
        virus_list=[]
        allsize=0
        for result in results:
            v = np.frombuffer(result[0],dtype=np.uint64)                  
            allsize+=v.size
            virus_list.append(v)
        virus = np.empty(allsize,dtype=np.uint64)                   
        size_now=0
        for v in virus_list:
            size = v.size
            virus[size_now:size_now+size]=v
            size_now+=size
        gid = np.full_like(virus,gid)
        
        alist = save_numbers(gid,get_first_number(virus,first_number_bite_of_OSM,all_bite_of_OSM),16,32)
        blist = get_second_number(virus,first_number_bite_of_OSM,all_bite_of_OSM)
        VS = save_numbers(alist,blist)
        return VS
    def findSequenceSource(self,nodes,first_number_bite_of_ONM=32,all_bite_of_ONM=64,first_number_bite_of_OSM=32,all_bite_of_OSM=64):
        nodes = np.array(nodes)
        op = self.find_virus_sub
        gid_of_nodes = get_first_number(nodes,first_number_bite_of_ONM,all_bite_of_ONM)
        gidset = set(gid_of_nodes)
        sourcelist = []
        allsize=0
        for gid in gidset:
            query_nodes = nodes[np.where(gid_of_nodes==gid)[0]]
            source = op(gid,query_nodes,first_number_bite_of_ONM,all_bite_of_ONM,first_number_bite_of_OSM,all_bite_of_OSM)
            sourcelist.append(source)
            allsize+=source.shape[0]
        source_matrix = np.full((allsize,),0,dtype=np.uint64)
        size_now = 0
        for v in sourcelist:
            col_size = v.shape[0]             
            source_matrix[size_now:size_now+col_size] = v            
            size_now += col_size

        return source_matrix
