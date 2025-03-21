# -*- coding: utf-8 -*-
import sqlite3
import numpy as np
import pandas
from DAG_tools import get_first_number,get_second_number,save_numbers
class sql_master():
    def __init__(self,graph,db="",mode = 'memory',dbidList=''):
        if db =="":
            self.conn = sqlite3.connect(":memory:")
            innerlinkset=[]
            for i in graph.linkset.keys():
                innerlinkset.append((i[0],i[1],graph.linkset[i]))
            df = pandas.DataFrame(innerlinkset,columns=['startnode','endnode','weight'])
            cursor = self.conn.cursor()
            sql = '''CREATE TABLE link(startnode int({0}),endnode int({0}),weight int({0}));'''.format(int(graph.fra)+3)
            cursor.execute(sql)
            sql = '''CREATE INDEX sin ON link (startnode);'''
            cursor.execute(sql)
            sql = '''CREATE INDEX ein ON link (endnode);'''
            cursor.execute(sql)
            df.to_sql('link', self.conn, if_exists='append', index=False)
            self.conn.commit()
        elif mode=='memory':
            tmpconn = sqlite3.connect(db)
            self.conn = sqlite3.connect(":memory:")
            with tmpconn, self.conn:
                for line in tmpconn.iterdump():
                    self.conn.execute(line)
        elif mode=='build':
            self.conn = sqlite3.connect(":memory:")
            cursor = self.conn.cursor()
            for setname in dbidList:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS vset{} (
                    node_id INTEGER PRIMARY KEY,
                    virus BLOB NOT NULL
                )
                '''.format(setname))
                self.conn.commit()
                arrays = np.load(db/'{}/osm.npy'.format(setname),allow_pickle=True)
                binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]
                df = pandas.DataFrame(binary_arrays, columns=['node_id','virus'])
                df.to_sql('vset'+str(setname), con=self.conn, if_exists='append', index=False)
                self.conn.commit()
                sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(setname)
                cursor.execute(sql)
                self.conn.commit()
        elif mode=='single':
            self.conn = sqlite3.connect(":memory:")
            cursor = self.conn.cursor()
            setname=1
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS vset{} (
                node_id INTEGER PRIMARY KEY,
                virus BLOB NOT NULL
            )
            '''.format(setname))
            self.conn.commit()
            arrays = np.load(db/'/osm.npy',allow_pickle=True)
            binary_arrays = [[index,array.tobytes()] for index,array in enumerate(arrays)]
            df = pandas.DataFrame(binary_arrays, columns=['node_id','virus'])
            df.to_sql('vset'+str(setname), con=self.conn, if_exists='append', index=False)
            self.conn.commit()
            sql = '''CREATE INDEX nodeid{0} ON vset{0} (node_id);'''.format(setname)
            cursor.execute(sql)
            self.conn.commit()
        elif isinstance(mode,list):
            disk_conn = sqlite3.connect(db)
            disk_cursor = disk_conn.cursor()
            self.conn = sqlite3.connect(':memory:')
            memory_cursor = self.conn .cursor()
            tables_to_copy = mode
            for table in tables_to_copy:
                disk_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                create_table_sql = disk_cursor.fetchone()[0]
                memory_cursor.execute(create_table_sql)
                disk_cursor.execute(f"SELECT * FROM {table}")
                rows = disk_cursor.fetchall()
                column_names = [description[0] for description in disk_cursor.description]
                columns = ', '.join(column_names)
                placeholders = ', '.join(['?'] * len(column_names))
                memory_cursor.executemany(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", rows)
            self.conn.commit()
            disk_conn.close()
            pass
        else:
            self.conn = sqlite3.connect(db)
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
        alist = save_numbers(gid,get_first_number(virus,first_number_bite_of_OSM,all_bite_of_OSM),32,64)
        blist = get_second_number(virus,first_number_bite_of_OSM,all_bite_of_OSM)
        VS=list(zip(alist, blist))
        return VS
    def findSequenceSource(self,nodes,first_number_bite_of_ONM=32,all_bite_of_ONM=64,first_number_bite_of_OSM=32,all_bite_of_OSM=64):
        results = list()
        nodes = np.array(nodes)
        op = self.find_virus_sub
        gid_of_nodes = get_first_number(nodes,first_number_bite_of_ONM,all_bite_of_ONM)
        gidset = set(gid_of_nodes)
        for gid in gidset:
            query_nodes = nodes[np.where(gid_of_nodes==gid)[0]]
            results.extend(op(gid,query_nodes,first_number_bite_of_ONM,all_bite_of_ONM,first_number_bite_of_OSM,all_bite_of_OSM))
        matrix = np.array(results,order='F',dtype=np.uint64).T
        return matrix