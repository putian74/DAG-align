# -*- coding: utf-8 -*-
from DAG_operator import *
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="DAGbuilder")
    parser.add_argument('input_file', type=str, help='input_file')
    parser.add_argument('output_dir', type=str, help='output_dir')
    parser.add_argument('--segment_length', type=int, default=32, 
                       help='segment_length (default: 32)')
    parser.add_argument('--threads', type=int, default=24,
                       help='threads (default: 24)')
    parser.add_argument('--chunk_size', type=int, default=5000,
                       help='chunk_size (default: 5000)')
    args = parser.parse_args()

    fragmentDAG_mutibuild(inpath=args.input_file,
        outpath=args.output_dir,
        fra=args.segment_length,
        threads=args.threads,
        chunk_size=args.chunk_size)