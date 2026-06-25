#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import numpy as np
import time
import networkx as nx
import pandas as pd

import graph
import boundary_cut

BASELINE = 'boundarycut'
DATA_PATH = '../data/real/'
DATA_TEST_NAME = ["crime", "HI-II-14", 'digg', 'Email-Enron', 'Gnutella31',
                  'Facebook', 'DBLP', 'Youtube', "Flickr"]
COST_TYPES = ['degree', 'random']

SAVE_ROOT = '../results/real/%s' % BASELINE


def cost_dir(costType):
    return '%s/%s_cost/' % (SAVE_ROOT, costType)


def build_py_graph(data_test, data_test_weight):
    weights = []
    with open(data_test_weight) as f:
        for line in f:
            line = line.strip()
            if line:
                weights.append(float(line))
    n = len(weights)
    A, B = [], []
    with open(data_test) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            A.append(int(parts[0]))
            B.append(int(parts[1]))
    A = np.array(A, dtype=np.int32)
    B = np.array(B, dtype=np.int32)
    W = np.array(weights, dtype=np.double)
    return graph.py_Graph(n, len(A), A, B, W), n


def GetSolution(gcc_threshold=0.01):
    bc = boundary_cut.py_BoundaryCut()
    for costType in COST_TYPES:
        save_dir = cost_dir(costType)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df = pd.DataFrame(np.zeros((1, len(DATA_TEST_NAME))), index=['time'],
                          columns=DATA_TEST_NAME)
        for j, name in enumerate(DATA_TEST_NAME):
            data_test = DATA_PATH + name + '.txt'
            data_test_weight = DATA_PATH + name + '_%s_weight.txt' % costType
            print('Generating %s solution for %s (%s cost)' % (BASELINE, name, costType))
            sys.stdout.flush()
            g, n = build_py_graph(data_test, data_test_weight)
            t1 = time.time()
            solution = bc.getSolution(g, gcc_threshold)
            t2 = time.time()
            sol_time = t2 - t1
            df.iloc[0, j] = sol_time
            result_file = save_dir + name + '.txt'
            with open(result_file, 'w') as f_out:
                for v in solution:
                    f_out.write('%d\n' % v)
            print('  nodes=%d  sol_len=%d  time=%.4fs' % (n, len(solution), sol_time))
            sys.stdout.flush()
        df.to_csv(save_dir + 'solution_%s_time.csv' % costType, encoding='utf-8', index=False)
    print('all %s solutions generated!' % BASELINE)



def main():
    GetSolution(0.01)


if __name__ == '__main__':
    main()
